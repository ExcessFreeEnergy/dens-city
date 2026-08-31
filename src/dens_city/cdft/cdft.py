"""
Pure tinygrad Classical Density Functional Theory (cDFT) solver.
Implements variational grand free energy minimization using JIT compilation,
BEAM-searchable kernels, exponential re-parameterization, physical steric masking,
and exact Irving-Kirkwood mechanical virial observables.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from tinygrad import GlobalCounters, Tensor, TinyJit, dtypes, nn
from tinygrad.helpers import getenv, trange

from dens_city.cdft.kernels import KernelBuilder

if TYPE_CHECKING:
    from dens_city.utils.materials import Material, MolecularBatch


class TinyCDFT:
    """
    Variational Classical Density Functional Theory Solver in pure tinygrad.
    Finds the equilibrium density profile rho*(r) that minimizes the Grand Potential Omega[rho].
    """

    def __init__(
        self,
        material: Material,
        n_grid: int = 128,
        slit_width_a: Optional[float] = None,
        temperature_k: Optional[float] = None,
        bulk_density_a3: Optional[float] = None,
        learning_rate: float = 0.01,
        wall_sigma: Optional[float] = None,
        wall_epsilon_k: float = 50.0,
    ):
        self.material = material
        self.n_grid = n_grid
        sigma = material.effective_sigma
        eps_k = material.effective_epsilon_k

        # Slit pore dimension scales with molecular diameter if not explicitly provided
        self.slit_width_a = slit_width_a if slit_width_a is not None else 12.0 * sigma
        self.dz_val = self.slit_width_a / n_grid
        self.dz = Tensor(self.dz_val, dtype=dtypes.float32).realize()
        self.temp_val = temperature_k if temperature_k is not None else material.temperature_k
        self.temperature_k = Tensor(self.temp_val, dtype=dtypes.float32).realize()
        self.bulk_rho_val = bulk_density_a3 if bulk_density_a3 is not None else material.bulk_density_a3
        self.bulk_density = Tensor(self.bulk_rho_val, dtype=dtypes.float32).realize()

        # 1. Build anti-aliased spatial kernels and realize them on device
        self.fmt_kernels = {
            k: v.realize() if isinstance(v, Tensor) else v
            for k, v in KernelBuilder.build_fmt_planar_kernels(sigma, self.dz_val).items()
        }
        self.fmt_pad = self.fmt_kernels["pad"]

        # WCA attractive dispersion kernel (scale-invariant cutoff r_cut = 5.0 * sigma)
        att_kern_raw, self.att_pad = KernelBuilder.build_wca_attraction_kernel(
            sigma=sigma, epsilon_k=eps_k, dz=self.dz_val
        )
        self.att_kernel = att_kern_raw.realize()

        # Confining slit external wall potential with exact physical boundary divergence
        # Lorentz-Berthelot collision diameter incorporates the fluid's effective size
        wall_sig = wall_sigma if wall_sigma is not None else 3.4
        v_ext_raw = (
            KernelBuilder.build_slit_wall_potential(
                n_grid=self.n_grid,
                dz=self.dz_val,
                fluid_sigma=sigma,
                wall_sigma=wall_sig,
                wall_epsilon_k=wall_epsilon_k,
            )
            / self.temp_val
        )  # in units of k_B * T
        self.v_ext = v_ext_raw.realize()

        # 2. Compute exact discrete excess chemical potential so grad(Omega) == 0 at rho = rho_bulk
        eta = (math.pi / 6.0) * self.bulk_rho_val * (sigma**3)
        one_minus_eta = max(1e-12, 1.0 - eta)
        mu_fmt = -math.log(one_minus_eta) + (eta * (14.0 - 13.0 * eta + 5.0 * (eta**2))) / (2.0 * (one_minus_eta**3))
        att_kernel_sum = float(self.att_kernel.numpy().sum()) * self.dz_val
        mu_att = (self.bulk_rho_val * att_kernel_sum) / self.temp_val
        mu_ex_val = mu_fmt + mu_att
        self.mu_ex = Tensor(mu_ex_val, dtype=dtypes.float32).realize()

        # 3. Dynamic Boltzmann Initialization: psi_0(z) = -beta * V_ext(z)
        # Allows psi_0 > 0 for attractive wells (V_ext < 0) bounded by physical close packing eta_max = 0.60
        psi_max = math.log(0.60 / max(1e-4, eta))
        v_ext_np = self.v_ext.numpy().reshape(self.n_grid)
        psi_init_vals = np.clip(-v_ext_np, -50.0, psi_max)
        self.psi = Tensor(psi_init_vals).reshape(1, 1, self.n_grid, 1).contiguous().realize()
        self.psi.requires_grad = True

        # 4. Setup optimizer and per-instance TinyJit compilation
        opt_type = nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam
        self.opt = opt_type([self.psi], lr=learning_rate)
        self.train_step = TinyJit(self._train_step)

    def compute_density(self) -> Tensor:
        """
        Forward transformation mapping latent field psi to positive density profile rho.
        Guarantees rho > 0 strictly throughout the domain with full autograd differentiability.
        """
        return (self.psi).exp() * self.bulk_density

    def compute_electrostatic_potential(self, charge_density: Tensor, dielectric_constant: float = 1.0) -> Tensor:
        r"""
        Solves 1D Poisson boundary value problem using exact Dirichlet Green's matrix G:
        \phi(z) = G * \rho_q(z) where \phi(0) = \phi(L_z) = 0.
        """
        g_matrix = KernelBuilder.build_coulomb_1d_greens_matrix(
            self.n_grid, self.dz_val, dielectric_constant=dielectric_constant
        ).realize()
        g_mat_2d = g_matrix.reshape(self.n_grid, self.n_grid)
        rho_q_vec = charge_density.reshape(self.n_grid, 1)
        phi_vec = g_mat_2d.matmul(rho_q_vec)
        return phi_vec.reshape(1, 1, self.n_grid, 1)

    def grand_potential(self) -> Tensor:
        r"""
        Evaluates the grand potential functional Omega[rho] / (k_B * T):
        Omega = F_ideal + F_ext + F_FMT + F_att - mu \int rho dz
        Executes continuous autograd without gradient-killing boolean masks.
        """
        rho = self.compute_density()

        # 1. Ideal gas free energy (log-free formulation: rho * psi - (rho - rho_b))
        f_ideal = (rho * self.psi - (rho - self.bulk_density)).sum() * self.dz

        # 2. External potential energy: rho * V_ext
        f_ext = (rho * self.v_ext).sum() * self.dz

        # 3. Rosenfeld FMT Hard-Sphere Excess Free Energy
        n3 = rho.conv2d(self.fmt_kernels["w3"], padding=(self.fmt_pad, 0))
        n2 = rho.conv2d(self.fmt_kernels["w2"], padding=(self.fmt_pad, 0))
        n1 = rho.conv2d(self.fmt_kernels["w1"], padding=(self.fmt_pad, 0))
        n0 = rho.conv2d(self.fmt_kernels["w0"], padding=(self.fmt_pad, 0))
        nv2 = rho.conv2d(self.fmt_kernels["wv2"], padding=(self.fmt_pad, 0))
        nv1 = rho.conv2d(self.fmt_kernels["wv1"], padding=(self.fmt_pad, 0))

        n3_star = n3.minimum(1.0 - 1e-5)
        one_minus_n3 = 1.0 - n3_star
        phi_fmt = (
            -n0 * one_minus_n3.log()
            + (n1 * n2 - nv1 * nv2) / one_minus_n3
            + (n2 * n2 * n2 - 3.0 * n2 * (nv2 * nv2)) / (24.0 * math.pi * (one_minus_n3 * one_minus_n3))
        )
        f_fmt = phi_fmt.sum() * self.dz

        # 4. Attractive dispersion excess free energy via convolution
        att_conv = rho.conv2d(self.att_kernel, padding=(self.att_pad, 0))
        f_att = 0.5 * (rho * att_conv).sum() * (self.dz * self.dz) / self.temperature_k

        # 5. Grand canonical excess chemical potential term
        f_mu = -(rho * self.mu_ex).sum() * self.dz

        return f_ideal + f_ext + f_fmt + f_att + f_mu

    def _train_step(self) -> Tensor:
        """
        Pure JIT-compiled optimization step.
        Computes forward free energy, executes reverse-mode autograd, and realizes optimizer step.
        """
        Tensor.training = True
        self.opt.zero_grad()
        loss = self.grand_potential().backward()
        return loss.realize(*self.opt.schedule_step())

    def solve(self, steps: int = 300, verbose: bool = True) -> Dict[str, Any]:
        """Runs the variational free energy minimization loop."""
        losses = []
        iterator = trange(steps) if verbose else range(steps)

        for i in iterator:
            GlobalCounters.reset()
            loss = self.train_step()
            loss_val = loss.item()
            losses.append(loss_val)

            if verbose and hasattr(iterator, "set_description") and (i % 20 == 0 or i == steps - 1):
                p_wall = self.get_wall_contact_pressure()
                gamma = self.get_excess_adsorption()
                iterator.set_description(
                    f"[{self.material.name}] Loss: {loss_val:8.2f} | P_wall: {p_wall:6.2f} bar | Gamma: {gamma:6.4f}"
                )

        rho_final = self.get_density_profile()
        return {
            "material": self.material.name,
            "loss_history": losses,
            "final_loss": losses[-1] if losses else 0.0,
            "rho": rho_final,
            "z_coords": np.linspace(0.5 * self.dz_val, self.slit_width_a - 0.5 * self.dz_val, self.n_grid),
            "wall_pressure_bar": self.get_wall_contact_pressure(),
            "excess_adsorption": self.get_excess_adsorption(),
            "peak_density": float(np.max(rho_final)),
            "bulk_density": self.bulk_rho_val,
        }

    def get_density_profile(self) -> np.ndarray:
        """Extracts the realized 1D density profile as a numpy array."""
        rho_tensor = self.compute_density().reshape(self.n_grid)
        return rho_tensor.numpy().copy()

    def get_wall_contact_pressure(self) -> float:
        r"""
        Calculates the exact statistical mechanical wall pressure via the
        Irving-Kirkwood mechanical virial force balance integral:
        P_wall = - \int_0^{z_bulk} \rho(z) \frac{d V_ext(z)}{dz} dz
        where z_bulk is dynamically detected where |\nabla V_ext(z)| < \epsilon_tol.
        Converted to bar (1 bar = 1e5 Pa).
        """
        rho_arr = self.get_density_profile()
        v_ext_arr = self.v_ext.numpy().reshape(self.n_grid) * self.temp_val

        # Numerical derivative of external wall potential: dV_ext / dz (in K / Å)
        dv_dz = np.gradient(v_ext_arr, self.dz_val)

        # Dynamically identify the wall interaction domain up to the bulk plateau
        mid = self.n_grid // 2
        min_grad_idx = int(np.argmin(dv_dz[:mid]))
        tol = 1.0  # K / Å
        plateau_candidates = np.where(np.abs(dv_dz[min_grad_idx:mid]) < tol)[0]
        bulk_cutoff_idx = (min_grad_idx + int(plateau_candidates[0])) if len(plateau_candidates) > 0 else mid

        # Integrate virial force density over the dynamically detected wall interaction domain
        f_integral = -float(
            np.sum(rho_arr[min_grad_idx:bulk_cutoff_idx] * dv_dz[min_grad_idx:bulk_cutoff_idx]) * self.dz_val
        )
        p_virial_bar = f_integral * (1e30 * 1.380649e-23 * 1e-5)

        return p_virial_bar

    def get_contact_ratio(self) -> float:
        r"""
        Calculates the exact dimensionless bulk-normalized contact ratio:
        R_contact = f_virial / rho_bulk = P_wall / (rho_bulk * k_B T)
        """
        rho_arr = self.get_density_profile()
        v_ext_arr = self.v_ext.numpy().reshape(self.n_grid)  # in units of k_B * T
        dv_dz = np.gradient(v_ext_arr, self.dz_val)

        mid = self.n_grid // 2
        min_grad_idx = int(np.argmin(dv_dz[:mid]))
        tol = 1.0 / max(1.0, self.temp_val)
        plateau_candidates = np.where(np.abs(dv_dz[min_grad_idx:mid]) < tol)[0]
        bulk_cutoff_idx = (min_grad_idx + int(plateau_candidates[0])) if len(plateau_candidates) > 0 else mid

        f_virial = -float(
            np.sum(rho_arr[min_grad_idx:bulk_cutoff_idx] * dv_dz[min_grad_idx:bulk_cutoff_idx]) * self.dz_val
        )
        return float(f_virial / max(1e-6, self.bulk_rho_val))

    def get_excess_adsorption(self) -> float:
        r"""
        Computes exact statistical mechanical Gibbs excess pore adsorption:
        \Gamma_excess = \int_0^L (\rho(z) - \rho_bulk) dz
        relative to the theoretical reservoir bulk density.
        """
        rho_arr = self.get_density_profile()
        return float(np.sum(rho_arr - self.bulk_rho_val) * self.dz_val)


class BatchedTinyCDFT:
    """
    Batched Classical Density Functional Theory Solver in pure tinygrad.
    Minimizes the Grand Potential functional Omega[psi] simultaneously for B <= 32 distinct
    materials in a single JIT graph execution via grouped planar convolutions.
    """

    def __init__(
        self,
        batch: MolecularBatch,
        n_grid: int = 128,
        learning_rate: float = 0.02,
        wall_sigma: float = 3.4,
        wall_epsilon_k: float = 50.0,
    ):
        self.batch = batch
        self.batch_size = batch.batch_size
        self.n_grid = n_grid
        self.materials = batch.materials

        dz_list = []
        temp_list = []
        rho_bulk_list = []
        mu_ex_list = []
        psi_init_list = []
        v_ext_list = []
        slit_widths = []

        # 1. Per-material parameter extraction and kernel building
        raw_fmt_w3, raw_fmt_w2, raw_fmt_w1, raw_fmt_w0, raw_fmt_wv2, raw_fmt_wv1 = [], [], [], [], [], []
        raw_att_kernels = []
        fmt_k_sizes = []
        att_k_sizes = []

        for b in range(self.batch_size):
            mat = self.materials[b]
            is_active = (mat is not None) or (
                hasattr(batch, "molecule_mask") and float(batch.molecule_mask.numpy()[b]) > 0.5
            )
            if is_active:
                if mat is not None:
                    p_data = getattr(mat, "precomputed_cdft_data", None)
                    if p_data is not None:
                        slit_w_b = p_data["slit_w"]
                        dz_b = p_data["dz"]
                        temp_b = mat.temperature_k
                        rho_b = mat.bulk_density_a3
                        sigma_b = mat.effective_sigma
                        w3_arr = p_data["fmt_w3"]
                        w2_arr = p_data["fmt_w2"]
                        w1_arr = p_data["fmt_w1"]
                        w0_arr = p_data["fmt_w0"]
                        wv2_arr = p_data["fmt_wv2"]
                        wv1_arr = p_data["fmt_wv1"]
                        att_arr = p_data["att_arr"]
                        v_ext_np = p_data["v_ext_np"]
                    else:
                        sigma_b = mat.effective_sigma
                        eps_b = mat.effective_epsilon_k
                        slit_w_b = max(40.0, 12.0 * sigma_b)
                        dz_b = slit_w_b / n_grid
                        temp_b = mat.temperature_k
                        rho_b = mat.bulk_density_a3

                        fmt_dict = KernelBuilder.build_fmt_planar_kernels_np(sigma_b, dz_b)
                        att_arr, att_pad = KernelBuilder.build_wca_attraction_kernel_np(sigma_b, eps_b, dz_b)
                        v_ext_np = (
                            KernelBuilder.build_slit_wall_potential_np(
                                n_grid=n_grid,
                                dz=dz_b,
                                fluid_sigma=sigma_b,
                                wall_sigma=wall_sigma,
                                wall_epsilon_k=wall_epsilon_k,
                            )
                            / temp_b
                        )
                        w3_arr = fmt_dict["w3"]
                        w2_arr = fmt_dict["w2"]
                        w1_arr = fmt_dict["w1"]
                        w0_arr = fmt_dict["w0"]
                        wv2_arr = fmt_dict["wv2"]
                        wv1_arr = fmt_dict["wv1"]
                else:
                    cond_np = (
                        batch.conditioning.numpy()[b]
                        if hasattr(batch, "conditioning") and batch.conditioning is not None
                        else np.array([3.4, 120.0, 300.0, 0.02, -8.0], dtype=np.float32)
                    )
                    sigma_b = float(cond_np[0]) if cond_np[0] > 0.0 else 3.4
                    eps_b = float(cond_np[1]) if cond_np[1] > 0.0 else 120.0
                    temp_b = float(batch.temperature_k.numpy()[b]) if hasattr(batch, "temperature_k") else 300.0
                    rho_b = (
                        float(batch.bulk_density_a3.numpy()[b])
                        if hasattr(batch, "bulk_density_a3")
                        else float(cond_np[3])
                    )
                    slit_w_b = (
                        float(batch.slit_width_a.numpy()[b])
                        if hasattr(batch, "slit_width_a")
                        else max(40.0, 12.0 * sigma_b)
                    )
                    dz_b = slit_w_b / n_grid
                    fmt_dict = KernelBuilder.build_fmt_planar_kernels_np(sigma_b, dz_b)
                    att_arr, att_pad = KernelBuilder.build_wca_attraction_kernel_np(sigma_b, eps_b, dz_b)
                    v_ext_np = (
                        KernelBuilder.build_slit_wall_potential_np(
                            n_grid=n_grid,
                            dz=dz_b,
                            fluid_sigma=sigma_b,
                            wall_sigma=wall_sigma,
                            wall_epsilon_k=wall_epsilon_k,
                        )
                        / temp_b
                    )
                    w3_arr = fmt_dict["w3"]
                    w2_arr = fmt_dict["w2"]
                    w1_arr = fmt_dict["w1"]
                    w0_arr = fmt_dict["w0"]
                    wv2_arr = fmt_dict["wv2"]
                    wv1_arr = fmt_dict["wv1"]

                eta_b = (math.pi / 6.0) * rho_b * (sigma_b**3)
                one_minus_eta = max(1e-12, 1.0 - eta_b)
                mu_fmt = -math.log(one_minus_eta) + (eta_b * (14.0 - 13.0 * eta_b + 5.0 * (eta_b**2))) / (
                    2.0 * (one_minus_eta**3)
                )
                att_sum = float(att_arr.sum()) * dz_b
                mu_att = (rho_b * att_sum) / temp_b
                mu_ex_b = mu_fmt + mu_att

                psi_max = math.log(0.60 / max(1e-4, eta_b))
                psi_init_b = np.clip(-v_ext_np, -50.0, psi_max)
            else:
                sigma_b = 3.4
                eps_b = 100.0
                slit_w_b = 40.0
                dz_b = slit_w_b / n_grid
                temp_b = 300.0
                rho_b = 0.0
                fmt_dict = KernelBuilder.build_fmt_planar_kernels_np(sigma_b, dz_b)
                att_arr, att_pad = KernelBuilder.build_wca_attraction_kernel_np(sigma_b, eps_b, dz_b)
                v_ext_np = np.zeros(n_grid, dtype=np.float32)
                mu_ex_b = 0.0
                psi_init_b = np.zeros(n_grid, dtype=np.float32)
                w3_arr = fmt_dict["w3"]
                w2_arr = fmt_dict["w2"]
                w1_arr = fmt_dict["w1"]
                w0_arr = fmt_dict["w0"]
                wv2_arr = fmt_dict["wv2"]
                wv1_arr = fmt_dict["wv1"]

            slit_widths.append(slit_w_b)
            dz_list.append(dz_b)
            temp_list.append(temp_b)
            rho_bulk_list.append(rho_b)
            mu_ex_list.append(mu_ex_b)
            psi_init_list.append(psi_init_b)
            v_ext_list.append(v_ext_np)

            raw_fmt_w3.append(w3_arr)
            raw_fmt_w2.append(w2_arr)
            raw_fmt_w1.append(w1_arr)
            raw_fmt_w0.append(w0_arr)
            raw_fmt_wv2.append(wv2_arr)
            raw_fmt_wv1.append(wv1_arr)
            fmt_k_sizes.append(len(w3_arr))

            raw_att_kernels.append(att_arr)
            att_k_sizes.append(len(att_arr))

        self.slit_widths = slit_widths
        self.dz_vals = dz_list
        self.temp_vals = temp_list
        self.rho_bulk_vals = rho_bulk_list

        # 2. Build grouped convolution kernels with symmetric center-padding
        max_fmt_k = max(fmt_k_sizes)
        self.fmt_pad = (max_fmt_k - 1) // 2

        def stack_grouped_kernels(kernel_arrays: List[np.ndarray], target_k: int) -> Tensor:
            stacked = np.zeros((self.batch_size, 1, target_k, 1), dtype=np.float32)
            for b, arr in enumerate(kernel_arrays):
                k_len = len(arr)
                start = (target_k - k_len) // 2
                stacked[b, 0, start : start + k_len, 0] = arr
            return Tensor(stacked, dtype=dtypes.float32).realize()

        self.fmt_w3 = stack_grouped_kernels(raw_fmt_w3, max_fmt_k)
        self.fmt_w2 = stack_grouped_kernels(raw_fmt_w2, max_fmt_k)
        self.fmt_w1 = stack_grouped_kernels(raw_fmt_w1, max_fmt_k)
        self.fmt_w0 = stack_grouped_kernels(raw_fmt_w0, max_fmt_k)
        self.fmt_wv2 = stack_grouped_kernels(raw_fmt_wv2, max_fmt_k)
        self.fmt_wv1 = stack_grouped_kernels(raw_fmt_wv1, max_fmt_k)

        max_att_k = max(att_k_sizes)
        self.att_pad = (max_att_k - 1) // 2
        self.att_kernel = stack_grouped_kernels(raw_att_kernels, max_att_k)

        # 3. Stacked static field buffers: shape (1, B, N_grid, 1) and (1, B, 1, 1)
        self.dz = Tensor(dz_list, dtype=dtypes.float32).reshape(1, self.batch_size, 1, 1).realize()
        self.temperature_k = Tensor(temp_list, dtype=dtypes.float32).reshape(1, self.batch_size, 1, 1).realize()
        self.bulk_density = Tensor(rho_bulk_list, dtype=dtypes.float32).reshape(1, self.batch_size, 1, 1).realize()
        self.mu_ex = Tensor(mu_ex_list, dtype=dtypes.float32).reshape(1, self.batch_size, 1, 1).realize()
        self.molecule_mask = (
            batch.molecule_mask.reshape(1, self.batch_size, 1, 1).realize()
            if hasattr(batch, "molecule_mask")
            else Tensor.ones(1, self.batch_size, 1, 1)
        )
        self.v_ext = (
            Tensor(np.array(v_ext_list, dtype=np.float32)).reshape(1, self.batch_size, n_grid, 1).contiguous().realize()
        )

        psi_init_np = np.array(psi_init_list, dtype=np.float32).reshape(1, self.batch_size, n_grid, 1)
        self.psi = Tensor(psi_init_np).contiguous().realize()
        self.psi.requires_grad = True

        opt_type = nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam
        self.opt = opt_type([self.psi], lr=learning_rate)
        self.train_step = TinyJit(self._train_step)

    def compute_density(self) -> Tensor:
        """Computes positive density field rho for all batch slots: (1, B, N_grid, 1)."""
        return (self.psi).exp() * self.bulk_density

    def grand_potential(self) -> Tensor:
        """
        Evaluates grand potential functional for all B fluids simultaneously via grouped conv2d.
        """
        rho = self.compute_density()  # (1, B, N_grid, 1)

        # 1. Ideal gas free energy
        f_ideal = (rho * self.psi - (rho - self.bulk_density)).sum(axis=2, keepdim=True) * self.dz

        # 2. External wall potential energy
        f_ext = (rho * self.v_ext).sum(axis=2, keepdim=True) * self.dz

        # 3. Batched Rosenfeld FMT Hard-Sphere Excess via Grouped Convolutions
        n3 = rho.conv2d(self.fmt_w3, groups=self.batch_size, padding=(self.fmt_pad, 0))
        n2 = rho.conv2d(self.fmt_w2, groups=self.batch_size, padding=(self.fmt_pad, 0))
        n1 = rho.conv2d(self.fmt_w1, groups=self.batch_size, padding=(self.fmt_pad, 0))
        n0 = rho.conv2d(self.fmt_w0, groups=self.batch_size, padding=(self.fmt_pad, 0))
        nv2 = rho.conv2d(self.fmt_wv2, groups=self.batch_size, padding=(self.fmt_pad, 0))
        nv1 = rho.conv2d(self.fmt_wv1, groups=self.batch_size, padding=(self.fmt_pad, 0))

        n3_star = n3.minimum(1.0 - 1e-5)
        one_minus_n3 = 1.0 - n3_star
        phi_fmt = (
            -n0 * one_minus_n3.log()
            + (n1 * n2 - nv1 * nv2) / one_minus_n3
            + (n2 * n2 * n2 - 3.0 * n2 * (nv2 * nv2)) / (24.0 * math.pi * (one_minus_n3 * one_minus_n3))
        )
        f_fmt = phi_fmt.sum(axis=2, keepdim=True) * self.dz

        # 4. Batched WCA Attractive Dispersion Excess
        att_conv = rho.conv2d(self.att_kernel, groups=self.batch_size, padding=(self.att_pad, 0))
        f_att = 0.5 * (rho * att_conv).sum(axis=2, keepdim=True) * (self.dz * self.dz) / self.temperature_k

        # 5. Excess Chemical Potential Reservoir term
        f_mu = -(rho * self.mu_ex).sum(axis=2, keepdim=True) * self.dz

        omega_b = (f_ideal + f_ext + f_fmt + f_att + f_mu) * self.molecule_mask
        n_active = self.molecule_mask.sum().maximum(1.0)
        return omega_b.sum() / n_active

    def _train_step(self) -> Tensor:
        Tensor.training = True
        self.opt.zero_grad()
        loss = self.grand_potential().backward()
        return loss.realize(*self.opt.schedule_step())

    def solve(self, steps: int = 50, verbose: bool = False) -> List[float]:
        losses = []
        iterator = trange(steps) if verbose else range(steps)
        for _ in iterator:
            GlobalCounters.reset()
            loss = self.train_step()
            losses.append(loss.item())
        return losses

    def get_density_profiles(self) -> List[np.ndarray]:
        rho_all = self.compute_density().reshape(self.batch_size, self.n_grid).numpy()
        return [rho_all[b].copy() for b in range(self.batch_size)]

    def get_wall_contact_pressures(self) -> List[float]:
        rho_profiles = self.get_density_profiles()
        v_ext_all = (self.v_ext * self.temperature_k).reshape(self.batch_size, self.n_grid).numpy()
        pressures = []

        for b in range(self.batch_size):
            is_active = (self.materials[b] is not None) or (
                hasattr(self.batch, "molecule_mask") and float(self.batch.molecule_mask.numpy()[b]) > 0.5
            )
            if not is_active:
                pressures.append(0.0)
                continue
            rho_arr = rho_profiles[b]
            v_ext_arr = v_ext_all[b]
            dz_val = self.dz_vals[b]

            dv_dz = np.gradient(v_ext_arr, dz_val)
            mid = self.n_grid // 2
            min_grad_idx = int(np.argmin(dv_dz[:mid]))
            tol = 1.0
            plateau_candidates = np.where(np.abs(dv_dz[min_grad_idx:mid]) < tol)[0]
            bulk_cutoff_idx = (min_grad_idx + int(plateau_candidates[0])) if len(plateau_candidates) > 0 else mid

            f_integral = -float(
                np.sum(rho_arr[min_grad_idx:bulk_cutoff_idx] * dv_dz[min_grad_idx:bulk_cutoff_idx]) * dz_val
            )
            p_virial_bar = f_integral * (1e30 * 1.380649e-23 * 1e-5)
            pressures.append(p_virial_bar)

        return pressures

    def get_contact_ratios(self) -> List[float]:
        rho_profiles = self.get_density_profiles()
        v_ext_all = self.v_ext.reshape(self.batch_size, self.n_grid).numpy()
        ratios = []

        for b in range(self.batch_size):
            is_active = (self.materials[b] is not None) or (
                hasattr(self.batch, "molecule_mask") and float(self.batch.molecule_mask.numpy()[b]) > 0.5
            )
            if not is_active:
                ratios.append(0.0)
                continue
            rho_arr = rho_profiles[b]
            v_ext_arr = v_ext_all[b]
            dz_val = self.dz_vals[b]
            temp_val = self.temp_vals[b]
            rho_bulk = self.rho_bulk_vals[b]

            dv_dz = np.gradient(v_ext_arr, dz_val)
            mid = self.n_grid // 2
            min_grad_idx = int(np.argmin(dv_dz[:mid]))
            tol = 1.0 / max(1.0, temp_val)
            plateau_candidates = np.where(np.abs(dv_dz[min_grad_idx:mid]) < tol)[0]
            bulk_cutoff_idx = (min_grad_idx + int(plateau_candidates[0])) if len(plateau_candidates) > 0 else mid

            f_integral = -float(
                np.sum(rho_arr[min_grad_idx:bulk_cutoff_idx] * dv_dz[min_grad_idx:bulk_cutoff_idx]) * dz_val
            )
            ratios.append(float(f_integral / max(1e-6, rho_bulk)))

        return ratios

    def get_excess_adsorptions(self) -> List[float]:
        rho_profiles = self.get_density_profiles()
        gammas = []
        for b in range(self.batch_size):
            is_active = (self.materials[b] is not None) or (
                hasattr(self.batch, "molecule_mask") and float(self.batch.molecule_mask.numpy()[b]) > 0.5
            )
            if not is_active:
                gammas.append(0.0)
                continue
            rho_arr = rho_profiles[b]
            gamma = float(np.sum(rho_arr - self.rho_bulk_vals[b]) * self.dz_vals[b])
            gammas.append(gamma)
        return gammas
