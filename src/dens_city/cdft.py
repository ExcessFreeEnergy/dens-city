"""
Pure tinygrad Classical Density Functional Theory (cDFT) solver.
Implements variational grand free energy minimization using JIT compilation,
BEAM-searchable kernels, exponential re-parameterization, physical steric masking,
and exact Irving-Kirkwood mechanical virial observables.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Context, dtypes
from tinygrad.helpers import getenv, trange

from dens_city.kernels import KernelBuilder
from dens_city.materials import Material, MaterialLoader


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
        v_ext_raw = KernelBuilder.build_slit_wall_potential(
            n_grid=self.n_grid,
            dz=self.dz_val,
            fluid_sigma=sigma,
            wall_sigma=wall_sig,
            wall_epsilon_k=wall_epsilon_k,
        ) / self.temp_val  # in units of k_B * T
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
        opt_type = (
            nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam
        )
        self.opt = opt_type([self.psi], lr=learning_rate)
        self.train_step = TinyJit(self._train_step)

    def compute_density(self) -> Tensor:
        """
        Forward transformation mapping latent field psi to positive density profile rho.
        Guarantees rho > 0 strictly throughout the domain with full autograd differentiability.
        """
        return (self.psi).exp() * self.bulk_density

    def compute_electrostatic_potential(
        self, charge_density: Tensor, dielectric_constant: float = 1.0
    ) -> Tensor:
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
        f_integral = -float(np.sum(rho_arr[min_grad_idx:bulk_cutoff_idx] * dv_dz[min_grad_idx:bulk_cutoff_idx]) * self.dz_val)
        p_virial_bar = f_integral * (1e30 * 1.380649e-23 * 1e-5)

        return p_virial_bar

    def get_excess_adsorption(self) -> float:
        r"""
        Computes exact statistical mechanical Gibbs excess pore adsorption:
        \Gamma_excess = \int_0^L (\rho(z) - \rho_bulk) dz
        relative to the theoretical reservoir bulk density.
        """
        rho_arr = self.get_density_profile()
        return float(np.sum(rho_arr - self.bulk_rho_val) * self.dz_val)

    def ascii_plot(self, width: int = 60, height: int = 15) -> str:
        """Renders an ASCII visualization of the density profile across the slit."""
        rho_arr = self.get_density_profile()
        r_min, r_max = 0.0, float(np.max(rho_arr)) * 1.15
        if r_max <= 0.0:
            r_max = 1.0

        grid = [[" " for _ in range(width)] for _ in range(height)]

        for col in range(width):
            idx = int(col * (self.n_grid - 1) / (width - 1))
            val = rho_arr[idx]
            row = int((val - r_min) / (r_max - r_min) * (height - 1))
            row = min(height - 1, max(0, height - 1 - row))
            grid[row][col] = "█"

        # Bulk density baseline
        bulk_row = int((self.bulk_density - r_min) / (r_max - r_min) * (height - 1))
        bulk_row = min(height - 1, max(0, height - 1 - bulk_row))
        for col in range(width):
            if grid[bulk_row][col] == " ":
                grid[bulk_row][col] = "-"

        lines = [f"=== cDFT Density Profile: {self.material.name} (Max: {r_max:.4f} Å⁻³, Bulk: {self.bulk_density:.4f} Å⁻³) ==="]
        lines.append(f"{r_max:>7.4f} ┌" + "─" * width + "┐")
        for row in grid:
            lines.append("        │" + "".join(row) + "│")
        lines.append(f"{0.0:>7.4f} └" + "─" * width + "┘")
        lines.append(f"       0.0 Å{' ' * (width - 10)}{self.slit_width_a:.1f} Å (Slit Width)")
        return "\n".join(lines)
