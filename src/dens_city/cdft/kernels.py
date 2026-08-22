"""
Anti-aliased kernel builders for Classical Density Functional Theory (cDFT).
Computes analytically integrated weight functions, scale-invariant WCA/Lennard-Jones dispersion,
Coulomb Green's functions, and exact steric confining wall potentials.
"""

import math
from typing import Dict, Optional, Tuple

from tinygrad import Tensor


class KernelBuilder:
    """Builds analytically cell-integrated kernels to eliminate grid aliasing and ringing."""

    @staticmethod
    def build_fmt_planar_kernels(sigma: float, dz: float) -> Dict[str, Tensor]:
        """
        Analytically integrates Rosenfeld Fundamental Measure Theory (FMT) 1D planar
        weight functions w_3, w_2, w_1, w_0, w_v2, w_v1 across each grid cell [z - dz/2, z + dz/2].

        Returns:
            Dict containing tinygrad Tensors of shape (1, 1, K, 1) ready for conv2d.
        """
        R = sigma / 2.0
        # Kernel half-width in bins
        k_half = int(math.ceil(R / dz)) + 1
        k_size = 2 * k_half + 1

        w3_vals = []
        w2_vals = []
        wv2_vals = []

        for i in range(-k_half, k_half + 1):
            z_center = i * dz
            z_left = z_center - dz / 2.0
            z_right = z_center + dz / 2.0

            # Clamp integration interval to [-R, R]
            z1 = max(-R, z_left)
            z2 = min(R, z_right)

            if z1 < z2:
                # \int \pi(R^2 - z^2) dz = \pi [ R^2(z2 - z1) - (z2^3 - z1^3)/3 ]
                int_w3 = math.pi * (R * R * (z2 - z1) - (z2**3 - z1**3) / 3.0)
                # \int 2\pi R dz = 2\pi R (z2 - z1)
                int_w2 = 2.0 * math.pi * R * (z2 - z1)
                # \int 2\pi z dz = \pi (z2^2 - z1^2)
                int_wv2 = math.pi * (z2**2 - z1**2)

                w3_vals.append(int_w3 / dz)
                w2_vals.append(int_w2 / dz)
                wv2_vals.append(int_wv2 / dz)
            else:
                w3_vals.append(0.0)
                w2_vals.append(0.0)
                wv2_vals.append(0.0)

        w1_vals = [w / (4.0 * math.pi * R) for w in w2_vals]
        w0_vals = [w / (4.0 * math.pi * R * R) for w in w2_vals]
        wv1_vals = [w / (4.0 * math.pi * R) for w in wv2_vals]

        # Reshape to (1, 1, K, 1) for tinygrad conv2d
        def to_tensor(vals):
            return Tensor(vals).reshape(1, 1, k_size, 1).contiguous()

        return {
            "w3": to_tensor(w3_vals),
            "w2": to_tensor(w2_vals),
            "w1": to_tensor(w1_vals),
            "w0": to_tensor(w0_vals),
            "wv2": to_tensor(wv2_vals),
            "wv1": to_tensor(wv1_vals),
            "pad": k_half,
        }

    @staticmethod
    def build_wca_attraction_kernel(
        sigma: float, epsilon_k: float, dz: float, r_cut: Optional[float] = None
    ) -> Tuple[Tensor, int]:
        r"""
        Analytically cell-integrates 1D planar WCA attractive dispersion potential across each grid bin [z - dz/2, z + dz/2]:
        \bar{v}_att(i) = \frac{1}{dz} \int_{z - dz/2}^{z + dz/2} v_att,1D(z') dz'
        where v_att,1D(z) = \int_{|z|}^{r_cut} 2\pi r v_att(r) dr.
        """
        cutoff = r_cut if r_cut is not None else 5.0 * sigma
        r_min = (2.0 ** (1.0 / 6.0)) * sigma
        k_half = int(math.ceil(cutoff / dz)) + 1
        k_size = 2 * k_half + 1

        # Indefinite integral V(r) = \int_0^r 2\pi r' v_att(r') dr'
        # and secondary anti-derivative W(r) = \int_0^r V(u) du
        def eval_v_and_w(r: float) -> Tuple[float, float]:
            if r <= 0.0:
                return 0.0, 0.0
            if r <= r_min:
                v_val = -math.pi * epsilon_k * (r**2)
                w_val = -(math.pi * epsilon_k / 3.0) * (r**3)
                return v_val, w_val
            else:
                # Core value at r_min
                v_rmin = -math.pi * epsilon_k * (r_min**2)
                w_rmin = -(math.pi * epsilon_k / 3.0) * (r_min**3)

                # Antiderivative of 8\pi \epsilon [ \sigma^12 / u^11 - \sigma^6 / u^5 ]
                # \int [ \sigma^12 / u^11 - \sigma^6 / u^5 ] du = -\sigma^12 / (10 u^10) + \sigma^6 / (4 u^4)
                anti_v_rmin = (
                    8.0 * math.pi * epsilon_k * (-(sigma**12) / (10.0 * (r_min**10)) + (sigma**6) / (4.0 * (r_min**4)))
                )
                c1 = anti_v_rmin - v_rmin

                v_val = 8.0 * math.pi * epsilon_k * (-(sigma**12) / (10.0 * (r**10)) + (sigma**6) / (4.0 * (r**4))) - c1

                # Second antiderivative \int [ -\sigma^12 / (10 u^10) + \sigma^6 / (4 u^4) ] du
                # = \sigma^12 / (90 u^9) - \sigma^6 / (12 u^3)
                anti_w_rmin = (
                    8.0 * math.pi * epsilon_k * ((sigma**12) / (90.0 * (r_min**9)) - (sigma**6) / (12.0 * (r_min**3)))
                )
                anti_w_r = 8.0 * math.pi * epsilon_k * ((sigma**12) / (90.0 * (r**9)) - (sigma**6) / (12.0 * (r**3)))

                w_val = w_rmin - c1 * (r - r_min) + (anti_w_r - anti_w_rmin)
                return v_val, w_val

        v_rcut, _ = eval_v_and_w(cutoff)

        kernel_vals = []
        for i in range(-k_half, k_half + 1):
            if i == 0:
                # Symmetric cell [-dz/2, dz/2]
                z_right = min(cutoff, 0.5 * dz)
                _, w_right = eval_v_and_w(z_right)
                # \int_{-dz/2}^{dz/2} v_1D(z) dz = 2 * [ z_right * V(r_cut) - W(z_right) ]
                int_cell = 2.0 * (z_right * v_rcut - w_right)
                kernel_vals.append(int_cell / dz)
            else:
                z_center = abs(i * dz)
                z1 = max(0.0, z_center - 0.5 * dz)
                z2 = min(cutoff, z_center + 0.5 * dz)

                if z1 < cutoff:
                    _, w1 = eval_v_and_w(z1)
                    _, w2 = eval_v_and_w(z2)
                    int_cell = (z2 - z1) * v_rcut - (w2 - w1)
                    kernel_vals.append(int_cell / dz)
                else:
                    kernel_vals.append(0.0)

        kernel_tensor = Tensor(kernel_vals).reshape(1, 1, k_size, 1).contiguous()
        return kernel_tensor, k_half

    @staticmethod
    def build_slit_wall_potential(
        n_grid: int,
        dz: float,
        fluid_sigma: Optional[float] = None,
        wall_sigma: float = 3.4,
        wall_epsilon_k: float = 50.0,
        wall_type: str = "stele93",
    ) -> Tensor:
        """
        Constructs the external confining slit wall potential V_ext(z) with exact physical divergence.
        Computes Lorentz-Berthelot collision diameter sigma_wf = 0.5 * (wall_sigma + fluid_sigma).
        For steric hard boundary overlap (z <= 0.5 * sigma_wf or z >= L_z - 0.5 * sigma_wf),
        V_ext(z) = 1e6 (exact impenetrable brick wall potential in units of k_B * T).
        """
        l_z = n_grid * dz
        v_vals = []
        f_sig = fluid_sigma if fluid_sigma is not None else wall_sigma
        sigma_wf = 0.5 * (wall_sigma + f_sig)
        prefactor = (2.0 * math.pi * wall_epsilon_k * (sigma_wf**3)) / 3.0
        steric_radius = 0.5 * sigma_wf
        v_wall_inf = 1e6  # Massive physical potential barrier (avoids IEEE 754 0 * inf NaN)

        for i in range(n_grid):
            z = (i + 0.5) * dz
            z_l = z
            z_r = l_z - z

            # Exact steric exclusion boundary
            if z_l <= steric_radius or z_r <= steric_radius:
                v_vals.append(v_wall_inf)
            elif wall_type == "stele93":
                s_l = sigma_wf / z_l
                s_r = sigma_wf / z_r
                v_l = prefactor * ((2.0 / 15.0) * (s_l**9) - (s_l**3))
                v_r = prefactor * ((2.0 / 15.0) * (s_r**9) - (s_r**3))
                v_total = min(v_wall_inf, v_l + v_r)
                v_vals.append(v_total)
            else:
                # Hard wall
                v_vals.append(v_wall_inf if (z_l < sigma_wf or z_r < sigma_wf) else 0.0)

        return Tensor(v_vals).reshape(1, 1, n_grid, 1).contiguous()

    @staticmethod
    def build_coulomb_1d_greens_matrix(n_grid: int, dz: float, dielectric_constant: float = 1.0) -> Tensor:
        r"""
        Constructs the exact 2-point 1D Poisson / Coulomb Green's function matrix G \in R^{N x N}
        for electrostatics in a confined slit pore [0, L_z] with grounded Dirichlet boundary conditions
        \phi(0) = \phi(L_z) = 0:
        G_{ij} = -(4\pi \Delta z / \epsilon L_z) * \min(z_i, z_j) * (L_z - \max(z_i, z_j))

        The electrostatic potential is evaluated via matrix multiplication: \phi = G * \rho_q.
        """
        l_z = n_grid * dz
        pref = -(4.0 * math.pi * dz) / (dielectric_constant * l_z) if l_z > 0 else 0.0
        g_matrix = []
        for i in range(n_grid):
            z_i = (i + 0.5) * dz
            row = []
            for j in range(n_grid):
                z_j = (j + 0.5) * dz
                min_z = min(z_i, z_j)
                max_z = max(z_i, z_j)
                g_val = pref * min_z * (l_z - max_z)
                row.append(g_val)
            g_matrix.append(row)
        return Tensor(g_matrix).reshape(1, 1, n_grid, n_grid).contiguous()

    @staticmethod
    def build_coulomb_1d_kernel(n_grid: int, dz: float, dielectric_constant: float = 1.0) -> Tuple[Tensor, int]:
        """Deprecated alias pointing to Greens matrix solver for backwards compatibility."""
        g_matrix = KernelBuilder.build_coulomb_1d_greens_matrix(n_grid, dz, dielectric_constant)
        return g_matrix, n_grid
