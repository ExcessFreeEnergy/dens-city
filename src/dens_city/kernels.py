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
        to_tensor = lambda vals: Tensor(vals).reshape(1, 1, k_size, 1).contiguous()

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
        Analytically integrates 1D planar WCA attractive dispersion potential:
        v_att(z) = \int_{|z|}^{r_cut} 2\pi r v_att(r) dr
        where v_att(r) = -epsilon for r <= r_min, and 4\epsilon [(\sigma/r)^12 - (\sigma/r)^6] for r > r_min.
        Cutoff radius scales with molecular diameter: r_cut = 5.0 * sigma.
        """
        cutoff = r_cut if r_cut is not None else 5.0 * sigma
        r_min = (2.0 ** (1.0 / 6.0)) * sigma
        k_half = int(math.ceil(cutoff / dz)) + 1
        k_size = 2 * k_half + 1

        def int_r_vatt(r: float) -> float:
            r"""Indefinite integral \int 2\pi r v_att(r) dr."""
            if r <= 0.0:
                return 0.0
            if r <= r_min:
                # \int -2\pi \epsilon r dr = -\pi \epsilon r^2
                return -math.pi * epsilon_k * r * r
            else:
                # Base value at r_min
                val_rmin = -math.pi * epsilon_k * (r_min**2)
                # \int_{r_min}^r 8\pi \epsilon [ \sigma^12 / r^11 - \sigma^6 / r^5 ] dr
                # = 8\pi \epsilon [ -\sigma^12 / (10 r^10) + \sigma^6 / (4 r^4) ]
                def anti_deriv(x: float) -> float:
                    return 8.0 * math.pi * epsilon_k * (-(sigma**12) / (10.0 * (x**10)) + (sigma**6) / (4.0 * (x**4)))

                return val_rmin + (anti_deriv(r) - anti_deriv(r_min))

        val_rcut = int_r_vatt(cutoff)

        kernel_vals = []
        for i in range(-k_half, k_half + 1):
            z_center = abs(i * dz)
            z1 = max(0.0, z_center - dz / 2.0)
            z2 = min(cutoff, z_center + dz / 2.0)

            if z1 < cutoff:
                z_mid = (z1 + z2) / 2.0
                v_1d_mid = val_rcut - int_r_vatt(z_mid)
                kernel_vals.append(v_1d_mid)
            else:
                kernel_vals.append(0.0)

        kernel_tensor = Tensor(kernel_vals).reshape(1, 1, k_size, 1).contiguous()
        return kernel_tensor, k_half

    @staticmethod
    def build_slit_wall_potential(
        n_grid: int,
        dz: float,
        wall_sigma: float = 3.4,
        wall_epsilon_k: float = 50.0,
        wall_type: str = "stele93",
    ) -> Tensor:
        """
        Constructs the external confining slit wall potential V_ext(z) with exact physical divergence.
        For steric hard boundary overlap (z <= 0.5 * wall_sigma or z >= L_z - 0.5 * wall_sigma),
        V_ext(z) = 1e6 (exact impenetrable brick wall potential in units of k_B * T).
        """
        l_z = n_grid * dz
        v_vals = []
        prefactor = (2.0 * math.pi * wall_epsilon_k * (wall_sigma**3)) / 3.0
        steric_radius = 0.5 * wall_sigma
        v_wall_inf = 1e6  # Massive physical potential barrier (avoids IEEE 754 0 * inf NaN)

        for i in range(n_grid):
            z = (i + 0.5) * dz
            z_l = z
            z_r = l_z - z

            # Exact steric exclusion boundary
            if z_l <= steric_radius or z_r <= steric_radius:
                v_vals.append(v_wall_inf)
            elif wall_type == "stele93":
                s_l = wall_sigma / z_l
                s_r = wall_sigma / z_r
                v_l = prefactor * ((2.0 / 15.0) * (s_l**9) - (s_l**3))
                v_r = prefactor * ((2.0 / 15.0) * (s_r**9) - (s_r**3))
                v_total = min(v_wall_inf, v_l + v_r)
                v_vals.append(v_total)
            else:
                # Hard wall
                v_vals.append(v_wall_inf if (z_l < wall_sigma or z_r < wall_sigma) else 0.0)

        return Tensor(v_vals).reshape(1, 1, n_grid, 1).contiguous()

    @staticmethod
    def build_coulomb_1d_kernel(n_grid: int, dz: float) -> Tuple[Tensor, int]:
        r"""
        1D Poisson / Coulomb Green's function for electrostatics:
        v_C(z) = -2\pi |z|
        """
        k_half = n_grid
        k_size = 2 * k_half + 1
        kernel_vals = [-2.0 * math.pi * abs(i * dz) for i in range(-k_half, k_half + 1)]
        kernel_tensor = Tensor(kernel_vals).reshape(1, 1, k_size, 1).contiguous()
        return kernel_tensor, k_half
