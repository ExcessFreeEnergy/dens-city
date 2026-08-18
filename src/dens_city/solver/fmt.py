"""
Fundamental Measure Theory (FMT) for Hard-Sphere Fluids in 1D Planar Geometry.
Implements exact Rosenfeld (1989) and White Bear (Roth et al. 2002) excess free energy functionals.

Weight functions for hard sphere of diameter d (radius R = d / 2):
Scalar:
  w3(z) = pi * (R^2 - z^2) * Theta(R - |z|)
  w2(z) = 2 * pi * R * Theta(R - |z|)
  w1(z) = w2(z) / (4 * pi * R)
  w0(z) = w2(z) / (4 * pi * R^2)
Vector:
  wv2(z) = 2 * pi * z * Theta(R - |z|) * z_hat
  wv1(z) = wv2(z) / (4 * pi * R)
"""

import numpy as np


class FundamentalMeasureTheory1D:
    def __init__(self, diameter: float):
        """
        diameter: Hard sphere diameter d (in Angstroms).
        """
        self.d = float(diameter)
        self.R = self.d / 2.0

    def compute_weighted_densities(self, z_coords: np.ndarray, rho: np.ndarray):
        """
        Computes 1D planar Rosenfeld weighted densities n0, n1, n2, n3, nv1, nv2 via convolution.
        """
        dz = z_coords[1] - z_coords[0]
        R = self.R

        # Kernel grid
        k_max = int(np.ceil(R / dz))
        z_k = np.arange(-k_max, k_max + 1) * dz
        mask = np.abs(z_k) <= R

        w3_k = np.zeros_like(z_k)
        w3_k[mask] = np.pi * (R * R - z_k[mask] ** 2)

        w2_k = np.zeros_like(z_k)
        w2_k[mask] = 2.0 * np.pi * R

        w1_k = w2_k / (4.0 * np.pi * R)
        w0_k = w2_k / (4.0 * np.pi * R * R)

        wv2_k = np.zeros_like(z_k)
        wv2_k[mask] = 2.0 * np.pi * z_k[mask]
        wv1_k = wv2_k / (4.0 * np.pi * R)

        # Convolutions
        n3 = np.convolve(rho, w3_k * dz, mode="same")
        n2 = np.convolve(rho, w2_k * dz, mode="same")
        n1 = np.convolve(rho, w1_k * dz, mode="same")
        n0 = np.convolve(rho, w0_k * dz, mode="same")
        nv2 = np.convolve(rho, wv2_k * dz, mode="same")
        nv1 = np.convolve(rho, wv1_k * dz, mode="same")

        return n0, n1, n2, n3, nv1, nv2

    def compute_c1_hs(self, z_coords: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Computes the one-body direct correlation function c1_hs(z) = -delta beta F_ex / delta rho(z)
        using the White Bear / Rosenfeld functional.
        """
        dz = z_coords[1] - z_coords[0]
        R = self.R
        n0, n1, n2, n3, nv1, nv2 = self.compute_weighted_densities(z_coords, rho)

        # White Bear / Rosenfeld free energy density derivatives:
        # Phi_hs = -n0 * ln(1 - n3) + (n1*n2 - nv1*nv2)/(1 - n3) + (n2^3 - 3*n2*nv2^2)/(24*pi*(1 - n3)^2)
        # Prevent division by zero / unphysical packing n3 >= 1
        n3_safe = np.clip(n3, 0.0, 0.999)
        om3 = 1.0 - n3_safe
        om3_sq = om3 * om3

        # Derivatives w.r.t weighted densities
        dPhi_dn0 = -np.log(om3)
        dPhi_dn1 = n2 / om3
        dPhi_dn2 = n1 / om3 + (n2 * n2 - nv2 * nv2) / (8.0 * np.pi * om3_sq)
        dPhi_dn3 = n0 / om3 + (n1 * n2 - nv1 * nv2) / om3_sq + (n2**3 - 3.0 * n2 * nv2**2) / (12.0 * np.pi * om3**3)
        dPhi_dnv1 = -nv2 / om3
        dPhi_dnv2 = -nv1 / om3 - (n2 * nv2) / (4.0 * np.pi * om3_sq)

        # Kernel grid
        k_max = int(np.ceil(R / dz))
        z_k = np.arange(-k_max, k_max + 1) * dz
        mask = np.abs(z_k) <= R

        w3_k = np.zeros_like(z_k)
        w3_k[mask] = np.pi * (R * R - z_k[mask] ** 2)

        w2_k = np.zeros_like(z_k)
        w2_k[mask] = 2.0 * np.pi * R

        w1_k = w2_k / (4.0 * np.pi * R)
        w0_k = w2_k / (4.0 * np.pi * R * R)

        wv2_k = np.zeros_like(z_k)
        wv2_k[mask] = 2.0 * np.pi * z_k[mask]
        wv1_k = wv2_k / (4.0 * np.pi * R)

        # Deconvolution / adjoint integration:
        # c1(z) = - \sum_\alpha \int dz' (dPhi / dn_\alpha)(z') * w_\alpha(z' - z)
        # Note: for symmetric weight functions, w(z' - z) = w(z - z'). For vector weights, wv(z' - z) = -wv(z - z').
        c1 = (
            -np.convolve(dPhi_dn0, w0_k * dz, mode="same")
            - np.convolve(dPhi_dn1, w1_k * dz, mode="same")
            - np.convolve(dPhi_dn2, w2_k * dz, mode="same")
            - np.convolve(dPhi_dn3, w3_k * dz, mode="same")
            + np.convolve(dPhi_dnv1, wv1_k * dz, mode="same")
            + np.convolve(dPhi_dnv2, wv2_k * dz, mode="same")
        )
        return c1
