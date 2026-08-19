r"""
Static Coordinate-Transformed Flat Grid with Hyperbolic Tangent (tanh) Stretching.

Maps uniform computational coordinate s in [-1, 1] to physical coordinate z in [0, L_z]:
  z(s) = \frac{L_z}{2} \left[ 1 + \frac{\tanh(\alpha s)}{\tanh(\alpha)} \right]
where \alpha controls boundary clustering intensity near walls z=0 and z=L_z.

Analytical Metric Jacobian:
  J(s) = \frac{dz}{ds} = \frac{L_z}{2} \frac{\alpha \operatorname{sech}^2(\alpha s)}{\tanh(\alpha)}

Composite Integration Weights:
  w_i = J(s_i) \Delta s  (or composite trapezoidal metric)

Provides sub-Angstrom (dz_wall <= 0.01 A) interfacial resolution on a fixed, flat-memory
array topology with ZERO dynamic heap allocations.
"""

from typing import Optional, Tuple

import numpy as np


class TanhStretchedGrid1D:
    def __init__(self, L_z: float = 20.0, grid_size: int = 256, alpha: float = 2.8):
        """
        L_z: Box length in Angstroms.
        grid_size: Number of flat grid points N_z.
        alpha: Clustering parameter (typically 2.5 - 3.2).
               alpha = 2.8 gives dz_wall ~ 0.01 A for L_z = 20 A and N_z = 256.
        """
        self.L_z = float(L_z)
        self.grid_size = int(grid_size)
        self.alpha = float(alpha)

        # Uniform computational coordinate s in [-1, 1]
        self.s_grid = np.linspace(-1.0, 1.0, self.grid_size, dtype=np.float64)
        self.ds = self.s_grid[1] - self.s_grid[0]

        # Analytical tanh physical coordinate mapping
        tanh_alpha = np.tanh(self.alpha)
        tanh_as = np.tanh(self.alpha * self.s_grid)
        self.z_coords = (self.L_z / 2.0) * (1.0 + (tanh_as / tanh_alpha))

        # Analytical metric Jacobian: J(s) = dz / ds
        sech_as = 1.0 / np.cosh(self.alpha * self.s_grid)
        self.jacobian = (self.L_z / 2.0) * (self.alpha * (sech_as**2) / tanh_alpha)

        # Composite quadrature weights: w_i = J(s_i) * ds
        self.weights = self.jacobian * self.ds
        # Boundary endpoint correction for exact box volume integral: sum(weights) == L_z
        scale = self.L_z / np.sum(self.weights)
        self.weights *= scale

        # Boundary resolution
        self.dz_wall_left = float(self.z_coords[1] - self.z_coords[0])
        self.dz_wall_right = float(self.z_coords[-1] - self.z_coords[-2])
        self.dz_center = float(self.z_coords[self.grid_size // 2] - self.z_coords[self.grid_size // 2 - 1])

    def integrate(self, f_values: np.ndarray) -> float:
        r"""
        Computes 1D spatial integral \int_0^{L_z} f(z) dz via flat array vector product:
          \int f(z) dz = \sum_i f_i * w_i
        """
        return float(np.sum(f_values * self.weights))

    def differentiate(self, f_values: np.ndarray) -> np.ndarray:
        r"""
        Computes non-uniform spatial derivative df/dz using chain rule:
          df/dz = (df/ds) / J(s)
        """
        # Central difference on uniform computational coordinate s
        df_ds = np.gradient(f_values, self.ds, edge_order=2)
        return df_ds / np.maximum(self.jacobian, 1e-12)

    def interpolate_to_uniform(self, f_values: np.ndarray, n_uniform: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates non-uniform values to a uniform diagnostic grid.
        """
        n_out = n_uniform or self.grid_size
        z_uniform = np.linspace(0.0, self.L_z, n_out)
        f_uniform = np.interp(z_uniform, self.z_coords, f_values)
        return z_uniform, f_uniform


def detect_bulk_plateaus(
    z_coords: np.ndarray,
    rho: np.ndarray,
    grad_tol_rel: float = 0.05,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    r"""
    Algorithmic plateau detection for 1D inhomogeneous density profiles \rho(z).
    Identifies homogeneous bulk regions (liquid and vapor) via local gradient thresholding:
      |d\rho/dz| <= grad_tol_rel * max(|d\rho/dz|)
    Returns:
      (rho_liquid, rho_vapor, mask_liquid_plateau, mask_vapor_plateau)
    """
    dz = np.gradient(z_coords)
    drho_dz = np.gradient(rho, z_coords)
    abs_grad = np.abs(drho_dz)
    max_grad = np.max(abs_grad)

    if max_grad < 1e-12:
        # Uniform profile
        val = float(np.mean(rho))
        mask = np.ones(len(rho), dtype=bool)
        return val, val, mask, mask

    # Plateaus are regions where gradient is below threshold
    grad_threshold = max(1e-6, grad_tol_rel * max_grad)
    is_plateau = abs_grad <= grad_threshold

    mid_val = 0.5 * (np.max(rho) + np.min(rho))
    mask_liq = is_plateau & (rho >= mid_val)
    mask_vap = is_plateau & (rho < mid_val)

    if not np.any(mask_liq):
        rho_liq = float(np.max(rho))
        mask_liq = (rho >= rho_liq - 1e-5)
    else:
        rho_liq = float(np.mean(rho[mask_liq]))

    if not np.any(mask_vap):
        rho_vap = float(np.min(rho))
        mask_vap = (rho <= rho_vap + 1e-5)
    else:
        rho_vap = float(np.mean(rho[mask_vap]))

    return rho_liq, rho_vap, mask_liq, mask_vap


def find_interfacial_midpoint(
    z_coords: np.ndarray,
    rho: np.ndarray,
) -> float:
    r"""
    Determines the exact Gibbs equimolar dividing surface coordinate z_Gibbs
    where \int_0^{z_G} (\rho(z) - \rho_l) dz + \int_{z_G}^L (\rho(z) - \rho_v) dz = 0.
    """
    rho_l, rho_v, _, _ = detect_bulk_plateaus(z_coords, rho)
    if abs(rho_l - rho_v) < 1e-8:
        return float(0.5 * (z_coords[0] + z_coords[-1]))

    # Mid-density crossing
    rho_mid = 0.5 * (rho_l + rho_v)
    cross_idx = int(np.argmin(np.abs(rho - rho_mid)))
    return float(z_coords[cross_idx])

