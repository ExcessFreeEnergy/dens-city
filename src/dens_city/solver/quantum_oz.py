r"""
Quantum Ornstein-Zernike (OZ) Inversion & Barker-Henderson Quantum Diameter Solver.

Provides:
1. Bidirectional Ornstein-Zernike Fourier transform:
     \hat{c}(k) = \frac{S(k) - 1}{\rho_b S(k)}
     S(k) = \frac{1}{1 - \rho_b \hat{c}(k)}
     c(r) = \frac{1}{2\pi^2 r} \int_0^\infty k \hat{c}(k) \sin(kr) dk
2. Effective quantum Barker-Henderson diameter d_eff(T) for arbitrary quantum potentials v_q(r).
3. Zero-wavenumber limit \hat{c}(k=0) evaluation via spatial volume integration to avoid 0/0 singularities.
"""

from typing import Callable, Optional, Tuple
import numpy as np

from dens_city.solver.quantum_surrogates import apply_hann_window, zbl_repulsive_core

KB = 1.380649e-23


def compute_quantum_barker_henderson_diameter(
    potential_fn: Callable[[np.ndarray], np.ndarray],
    T: float,
    r_min_search: float = 5.0,
    n_points: int = 2000,
    r_core: float = 0.8,
) -> float:
    r"""
    Computes the temperature-dependent Barker-Henderson effective hard-sphere diameter:
      d_eff(T) = \int_0^{r_min} \left[ 1 - \exp\left(-\frac{u_0(r)}{k_B T}\right) \right] dr
    where u_0(r) is the repulsive reference potential (WCA split or shifted-force).
    Includes universal ZBL repulsive core shield for r <= r_core.
    """
    r_grid = np.linspace(1e-4, r_min_search, n_points)
    dr = r_grid[1] - r_grid[0]

    u_raw = potential_fn(r_grid)  # in Kelvin
    # Add ZBL repulsive shield
    u_zbl = zbl_repulsive_core(r_grid, r_core=r_core)
    u_total = u_raw + u_zbl

    # Find first minimum
    min_idx = int(np.argmin(u_total))
    r_min = float(r_grid[min_idx])
    u_min = float(u_total[min_idx])

    # WCA split: u_0(r) = u(r) - u(r_min) for r < r_min
    r_sub = r_grid[:min_idx + 1]
    u_0 = u_total[:min_idx + 1] - u_min
    u_0 = np.maximum(u_0, 0.0)

    beta = 1.0 / max(1e-3, T)
    integrand = 1.0 - np.exp(-np.clip(beta * u_0, 0.0, 100.0))
    d_eff = float(np.sum(integrand) * dr)

    return max(0.5 * r_core, min(d_eff, r_min))


def invert_structure_factor_to_c_hat(
    s_k: np.ndarray,
    rho_bulk: float,
) -> np.ndarray:
    r"""
    Inverts the static structure factor S(k) to the direct correlation Fourier modes \hat{c}(k):
      \hat{c}(k) = \frac{S(k) - 1}{\rho_bulk * S(k)}
    """
    s_safe = np.maximum(s_k, 1e-6)
    c_hat = (s_safe - 1.0) / (rho_bulk * s_safe)
    return c_hat


def compute_s_k_from_c_hat(
    c_hat_k: np.ndarray,
    rho_bulk: float,
) -> np.ndarray:
    r"""
    Computes S(k) from direct correlation Fourier modes \hat{c}(k):
      S(k) = \frac{1}{1 - \rho_bulk * \hat{c}(k)}
    """
    denom = 1.0 - rho_bulk * c_hat_k
    s_k = np.where(denom > 1e-4, 1.0 / denom, 100.0)
    return s_k


def invert_c_hat_to_c_radial(
    k_grid: np.ndarray,
    c_hat_k: np.ndarray,
    r_grid: np.ndarray,
    apply_window: bool = True,
    r_box: Optional[float] = None,
) -> np.ndarray:
    r"""
    Computes the 3D isotropic radial direct correlation function c(r) from \hat{c}(k)
    via continuous spherical Bessel transform:
      c(r) = \frac{1}{2\pi^2 r} \int_0^\infty k \hat{c}(k) \sin(kr) dk
    """
    dk = k_grid[1] - k_grid[0]
    c_r = np.zeros_like(r_grid)

    for i, r in enumerate(r_grid):
        if r < 1e-6:
            # L'Hopital limit at r -> 0: (1 / (2*pi^2)) \int k^2 c_hat(k) dk
            c_r[i] = float(np.sum((k_grid**2) * c_hat_k) * dk / (2.0 * np.pi**2))
        else:
            integrand = k_grid * c_hat_k * np.sin(k_grid * r)
            c_r[i] = float(np.sum(integrand) * dk / (2.0 * (np.pi**2) * r))

    if apply_window:
        box_size = r_box if r_box is not None else float(r_grid[-1])
        c_r = apply_hann_window(c_r, r_grid, box_size)

    return c_r


def compute_c_hat_zero_volume_integral(
    r_grid: np.ndarray,
    c_r: np.ndarray,
) -> float:
    r"""
    Evaluates \hat{c}(k=0) strictly via 3D spherical volume integration:
      \hat{c}(k=0) = \int c(r) d^3r = 4\pi \int_0^\infty r^2 c(r) dr
    Guarantees no 0/0 division by k or Fourier origin singularities.
    """
    dr = r_grid[1] - r_grid[0]
    integral = 4.0 * np.pi * float(np.sum((r_grid**2) * c_r * dr))
    return integral
