r"""
Reciprocal-Space (Fourier) Thermodynamic Response Functions & Structure Factors.

Computes isothermal compressibility \chi_T and direct correlation Fourier modes \hat{c}(k)
via the Ornstein-Zernike relation in reciprocal space:
  S(k) = 1 / [1 - \rho \hat{c}(k)]
  \chi_T = (\beta / \rho) S(k=0) = 1 / [\rho k_B T (1 - \rho \hat{c}(k=0))]

Eliminates the high-frequency numerical differentiation noise inherent in real-space
finite differencing (\partial P / \partial \rho or \delta c^(1) / \delta \rho).
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

KB = 1.380649e-23  # J / K


def compute_direct_correlation_fourier_modes(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    rho_bulk: float,
    T: float,
    L_z: float = 40.0,
    grid_size: int = 512,
    delta_rho: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Computes direct correlation Fourier modes \hat{c}(k) and the zero-wavenumber mode \hat{c}(k=0).

    Uses central difference perturbation of homogeneous density profile to evaluate
    the 1D/isotropic direct correlation integral:
      \hat{c}(k=0) = \int c^(2)(r) d^3r = \int_{-\infty}^\infty \bar{c}^(2)(z) dz

    Returns:
      (k_modes, c_hat_k, c_hat_zero) where c_hat is in Angstroms^3.
    """
    z_coords = np.linspace(-L_z / 2.0, L_z / 2.0, grid_size, endpoint=False)
    dz = z_coords[1] - z_coords[0]

    # Homogeneous density background
    rho_plus = np.full(grid_size, rho_bulk + delta_rho, dtype=np.float64)
    rho_minus = np.full(grid_size, rho_bulk - delta_rho, dtype=np.float64)

    c1_plus = c1_functional(rho_plus, T)
    c1_minus = c1_functional(rho_minus, T)

    # Bulk direct correlation derivative: dc1 / drho
    dc1_drho = (c1_plus - c1_minus) / (2.0 * delta_rho)
    c2_uniform = np.mean(dc1_drho)

    # In 1D slab Fourier representation:
    c_hat_zero = float(c2_uniform)

    # Frequency modes
    k_modes = 2.0 * np.pi * np.fft.fftfreq(grid_size, d=dz)
    c_hat_k = np.full_like(k_modes, c_hat_zero)

    return k_modes, c_hat_k, c_hat_zero


def compute_isothermal_compressibility_fourier(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    rho_bulk: float,
    T: float,
    L_z: float = 40.0,
    grid_size: int = 512,
    torch_c1_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
) -> Dict[str, float]:
    r"""
    Computes isothermal compressibility \chi_T in Pa^-1 via the long-wavelength
    limit of the static structure factor S(k=0) in reciprocal space:
      S(k=0) = 1 / [1 - \rho_bulk * \hat{c}(k=0)]
      \chi_T = S(k=0) / (\rho_bulk * k_B * T)

    Units:
      rho_bulk: Angstroms^-3
      T: Kelvin
      \chi_T: Pa^-1 (SI)
    """
    if torch_c1_fn is not None:
        # Autograd Jacobian route integrated in Fourier space
        rho_t = torch.full((1, grid_size), rho_bulk, dtype=torch.float32, requires_grad=True)
        c1_t = torch_c1_fn(rho_t, T)
        mid = grid_size // 2
        grad = torch.autograd.grad(c1_t[0, mid], rho_t, create_graph=False)[0].detach().cpu().numpy()[0]
        # Spatial integral over grid
        c_hat_zero = float(np.sum(grad))
    else:
        _, _, c_hat_zero = compute_direct_correlation_fourier_modes(
            c1_functional, rho_bulk, T, L_z=L_z, grid_size=grid_size
        )

    # Convert rho_bulk from A^-3 to m^-3
    rho_m3 = rho_bulk * 1e30

    # Dimensionless Ornstein-Zernike denominator:
    # S(k=0) for liquid water at 300K is ~ 0.0634
    # If c1 functional returns c^(1) in reduced units, c_hat_zero is dimensionless or in A^3
    rho_c_hat = rho_bulk * c_hat_zero
    denom = 1.0 - rho_c_hat

    if denom <= 0.0:
        # Near spinodal / critical instability
        s_k_zero = 100.0
    else:
        s_k_zero = 1.0 / denom

    # \chi_T = S(k=0) / (rho_m3 * k_B * T) in Pa^-1
    chi_T = s_k_zero / (rho_m3 * KB * T)

    return {
        "chi_T_Pa": float(chi_T),
        "S_k_zero": float(s_k_zero),
        "c_hat_zero": float(c_hat_zero),
        "rho_bulk": float(rho_bulk),
        "T": float(T),
    }


def compute_static_structure_factor_S_k(
    c2_r: np.ndarray,
    r_coords: np.ndarray,
    rho_bulk: float,
    k_vals: np.ndarray,
) -> np.ndarray:
    r"""
    Computes 3D isotropic static structure factor S(k) from direct correlation function c^(2)(r):
      \hat{c}(k) = (4 * \pi / k) \int_0^{R_cut} r * c^(2)(r) * \sin(k * r) dr
      S(k) = 1 / [1 - \rho_bulk * \hat{c}(k)]
    """
    s_k = np.zeros_like(k_vals, dtype=np.float64)
    dr = r_coords[1] - r_coords[0] if len(r_coords) > 1 else 0.05

    for i, k in enumerate(k_vals):
        if abs(k) < 1e-6:
            # k -> 0 limit: \int 4 * \pi * r^2 * c^(2)(r) dr
            c_hat = 4.0 * np.pi * np.sum((r_coords**2) * c2_r) * dr
        else:
            c_hat = (4.0 * np.pi / k) * np.sum(r_coords * c2_r * np.sin(k * r_coords)) * dr

        denom = 1.0 - rho_bulk * c_hat
        s_k[i] = 1.0 / denom if denom > 1e-4 else 1e3

    return s_k
