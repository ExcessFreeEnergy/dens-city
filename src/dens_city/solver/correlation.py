from typing import Callable, Optional, Tuple

import numpy as np
import torch

from dens_city.solver.quantum_surrogates import apply_hann_window
from dens_city.solver.response_functions import (
    compute_static_structure_factor_S_k,
)

KB = 1.380649e-23


def compute_radial_c2(
    torch_c1_fn: Callable[[torch.Tensor, float], torch.Tensor],
    rho_bulk: float,
    T: float,
    grid_size: int = 256,
    L_z: float = 20.0,
    apply_window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 1D direct correlation function c^(2)(|z - z'|)
    via automatic differentiation of the neural functional c^(1)(z; [rho]).
    Includes smooth Hann windowing to suppress boundary Gibbs ringing.
    """
    dz = L_z / grid_size
    z_coords = np.linspace(0, L_z, grid_size, endpoint=False)

    rho_t = torch.full((1, grid_size), rho_bulk, dtype=torch.float32, requires_grad=True)
    c1_t = torch_c1_fn(rho_t, T)

    mid = grid_size // 2
    grad = torch.autograd.grad(c1_t[0, mid], rho_t, create_graph=False)[0].detach().cpu().numpy()[0]

    c2_profile = grad / dz
    r_coords = np.abs(z_coords - z_coords[mid])

    sort_idx = np.argsort(r_coords)
    r_sorted = r_coords[sort_idx]
    c2_sorted = c2_profile[sort_idx]

    if apply_window:
        c2_sorted = apply_hann_window(c2_sorted, r_sorted, L_z / 2.0)

    return r_sorted, c2_sorted


def compute_structure_factor(
    c2_r: np.ndarray,
    r_coords: np.ndarray,
    rho_bulk: float,
    k_vals: np.ndarray,
) -> np.ndarray:
    r"""
    Computes the bulk structure factor S(k) via the Ornstein-Zernike relation in Fourier space.
    """
    return compute_static_structure_factor_S_k(c2_r, r_coords, rho_bulk, k_vals)


def compute_c_hat_zero_volume(
    c2_r: np.ndarray,
    r_coords: np.ndarray,
) -> float:
    r"""
    Evaluates \hat{c}(k=0) strictly via spatial volume integration:
      \hat{c}(0) = 4\pi \int_0^\infty r^2 c^{(2)}(r) dr
    Guarantees no 0/0 division by k.
    """
    dr = r_coords[1] - r_coords[0]
    return float(4.0 * np.pi * np.sum((r_coords**2) * c2_r * dr))


def compute_isothermal_compressibility(
    s_k_zero: float,
    rho_bulk: float,
    T: float,
) -> float:
    r"""
    Computes isothermal compressibility \chi_T = (\beta / \rho_b) S(k=0).
    """
    beta = 1.0 / (KB * T)
    rho_m3 = rho_bulk * 1e30 if rho_bulk < 1.0 else rho_bulk
    return (beta / rho_m3) * s_k_zero
