from typing import Callable, Tuple

import numpy as np
import torch

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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 1D direct correlation function c^(2)(|z - z'|)
    via automatic differentiation of the neural functional c^(1)(z; [rho]).
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
    return r_coords[sort_idx], c2_profile[sort_idx]


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
