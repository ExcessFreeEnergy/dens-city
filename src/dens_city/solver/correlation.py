from typing import Callable, Tuple
import numpy as np
import torch

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
    c1_t = torch_c1_fn(rho_t, T) # [1, grid_size]

    # Compute Jacobian row by row or centered at midpoint
    mid = grid_size // 2
    c1_mid = c1_t[0, mid]

    grad = torch.autograd.grad(c1_mid, rho_t, create_graph=False)[0].detach().cpu().numpy()[0]

    # c^(2)(z, z') = - \delta c^(1)(z) / \delta \rho(z') / dz
    c2_profile = grad / dz
    r_coords = np.abs(z_coords - z_coords[mid])

    # Sort by distance r
    sort_idx = np.argsort(r_coords)
    return r_coords[sort_idx], c2_profile[sort_idx]


def compute_structure_factor(
    c2_r: np.ndarray,
    r_coords: np.ndarray,
    rho_bulk: float,
    k_vals: np.ndarray,
) -> np.ndarray:
    r"""
    Computes the bulk structure factor S(k) via the Ornstein-Zernike relation:
    S(k) = 1 / (1 - \rho_b \hat{c}_r^(2)(k))
    """
    dr = r_coords[1] - r_coords[0] if len(r_coords) > 1 else 1.0
    s_k = np.zeros_like(k_vals, dtype=np.float64)

    for i, k in enumerate(k_vals):
        if k == 0.0:
            c2_k = 2.0 * np.sum(c2_r) * dr # 1D Fourier zero mode
        else:
            c2_k = 2.0 * np.sum(c2_r * np.cos(k * r_coords)) * dr

        denom = 1.0 - rho_bulk * c2_k
        s_k[i] = 1.0 / denom if abs(denom) > 1e-6 else 1e3

    return s_k


def compute_isothermal_compressibility(
    s_k_zero: float,
    rho_bulk: float,
    T: float,
) -> float:
    r"""
    Computes isothermal compressibility \chi_T = (\beta / \rho_b) S(k=0).
    """
    beta = 1.0 / (KB * T)
    return (beta / rho_bulk) * s_k_zero
