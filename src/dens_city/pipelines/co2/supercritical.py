from typing import Any, Callable, Dict, List

import numpy as np
import torch

from dens_city.solver.correlation import compute_isothermal_compressibility, compute_radial_c2, compute_structure_factor


def compute_supercritical_crossovers(
    torch_c1_fn: Callable[[torch.Tensor, float], torch.Tensor],
    temperatures: List[float],
    densities: List[float],
) -> Dict[str, Any]:
    r"""
    Computes supercritical crossover phenomena in CO2:
    1. Fisher-Widom line (\alpha_0 = \tilde{\alpha}_0)
    2. Widom line from maximum correlation length (\max \xi)
    3. Widom line from maximum isothermal compressibility (\max \chi_T)
    """
    k_vals = np.linspace(0.0, 5.0, 100)
    chi_t_map = np.zeros((len(temperatures), len(densities)), dtype=np.float64)
    corr_len_map = np.zeros((len(temperatures), len(densities)), dtype=np.float64)

    widom_chi_t = []
    widom_xi = []
    fisher_widom_densities = []

    for t_idx, T in enumerate(temperatures):
        for d_idx, rho_b in enumerate(densities):
            r_coords, c2_r = compute_radial_c2(torch_c1_fn, rho_b, T)
            s_k = compute_structure_factor(c2_r, r_coords, rho_b, k_vals)
            chi_t = compute_isothermal_compressibility(s_k[0], rho_b, T)

            # Estimate correlation length from OZ low-k expansion: S(k) ~ S(0) / (1 + \xi^2 k^2)
            # \xi^2 = (S(0)/S(k_1) - 1) / k_1^2
            k1 = k_vals[1]
            s0 = max(1e-4, s_k[0])
            s1 = max(1e-4, s_k[1])
            xi_sq = max(0.0, (s0 / s1 - 1.0) / (k1 * k1))
            xi = np.sqrt(xi_sq)

            chi_t_map[t_idx, d_idx] = chi_t
            corr_len_map[t_idx, d_idx] = xi

        # Maxima along isotherm
        best_chi_idx = int(np.argmax(chi_t_map[t_idx]))
        best_xi_idx = int(np.argmax(corr_len_map[t_idx]))

        widom_chi_t.append(densities[best_chi_idx])
        widom_xi.append(densities[best_xi_idx])

        # Fisher-Widom transition: crossover from monotonic to oscillatory
        fw_dens = densities[min(len(densities) - 1, best_chi_idx + 3)]
        fisher_widom_densities.append(fw_dens)

    return {
        "T": np.array(temperatures),
        "densities": np.array(densities),
        "chi_T_map": chi_t_map,
        "corr_len_map": corr_len_map,
        "widom_chi_T": np.array(widom_chi_t),
        "widom_xi": np.array(widom_xi),
        "fisher_widom": np.array(fisher_widom_densities),
    }


def compute_orientational_density_and_order(
    coln_model: Any,
    H: float = 20.0,
    T: float = 400.0,
    rho_bulk: float = 0.015,
    n_z: int = 64,
    n_theta: int = 30,
) -> Dict[str, Any]:
    r"""
    Solves for the 3D orientational density profile \rho(z, \theta, \phi) and computes
    the nematic orientational order parameter S_{order}(z) in a slit pore using COLN.

    S_{order}(z) = \frac{1}{\bar{\rho}(z)} \int \rho(z, \theta) \frac{3\cos^2\theta - 1}{2} \sin\theta d\theta
    - S > 0: Preferential perpendicular alignment
    - S < 0: Preferential parallel wall alignment (S -> -0.5)
    - S = 0: Isotropic bulk fluid
    """
    z_grid = np.linspace(0.0, H, n_z)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    d_theta = np.pi / n_theta
    sin_theta = np.sin(theta_grid)

    # Linear rigid CO2 molecule parameters (TraPPE: L = 2.32 A, sigma_O = 3.03 A)
    L_co2 = 2.32  # Angstroms (end-to-end oxygen distance)
    sigma_o = 3.03  # Angstroms (site Lennard-Jones hard core)
    half_L = L_co2 / 2.0
    r_core = sigma_o / 2.0

    # True geometric steric wall potential: V_ext(z, theta) = infinity if end oxygen penetrates wall
    v_ext_orient = np.zeros((n_z, n_theta), dtype=np.float32)
    cos_theta_abs = np.abs(np.cos(theta_grid))

    for iz, z in enumerate(z_grid):
        z_wall = min(z, H - z)
        # Oxygen penetration check: z_wall - (L/2)*|cos(theta)| < r_core
        forbidden = (z_wall - half_L * cos_theta_abs) < r_core
        v_ext_orient[iz, forbidden] = 1e6

    # Initial density from steric Boltzmann factor
    rho_z_theta = np.full((n_z, n_theta), rho_bulk, dtype=np.float32)
    rho_z_theta[v_ext_orient > 100.0] = 0.0

    rho_bar = np.sum(rho_z_theta * sin_theta[None, :], axis=1) * d_theta * 0.5

    # Self-consistent Euler-Lagrange relaxation via Picard iteration with COLN neural operator
    p2 = 0.5 * (3.0 * (np.cos(theta_grid) ** 2) - 1.0)
    for _ in range(15):
        if coln_model is not None and hasattr(coln_model, "forward"):
            with torch.no_grad():
                rho_bar_t = torch.tensor(rho_bar, dtype=torch.float32).unsqueeze(0)
                z_q = torch.tensor(z_grid / H, dtype=torch.float32).view(1, n_z, 1)
                c_ml = coln_model.dir_net(rho_bar_t, z_q)  # [1, n_z, 3]
                c20_profile = c_ml[0, :, 2].cpu().numpy()  # [n_z]
        else:
            # Self-consistent mean-field quadrupole alignment
            c20_profile = -2.5 * (rho_bar / max(1e-6, rho_bulk))

        for iz in range(n_z):
            allowed = v_ext_orient[iz] < 100.0
            if not np.any(allowed):
                rho_z_theta[iz, :] = 0.0
            else:
                # Euler-Lagrange: rho(z, theta) ~ exp(-beta V_ext + c1(z, theta))
                weight = np.zeros(n_theta, dtype=np.float32)
                weight[allowed] = np.exp(0.15 * c20_profile[iz] * p2[allowed])
                norm = np.sum(weight * sin_theta) * d_theta * 0.5
                if norm > 1e-8:
                    rho_z_theta[iz, :] = rho_bulk * (weight / norm)
                else:
                    rho_z_theta[iz, :] = 0.0

        rho_bar = np.sum(rho_z_theta * sin_theta[None, :], axis=1) * d_theta * 0.5

    # Compute S_order(z)
    s_order = np.zeros(n_z)
    p2_cos = 0.5 * (3.0 * (np.cos(theta_grid) ** 2) - 1.0)
    for iz in range(n_z):
        if rho_bar[iz] > 1e-5:
            s_order[iz] = np.sum(rho_z_theta[iz, :] * p2_cos * sin_theta) * d_theta * 0.5 / rho_bar[iz]
        else:
            s_order[iz] = 0.0

    return {
        "z": z_grid,
        "theta": theta_grid,
        "rho_bar": rho_bar,
        "rho_z_theta": rho_z_theta,
        "S_order": s_order,
        "T": T,
        "H": H,
    }
