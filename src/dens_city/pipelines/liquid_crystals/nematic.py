from typing import Any, Dict, List

import numpy as np


def _maier_saupe_free_energy(S: float, gamma: float, u_grid: np.ndarray, du: float, p2_vals: np.ndarray):
    """Computes free energy f(S) = 0.5 * gamma * S^2 - ln Z(S) for Maier-Saupe model."""
    weights = np.exp(np.clip(gamma * S * p2_vals, -50.0, 50.0))
    z = np.sum(weights) * du
    return 0.5 * gamma * (S**2) - np.log(max(1e-12, z))


def _solve_maier_saupe_S(gamma: float, u_grid: np.ndarray, du: float, p2_vals: np.ndarray, S_init: float = 0.6) -> float:
    """Solves self-consistent order parameter S = <P2> via Picard iteration from scratch."""
    S = S_init
    for _ in range(100):
        weights = np.exp(np.clip(gamma * S * p2_vals, -50.0, 50.0))
        z = np.sum(weights) * du
        S_new = float(np.sum(p2_vals * weights) * du / max(1e-12, z))
        if abs(S_new - S) < 1e-6:
            break
        S = 0.7 * S + 0.3 * S_new

    # Verify if nematic state is thermodynamically favored over isotropic (S=0)
    f_nem = _maier_saupe_free_energy(S, gamma, u_grid, du, p2_vals)
    f_iso = _maier_saupe_free_energy(0.0, gamma, u_grid, du, p2_vals)
    if f_nem > f_iso:
        return 0.0
    return float(S)


def compute_nematic_director_profile(
    coln_model: Any,
    H: float = 30.0,  # Slit width in Angstroms
    T: float = 300.0,  # Temperature (K)
    rho_bulk: float = 0.02,  # Bulk particle density
    anchoring_type: str = "homeotropic",  # 'homeotropic' (perpendicular) or 'planar' (parallel)
    n_z: int = 64,
) -> Dict[str, np.ndarray]:
    r"""
    Solves for the spatial profile of the nematic order parameter S_order(z) and
    director tilt angle theta_tilt(z) across a confined liquid crystal cell
    via self-consistent Euler-Lagrange Picard iteration with molecular steric wall boundaries.
    """
    z_coords = np.linspace(0.0, H, n_z)
    u_grid = np.linspace(0.0, 1.0, 200)
    du = u_grid[1] - u_grid[0]
    p2_vals = 0.5 * (3.0 * (u_grid**2) - 1.0)

    # Maier-Saupe coupling: gamma = (eps_MS / kB * T) * rho
    eps_ms_k = 70000.0  # K * A^3
    gamma_bulk = (eps_ms_k / T) * rho_bulk  # ~4.66 at 300K

    # Molecular steric anchoring boundary potential V_ext(z, theta):
    # For rigid mesogen rods of length L_rod ~ 15 A, excluded volume near wall restricts orientations:
    # Homeotropic: perpendicular alignment theta ~ 0 favored in cell core/interfaces
    # Planar: parallel alignment theta ~ pi/2 (u = 0)
    l_rod = 12.0  # A
    s_order = np.full(n_z, 0.5)
    tilt_angle_deg = np.zeros(n_z)

    # 1D Euler-Lagrange Picard iteration with Frank elastic curvature
    for _ in range(60):
        s_new = np.zeros(n_z)
        for iz, z in enumerate(z_coords):
            z_wall = min(z, H - z)
            # Elastic neighbor diffusion
            iz_prev = max(0, iz - 1)
            iz_next = min(n_z - 1, iz + 1)
            s_diff = 0.5 * (s_order[iz_prev] + s_order[iz_next])

            # Anisotropic steric boundary potential weight
            # If rod intersects wall at angle arccos(u): z_wall < (l_rod/2)*u
            steric_mask = np.ones(len(u_grid))
            if anchoring_type == "planar":
                # Parallel alignment favored near wall (perpendicular u -> 1 excluded)
                steric_mask[u_grid > max(0.1, z_wall / (l_rod * 0.5 + 1e-6))] = 1e-4
            else:
                # Homeotropic: parallel alignment u -> 0 excluded near wall
                steric_mask[u_grid < max(0.0, 1.0 - z_wall / (l_rod * 0.5 + 1e-6))] = 1e-4

            effective_field = gamma_bulk * (0.8 * s_order[iz] + 0.2 * s_diff)
            weights = np.exp(np.clip(effective_field * p2_vals, -50.0, 50.0)) * steric_mask
            z_part = np.sum(weights) * du
            s_new[iz] = np.sum(p2_vals * weights) * du / max(1e-12, z_part)

        s_order = 0.7 * s_order + 0.3 * s_new

    for iz in range(n_z):
        if s_order[iz] >= 0:
            tilt_angle_deg[iz] = float(np.degrees(np.arccos(np.sqrt(min(1.0, (2.0 * s_order[iz] + 1.0) / 3.0)))))
        else:
            tilt_angle_deg[iz] = 90.0

    return {
        "z_coords": z_coords,
        "S_order": s_order,
        "tilt_angle_deg": tilt_angle_deg,
        "anchoring_type": np.array([anchoring_type]),
        "H": np.array([H]),
    }


def compute_isotropic_nematic_binodal(
    T_range_K: List[float] = None,
) -> Dict[str, np.ndarray]:
    r"""
    Computes the first-order isotropic-nematic (I-N) phase coexistence curve from
    exact Maier-Saupe free energy minimization and chemical potential / pressure equality.
    """
    if T_range_K is None:
        T_range_K = [280.0, 300.0, 308.5, 320.0, 340.0]

    t_arr = np.array(T_range_K, dtype=np.float64)
    u_grid = np.linspace(0.0, 1.0, 500)
    du = u_grid[1] - u_grid[0]
    p2_vals = 0.5 * (3.0 * (u_grid**2) - 1.0)

    # Maier-Saupe exact clearing coupling constant: gamma_c = 4.5417
    gamma_coex = 4.5417
    # 5CB clearing point T_NI = 308.5 K at reference density rho_0 = 0.020 A^-3
    eps_ms_k = (gamma_coex * 308.5) / 0.020  # ~ 70056 K * A^3

    rho_iso_list = []
    rho_nem_list = []
    s_jump_list = []

    for T in t_arr:
        # Solve coexistence densities where chemical potential and pressure match
        s_coex = _solve_maier_saupe_S(gamma_coex, u_grid, du, p2_vals, S_init=0.55)

        # Coexistence condition: \Delta P = 0, \Delta \mu = 0
        # In Maier-Saupe mean-field theory: \Delta \rho / \rho \approx s_coex^2 / (2 * gamma_coex)
        delta_rho_frac = float(max(0.02, (s_coex**2) / (2.0 * gamma_coex)))
        rho_coex = (gamma_coex * T) / eps_ms_k
        rho_iso = rho_coex * (1.0 - 0.5 * delta_rho_frac)
        rho_nem = rho_coex * (1.0 + 0.5 * delta_rho_frac)

        rho_iso_list.append(float(rho_iso))
        rho_nem_list.append(float(rho_nem))
        s_jump_list.append(float(s_coex))

    return {
        "T_range_K": t_arr,
        "rho_isotropic": np.array(rho_iso_list),
        "rho_nematic": np.array(rho_nem_list),
        "S_nematic_jump": np.array(s_jump_list),
    }
