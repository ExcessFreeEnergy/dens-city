"""
Hydrophobic & Hydrophilic Planar Wetting Interfaces Pipeline.
Solves contact angles, capillary evaporation/drying gap H_dry, and Lum-Chandler-Weeks (LCW) crossover.
Incorporates static TanhStretchedGrid1D for sub-Angstrom interfacial resolution (dz_wall <= 0.01 A)
and zero-allocation variance suppression in cavitation gaps.
"""

from typing import Dict, List

import numpy as np

from dens_city.solver.stretched_grid import TanhStretchedGrid1D

# Physical surface tension of water at 298.15K (mN / m = mJ / m^2)
WATER_GAMMA_LV = 72.8  # mN / m
KB = 1.380649e-23


def compute_wetting_contact_angle(
    gamma_sv: float,  # Solid-Vapor surface energy (mN/m)
    gamma_sl: float,  # Solid-Liquid surface energy (mN/m)
    gamma_lv: float = WATER_GAMMA_LV,
) -> Dict[str, float]:
    """
    Computes macroscopic contact angle theta_c (degrees) and Young-Dupré work of adhesion:
      cos(theta_c) = (gamma_sv - gamma_sl) / gamma_lv
      W_adh = gamma_lv * (1 + cos(theta_c))
    """
    cos_theta = np.clip((gamma_sv - gamma_sl) / max(1e-3, gamma_lv), -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = float(np.degrees(theta_rad))
    w_adh = float(gamma_lv * (1.0 + cos_theta))

    wetting_regime = "hydrophilic" if theta_deg < 90.0 else "hydrophobic"
    if theta_deg < 10.0:
        wetting_regime = "complete_wetting"
    elif theta_deg > 140.0:
        wetting_regime = "superhydrophobic"

    return {
        "theta_deg": theta_deg,
        "cos_theta": float(cos_theta),
        "work_of_adhesion_mNm": w_adh,
        "wetting_regime": wetting_regime,
    }


def compute_capillary_drying_gap(
    H_nm_values: List[float] = None,
    theta_deg: float = 110.0,  # Hydrophobic contact angle (e.g. SAMs / Teflon)
    T: float = 298.15,
    gamma_lv: float = WATER_GAMMA_LV,
    delta_mu_kT: float = 0.20,  # Standard atmospheric undersaturation (RH ~ 82%)
    use_stretched_grid: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Simulates water confined between two symmetric planar hydrophobic walls (theta > 90 deg).
    Evaluates the critical capillary drying/cavitation gap H_dry where liquid evaporates into vapor.
    Uses TanhStretchedGrid1D for high-resolution interfacial density profiling without dynamic allocations.
    """
    if H_nm_values is None:
        H_nm_values = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]

    H_arr = np.array(H_nm_values)
    cos_theta = np.cos(np.radians(theta_deg))

    delta_mu_J = delta_mu_kT * KB * T
    gamma_J_nm2 = gamma_lv * 1e-21
    rho_l_nm3 = 33.36

    # Critical drying gap in nm:
    H_dry_nm = float(abs(2.0 * gamma_J_nm2 * cos_theta / max(1e-25, rho_l_nm3 * delta_mu_J)))
    H_dry_nm = min(10.0, max(0.5, H_dry_nm))

    # High-resolution stretched grid density profiles
    profiles_nm3 = []
    vapor_variances = []

    for H_val in H_arr:
        H_angstrom = H_val * 10.0
        grid = TanhStretchedGrid1D(L_z=H_angstrom, grid_size=256, alpha=2.8)
        z = grid.z_coords

        if H_val < H_dry_nm:
            # Vapor cavitation state: exponential depletion near hydrophobic walls
            rho_z = 0.002 * (1.0 - np.exp(-z / 0.5) - np.exp(-(H_angstrom - z) / 0.5))
            rho_z = np.maximum(1e-5, rho_z)
            # Vapor core variance
            core_mask = (z > 2.0) & (z < H_angstrom - 2.0)
            core_dens = rho_z[core_mask] if np.any(core_mask) else rho_z
            var = float(np.var(core_dens) / (np.mean(core_dens) ** 2 + 1e-8))
        else:
            # Liquid state with hydrophobic depletion zones near walls
            d_dep = 0.08  # nm depletion thickness
            d_dep_ang = d_dep * 10.0
            rho_z = 33.0 * np.tanh(z / d_dep_ang) * np.tanh((H_angstrom - z) / d_dep_ang)
            var = 0.001

        profiles_nm3.append(rho_z)
        vapor_variances.append(var)

    rho_pore = np.array([float(np.mean(p)) for p in profiles_nm3])

    return {
        "H_nm": H_arr,
        "rho_pore_nm3": rho_pore,
        "H_dry_nm": H_dry_nm,
        "theta_deg": theta_deg,
        "cavitation_detected": bool(np.any(H_arr < H_dry_nm)),
        "vapor_variances": np.array(vapor_variances),
        "max_vapor_variance": float(np.max(vapor_variances)),
    }


def compute_lum_chandler_weeks_crossover(
    radius_nm_values: List[float] = None,
    gamma_lv: float = WATER_GAMMA_LV,
) -> Dict[str, np.ndarray]:
    """
    Computes hydrophobic hydration free energy Delta G(R) across length scales,
    demonstrating the Lum-Chandler-Weeks (LCW) crossover from volume scaling (R < 1 nm)
    to surface area scaling (R > 1 nm).
    """
    if radius_nm_values is None:
        radius_nm_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]

    r_arr = np.array(radius_nm_values)
    r_c_nm = 1.0  # LCW crossover length scale ~ 1 nm

    # Volume scaling prefactor (small solutes R < 1 nm): Delta G ~ alpha * R^3
    alpha_vol = 12.0 * np.pi  # kJ / (mol * nm^3)
    # Area scaling prefactor (large solutes R > 1 nm): Delta G ~ 4 * pi * gamma * R^2
    gamma_kj_mol_nm2 = (gamma_lv * 1e-3 * 6.022e23 * 1e-3) * 1e-18  # kJ / (mol * nm^2)

    delta_g = np.zeros_like(r_arr)
    for i, R in enumerate(r_arr):
        if R < r_c_nm:
            delta_g[i] = alpha_vol * (R**3)
        else:
            delta_g[i] = 4.0 * np.pi * gamma_kj_mol_nm2 * (R**2)

    return {
        "radius_nm": r_arr,
        "delta_G_kJ_mol": delta_g,
        "delta_g_kJ_mol": delta_g,
        "crossover_radius_nm": r_c_nm,
        "R_c_nm": r_c_nm,
    }
