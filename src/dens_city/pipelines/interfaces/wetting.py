"""
Hydrophobic & Hydrophilic Planar Wetting Interfaces Pipeline.
Solves contact angles, capillary evaporation/drying gap H_dry, and Lum-Chandler-Weeks (LCW) crossover.
"""

from typing import Dict, List, Tuple
import numpy as np

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
) -> Dict[str, np.ndarray]:
    """
    Simulates water confined between two symmetric planar hydrophobic walls (theta > 90 deg).
    Evaluates the critical capillary drying/cavitation gap H_dry where liquid evaporates into vapor.
    """
    if H_nm_values is None:
        H_nm_values = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]

    H_arr = np.array(H_nm_values)
    cos_theta = np.cos(np.radians(theta_deg))

    # Grand potential per area:
    # Liquid state: Omega_liquid / A = 2 * gamma_sl - (rho_l * delta_mu) * H
    # Vapor state:  Omega_vapor  / A = 2 * gamma_sv
    # Difference: Delta Omega = 2 * (gamma_sl - gamma_sv) - (rho_l * delta_mu) * H = - 2 * gamma_lv * cos(theta) - (rho_l * delta_mu) * H
    # Cavitation occurs when Delta Omega > 0 (vapor is thermodynamically favored):
    # H_dry = - 2 * gamma_lv * cos(theta) / (rho_l * delta_mu)

    # For water: rho_l = 33.36 nm^-3, delta_mu in J
    delta_mu_J = delta_mu_kT * KB * T
    # Convert gamma_lv from mN/m (mJ/m^2) to J/nm^2: 1 mJ/m^2 = 1e-3 J/m^2 = 1e-21 J/nm^2
    gamma_J_nm2 = gamma_lv * 1e-21
    rho_l_nm3 = 33.36

    # Critical drying gap in nm:
    H_dry_nm = float(abs(2.0 * gamma_J_nm2 * cos_theta / max(1e-25, rho_l_nm3 * delta_mu_J)))
    H_dry_nm = min(10.0, max(0.5, H_dry_nm))

    # Average density in pore as a function of H: drops from liquid (33.0 nm^-3) to vapor (~0.01 nm^-3) below H_dry
    rho_pore = np.where(H_arr < H_dry_nm, 0.002, 33.0 * (1.0 - 0.2 / (H_arr + 0.1)))

    return {
        "H_nm": H_arr,
        "rho_pore_nm3": rho_pore,
        "H_dry_nm": H_dry_nm,
        "theta_deg": theta_deg,
        "cavitation_detected": bool(np.any(H_arr < H_dry_nm)),
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

    R_arr = np.array(radius_nm_values)
    R_c = 1.0  # nm (LCW crossover length scale)

    # Free energy per surface area: Delta G / (4 * pi * R^2)
    # At small R: scales with R (Delta G ~ (4/3)*pi*R^3 * P_effective) -> Delta G / Area ~ (1/3)*P*R
    # At large R: scales with gamma_lv -> Delta G / Area -> gamma_lv
    delta_g_per_area = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        if R <= R_c:
            # Volume-dominated regime (cavity formation inside hydrogen bond network)
            delta_g_per_area[i] = gamma_lv * (R / R_c) ** 0.85
        else:
            # Surface-dominated dewetting regime (macroscopic interface)
            delta_g_per_area[i] = gamma_lv * (1.0 - 0.2 * (R_c / R))

    delta_g_kJ_mol = delta_g_per_area * (4.0 * np.pi * (R_arr**2)) * 6.022e23 * 1e-21 * 1e-3  # kJ / mol

    return {
        "radius_nm": R_arr,
        "R_c_nm": R_c,
        "delta_g_per_area_mNm": delta_g_per_area,
        "delta_g_kJ_mol": delta_g_kJ_mol,
    }
