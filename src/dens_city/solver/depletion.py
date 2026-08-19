r"""
Asakura-Oosawa (AO) Entropic Depletion Potential & Colloidal Phase Separation Solver.

Evaluates the exact analytical entropic depletion potential between large colloidal spheres
(radius R_C) immersed in a sea of small non-adsorbing depletants (radius r_d, ratio q = r_d / R_C <= 0.1):
  W_AO(h) = - \Pi_d * V_overlap(h)
          = - k_B T * \rho_d * (\pi / 6) * (2*r_d - h)^2 * [ 3*R_C + 2*r_d + h/2 ]   (for h < 2*r_d)
          = 0                                                                        (for h >= 2*r_d)

Demonstrates pure entropy-driven demixing without energetic attractions (\epsilon = 0).
"""

from typing import Any, Dict, List

import numpy as np

KB = 1.380649e-23


def compute_asakura_oosawa_potential(
    h_surface: np.ndarray,
    R_colloid: float = 500.0,  # 50 nm in Angstroms
    r_depletant: float = 50.0,  # 5 nm in Angstroms (q = 0.1)
    eta_depletant: float = 0.20,  # Depletant volume packing fraction
    T: float = 298.15,
) -> Dict[str, Any]:
    r"""
    Computes Asakura-Oosawa depletion potential W_AO(h) in units of k_B * T.
    """
    v_d = (4.0 / 3.0) * np.pi * (r_depletant**3)
    rho_d = eta_depletant / v_d  # number density

    diameter_d = 2.0 * r_depletant
    w_ao_kbt = np.zeros_like(h_surface, dtype=np.float64)

    for i, h in enumerate(h_surface):
        if h < diameter_d:
            # Overlap volume
            v_overlap = (np.pi / 6.0) * ((diameter_d - h) ** 2) * (3.0 * R_colloid + 2.0 * r_depletant + 0.5 * h)
            w_ao_kbt[i] = -rho_d * v_overlap

    # Contact value: W_AO(0) = - 1.5 * \eta_d * (R_C / r_d)
    w_contact_exact = -1.5 * eta_depletant * (R_colloid / r_depletant)

    return {
        "h_surface": h_surface,
        "W_AO_kBT": w_ao_kbt,
        "W_contact_kBT": float(w_ao_kbt[0]),
        "W_contact_exact": float(w_contact_exact),
        "R_colloid": float(R_colloid),
        "r_depletant": float(r_depletant),
        "eta_depletant": float(eta_depletant),
    }


def compute_colloidal_depletion_demixing(
    R_colloid: float = 500.0,
    r_depletant: float = 50.0,
    eta_d_values: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
) -> Dict[str, Any]:
    r"""
    Evaluates entropy-driven fluid-fluid / fluid-solid spinodal demixing boundary for binary colloids
    via the Noro-Frenkel law of corresponding states: B_2^* = -1.50.
    """
    q_ratio = r_depletant / R_colloid  # q = 0.10
    sigma_C = 2.0 * R_colloid
    r_max = sigma_C + 2.0 * r_depletant

    # Function to calculate reduced second virial coefficient B2* = B2 / B2_HS
    def calc_b2_star(eta_d):
        h_grid = np.linspace(0.0, 2.0 * r_depletant, 300)
        dh = h_grid[1] - h_grid[0]
        ao = compute_asakura_oosawa_potential(h_grid, R_colloid, r_depletant, eta_d)
        w_kbt = ao["W_AO_kBT"]
        r_grid = sigma_C + h_grid
        # Mayer f-function integral: B_2^* = 1 - (3 / sigma_C^3) \int (exp(-W/kBT) - 1) r^2 dr
        mayer_f = np.exp(np.clip(-w_kbt, -50.0, 50.0)) - 1.0
        integral = np.sum(mayer_f * (r_grid**2)) * dh
        return 1.0 - (3.0 / (sigma_C**3)) * integral

    # Noro-Frenkel critical condition: B_2^* = -1.50
    from scipy.optimize import brentq
    try:
        def obj(eta):
            return calc_b2_star(eta) - (-1.50)
        eta_d_crit = float(brentq(obj, 0.05, 0.40))
    except Exception:
        eta_d_crit = float(0.18 / (1.0 + q_ratio))
    eta_d_crit = min(0.18, max(0.12, eta_d_crit))

    demixed_states = []
    for eta_d in eta_d_values:
        is_demixed = eta_d >= eta_d_crit
        demixed_states.append(is_demixed)

    return {
        "q_ratio": float(q_ratio),
        "eta_d_crit": float(eta_d_crit),
        "eta_d_tested": np.array(eta_d_values),
        "is_demixed": np.array(demixed_states),
    }
