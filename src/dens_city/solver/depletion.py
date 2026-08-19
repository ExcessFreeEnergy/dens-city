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
    # Noro-Frenkel sticky sphere / generalized van der Waals critical condition (Lekkerkerker 1992):
    # \tau_c = 0.1189 maps to critical depletant packing \eta_d^crit(q)
    # \tau(q, \eta_d) = \frac{1}{4 q (exp(-W_contact / kBT) - 1)} = \tau_c = 0.1189
    tau_c = 0.1189
    # W_contact = - 1.5 * \eta_d * (1/q + 2/3)
    # exp(-W_contact) = 1 + 1 / (4 * q * \tau_c)
    w_crit_target = -float(np.log(1.0 + 1.0 / (4.0 * max(1e-4, q_ratio) * tau_c)))
    eta_d_crit = float(abs(w_crit_target) / (1.5 * (1.0 / max(1e-4, q_ratio) + 2.0 / 3.0)))

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
