r"""
Binary Hard-Sphere Colloids Pipeline: Asakura-Oosawa Depletion Forces.

Simulates extreme size asymmetry (R_colloid / r_depletant >= 10), predicting
purely entropy-driven attraction and phase separation with zero energetic parameters (\epsilon = 0).
"""

from typing import Any, Dict

import numpy as np

from dens_city.solver.depletion import (
    compute_asakura_oosawa_potential,
    compute_colloidal_depletion_demixing,
)


def run_colloidal_depletion_simulation(
    R_colloid_nm: float = 50.0,
    r_depletant_nm: float = 5.0,
    eta_depletant: float = 0.20,
) -> Dict[str, Any]:
    r"""
    Solves Asakura-Oosawa depletion potential and verifies colloidal demixing.
    """
    R_colloid_A = R_colloid_nm * 10.0
    r_depletant_A = r_depletant_nm * 10.0

    h_surface_A = np.linspace(0.0, 2.5 * r_depletant_A, 250)
    ao_res = compute_asakura_oosawa_potential(
        h_surface_A,
        R_colloid=R_colloid_A,
        r_depletant=r_depletant_A,
        eta_depletant=eta_depletant,
    )

    demixing_res = compute_colloidal_depletion_demixing(
        R_colloid=R_colloid_A,
        r_depletant=r_depletant_A,
    )

    return {
        "species": "colloidal_depletion",
        "R_colloid_nm": R_colloid_nm,
        "r_depletant_nm": r_depletant_nm,
        "size_ratio_q": float(r_depletant_nm / R_colloid_nm),
        "eta_depletant": eta_depletant,
        "W_contact_kBT": ao_res["W_contact_kBT"],
        "W_contact_exact": ao_res["W_contact_exact"],
        "eta_d_crit": demixing_res["eta_d_crit"],
        "is_phase_separated": bool(eta_depletant >= demixing_res["eta_d_crit"]),
        "h_surface_A": h_surface_A,
        "W_AO_kBT": ao_res["W_AO_kBT"],
    }
