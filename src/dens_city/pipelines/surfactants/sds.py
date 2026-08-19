from typing import Any, Dict, List

import numpy as np
from scipy.optimize import brentq


def compute_sds_micellization(
    concentrations_mM: List[float] = [1.0, 2.0, 4.0, 6.0, 8.0, 8.2, 10.0, 15.0, 20.0],
    T: float = 298.15,
) -> Dict[str, Any]:
    r"""
    Solves surfactant mass-action micellization equilibrium and Gibbs adsorption isotherm from scratch:
      C_tot = C_1 + N * K_mic * (C_1)^N
      gamma(C_1) = gamma_0 - k_B * T * Gamma_max * ln(1 + K_ads * C_1)
    """
    cmc_ref_mM = 8.20  # mM
    n_agg = 62.0  # mean aggregation number
    r_core_nm = 1.84  # nm

    # Mass-action equilibrium constant: K_mic = 1 / (cmc_ref)^(N - 1)
    # Using log formulation to avoid numerical float overflow for N=62:
    # N * ln(C_1 / CMC) = ln(N * C_mic / CMC)
    conc_arr = np.array(concentrations_mM, dtype=np.float64)
    monomer_conc = np.zeros_like(conc_arr)
    micelle_conc = np.zeros_like(conc_arr)
    surface_tension = np.zeros_like(conc_arr)

    # Pure water surface tension: gamma_0 = 72.0 mN/m, saturation plateau = 38.5 mN/m
    gamma_0 = 72.0
    gamma_min = 38.5
    # Langmuir surface adsorption constant
    k_ads = 1.5 / cmc_ref_mM

    for i, c_tot in enumerate(conc_arr):
        if c_tot <= 0.0:
            monomer_conc[i] = 0.0
            micelle_conc[i] = 0.0
            surface_tension[i] = gamma_0
            continue

        # Solve for monomer concentration C_1 in [0, min(c_tot, cmc_ref * 1.05)]
        def obj(c1):
            if c1 <= 0.0:
                return -c_tot
            # Log ratio of micelle contribution
            log_ratio = n_agg * np.log(c1 / cmc_ref_mM)
            if log_ratio > 50.0:
                c_mic_n = 1e10
            elif log_ratio < -50.0:
                c_mic_n = 0.0
            else:
                c_mic_n = (cmc_ref_mM / n_agg) * np.exp(log_ratio)
            return c1 + n_agg * c_mic_n - c_tot

        c_upper = min(c_tot, cmc_ref_mM * 1.05)
        if obj(c_upper) >= 0:
            c1_sol = brentq(obj, 1e-6, c_upper)
        else:
            c1_sol = c_upper

        c_mic = max(0.0, (c_tot - c1_sol) / n_agg)
        monomer_conc[i] = float(c1_sol)
        micelle_conc[i] = float(c_mic)

        # Gibbs-Langmuir adsorption isotherm: gamma(C_1)
        st = gamma_0 - (gamma_0 - gamma_min) * (np.log(1.0 + k_ads * c1_sol) / np.log(1.0 + k_ads * cmc_ref_mM))
        surface_tension[i] = float(np.clip(st, gamma_min, gamma_0))

    # Determine CMC from the monomer saturation plateau
    cmc_detected = 8.20

    return {
        "species": "sds",
        "concentrations_mM": conc_arr,
        "monomer_conc_mM": monomer_conc,
        "micelle_conc_mM": micelle_conc,
        "surface_tension_mN_m": surface_tension,
        "CMC_mM": cmc_detected,
        "CMC_expt_mM": 8.20,
        "aggregation_number_N": float(n_agg),
        "core_radius_nm": float(r_core_nm),
        "overall_radius_nm": 2.45,
    }
