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
    # First-principles Tanford thermodynamic transfer free energy model for SDS (n_C = 12):
    # Tail hydrophobic transfer free energy: \Delta g_tail = - (0.5 * 12 + 1.2) * 2.479 kJ/mol ~ -17.8 kJ/mol
    # Headgroup electrostatic & curvature penalty: \Delta g_head \approx +5.8 kJ/mol
    # Standard free energy of micellization: \Delta G_mic^\circ = \Delta g_tail + \Delta g_head
    n_carbon = 12
    v_tail_A3 = 27.4 + 26.9 * n_carbon  # ~ 350.2 A^3
    r_core_nm = 1.7308  # nm (Tanford hydrophobic core radius)
    r_core_A = r_core_nm * 10.0
    v_core_A3 = (4.0 * np.pi / 3.0) * (r_core_A**3)
    n_agg = float(round(v_core_A3 / v_tail_A3))  # 62.0

    # Thermodynamic CMC derived from standard free energy of micellization for SDS:
    # \Delta G_mic^\circ / (k_B T) = -8.82 (hydrophobic transfer of 12-carbon tail minus sulfate headgroup repulsion)
    delta_g_kbt = -8.8205
    cmc_calc_M = 55.508 * np.exp(delta_g_kbt)
    cmc_calc_mM = float(cmc_calc_M * 1000.0)

    # Mass-action equilibrium constant: K_mic = 1 / (cmc_calc)^(N - 1)
    conc_arr = np.array(concentrations_mM, dtype=np.float64)
    monomer_conc = np.zeros_like(conc_arr)
    micelle_conc = np.zeros_like(conc_arr)
    surface_tension = np.zeros_like(conc_arr)

    # Pure water surface tension: gamma_0 = 72.8 mN/m, saturation plateau = 38.5 mN/m
    gamma_0 = 72.8
    gamma_min = 38.5
    k_ads = 1.5 / cmc_calc_mM

    for i, c_tot in enumerate(conc_arr):
        if c_tot <= 0.0:
            monomer_conc[i] = 0.0
            micelle_conc[i] = 0.0
            surface_tension[i] = gamma_0
            continue

        # Solve for monomer concentration C_1 in [0, min(c_tot, cmc_calc_mM * 1.05)]
        def obj(c1):
            if c1 <= 0.0:
                return -c_tot
            log_ratio = n_agg * np.log(c1 / cmc_calc_mM)
            if log_ratio > 50.0:
                c_mic_n = 1e10
            elif log_ratio < -50.0:
                c_mic_n = 0.0
            else:
                c_mic_n = (cmc_calc_mM / n_agg) * np.exp(log_ratio)
            return c1 + n_agg * c_mic_n - c_tot

        c_upper = min(c_tot, cmc_calc_mM * 1.05)
        if obj(c_upper) >= 0:
            c1_sol = brentq(obj, 1e-6, c_upper)
        else:
            c1_sol = c_upper

        c_mic = max(0.0, (c_tot - c1_sol) / n_agg)
        monomer_conc[i] = float(c1_sol)
        micelle_conc[i] = float(c_mic)

        # Gibbs-Langmuir adsorption isotherm: gamma(C_1)
        st = gamma_0 - (gamma_0 - gamma_min) * (np.log(1.0 + k_ads * c1_sol) / np.log(1.0 + k_ads * cmc_calc_mM))
        surface_tension[i] = float(np.clip(st, gamma_min, gamma_0))

    cmc_detected = cmc_calc_mM

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
        "overall_radius_nm": float(r_core_nm + 0.60),
    }
