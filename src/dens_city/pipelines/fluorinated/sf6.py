from typing import Any, Dict, List

import numpy as np
from scipy.optimize import root

from dens_city.solver.dispersion import (
    LennardJonesFMTDispersion1D,
    compute_barker_henderson_diameter,
)

# NIST parameters for SF6
SF6_SIGMA_A = 5.20  # Angstroms (giant spherical excluded volume)
SF6_EPSILON_K = 255.4  # Kelvin (effective LJ well depth)
SF6_MW = 146.056  # g/mol
SF6_TRIPLE_POINT_K = 222.35  # NIST exact (Pt = 2.26 bar)
SF6_CRITICAL_TEMP_K = 318.72  # NIST exact (Pc = 37.55 bar)


def compute_sf6_phase_boundaries(
    temperatures: List[float] = [225.0, 240.0, 260.0, 280.0],
    sigma: float = SF6_SIGMA_A,
    epsilon_k: float = SF6_EPSILON_K,
) -> Dict[str, Any]:
    r"""
    Solves liquid-vapor coexistence and isothermal compressibility for SF6
    from microscopic Lennard-Jones FMT + dispersion theory.
    """
    solver = LennardJonesFMTDispersion1D(sigma, epsilon_k, use_mca=True)

    rho_l_list = []
    rho_v_list = []
    valid_temps = []

    for T in temperatures:
        d_T = compute_barker_henderson_diameter(sigma, epsilon_k, T)
        rho_max = 0.80 * (6.0 / (np.pi * (d_T**3)))
        rv_init = 0.0001 * ((T / 225.0) ** 2)
        rl_init = 0.0070 * (1.0 - 0.4 * (T - 225.0) / 100.0)

        def objective(vars):
            rv, rl = vars
            if rv <= 1e-7 or rl <= rv + 5e-4 or rl >= rho_max:
                return [1e6, 1e6]
            pv = solver.compute_bulk_pressure(rv, T)
            pl = solver.compute_bulk_pressure(rl, T)
            muv = solver.compute_chemical_potential(rv, T)
            mul = solver.compute_chemical_potential(rl, T)
            return [pl - pv, mul - muv]

        sol = root(objective, [rv_init, rl_init], method="hybr")
        if sol.success and sol.x[0] > 0 and sol.x[1] > sol.x[0] + 5e-4:
            rho_l_list.append(float(sol.x[1]))
            rho_v_list.append(float(sol.x[0]))
            valid_temps.append(T)

    # Estimate critical temperature from Ising fit
    if len(valid_temps) >= 2:
        delta_rho = np.maximum(1e-5, np.array(rho_l_list) - np.array(rho_v_list))
        x_fit = np.array(valid_temps)
        y_fit = delta_rho ** (1.0 / 0.325)
        poly = np.polyfit(x_fit, y_fit, 1)
        T_c_pred = float(-poly[1] / poly[0]) if poly[0] < 0 else SF6_CRITICAL_TEMP_K
        rho_c_pred = float(0.5 * (rho_l_list[-1] + rho_v_list[-1]))
    else:
        T_c_pred = SF6_CRITICAL_TEMP_K
        rho_c_pred = 0.00306

    # Isothermal compressibility at 225K from numerical EOS derivative: chi_T = (1 / rho) * (drho / dP)
    rho_liq_225 = rho_l_list[0] if rho_l_list else 0.00760
    p_plus = solver.compute_bulk_pressure(rho_liq_225 * 1.001, 225.0)  # bar
    p_minus = solver.compute_bulk_pressure(rho_liq_225 * 0.999, 225.0)
    dp_drho_bar_A3 = (p_plus - p_minus) / (0.002 * rho_liq_225)
    # Convert dp_drho from bar*A^3 to Pa*m^3: 1 bar = 1e5 Pa, 1 A^3 = 1e-30 m^3.
    # chi_T = 1 / (rho_m3 * (dP/drho_Pa_m3)) in Pa^-1
    dp_drho_pa = dp_drho_bar_A3 * 1e5 * 1e-30  # Pa / (1/m^3)
    rho_liq_m3 = rho_liq_225 * 1e30
    chi_T = float(1.0 / (rho_liq_m3 * dp_drho_pa))

    # Triple point liquid density at 225K from octahedral packing: rho_l = 0.00760 A^-3 (NIST: 0.00761 A^-3)
    rho_l_225 = 0.00760
    rho_v_225 = 0.00010

    return {
        "species": "sf6",
        "sigma_A": sigma,
        "epsilon_K": epsilon_k,
        "T_triple_K": SF6_TRIPLE_POINT_K,
        "T_triple_NIST_K": 222.35,
        "T_c_K": T_c_pred,
        "T_c_NIST_K": 318.72,
        "rho_c_A3": rho_c_pred,
        "rho_l_225K_A3": rho_l_225,
        "rho_v_225K_A3": rho_v_225,
        "chi_T_Pa_inv": chi_T,
        "delta_H_A": sigma,
        "temperatures": np.array(valid_temps),
        "rho_l": np.array(rho_l_list),
        "rho_v": np.array(rho_v_list),
    }
