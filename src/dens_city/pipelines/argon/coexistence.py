"""
Argon Coexistence & Thermodynamic Isotherm Solver.
Derived strictly from the microscopic Lennard-Jones pair potential (sigma = 3.405 A, eps/kB = 119.8 K)
coupled with Axilrod-Teller-Muto (ATM) 3-body non-additive quantum dispersion.
Combines analytical Barker-Henderson effective diameter with WCA attractive perturbation theory
and Macroscopic Compressibility Approximation (MCA) for second-order fluctuations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root

from dens_city.solver.dispersion import (
    LennardJonesFMTDispersion1D,
    compute_barker_henderson_diameter,
)

# Microscopic Lennard-Jones parameters for Argon (NIST / White 1999)
ARGON_SIGMA = 3.405  # Angstroms
ARGON_EPSILON_K = 119.8  # Kelvin (eps / kB)
ARGON_NU_ATM = 8.495e5  # Kelvin * Angstrom^9 (73.2 eV * A^9)

_SOLVER = LennardJonesFMTDispersion1D(
    ARGON_SIGMA, ARGON_EPSILON_K, use_mca=True, use_atm=True, nu_atm=ARGON_NU_ATM
)


def compute_argon_chemical_potential(rho: float, T: float) -> float:
    """
    Computes chemical potential mu(rho, T) in Kelvin using CS + MCA second-order dispersion
    and ATM 3-body non-additive dispersion.
    """
    return _SOLVER.compute_chemical_potential(rho, T)


def compute_argon_pressure(rho: float, T: float) -> float:
    """
    Computes bulk pressure P(rho, T) in bar using CS + MCA second-order dispersion
    and ATM 3-body non-additive dispersion.
    """
    return _SOLVER.compute_bulk_pressure(rho, T)


def solve_argon_coexistence_point(T: float) -> Optional[Tuple[float, float, float]]:
    """
    Solves the exact coexistence condition at temperature T:
      P(rho_l, T) = P(rho_v, T) = P_sat
      mu(rho_l, T) = mu(rho_v, T)
    Returns (rho_l, rho_v, P_sat_bar) or None if above critical point.
    """
    d_T = compute_barker_henderson_diameter(ARGON_SIGMA, ARGON_EPSILON_K, T)
    rho_max = 0.85 * (6.0 / (np.pi * (d_T**3)))
    rho_grid = np.linspace(1e-5, rho_max, 500)
    p_grid = np.array([compute_argon_pressure(r, T) for r in rho_grid])
    dp = np.diff(p_grid)

    min_idx = np.where((dp[:-1] < 0) & (dp[1:] > 0))[0]
    max_idx = np.where((dp[:-1] > 0) & (dp[1:] < 0))[0]

    if len(min_idx) == 0 or len(max_idx) == 0:
        return None

    r_v_init = rho_grid[max_idx[0]] * 0.1
    r_l_init = rho_grid[min_idx[0]] * 1.3

    def objective(vars):
        rv, rl = vars
        if rv <= 1e-7 or rl <= rv or rl >= rho_max:
            return [1e6, 1e6]
        p_v = compute_argon_pressure(rv, T)
        p_l = compute_argon_pressure(rl, T)
        mu_v = compute_argon_chemical_potential(rv, T)
        mu_l = compute_argon_chemical_potential(rl, T)
        return [p_l - p_v, mu_l - mu_v]

    sol = root(objective, [r_v_init, r_l_init], method="hybr")
    if sol.success and sol.x[0] < sol.x[1] and sol.x[0] > 0:
        rv, rl = float(sol.x[0]), float(sol.x[1])
        p_sat = float(compute_argon_pressure(rv, T))
        return rl, rv, p_sat

    return None


def compute_argon_binodal(temperatures: List[float] = None) -> Dict[str, np.ndarray]:
    """
    Computes the liquid-vapor binodal curve for Argon across a range of subcritical temperatures.
    """
    if temperatures is None:
        temperatures = [85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0]

    rho_l_list, rho_v_list, p_sat_list, valid_temps = [], [], [], []

    for T in temperatures:
        res = solve_argon_coexistence_point(T)
        if res is not None:
            rl, rv, psat = res
            rho_l_list.append(rl)
            rho_v_list.append(rv)
            p_sat_list.append(psat)
            valid_temps.append(T)

    delta_rho = np.maximum(1e-5, np.array(rho_l_list) - np.array(rho_v_list))
    x_fit = np.array(valid_temps)
    y_fit = delta_rho ** (1.0 / 0.325)
    poly = np.polyfit(x_fit, y_fit, 1)
    T_c_pred = float(-poly[1] / poly[0])

    rho_mid = 0.5 * (np.array(rho_l_list) + np.array(rho_v_list))
    poly_d = np.polyfit(x_fit, rho_mid, 1)
    rho_c_pred = float(np.polyval(poly_d, T_c_pred))
    P_c_pred = float(compute_argon_pressure(rho_c_pred, T_c_pred))

    return {
        "temperatures": np.array(valid_temps),
        "rho_l": np.array(rho_l_list),
        "rho_v": np.array(rho_v_list),
        "P_sat_bar": np.array(p_sat_list),
        "T_c_K": T_c_pred,
        "rho_c": rho_c_pred,
        "P_c_bar": P_c_pred,
    }


def compute_argon_isotherms(
    temperatures: List[float] = None,
    rho_range: Tuple[float, float] = (0.0005, 0.025),
    n_points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Computes equation-of-state isotherms P(rho) for Argon across subcritical and supercritical regimes.
    """
    if temperatures is None:
        temperatures = [100.0, 130.0, 150.86, 180.0, 220.0]

    rho_grid = np.linspace(rho_range[0], rho_range[1], n_points)
    isotherms = {}

    for T in temperatures:
        p_iso = np.array([compute_argon_pressure(r, T) for r in rho_grid])
        isotherms[f"T_{int(T)}K"] = p_iso

    return {
        "rho": rho_grid,
        "temperatures": np.array(temperatures),
        "isotherms": isotherms,
    }
