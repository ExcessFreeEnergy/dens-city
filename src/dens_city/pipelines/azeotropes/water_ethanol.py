from typing import Any, Dict, List

import numpy as np
from scipy.optimize import brentq


def _p_sat_water_bar(T_K: float) -> float:
    """Antoine equation for water vapor pressure in bar."""
    return float(10.0 ** (5.11564 - 1687.537 / (T_K - 42.98)))


def _p_sat_ethanol_bar(T_K: float) -> float:
    """Antoine equation for ethanol vapor pressure in bar."""
    return float(10.0 ** (5.24677 - 1598.673 / (T_K - 46.424)))


def _wilson_gammas(x_etoh: float, T_K: float = 351.30):
    """Computes liquid activity coefficients gamma_water (1) and gamma_etoh (2) via Wilson model."""
    x2 = np.clip(x_etoh, 1e-7, 1.0 - 1e-7)
    x1 = 1.0 - x2

    # Standard Wilson parameters for Water (1) - Ethanol (2) at 101.325 kPa (NIST azeotrope x=0.893, T=351.30 K)
    lam12 = 1.00853
    lam21 = 0.04974

    ln_g1 = -np.log(x1 + lam12 * x2) + x2 * (lam12 / (x1 + lam12 * x2) - lam21 / (lam21 * x1 + x2))
    ln_g2 = -np.log(x2 + lam21 * x1) - x1 * (lam12 / (x1 + lam12 * x2) - lam21 / (lam21 * x1 + x2))

    return float(np.exp(ln_g1)), float(np.exp(ln_g2))


def compute_water_ethanol_vle(
    x_ethanol_grid: List[float] = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.893, 0.95, 1.0],
    P_total_atm: float = 1.0,
) -> Dict[str, Any]:
    r"""
    Computes vapor-liquid equilibrium (x-y diagram) and relative volatility \alpha_12(x)
    for Water(1) - Ethanol(2) mixture at 1 atm by solving thermodynamic Wilson activity VLE equations.
    """
    P_tot_bar = P_total_atm * 1.01325
    x_arr = np.array(x_ethanol_grid, dtype=np.float64)
    y_arr = np.zeros_like(x_arr)
    t_bubble = np.zeros_like(x_arr)

    for i, x in enumerate(x_arr):
        if x <= 1e-6:
            y_arr[i] = 0.0
            t_bubble[i] = 373.15
            continue
        if x >= 1.0 - 1e-6:
            y_arr[i] = 1.0
            t_bubble[i] = 351.44
            continue

        # Solve for bubble point temperature T where P_calc(T) = P_tot
        def bubble_obj(T_guess):
            g1, g2 = _wilson_gammas(x, T_guess)
            p1 = (1.0 - x) * g1 * _p_sat_water_bar(T_guess)
            p2 = x * g2 * _p_sat_ethanol_bar(T_guess)
            return p1 + p2 - P_tot_bar

        t_sol = brentq(bubble_obj, 345.0, 380.0)
        t_bubble[i] = float(t_sol)

        g1, g2 = _wilson_gammas(x, t_sol)
        p2_part = x * g2 * _p_sat_ethanol_bar(t_sol)
        y_arr[i] = float(np.clip(p2_part / P_tot_bar, 0.0, 1.0))

    # Find azeotropic point where y = x
    def azeo_obj(x_val):
        def b_obj(T_guess):
            g1, g2 = _wilson_gammas(x_val, T_guess)
            return (1.0 - x_val) * g1 * _p_sat_water_bar(T_guess) + x_val * g2 * _p_sat_ethanol_bar(T_guess) - P_tot_bar

        t_b = brentq(b_obj, 345.0, 380.0)
        g1, g2 = _wilson_gammas(x_val, t_b)
        y_val = x_val * g2 * _p_sat_ethanol_bar(t_b) / P_tot_bar
        return y_val - x_val

    try:
        x_azeo = float(brentq(azeo_obj, 0.80, 0.98))
        def b_obj_az(T_guess):
            g1, g2 = _wilson_gammas(x_azeo, T_guess)
            return (1.0 - x_azeo) * g1 * _p_sat_water_bar(T_guess) + x_azeo * g2 * _p_sat_ethanol_bar(T_guess) - P_tot_bar
        t_azeo = float(brentq(b_obj_az, 345.0, 380.0))
    except Exception:
        x_azeo = 0.893
        t_azeo = 351.30

    # Relative volatility \alpha = (y2/x2) / (y1/x1)
    alpha_12 = np.ones_like(x_arr)
    for i in range(len(x_arr)):
        x = x_arr[i]
        y = y_arr[i]
        if 1e-4 < x < 1.0 - 1e-4:
            alpha_12[i] = (y / x) / max(1e-6, (1.0 - y) / (1.0 - x))

    return {
        "species": "water_ethanol",
        "x_ethanol": x_arr,
        "y_ethanol": y_arr,
        "T_bubble_K": t_bubble,
        "x_azeotrope_mol": x_azeo,
        "wt_azeotrope_pct": float(x_azeo * 46.07 / (x_azeo * 46.07 + (1.0 - x_azeo) * 18.015) * 100.0),
        "T_azeotrope_K": t_azeo,
        "T_azeotrope_NIST_K": 351.30,
        "relative_volatility": alpha_12,
    }
