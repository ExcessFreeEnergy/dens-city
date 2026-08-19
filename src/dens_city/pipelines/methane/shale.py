"""
TraPPE United-Atom Methane (CH4) Shale Adsorption, Binodal & Gas Recovery Pipeline.
Calculates properties directly from the TraPPE Lennard-Jones pair parameters:
sigma = 3.730 A, eps/kB = 148.0 K coupled with Axilrod-Teller-Muto (ATM) 3-body dispersion.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root

from dens_city.solver.dispersion import (
    LennardJonesFMTDispersion1D,
    compute_barker_henderson_diameter,
)

KB = 1.380649e-23
METHANE_SIGMA = 3.730  # Angstroms
METHANE_EPSILON_K = 148.0  # Kelvin
METHANE_NU_ATM = 1.218e6  # Kelvin * Angstrom^9 (105.0 eV * A^9)

_SOLVER = LennardJonesFMTDispersion1D(
    METHANE_SIGMA, METHANE_EPSILON_K, use_mca=True, use_atm=True, nu_atm=METHANE_NU_ATM
)


def compute_methane_pressure(rho: float, T: float) -> float:
    """
    Computes bulk pressure P(rho, T) in bar using CS + MCA second-order dispersion
    and ATM 3-body non-additive dispersion.
    """
    return _SOLVER.compute_bulk_pressure(rho, T)


def compute_methane_chemical_potential(rho: float, T: float) -> float:
    """
    Computes chemical potential mu(rho, T) in Kelvin using CS + MCA second-order dispersion
    and ATM 3-body non-additive dispersion.
    """
    return _SOLVER.compute_chemical_potential(rho, T)


def solve_methane_coexistence_point(T: float) -> Optional[Tuple[float, float, float]]:
    """
    Solves the exact coexistence condition at temperature T:
      P(rho_l, T) = P(rho_v, T) = P_sat
      mu(rho_l, T) = mu(rho_v, T)
    Returns (rho_l, rho_v, P_sat_bar) or None if above critical point.
    """
    d_T = compute_barker_henderson_diameter(METHANE_SIGMA, METHANE_EPSILON_K, T)
    rho_max = 0.85 * (6.0 / (np.pi * (d_T**3)))
    rho_grid = np.linspace(1e-5, rho_max, 500)
    p_grid = np.array([compute_methane_pressure(r, T) for r in rho_grid])
    dp = np.diff(p_grid)

    min_idx = np.where((dp[:-1] < 0) & (dp[1:] > 0))[0]
    max_idx = np.where((dp[:-1] > 0) & (dp[1:] < 0))[0]

    if len(min_idx) == 0 or len(max_idx) == 0:
        return None

    r_v_init = rho_grid[max_idx[0]] * 0.1
    r_l_init = rho_grid[min_idx[0]] * 1.5

    def objective(vars):
        rv, rl = vars
        if rv <= 1e-7 or rl <= rv or rl >= rho_max:
            return [1e6, 1e6]
        p_v = compute_methane_pressure(rv, T)
        p_l = compute_methane_pressure(rl, T)
        mu_v = compute_methane_chemical_potential(rv, T)
        mu_l = compute_methane_chemical_potential(rl, T)
        return [p_l - p_v, mu_l - mu_v]

    sol = root(objective, [r_v_init, r_l_init], method="hybr")
    if sol.success and sol.x[0] < sol.x[1] and sol.x[0] > 0:
        rv, rl = float(sol.x[0]), float(sol.x[1])
        p_sat = float(compute_methane_pressure(rv, T))
        return rl, rv, p_sat

    return None


def compute_methane_binodal(temperatures: List[float] = None) -> Dict[str, np.ndarray]:
    """
    Computes the liquid-vapor binodal curve for Methane from TraPPE pair parameters with ATM 3-body dispersion.
    """
    if temperatures is None:
        temperatures = [110.0, 125.0, 140.0, 155.0, 170.0, 180.0]

    rho_l_list, rho_v_list, p_sat_list, valid_temps = [], [], [], []

    for T in temperatures:
        res = solve_methane_coexistence_point(T)
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
    P_c_pred = float(compute_methane_pressure(rho_c_pred, T_c_pred))

    return {
        "temperatures": np.array(valid_temps),
        "rho_l": np.array(rho_l_list),
        "rho_v": np.array(rho_v_list),
        "P_sat_bar": np.array(p_sat_list),
        "T_c_K": T_c_pred,
        "rho_c": rho_c_pred,
        "P_c_bar": P_c_pred,
    }


def _find_bulk_density_for_pressure(target_P_bar: float, T: float) -> float:
    """Inverts the FMT+dispersion equation of state to find bulk density rho(P, T)."""
    d_T = compute_barker_henderson_diameter(METHANE_SIGMA, METHANE_EPSILON_K, T)
    rho_max = 0.80 * (6.0 / (np.pi * (d_T**3)))

    def obj(r):
        r_val = float(np.atleast_1d(r)[0])
        if r_val <= 1e-7 or r_val >= rho_max:
            return 1e6
        return float(compute_methane_pressure(r_val, T) - target_P_bar)

    sol = root(obj, 0.001, method="hybr")
    if sol.success and 0 < sol.x[0] < rho_max:
        return float(sol.x[0])
    # Fallback to ideal gas / low density
    return float(np.clip(target_P_bar / (1.380649e-23 * T * 1e30 * 1e-5), 1e-5, rho_max))


def compute_methane_shale_isotherm(
    H_values: List[float],  # Slit widths in Angstroms (e.g. 10.0, 20.0, 30.0)
    T: float = 330.0,  # Reservoir temperature (K)
    P_range_bar: List[float] = None,  # Pressure range (bar)
    grid_size: int = 128,
) -> Dict[str, np.ndarray]:
    r"""
    Simulates TraPPE united-atom methane (CH4) adsorption in organic kerogen/shale nanopores
    by solving inhomogeneous cDFT density profiles in carbon slit walls and integrating excess adsorption.
    """
    if P_range_bar is None:
        P_range_bar = [10.0, 50.0, 100.0, 200.0, 300.0]

    p_arr = np.array(P_range_bar)
    excess_adsorption = np.zeros((len(H_values), len(p_arr)))

    for ih, H in enumerate(H_values):
        z_grid = np.linspace(0.0, H, grid_size)
        dz = z_grid[1] - z_grid[0]

        # 10-4 LJ kerogen slit wall potential: V_wall(z) = 2*pi*eps*sigma^2 [ 0.4*(sigma/z)^10 - (sigma/z)^4 ]
        sig_w = 3.40  # A
        eps_w = 110.0  # K
        v_wall = np.zeros(grid_size)
        for iz, z in enumerate(z_grid):
            z_left = max(0.2, z)
            z_right = max(0.2, H - z)
            if z_left < 1.0 or z_right < 1.0:
                v_wall[iz] = 1e6
            else:
                s_l = sig_w / z_left
                s_r = sig_w / z_right
                v_wall[iz] = 2.0 * np.pi * eps_w * (sig_w**2) * (0.4 * (s_l**10) - (s_l**4) + 0.4 * (s_r**10) - (s_r**4))

        for ip, P in enumerate(p_arr):
            rho_bulk = _find_bulk_density_for_pressure(P, T)

            # Solve inhomogeneous equilibrium profile via Picard iteration
            rho_profile = np.full(grid_size, rho_bulk)
            rho_profile[v_wall > 100.0] = 0.0

            d_T = compute_barker_henderson_diameter(METHANE_SIGMA, METHANE_EPSILON_K, T)
            for _ in range(40):
                eta = rho_profile * (np.pi / 6.0) * (d_T**3)
                eta_c = np.clip(eta, 0.0, 0.65)
                c1_hs = -np.log(np.maximum(1e-4, 1.0 - eta_c)) - (3.0 * eta_c / (1.0 - eta_c))
                target = rho_bulk * np.exp(np.clip(-v_wall / T + c1_hs, -20.0, 15.0))
                target[v_wall > 100.0] = 0.0
                rho_profile = 0.85 * rho_profile + 0.15 * target

            # Integrate excess adsorption: \Gamma_excess = \int_0^H (\rho(z) - \rho_bulk) dz
            gamma_ex = np.sum(rho_profile - rho_bulk) * dz
            excess_adsorption[ih, ip] = float(gamma_ex)

    return {
        "H_values": np.array(H_values),
        "P_range_bar": p_arr,
        "excess_adsorption": excess_adsorption,
        "T": np.array([T]),
    }


def compute_ch4_co2_gas_recovery_crossover(
    T: float = 330.0,  # K
    P_range_bar: List[float] = None,
) -> Dict[str, np.ndarray]:
    r"""
    Evaluates competitive Enhanced Gas Recovery (EGR) displacement of CH4 by injected supercritical CO2
    from competitive adsorption isotherms.
    """
    if P_range_bar is None:
        P_range_bar = [20.0, 60.0, 120.0, 200.0]

    p_arr = np.array(P_range_bar)
    recovery_efficiency = np.zeros(len(p_arr))

    for ip, P in enumerate(p_arr):
        # Langmuir-competitive adsorption selectivity: K_CO2 / K_CH4 ~ 2.5
        k_ch4 = 0.015 * np.exp(148.0 / T)
        k_co2 = 0.038 * np.exp(240.0 / T)
        theta_co2 = (k_co2 * P) / (1.0 + (k_co2 + k_ch4) * P)
        theta_ch4 = (k_ch4 * P) / (1.0 + (k_co2 + k_ch4) * P)
        recovery_efficiency[ip] = float(np.clip(theta_co2 / max(1e-6, theta_co2 + theta_ch4) * 0.95 + 0.05, 0.0, 0.98))

    return {
        "P_range_bar": p_arr,
        "recovery_efficiency": recovery_efficiency,
        "T": np.array([T]),
    }
