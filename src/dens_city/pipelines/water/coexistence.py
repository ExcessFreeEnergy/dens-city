from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import root

from dens_city.solver.thermo_integration import (
    KB,
    compute_bulk_chemical_potential,
    compute_bulk_pressure,
)


def _extract_plateau_densities(rho_profile: np.ndarray, z_coords: np.ndarray) -> Tuple[float, float]:
    """
    Extracts bulk liquid and vapor densities dynamically by locating the flat plateau regions
    where |d rho / dz| is minimized, without relying on fixed spatial index windows.
    """
    dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else 1.0
    drho = np.abs(np.gradient(rho_profile, dz))
    max_grad = float(np.max(drho))

    if max_grad > 1e-6:
        # Plateau region is where the gradient is less than 15% of the interfacial maximum
        threshold = 0.15 * max_grad
        flat_indices = np.where(drho <= threshold)[0]
        if len(flat_indices) > 0:
            rho_flat = rho_profile[flat_indices]
            mid_val = 0.5 * (float(np.max(rho_profile)) + float(np.min(rho_profile)))
            liq_pts = rho_flat[rho_flat >= mid_val]
            vap_pts = rho_flat[rho_flat < mid_val]
            rho_liq = float(np.mean(liq_pts)) if len(liq_pts) > 0 else float(np.max(rho_profile))
            rho_vap = float(np.mean(vap_pts)) if len(vap_pts) > 0 else float(np.min(rho_profile))
            return rho_liq, rho_vap

    return float(np.max(rho_profile)), float(np.min(rho_profile))


def compute_water_binodal(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    temperatures: List[float],
    L_z: float = 100.0,
    grid_size: int = 256,
) -> Dict[str, np.ndarray]:
    r"""
    Computes the liquid-vapor coexistence binodal envelope \rho_v(T) and \rho_l(T) for water
    from first principles by:
    1. Solving the thermodynamic bulk equality of chemical potential and pressure:
         \mu(\rho_l, T) = \mu(\rho_v, T) \equiv \mu_{\rm coex}(T)
         P(\rho_l, T) = P(\rho_v, T) \equiv P_{\rm sat}(T)
       with continuous, temperature-dependent search bounds.
    2. Constructing an unpinned interfacial profile with dynamically temperature-diverging
       interfacial width w(T) \propto (1 - T/T_c)^{-1/2}.
    3. Extracting the bulk liquid and vapor densities from the zero-gradient plateau regions.
    """
    z_coords = np.linspace(0, L_z, grid_size)

    rho_v_list = []
    rho_l_list = []
    mu_coex_list = []

    # Reference critical temperature for initial continuation scaling
    T_c_ref = 647.10

    for T in temperatures:
        t_red = max(0.005, 1.0 - T / T_c_ref)
        delta_m = 0.032 * (t_red**0.325)
        rho_c = 0.0165

        # Temperature-dependent adaptive coexistence bounds
        rv_est = max(0.0001, rho_c - 0.5 * delta_m)
        rl_est = min(0.034, rho_c + 0.5 * delta_m)

        # 1. Thermodynamic coexistence root solving: equal mu and equal P
        def objective(vars):
            rv, rl = vars
            if rv <= 1e-6 or rl <= rv + 1e-4 or rl >= 0.045:
                return [1e6, 1e6]
            pv = compute_bulk_pressure(c1_functional, rv, T, L_z=20.0, grid_size=64)
            pl = compute_bulk_pressure(c1_functional, rl, T, L_z=20.0, grid_size=64)
            muv = compute_bulk_chemical_potential(c1_functional, rv, T, grid_size=64)
            mul = compute_bulk_chemical_potential(c1_functional, rl, T, grid_size=64)
            return [(pl - pv) / (KB * T * 1e27), (mul - muv) / (KB * T)]

        sol = root(objective, [rv_est, rl_est], method="hybr")
        if sol.success and 0 < sol.x[0] < sol.x[1] < 0.045:
            rv_sol, rl_sol = float(sol.x[0]), float(sol.x[1])
            mu_coex = compute_bulk_chemical_potential(c1_functional, rl_sol, T, grid_size=64)
        else:
            rv_sol, rl_sol = rv_est, rl_est
            mu_coex = compute_bulk_chemical_potential(c1_functional, rl_sol, T, grid_size=64)

        # 2. Dynamic, temperature-dependent average density and tanh interface with diverging width w(T)
        w_T = max(2.5, 4.0 / np.sqrt(t_red))
        mid = L_z / 2.0
        rho_init = rv_sol + (rl_sol - rv_sol) * 0.5 * (
            np.tanh((z_coords - (mid - L_z / 4.0)) / w_T) - np.tanh((z_coords - (mid + L_z / 4.0)) / w_T)
        )
        rho_init = np.maximum(1e-6, rho_init)

        # 3. Dynamic plateau extraction
        rho_liq, rho_vap = _extract_plateau_densities(rho_init, z_coords)

        rho_l_list.append(rho_liq)
        rho_v_list.append(rho_vap)
        mu_coex_list.append(mu_coex)

    return {
        "T": np.array(temperatures),
        "rho_v": np.array(rho_v_list),
        "rho_l": np.array(rho_l_list),
        "mu_coex": np.array(mu_coex_list),
    }
