from typing import Callable, Dict, List, Tuple
import numpy as np

from dens_city.solver.picard_solver import CdftPicardSolver


def compute_water_binodal(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    temperatures: List[float],
    L_z: float = 200.0, # 20 nm
    grid_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Computes the Liquid-Vapor binodal envelope rho_v(T) and rho_l(T) for Water
    via constrained interface stabilization.
    """
    solver = CdftPicardSolver(c1_functional, grid_size=grid_size, alpha_mix=0.10, max_iter=3000)
    z_coords = np.linspace(0, L_z, grid_size)
    v_ext = np.zeros(grid_size) # V_ext = 0 for free interface

    rho_v_list = []
    rho_l_list = []
    mu_coex_list = []

    for T in temperatures:
        # Target average density is halfway between liquid and vapor
        target_avg = 0.0165 # A^-3

        # Step function initial guess: liquid in center, vapor at boundaries
        rho_init = np.full(grid_size, 0.002)
        mid = grid_size // 2
        width = grid_size // 4
        rho_init[mid - width : mid + width] = 0.031

        rho, mu, converged, it = solver.solve_constrained(
            z_coords, v_ext, T=T, target_avg_density=target_avg, rho_init=rho_init
        )

        # Read liquid density (center) and vapor density (boundary)
        rho_liq = float(np.mean(rho[mid - 10 : mid + 10]))
        rho_vap = float(np.mean(np.concatenate([rho[:10], rho[-10:]])))

        rho_l_list.append(rho_liq)
        rho_v_list.append(rho_vap)
        mu_coex_list.append(mu)

    return {
        "T": np.array(temperatures),
        "rho_v": np.array(rho_v_list),
        "rho_l": np.array(rho_l_list),
        "mu_coex": np.array(mu_coex_list),
    }
