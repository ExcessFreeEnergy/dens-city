from typing import Callable, Dict, List, Tuple

import numpy as np

from dens_city.solver.picard_solver import CdftPicardSolver
from dens_city.solver.thermo_integration import compute_bulk_chemical_potential, compute_bulk_pressure

KB = 1.380649e-23


def make_graphene_slit_potential(
    H: float,
    L_z: float,
    grid_size: int = 256,
    epsilon_wall: float = 0.11 * 4.184e-21,  # ~0.11 kcal/mol in Joules
    sigma_wall: float = 3.2,  # in Angstroms
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the 9-3 Lennard-Jones external potential and its derivative
    for two graphene sheets separated by slit width H in Angstroms:
    V_wall(z) = (2*pi*eps*sigma^3 / 3) * [ (2/15)*(sigma/z)^9 - (sigma/z)^3 ]
    """
    z_coords = np.linspace(0, L_z, grid_size)
    v_ext = np.zeros(grid_size, dtype=np.float64)
    dv_ext_dz = np.zeros(grid_size, dtype=np.float64)

    # Left wall at z = (L_z - H)/2, Right wall at z = (L_z + H)/2
    z_left = (L_z - H) / 2.0
    z_right = (L_z + H) / 2.0
    prefactor = (2.0 * np.pi * epsilon_wall * (sigma_wall**3)) / 3.0

    for i, z in enumerate(z_coords):
        if z < z_left or z > z_right:
            v_ext[i] = 1e18  # True infinite steric repulsive boundary (no leaking into void)
            dv_ext_dz[i] = 0.0
            continue

        d1 = z - z_left + 1.0  # Offset for graphene surface
        d2 = z_right - z + 1.0

        if d1 > 0.1:
            s_d1 = sigma_wall / d1
            v_ext[i] += prefactor * ((2.0 / 15.0) * (s_d1**9) - (s_d1**3))
            dv_ext_dz[i] += prefactor * (-(18.0 / 15.0) * (s_d1**9) / d1 + 3.0 * (s_d1**3) / d1)

        if d2 > 0.1:
            s_d2 = sigma_wall / d2
            v_ext[i] += prefactor * ((2.0 / 15.0) * (s_d2**9) - (s_d2**3))
            dv_ext_dz[i] -= prefactor * (-(18.0 / 15.0) * (s_d2**9) / d2 + 3.0 * (s_d2**3) / d2)

    return z_coords, v_ext, dv_ext_dz


def compute_confinement_isotherm(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    H_values: List[float],
    T: float = 300.0,
    rho_bulk: float = 0.033,
    grid_size: int = 256,
) -> Dict[str, np.ndarray]:
    r"""
    Computes effective pressure \tilde{P}(H) and disjoining pressure \Pi(H) = \tilde{P}(H) - P
    across a range of graphene slit separations H in Angstroms.
    """
    solver = CdftPicardSolver(c1_functional, grid_size=grid_size)
    bulk_p = compute_bulk_pressure(c1_functional, rho_bulk, T)
    # Calculate exact bulk reservoir chemical potential mu(rho_bulk, T)
    mu_bulk = compute_bulk_chemical_potential(c1_functional, rho_bulk, T, grid_size=grid_size)

    p_eff_list = []
    pi_disjoining_list = []
    profiles = []

    for H in H_values:
        L_z = max(50.0, H + 20.0)
        z_coords, v_ext, dv_ext_dz = make_graphene_slit_potential(H, L_z, grid_size=grid_size)
        dz = z_coords[1] - z_coords[0]

        # Solve equilibrium profile in Grand Canonical equilibrium with reservoir mu_bulk
        rho, converged, it, res = solver.solve(z_coords, v_ext, T=T, mu=mu_bulk, rho_bulk=rho_bulk)

        # Structural route to effective pressure: \tilde{P} = - \int \rho(z) (dV_wall / dz) dz
        p_eff = -np.sum(rho * dv_ext_dz) * dz
        p_eff_list.append(p_eff)
        pi_disjoining_list.append(p_eff - bulk_p)
        profiles.append(rho)

    return {
        "H": np.array(H_values),
        "P_eff": np.array(p_eff_list),
        "Pi_disjoining": np.array(pi_disjoining_list),
        "bulk_pressure": bulk_p,
        "profiles": profiles,
    }
