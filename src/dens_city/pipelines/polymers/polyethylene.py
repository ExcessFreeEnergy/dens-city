r"""
Flexible Macromolecules Pipeline: Polyethylene Chains (N > 100).

Applies Wertheim's TPT1 chain connectivity functional to model long flexible polymers,
predicting the entropic confinement penalty, radius of gyration scaling R_g ~ b * N^(3/5),
and near-wall density depletion layers.
"""

from typing import Any, Dict

import numpy as np

from dens_city.solver.wertheim import (
    compute_polymer_wall_depletion,
    compute_wertheim_tpt1_chain_potential,
)


def run_polyethylene_confinement_simulation(
    m_chain: int = 100,
    L_z: float = 100.0,
    grid_size: int = 256,
    rho_bulk: float = 0.033,
) -> Dict[str, Any]:
    r"""
    Simulates polyethylene chain confinement near a planar hard wall.
    """
    z_coords = np.linspace(0.0, L_z, grid_size)
    depletion_res = compute_polymer_wall_depletion(z_coords, m_chain=m_chain, rho_bulk=rho_bulk)

    # Chain potential
    rho_arr = np.linspace(0.001, 0.035, 50)
    c1_chain = compute_wertheim_tpt1_chain_potential(rho_arr, m_chain=m_chain)

    return {
        "species": "polyethylene",
        "m_chain": m_chain,
        "R_g_nm": depletion_res["R_g_nm"],
        "R_g_A": depletion_res["R_g_A"],
        "depletion_thickness_nm": depletion_res["depletion_thickness_nm"],
        "rho_profile": depletion_res["rho_profile"],
        "z_coords": z_coords,
        "c1_chain": c1_chain,
    }
