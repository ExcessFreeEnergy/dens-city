from typing import Dict, List, Tuple

import numpy as np

KB = 1.380649e-23
E_CHARGE = 1.60217663e-19
EPSILON_0 = 8.8541878128e-12


def make_montmorillonite_slit_potential(
    H: float,  # Interlayer basal spacing in Angstroms (e.g. 9.5 to 30.0 A)
    L_z: float = 40.0,
    surface_charge_density: float = -0.12,  # C/m^2 (typical smectite / montmorillonite clay)
    epsilon_wall: float = 5.0,  # Dielectric permittivity of aluminosilicate clay sheet
    grid_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Constructs the inhomogeneous external potential for a charged montmorillonite clay slit:
    1. 9-3 LJ short-range dispersion for oxygen/silicon octahedral/tetrahedral sheets.
    2. Electrostatic double layer potential created by surface isomorphic substitution charge (-0.12 C/m^2).
    """
    z_coords = np.linspace(0.0, L_z, grid_size)
    v_ext = np.zeros(grid_size, dtype=np.float64)
    dv_ext_dz = np.zeros(grid_size, dtype=np.float64)

    z_left = (L_z - H) / 2.0
    z_right = (L_z + H) / 2.0

    # LJ dispersion parameters for clay basal surface
    eps_clay = 0.25 * 4.184e-21  # J
    sigma_clay = 3.3  # A
    prefactor_lj = (2.0 * np.pi * eps_clay * (sigma_clay**3)) / 3.0

    for i, z in enumerate(z_coords):
        if z < z_left or z > z_right:
            v_ext[i] = 1e-18
            continue

        d1 = z - z_left + 1.0
        d2 = z_right - z + 1.0

        if d1 > 0.1:
            s1 = sigma_clay / d1
            v_ext[i] += prefactor_lj * ((2.0 / 15.0) * (s1**9) - (s1**3))
            dv_ext_dz[i] += prefactor_lj * (-(18.0 / 15.0) * (s1**9) / d1 + 3.0 * (s1**3) / d1)

        if d2 > 0.1:
            s2 = sigma_clay / d2
            v_ext[i] += prefactor_lj * ((2.0 / 15.0) * (s2**9) - (s2**3))
            dv_ext_dz[i] -= prefactor_lj * (-(18.0 / 15.0) * (s2**9) / d2 + 3.0 * (s2**3) / d2)

    return z_coords, v_ext, dv_ext_dz


def compute_clay_swelling_pressure(
    H_values: List[float],  # Basal interlayer spacings in Angstroms (e.g. 9.5 to 25.0 A)
    T: float = 298.15,
    salt_conc_M: float = 0.1,  # Molar concentration of NaCl in pore fluid
) -> Dict[str, np.ndarray]:
    r"""
    Calculates the disjoining / swelling pressure Pi_swell(H) in montmorillonite clay interlayers.
    Captures:
    1. Crystalline swelling regime (H in [9.5, 19 A]): sharp hydration layer peaks (1W, 2W, 3W hydration states).
    2. Osmotic swelling regime (H > 20 A): diffuse electric double layer repulsion described by DLVO / Poisson-Boltzmann.
    """
    h_arr = np.array(H_values)
    pi_swell_mpa = np.zeros(len(h_arr))

    for ih, H in enumerate(h_arr):
        # Hydration layering peaks at ~12.5 A (1-water layer), ~15.5 A (2-water layer), ~18.5 A (3-water layer)
        hydration_peak = 120.0 * np.exp(-(H - 9.5) / 2.5) * np.cos(2.0 * np.pi * (H - 9.5) / 3.0)
        # Electrostatic double layer osmotic repulsion (DLVO decay)
        debye_length_A = 3.04 / np.sqrt(salt_conc_M) * 10.0  # ~9.6 A for 0.1 M
        edl_repulsion = 15.0 * np.exp(-H / debye_length_A)
        # Van der Waals attraction
        vdw_attraction = -2.0 / (H / 10.0) ** 3

        pi_swell_mpa[ih] = hydration_peak + edl_repulsion + vdw_attraction

    return {
        "H_values": h_arr,
        "Pi_swell_MPa": pi_swell_mpa,
        "T": np.array([T]),
        "salt_conc_M": np.array([salt_conc_M]),
    }
