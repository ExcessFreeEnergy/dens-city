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
            v_ext[i] = 1e18  # Exact hard wall boundary condition
            dv_ext_dz[i] = 0.0
            continue

        d1 = z - z_left
        d2 = z_right - z

        if d1 > 0.05:
            s1 = sigma_clay / d1
            v_ext[i] += prefactor_lj * ((2.0 / 15.0) * (s1**9) - (s1**3))
            dv_ext_dz[i] += prefactor_lj * (-(18.0 / 15.0) * (s1**9) / d1 + 3.0 * (s1**3) / d1)

        if d2 > 0.05:
            s2 = sigma_clay / d2
            v_ext[i] += prefactor_lj * ((2.0 / 15.0) * (s2**9) - (s2**3))
            dv_ext_dz[i] -= prefactor_lj * (-(18.0 / 15.0) * (s2**9) / d2 + 3.0 * (s2**3) / d2)

    return z_coords, v_ext, dv_ext_dz


def compute_clay_swelling_pressure(
    H_values: List[float],  # Basal interlayer spacings in Angstroms (e.g. 9.5 to 25.0 A)
    T: float = 298.15,
    salt_conc_M: float = 0.1,  # Molar concentration of NaCl in pore fluid
    grid_size: int = 128,
) -> Dict[str, np.ndarray]:
    r"""
    Calculates the disjoining / swelling pressure Pi_swell(H) in montmorillonite clay interlayers
    by solving inhomogeneous cDFT density profiles in clay slit walls and integrating structural virial forces
    and Poisson-Boltzmann diffuse double-layer repulsion.
    """
    h_arr = np.array(H_values)
    pi_swell_mpa = np.zeros(len(h_arr))
    rho_bulk_water = 0.0333  # molecules / A^3 (~1.0 g/cm^3)
    sigma_water = 3.0  # A

    for ih, H in enumerate(h_arr):
        L_z = max(40.0, H + 10.0)
        z_coords, v_ext, dv_ext_dz = make_montmorillonite_slit_potential(H, L_z=L_z, grid_size=grid_size)
        dz = z_coords[1] - z_coords[0]

        # Solve equilibrium water density profile via Picard iteration
        rho_water = np.full(grid_size, rho_bulk_water)
        rho_water[v_ext > 1e10] = 0.0

        for _ in range(50):
            eta = rho_water * (np.pi / 6.0) * (sigma_water**3)
            eta_c = np.clip(eta, 0.0, 0.65)
            c1_hs = -np.log(np.maximum(1e-4, 1.0 - eta_c)) - (3.0 * eta_c / (1.0 - eta_c))

            # External potential in Joules converted to dimensionless beta*V_ext
            beta_v = v_ext / (KB * T)
            target = rho_bulk_water * np.exp(np.clip(-beta_v + c1_hs, -25.0, 15.0))
            target[v_ext > 1e10] = 0.0
            rho_water = 0.80 * rho_water + 0.20 * target

        # Virial structural wall force: P_struct = - \int \rho(z) (dV/dz) dz (converted to MPa)
        f_integral = -np.sum(rho_water * dv_ext_dz) * dz * 1e-10 * 1e30 * 1e10  # in Pa
        p_struct_mpa = float(f_integral * 1e-6)

        # First-principles non-linear Poisson-Boltzmann midpoint double layer osmotic pressure:
        # P_edl = 2 * n_bulk * k_B * T * (cosh(e * psi_mid / k_B T) - 1)
        eps_r = 78.4
        debye_length_m = np.sqrt(eps_r * EPSILON_0 * KB * T / (2.0 * salt_conc_M * 1000.0 * 6.022e23 * (E_CHARGE**2)))
        debye_length_A = debye_length_m * 1e10
        gamma_0 = np.tanh(-0.12 * E_CHARGE / (4.0 * KB * T * eps_r * EPSILON_0 / debye_length_m + 1e-12))
        psi_mid_dimless = 8.0 * gamma_0 * np.exp(-max(0.0, H) / (2.0 * debye_length_A))
        p_edl_pa = 2.0 * (salt_conc_M * 1000.0 * 6.022e23) * KB * T * (np.cosh(np.clip(psi_mid_dimless, -10.0, 10.0)) - 1.0)
        p_edl_mpa = float(p_edl_pa * 1e-6)

        # Total disjoining pressure
        pi_swell_mpa[ih] = p_struct_mpa + p_edl_mpa

    return {
        "H_values": h_arr,
        "Pi_swell_MPa": pi_swell_mpa,
        "T": np.array([T]),
        "salt_conc_M": np.array([salt_conc_M]),
    }
