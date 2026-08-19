from typing import Any, Dict

import numpy as np

from dens_city.solver.thermo_integration import compute_excess_free_energy

KB = 1.380649e-23


def compute_solvation_free_energy(
    c1_functional_water: Any,
    T: float = 300.0,
    rho_water_bulk: float = 0.033,  # A^-3
    grid_size: int = 128,
) -> float:
    r"""
    Computes the excess chemical potential / solvation free energy Delta mu_ex of a
    solute molecule in liquid water via thermodynamic line integration:
    Delta mu^ex = -k_B T \int_0^1 d\lambda \int dz c1(z; \lambda rho) * rho(z) / N
    """
    z_coords = np.linspace(0.0, 10.0, grid_size)
    rho_profile = np.full(grid_size, rho_water_bulk)
    f_ex = compute_excess_free_energy(c1_functional_water, rho_profile, T=T, z_coords=z_coords, num_lambda_steps=20)
    # Solvation free energy per molecule in kJ/mol
    n_molecules = rho_water_bulk * grid_size
    delta_mu_ex_kj = (f_ex / max(1e-6, n_molecules)) * 6.022e23 * 1e-3
    return float(delta_mu_ex_kj)


def compute_mutual_solubility(
    T: float = 310.0,  # K
    P_atm: float = 50.0,  # atm
) -> Dict[str, float]:
    r"""
    Computes the mutual solubility of CO2 in liquid water (x_CO2) and H2O in dense CO2 (y_H2O)
    at given (T, P) using thermodynamic chemical potential equalization:
    mu_CO2(aq) = mu_CO2(dense gas), mu_H2O(liq) = mu_H2O(dense gas)
    """
    # Henry's law constant for CO2 in water at temperature T (K)
    # k_H(T) = k_H^0 * exp(C * (1/T - 1/298.15))
    k_h_0 = 0.034  # mol / (L * atm) at 298.15 K
    c_const = 2400.0
    k_h_t = k_h_0 * np.exp(c_const * (1.0 / T - 1.0 / 298.15))

    # Molality and mole fraction of dissolved CO2
    molality_co2 = k_h_t * P_atm  # mol CO2 / kg water
    moles_water_1kg = 1000.0 / 18.01528  # ~55.51 mol
    x_co2_aq = molality_co2 / (molality_co2 + moles_water_1kg)

    # Water solubility in dense CO2 gas phase (Poynting-corrected Raoult/Virial)
    p_sat_water = 0.006 * np.exp(17.27 * (T - 273.15) / (T - 35.85))  # atm
    y_h2o_gas = (p_sat_water / P_atm) * np.exp(0.018 * (P_atm - 1.0) / (0.08206 * T))

    return {
        "T_K": T,
        "P_atm": P_atm,
        "x_CO2_liquid": float(x_co2_aq),
        "y_H2O_vapor": float(y_h2o_gas),
    }


def compute_competitive_pore_adsorption(
    H: float = 20.0,  # Slit width in Angstroms
    T: float = 300.0,  # K
    x_co2_feed: float = 0.15,  # 15% CO2 feed
    grid_size: int = 128,
) -> Dict[str, np.ndarray]:
    r"""
    Computes binary competitive nanoconfined pore filling for CO2 / H2O mixture in a slit pore.
    Water strongly wets the hydrophilic walls while CO2 accumulates in the slit center.
    """
    z = np.linspace(0.0, H, grid_size)
    dz = z[1] - z[0]

    # Bulk densities (A^-3)
    rho_w_bulk = 0.033 * (1.0 - x_co2_feed)
    rho_co2_bulk = 0.015 * x_co2_feed

    # Molecular diameters: sigma_water = 3.15 A, sigma_co2 = 3.75 A
    sig_w = 3.15
    sig_c = 3.75

    # Slit wall potentials (hard repulsive walls with selective hydrophilic attraction for water)
    v_wall_w = np.zeros(grid_size)
    v_wall_c = np.zeros(grid_size)
    for iz, zi in enumerate(z):
        z_wall = min(zi, H - zi)
        if z_wall < 1.2:
            v_wall_w[iz] = 1e6
            v_wall_c[iz] = 1e6
        else:
            # Hydrophilic wall well for water (-4.2 kB*T) vs weak wall attraction for CO2 (-0.8 kB*T)
            v_wall_w[iz] = -4.2 * np.exp(-(z_wall - 1.2) / 1.5)
            v_wall_c[iz] = -0.8 * np.exp(-(z_wall - 1.2) / 2.0)

    # Initial density guesses
    rho_water = np.full(grid_size, rho_w_bulk)
    rho_co2 = np.full(grid_size, rho_co2_bulk)
    rho_water[v_wall_w > 100.0] = 0.0
    rho_co2[v_wall_c > 100.0] = 0.0

    # 2-Component Picard iteration with Boublík-Mansoori-Carnahan-Starling-Leland (BMCSL) / FMT packing
    for _ in range(80):
        eta = (np.pi / 6.0) * (rho_water * (sig_w**3) + rho_co2 * (sig_c**3))
        eta_c = np.clip(eta, 0.0, 0.65)
        # Excluded volume excess potential
        c1_pack_w = -np.log(np.maximum(1e-4, 1.0 - eta_c)) - (3.0 * eta_c / (1.0 - eta_c))
        c1_pack_c = -np.log(np.maximum(1e-4, 1.0 - eta_c)) - (3.0 * eta_c / (1.0 - eta_c)) * (sig_c / sig_w)

        # Euler-Lagrange update
        target_w = rho_w_bulk * np.exp(np.clip(-v_wall_w + c1_pack_w, -20.0, 15.0))
        target_c = rho_co2_bulk * np.exp(np.clip(-v_wall_c + c1_pack_c, -20.0, 15.0))

        target_w[v_wall_w > 100.0] = 0.0
        target_c[v_wall_c > 100.0] = 0.0

        rho_water = 0.85 * rho_water + 0.15 * target_w
        rho_co2 = 0.85 * rho_co2 + 0.15 * target_c

    return {
        "z_coords": z,
        "rho_water": rho_water,
        "rho_co2": rho_co2,
        "H": np.array([H]),
    }
