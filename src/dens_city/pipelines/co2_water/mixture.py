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
      \mu_CO2(x, T, P) = \mu_CO2(dense gas, T, P)
      \mu_H2O(liq, T, P) = \mu_H2O(vapor, T, P)
    """
    P_bar = P_atm * 1.01325
    # Temperature-dependent Henry constant from Van 't Hoff relation:
    # \Delta H_solv \approx -19.9 kJ/mol
    delta_h_solv = -19.9e3  # J / mol
    r_gas = 8.314462618
    k_h_0 = 0.034  # mol / (kg * atm) at 298.15 K
    k_h_t = k_h_0 * np.exp(-(delta_h_solv / r_gas) * (1.0 / T - 1.0 / 298.15))

    # Fugacity of CO2 at given pressure (phi ~ 0.85 at 50 atm)
    phi_co2 = 0.85
    f_co2_atm = P_atm * phi_co2

    # Poynting factor: exp(P * V_inf / (R * T)) where V_inf ~ 32 cm^3/mol
    v_inf_co2 = 32.0e-6  # m^3 / mol
    poynting = np.exp((P_bar * 1e5 * v_inf_co2) / (r_gas * T))

    # Molality and dissolved mole fraction
    molality_co2 = (k_h_t * f_co2_atm) / poynting
    x_co2_aq = float(molality_co2 / (molality_co2 + 55.508))

    # Water vapor pressure from exact Clausius-Clapeyron equation:
    # P_sat(T) = P_0 * exp( - \Delta H_vap / R * (1/T - 1/373.15) )
    delta_h_vap = 40.65e3  # J / mol
    p_sat_water_bar = 1.01325 * np.exp(-(delta_h_vap / r_gas) * (1.0 / T - 1.0 / 373.15))
    v_molar_water = 18.0e-6
    poynting_water = np.exp((P_bar - p_sat_water_bar) * 1e5 * v_molar_water / (r_gas * T))
    y_h2o_gas = float((p_sat_water_bar * poynting_water) / max(1e-2, P_bar))

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
    Computes binary competitive nanoconfined pore filling for CO2 / H2O mixture in a slit pore
    using exact 9-3 Lennard-Jones substrate potentials and multicomponent cDFT excluded volume.
    """
    z = np.linspace(0.0, H, grid_size)

    # Bulk densities (A^-3)
    rho_w_bulk = 0.033 * (1.0 - x_co2_feed)
    rho_co2_bulk = 0.015 * x_co2_feed

    # Molecular diameters: sigma_water = 3.15 A, sigma_co2 = 3.75 A
    sig_w = 3.15
    sig_c = 3.75

    # 9-3 Lennard-Jones slit wall potentials
    v_wall_w = np.zeros(grid_size)
    v_wall_c = np.zeros(grid_size)
    eps_w_wall = 1.2 * 4.184e-21 / (KB * T)  # Dimensionless
    eps_c_wall = 0.5 * 4.184e-21 / (KB * T)
    pre_w = (2.0 * np.pi * eps_w_wall * (sig_w**3)) / 3.0
    pre_c = (2.0 * np.pi * eps_c_wall * (sig_c**3)) / 3.0

    for iz, zi in enumerate(z):
        z_left = zi
        z_right = H - zi
        if z_left < 0.5 or z_right < 0.5:
            v_wall_w[iz] = 1e6
            v_wall_c[iz] = 1e6
        else:
            sw_l = sig_w / z_left
            sw_r = sig_w / z_right
            v_wall_w[iz] = pre_w * ((2.0 / 15.0) * (sw_l**9 + sw_r**9) - (sw_l**3 + sw_r**3))

            sc_l = sig_c / z_left
            sc_r = sig_c / z_right
            v_wall_c[iz] = pre_c * ((2.0 / 15.0) * (sc_l**9 + sc_r**9) - (sc_l**3 + sc_r**3))

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
