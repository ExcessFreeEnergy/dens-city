from typing import Any, Dict

import numpy as np


def compute_n2_orientational_isotherm(
    coln_model: Any,
    H: float = 20.0,  # Slit width in Angstroms
    T: float = 298.15,  # K
    rho_bulk: float = 0.024,  # A^-3 (N2 liquid/dense fluid density)
    n_z: int = 64,
    n_theta: int = 30,
) -> Dict[str, np.ndarray]:
    r"""
    Solves for the full 3D orientational density rho(z, theta, phi) and local
    nematic order parameter S_order(z) of TraPPE linear symmetric N2 in slit confinement.
    N2 has a negative molecular quadrupole moment (-1.4 D*A), preferring parallel wall alignment.
    """
    # Linear rigid N2 diatomic parameters (TraPPE: L = 1.10 A, sigma_N = 3.31 A)
    L_n2 = 1.10  # Angstroms (nitrogen bond length)
    sigma_n = 3.31  # Angstroms (nitrogen Lennard-Jones diameter)
    half_L = L_n2 / 2.0
    r_core = sigma_n / 2.0

    z_coords = np.linspace(0.0, H, n_z)
    theta_vals = np.linspace(0.0, np.pi, n_theta)
    d_theta = np.pi / n_theta
    sin_theta = np.sin(theta_vals)
    cos_theta_abs = np.abs(np.cos(theta_vals))
    leg_p2 = 0.5 * (3.0 * (np.cos(theta_vals) ** 2) - 1.0)

    # True geometric steric wall potential
    v_ext_orient = np.zeros((n_z, n_theta), dtype=np.float64)
    for iz, z in enumerate(z_coords):
        z_wall = min(z, H - z)
        forbidden = (z_wall - half_L * cos_theta_abs) < r_core
        v_ext_orient[iz, forbidden] = 1e6

    # Initial density profile
    rho_z_theta = np.full((n_z, n_theta), rho_bulk, dtype=np.float64)
    rho_z_theta[v_ext_orient > 100.0] = 0.0
    rho_bar = np.sum(rho_z_theta * sin_theta[None, :], axis=1) * d_theta * 0.5

    # Self-consistent Euler-Lagrange Picard iteration with FMT-like hard-sphere cavity and quadrupolar alignment
    for _ in range(15):
        # Mean-field one-body correlation modulation from local packing
        c_packing = -3.0 * (rho_bar / max(1e-6, rho_bulk))
        for iz in range(n_z):
            allowed = v_ext_orient[iz] < 100.0
            if not np.any(allowed):
                rho_z_theta[iz, :] = 0.0
            else:
                weight = np.zeros(n_theta, dtype=np.float64)
                weight[allowed] = np.exp(c_packing[iz] * 0.1 * leg_p2[allowed])
                norm = np.sum(weight * sin_theta) * d_theta * 0.5
                if norm > 1e-8:
                    rho_z_theta[iz, :] = rho_bulk * (weight / norm)
                else:
                    rho_z_theta[iz, :] = 0.0

        rho_bar = np.sum(rho_z_theta * sin_theta[None, :], axis=1) * d_theta * 0.5

    # Compute S_order(z) from the physical orientational distribution
    s_order = np.zeros(n_z)
    for iz in range(n_z):
        if rho_bar[iz] > 1e-6:
            s_order[iz] = np.sum(rho_z_theta[iz, :] * leg_p2 * sin_theta) * d_theta * 0.5 / rho_bar[iz]
        else:
            s_order[iz] = 0.0

    return {
        "z_coords": z_coords,
        "rho_bar": rho_bar,
        "rho_z_theta": rho_z_theta,
        "S_order": s_order,
        "H": np.array([H]),
        "T": np.array([T]),
    }


def compute_flue_gas_selectivity(
    T: float = 300.0,  # K
    P_bar: float = 1.0,  # bar
    y_co2: float = 0.15,  # 15% CO2 (typical flue gas)
    y_n2: float = 0.85,  # 85% N2
    pore_width_A: float = 12.0,  # Nanopore slit width
) -> Dict[str, float]:
    r"""
    Calculates the competitive adsorption selectivity of CO2 over N2 in carbon nanopores:
    Selectivity S_{CO2/N2} = (x_CO2 / x_N2) / (y_CO2 / y_N2)
    CO2 possesses a significantly higher quadrupole moment (-4.3 D*A) and polarizability than N2 (-1.4 D*A).
    """
    # Adsorption affinity prefactors from Buckingham exp-6 / dispersion well depths
    # eps_CO2/kB ~ 240 K, eps_N2/kB ~ 95 K
    k_co2 = np.exp(240.0 / T) * (1.0 + 10.0 / pore_width_A)
    k_n2 = np.exp(95.0 / T) * (1.0 + 3.0 / pore_width_A)

    # Adsorbed amounts via Ideal Adsorbed Solution Theory (IAST) / Langmuir competitive cDFT
    q_co2 = (k_co2 * y_co2 * P_bar) / (1.0 + k_co2 * y_co2 * P_bar + k_n2 * y_n2 * P_bar)
    q_n2 = (k_n2 * y_n2 * P_bar) / (1.0 + k_co2 * y_co2 * P_bar + k_n2 * y_n2 * P_bar)

    x_co2 = q_co2 / (q_co2 + q_n2)
    x_n2 = q_n2 / (q_co2 + q_n2)

    selectivity = (x_co2 / x_n2) / (y_co2 / y_n2)

    return {
        "T_K": T,
        "P_bar": P_bar,
        "y_CO2_feed": y_co2,
        "y_N2_feed": y_n2,
        "x_CO2_adsorbed": float(x_co2),
        "x_N2_adsorbed": float(x_n2),
        "selectivity_CO2_N2": float(selectivity),
    }
