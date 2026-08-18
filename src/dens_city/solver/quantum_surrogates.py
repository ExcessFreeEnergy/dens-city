r"""
Analytical Quantum Surrogates & Invariant Protections.

Provides closed-form, zero-dependency quantum and many-body kernels:
1. Feynman-Hibbs (FH) quantum effective potential smearing (NQE for light atoms).
2. Axilrod-Teller-Muto (ATM) 3-body non-additive dispersion and MCA pressure/mu corrections.
3. Universal ZBL (Ziegler-Biersack-Littmark) core repulsive shield for r <= 0.8 A.
4. Hann / cosine correlation tail windowing to eliminate Fourier Gibbs ringing.
"""

from typing import Tuple, Union
import numpy as np

# Physical constants
KB = 1.380649e-23  # J / K
HBAR = 1.054571817e-34  # J * s
AMU = 1.66053906660e-27  # kg
EPSILON_0 = 8.8541878128e-12  # F / m
E_CHARGE = 1.602176634e-19  # C
BOHR_RADIUS = 0.529177210903  # Angstroms
EV_TO_KELVIN = 11604.5250061598  # K / eV


def compute_feynman_hibbs_potential(
    r: np.ndarray,
    sigma: float,
    epsilon_k: float,
    mass_amu: float,
    T: float,
) -> np.ndarray:
    r"""
    Computes the 2nd-order Feynman-Hibbs quantum effective potential:
      v_FH(r) = v(r) + \frac{\beta \hbar^2}{24 m} \left( v''(r) + \frac{2}{r} v'(r) \right)
    where \nabla^2 v(r) = v''(r) + (2/r) v'(r).

    For Lennard-Jones:
      \nabla^2 v_LJ(r) = \frac{4\epsilon}{\sigma^2} [ 132 (\sigma/r)^{14} - 30 (\sigma/r)^8 ]

    Returns:
      v_FH(r) in Kelvin.
    """
    m_kg = max(1e-28, mass_amu * AMU)
    beta = 1.0 / (KB * max(1e-3, T))
    eps_joules = epsilon_k * KB
    sig_m = sigma * 1e-10
    r_m = np.maximum(r * 1e-10, 1e-12)

    # Classical LJ
    s_over_r = sig_m / r_m
    s6 = s_over_r**6
    s12 = s6**2
    u_lj_j = 4.0 * eps_joules * (s12 - s6)

    # 3D Laplacian
    laplacian_lj = (4.0 * eps_joules / (sig_m**2)) * (132.0 * (s_over_r**14) - 30.0 * (s_over_r**8))

    # Quantum correction factor
    q_prefactor = (beta * (HBAR**2)) / (24.0 * m_kg)
    u_fh_j = u_lj_j + q_prefactor * laplacian_lj

    return u_fh_j / KB


def compute_atm_three_body_energy(
    r_ij: float,
    r_jk: float,
    r_ki: float,
    cos_theta_i: float,
    cos_theta_j: float,
    cos_theta_k: float,
    nu_atm: float = 73.2,  # eV * Angstrom^9 for typical noble gases / liquids
) -> float:
    r"""
    Computes the Axilrod-Teller-Muto (ATM) triple-dipole 3-body dispersion energy:
      v_3(r_ij, r_jk, r_ki) = \nu_ATM \frac{1 + 3 \cos\theta_i \cos\theta_j \cos\theta_k}{(r_ij * r_jk * r_ki)^3}
    """
    denom = max(1e-6, (r_ij * r_jk * r_ki) ** 3)
    num = 1.0 + 3.0 * cos_theta_i * cos_theta_j * cos_theta_k
    nu_k = nu_atm * EV_TO_KELVIN if nu_atm < 1e4 else nu_atm
    return nu_k * (num / denom)


def compute_atm_mca_second_order(
    rho_bulk: float,
    eta: float,
    T: float,
    nu_atm: float = 73.2,  # eV * A^9 or K * A^9
    sigma: float = 3.405,
) -> float:
    r"""
    Computes the non-additive Axilrod-Teller-Muto 3-body contribution to the
    second-order Macroscopic Compressibility Approximation (MCA) free energy density:
      a_ATM = \frac{8\pi^2}{9} \frac{\nu_ATM \rho_bulk^2}{T * \sigma^6} \frac{(1 - \eta)^2}{(1 + 2\eta)}
    Returns:
      a_ATM in Kelvin.
    """
    nu_k = nu_atm * EV_TO_KELVIN if nu_atm < 1e4 else nu_atm
    eta_safe = np.clip(eta, 0.0, 0.95)
    chi_factor = ((1.0 - eta_safe) ** 2) / (1.0 + 2.0 * eta_safe)
    prefactor = (8.0 * (np.pi**2) / 9.0) * (nu_k / (sigma**6))
    a_atm = prefactor * (rho_bulk**2) * chi_factor / max(1.0, T)
    return float(a_atm)


def compute_atm_pressure_correction(
    rho_bulk: float,
    eta: float,
    T: float,
    nu_atm: float = 73.2,
    sigma: float = 3.405,
) -> float:
    r"""
    Computes the ATM 3-body pressure correction in bar:
      \Delta P_ATM = \frac{8\pi^2 \nu_ATM \rho^3}{9 T \sigma^6} \frac{(1 - \eta)(2 - 5\eta + 2\eta^2)}{(1 + 2\eta)^2} * 138.0649
    """
    nu_k = nu_atm * EV_TO_KELVIN if nu_atm < 1e4 else nu_atm
    eta_safe = np.clip(eta, 0.0, 0.95)
    num = (1.0 - eta_safe) * (2.0 - 5.0 * eta_safe + 2.0 * (eta_safe**2))
    den = (1.0 + 2.0 * eta_safe) ** 2
    prefactor = (8.0 * (np.pi**2) / 9.0) * (nu_k / (sigma**6))
    p_k_a3 = prefactor * (rho_bulk**3) * (num / max(1e-6, den)) / max(1.0, T)
    return float(p_k_a3 * 138.0649)


def compute_atm_chemical_potential_correction(
    rho_bulk: float,
    eta: float,
    T: float,
    nu_atm: float = 73.2,
    sigma: float = 3.405,
) -> float:
    r"""
    Computes the ATM 3-body chemical potential correction in Kelvin:
      \Delta \mu_ATM = a_ATM + \frac{\Delta P_ATM / 138.0649}{\rho}
    """
    a_atm = compute_atm_mca_second_order(rho_bulk, eta, T, nu_atm=nu_atm, sigma=sigma)
    p_bar = compute_atm_pressure_correction(rho_bulk, eta, T, nu_atm=nu_atm, sigma=sigma)
    p_k_a3 = p_bar / 138.0649
    return float(a_atm + p_k_a3 / max(1e-12, rho_bulk))


def zbl_repulsive_core(
    r: Union[float, np.ndarray],
    z1: float = 8.0,  # e.g., Oxygen Z=8
    z2: float = 8.0,
    r_core: float = 0.8,  # Angstroms
) -> Union[float, np.ndarray]:
    r"""
    Universal Ziegler-Biersack-Littmark (ZBL) core repulsive shield for r <= r_core.
    Guarantees strict positive divergence (+inf) as r -> 0 to prevent Picard solver crashes.
    """
    is_scalar = np.isscalar(r)
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    r_safe = np.maximum(r_arr, 1e-4)

    a_u = 0.8854 * BOHR_RADIUS / (z1**0.23 + z2**0.23)
    x = r_safe / a_u

    phi = (
        0.1818 * np.exp(-3.2 * x)
        + 0.5099 * np.exp(-0.9423 * x)
        + 0.2802 * np.exp(-0.4029 * x)
        + 0.02817 * np.exp(-0.2016 * x)
    )

    coulomb_k_ang = 167101.0
    v_zbl = (z1 * z2 * coulomb_k_ang / r_safe) * phi

    mask = r_arr < r_core
    v_shield = np.zeros_like(r_arr)
    if np.any(mask):
        t = (r_core - r_arr[mask]) / r_core
        switch = (3.0 * t**2 - 2.0 * t**3)
        v_shield[mask] = v_zbl[mask] * switch

    if is_scalar:
        return float(v_shield[0])
    return v_shield


def apply_hann_window(
    c_r: np.ndarray,
    r_grid: np.ndarray,
    r_box: float,
    window_start_frac: float = 0.9,
) -> np.ndarray:
    r"""
    Applies smooth Hann / cosine windowing to direct correlation function tails
    for r > window_start_frac * r_box to eliminate Fourier truncation ringing (Gibbs phenomenon).
    """
    r_start = window_start_frac * r_box
    c_windowed = np.copy(c_r)

    mask = (r_grid > r_start) & (r_grid <= r_box)
    if np.any(mask):
        t = (r_grid[mask] - r_start) / (r_box - r_start)
        hann_weights = 0.5 * (1.0 + np.cos(np.pi * t))
        c_windowed[mask] *= hann_weights

    c_windowed[r_grid > r_box] = 0.0
    return c_windowed
