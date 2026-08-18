from typing import Callable
import numpy as np

KB = 1.380649e-23


def compute_excess_free_energy(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    rho: np.ndarray,
    T: float,
    z_coords: np.ndarray,
    num_lambda_steps: int = 20,
) -> float:
    r"""
    Computes the excess Helmholtz free energy functional F_intr^(ex)[rho]
    via numerical functional line integration:
    F_intr^(ex)[rho] = - k_B T \int_0^1 d\lambda \int_0^L dz c^(1)(z; [\lambda \rho], T) \rho(z)
    """
    N = len(rho)
    dz = z_coords[1] - z_coords[0] if N > 1 else 1.0

    lambdas, weights = np.polynomial.legendre.leggauss(num_lambda_steps)
    # Map from [-1, 1] to [0, 1]
    lambdas = 0.5 * (lambdas + 1.0)
    weights = 0.5 * weights

    f_ex_integral = 0.0
    for lam, w in zip(lambdas, weights):
        rho_scaled = lam * rho
        c1 = c1_functional(rho_scaled, T)
        spatial_integral = np.sum(c1 * rho) * dz
        f_ex_integral += w * spatial_integral

    return -KB * T * f_ex_integral


def compute_grand_potential(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    rho: np.ndarray,
    T: float,
    mu: float,
    v_ext: np.ndarray,
    z_coords: np.ndarray,
    lambda_db: float = 1.0,
) -> float:
    r"""
    Computes the Grand Potential functional \Omega[\rho] = F_id + F_ex + \int \rho (V_ext - \mu) dz.
    """
    N = len(rho)
    dz = z_coords[1] - z_coords[0] if N > 1 else 1.0

    # Ideal free energy: F_id = k_B T \int \rho(z) [ln(\Lambda^3 \rho(z)) - 1] dz
    rho_safe = np.maximum(rho, 1e-15)
    f_id = KB * T * np.sum(rho_safe * (np.log(rho_safe * (lambda_db ** 3)) - 1.0)) * dz

    # Excess free energy via line integration
    f_ex = compute_excess_free_energy(c1_functional, rho, T, z_coords)

    # External coupling: \int \rho(z) [V_ext(z) - \mu] dz
    coupling = np.sum(rho * (v_ext - mu)) * dz

    return f_id + f_ex + coupling


def compute_bulk_pressure(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    rho_bulk: float,
    T: float,
    L_z: float = 20.0,
    grid_size: int = 256,
) -> float:
    """
    Computes the bulk pressure P(rho_b, T) directly from c^(1) and F_ex:
    P(rho_b, T) = k_B T \rho_b (1 - c^(1)(rho_b, T)) - F_ex(rho_b) / V
    """
    z_coords = np.linspace(0, L_z, grid_size)
    rho_arr = np.full(grid_size, rho_bulk)

    c1_arr = c1_functional(rho_arr, T)
    c1_val = float(np.mean(c1_arr))

    f_ex = compute_excess_free_energy(c1_functional, rho_arr, T, z_coords)
    volume = L_z * 1.0 * 1.0 # 1D per unit area
    f_ex_density = f_ex / volume

    pressure = KB * T * rho_bulk * (1.0 - c1_val) - f_ex_density
    return pressure
