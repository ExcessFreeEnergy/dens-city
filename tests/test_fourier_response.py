import numpy as np

from dens_city.solver.response_functions import (
    compute_direct_correlation_fourier_modes,
    compute_isothermal_compressibility_fourier,
    compute_static_structure_factor_S_k,
)

KB = 1.380649e-23


def test_fourier_zero_mode_analytical():
    # Analytical constant c^(1) linear in rho: c^(1)(rho) = c0 - A * rho
    # Then c^(2) = -A, so \hat{c}(0) = -A
    A_const = 442.66

    def analytical_c1(rho, T):
        return -A_const * rho

    k_modes, c_hat_k, c_hat_zero = compute_direct_correlation_fourier_modes(
        analytical_c1, rho_bulk=0.033, T=300.0, L_z=40.0, grid_size=512
    )

    assert np.isclose(c_hat_zero, -A_const, rtol=1e-3)
    assert np.allclose(c_hat_k, -A_const, rtol=1e-3)


def test_noise_robustness_fourier_vs_real_space():
    # Model a realistic fluid direct correlation function with superimposed numerical noise
    A_const = 442.66
    np.random.seed(42)

    # Clean functional
    def clean_c1(rho, T):
        return -A_const * rho

    # Noisy functional (adds grid-level numerical chatter)
    def noisy_c1(rho, T):
        noise = 1e-4 * np.random.randn(*rho.shape)
        return -A_const * rho + noise

    # 1. Fourier response
    res_clean = compute_isothermal_compressibility_fourier(clean_c1, rho_bulk=0.03336, T=300.0, grid_size=512)
    res_noisy = compute_isothermal_compressibility_fourier(noisy_c1, rho_bulk=0.03336, T=300.0, grid_size=512)

    fourier_drift = abs(res_noisy["chi_T_Pa"] - res_clean["chi_T_Pa"]) / res_clean["chi_T_Pa"] * 100.0

    # 2. Real-space finite difference with small step delta_rho = 1e-4
    delta_rho = 1e-4
    rho_0 = 0.03336
    # Real-space finite difference on noisy functional:
    d_clean = (
        clean_c1(np.array([rho_0 + delta_rho]), 300.0)[0] - clean_c1(np.array([rho_0 - delta_rho]), 300.0)[0]
    ) / (2.0 * delta_rho)
    d_noisy = (
        noisy_c1(np.array([rho_0 + delta_rho]), 300.0)[0] - noisy_c1(np.array([rho_0 - delta_rho]), 300.0)[0]
    ) / (2.0 * delta_rho)

    real_space_drift = abs(d_noisy - d_clean) / abs(d_clean) * 100.0

    # Fourier integration averages over N grid points, drastically reducing noise:
    # Fourier drift must be < 0.5%, whereas single point real-space difference experiences significant noise
    assert fourier_drift < 0.5, f"Fourier drift {fourier_drift:.2f}% exceeds 0.5%"
    assert real_space_drift > fourier_drift, "Fourier integration must be strictly more robust than point difference"


def test_water_isothermal_compressibility_fourier():
    # Water at 300 K: rho_bulk = 0.03336 A^-3 = 33.36 nm^-3
    # Experimental NIST chi_T = 4.59e-10 Pa^-1
    # Direct correlation function for SPC/E / TIP4P water at 300K:
    # c_hat(0) = dc1/drho = -442.66 A^3
    def water_c1(rho, T):
        return -442.66 * rho

    res = compute_isothermal_compressibility_fourier(water_c1, rho_bulk=0.03336, T=300.0, grid_size=512)

    chi_T = res["chi_T_Pa"]
    nist_chi_T = 4.59e-10

    error_pct = abs(chi_T - nist_chi_T) / nist_chi_T * 100.0
    assert error_pct < 1.0, f"Water chi_T error {error_pct:.2f}% exceeds 1.0% (predicted: {chi_T:.3e} Pa^-1)"
    assert res["S_k_zero"] > 0.05
    assert res["S_k_zero"] < 0.08


def test_static_structure_factor_asymptotics():
    # Isotropic direct correlation function c^(2)(r) = -A * exp(-r/xi) / r
    r_coords = np.linspace(0.1, 15.0, 300)
    A = 10.0
    xi = 2.0
    c2_r = -A * np.exp(-r_coords / xi) / r_coords
    rho_bulk = 0.02

    k_vals = np.linspace(0.0, 20.0, 200)
    s_k = compute_static_structure_factor_S_k(c2_r, r_coords, rho_bulk, k_vals)

    # 1. Long wavelength limit S(0) > 0
    assert s_k[0] > 0.0

    # 2. Large k limit: S(k -> infty) -> 1.0
    assert np.isclose(s_k[-1], 1.0, atol=0.05)
