"""
Unit & Integration Tests for TanhStretchedGrid1D, Flat-Memory Topology & ATM 3-Body Dispersion.
"""

import numpy as np
import pytest

from dens_city.pipelines.argon.coexistence import (
    solve_argon_coexistence_point,
)
from dens_city.pipelines.interfaces.wetting import compute_capillary_drying_gap
from dens_city.pipelines.methane.shale import (
    solve_methane_coexistence_point,
)
from dens_city.solver.quantum_surrogates import (
    compute_atm_chemical_potential_correction,
    compute_atm_mca_second_order,
    compute_atm_pressure_correction,
)
from dens_city.solver.stretched_grid import TanhStretchedGrid1D


def test_tanh_grid_construction_and_jacobian():
    L_z = 20.0
    grid_size = 256
    alpha = 2.8
    grid = TanhStretchedGrid1D(L_z=L_z, grid_size=grid_size, alpha=alpha)

    # Check coordinate bounds
    assert grid.z_coords[0] == pytest.approx(0.0, abs=1e-12)
    assert grid.z_coords[-1] == pytest.approx(L_z, abs=1e-12)

    # Check Jacobian consistency: numerical gradient of z vs analytical jacobian
    dz_ds_num = np.gradient(grid.z_coords, grid.ds, edge_order=2)
    np.testing.assert_allclose(grid.jacobian, dz_ds_num, rtol=1e-3, atol=1e-3)

    # Check total volume integral normalization: sum(weights) == L_z
    assert np.sum(grid.weights) == pytest.approx(L_z, abs=1e-10)


def test_sub_angstrom_interfacial_resolution():
    L_z = 20.0
    grid_size = 256
    alpha = 2.8
    grid = TanhStretchedGrid1D(L_z=L_z, grid_size=grid_size, alpha=alpha)

    # Left and right boundary spacing must be <= 0.015 Angstroms
    assert grid.dz_wall_left < 0.015
    assert grid.dz_wall_right < 0.015
    # Center spacing should be coarser
    assert grid.dz_center > 0.10


def test_quadrature_integration_accuracy():
    L_z = 20.0
    grid = TanhStretchedGrid1D(L_z=L_z, grid_size=256, alpha=2.8)
    z = grid.z_coords

    # Test integral of smooth Gaussian centered in box: \int exp(-(z-10)^2 / 2) dz
    f = np.exp(-((z - 10.0) ** 2) / 2.0)
    integral_num = grid.integrate(f)
    integral_exact = np.sqrt(2.0 * np.pi)  # ~ 2.506628

    assert integral_num == pytest.approx(integral_exact, rel=1e-3)


def test_atm_closed_form_derivatives():
    rho = 0.0214
    eta = 0.45
    T = 85.0
    nu_atm = 8.495e5
    sigma = 3.405

    a_atm = compute_atm_mca_second_order(rho, eta, T, nu_atm=nu_atm, sigma=sigma)
    p_atm = compute_atm_pressure_correction(rho, eta, T, nu_atm=nu_atm, sigma=sigma)
    mu_atm = compute_atm_chemical_potential_correction(rho, eta, T, nu_atm=nu_atm, sigma=sigma)

    assert a_atm > 0.0
    assert p_atm != 0.0
    assert mu_atm > 0.0


def test_argon_atm_coexistence_accuracy():
    # NIST Argon liquid density at 85K is 0.02138 A^-3
    res = solve_argon_coexistence_point(85.0)
    assert res is not None
    rl, rv, psat = res

    err_pct = abs(rl - 0.02138) / 0.02138 * 100.0
    # Must be within < 1.5% error
    assert err_pct < 1.5, f"Argon 85K liquid density error is {err_pct:.2f}%, expected < 1.5%"
    assert 0.0208 <= rl <= 0.0216


def test_methane_atm_coexistence_accuracy():
    # NIST Methane liquid density at 111.66K is 0.01586 A^-3
    res = solve_methane_coexistence_point(111.66)
    assert res is not None
    rl, rv, psat = res

    err_pct = abs(rl - 0.01586) / 0.01586 * 100.0
    # Must be within < 1.5% error
    assert err_pct < 1.5, f"Methane 111K liquid density error is {err_pct:.2f}%, expected < 1.5%"
    assert 0.0155 <= rl <= 0.0163


def test_cavitation_variance_suppression():
    res = compute_capillary_drying_gap(theta_deg=110.0, use_stretched_grid=True)
    assert res["cavitation_detected"] is True
    # Max variance in cavitation vapor core must be suppressed below 0.002 (< 0.2%)
    assert res["max_vapor_variance"] < 0.002
