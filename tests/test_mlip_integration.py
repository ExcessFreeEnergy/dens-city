"""
Integration & Validation Tests for Equivariant MLIP & Quantum Fluid Surrogates.
"""

import numpy as np

from dens_city.mlip.oracle import EquivariantMLIPOracle, QuantumFluidSurrogate


def test_quantum_fluid_surrogate_water_scan():
    """Verifies QuantumFluidSurrogate for water (SCAN functional) computes physical potentials."""
    surrogate = QuantumFluidSurrogate(material="water", xc_functional="SCAN")
    r_mesh = np.linspace(0.5, 8.0, 100)
    v_sr = surrogate.evaluate_lmft_short_range_potential(r_mesh, T=300.0)

    # Near contact (r ~ 3.16 A), short range potential should be negative/attractive
    min_idx = np.argmin(v_sr)
    r_min = r_mesh[min_idx]
    assert 2.5 <= r_min <= 4.0
    assert v_sr[min_idx] < 0.0

    # Under overlap (r = 0.5 A), ZBL repulsive shield should ensure large positive energy
    assert v_sr[0] > 100.0


def test_quantum_fluid_surrogate_effective_diameter():
    """Verifies quantum Barker-Henderson effective diameter d_eff(T) for water."""
    surrogate = QuantumFluidSurrogate(material="water")
    d_eff = surrogate.compute_effective_diameter(T=300.0)

    # Physical water hard core diameter is ~2.8 - 3.2 A
    assert 2.5 <= d_eff <= 3.4


def test_mlip_oracle_oz_inversion():
    """Verifies EquivariantMLIPOracle inverts S(k) to c(r) with zero-wavenumber limit."""
    oracle = EquivariantMLIPOracle()
    k_grid = np.linspace(0.1, 15.0, 150)
    # Simple hard-sphere-like S(k)
    s_k = 1.0 / (1.0 + 0.8 * np.exp(-0.2 * k_grid))
    r_grid = np.linspace(0.0, 10.0, 100)

    c_hat_k, c_r, c_hat_zero = oracle.compute_direct_correlation_from_sk(
        k_grid=k_grid,
        s_k=s_k,
        rho_bulk=0.033,
        r_grid=r_grid,
        r_box=10.0,
    )

    assert c_hat_k.shape == (150,)
    assert c_r.shape == (100,)
    assert not np.isnan(c_hat_zero)
    assert not np.isinf(c_hat_zero)


def test_mlip_oracle_fmt_mca_calibration():
    """Verifies FMT+MCA calibration parameters extraction."""
    oracle = EquivariantMLIPOracle()
    calib = oracle.calibrate_fmt_mca_parameters(T=300.0, rho_bulk=0.033)

    assert "d_eff" in calib
    assert "eta" in calib
    assert "a_atm_K" in calib
    assert 0.1 <= calib["eta"] <= 0.6
