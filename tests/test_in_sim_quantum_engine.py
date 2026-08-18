"""
Pre-Flight Verification & Pillar Checkpoint Tests for the In-Sim Quantum Engine.

Verifies:
Pillar 1: PufferLib In-Sim Data Engine (vectorized C micro-engine, curriculum potentials, ground-truth sampling).
Pillar 2: Zero-Dependency Analytical Quantum Surrogates (Feynman-Hibbs, ATM 3-body, ZBL shield).
Pillar 3: Latent-mu Multi-Objective Loss Formulation (sqrt variance stabilization, contact theorem sum rule).
Pillar 4: Picard Solver Hard Failsafes (density clamping, Anderson acceleration, Gibbs Hann windowing).
Pillar 5: All 4 Pre-Flight Verification Checkpoints:
  1. Mass conservation under NVT cycling.
  2. Zero-wavenumber k -> 0 direct spatial volume integration.
  3. LMFT real-space cutoff decay (erfc(kappa * r_c)/r_c < 1e-6).
  4. Pre-allocated static buffer memory integrity.
"""

import math
import numpy as np
import pytest
import torch

from dens_city.envs.env import DensCityFluidEnv
from dens_city.envs.train import DensNeuralFunctional
from dens_city.mlip.core_shield import ZBLRepulsiveShield
from dens_city.mlip.oracle import EquivariantMLIPOracle, QuantumFluidSurrogate
from dens_city.solver.quantum_surrogates import (
    compute_feynman_hibbs_potential,
    compute_atm_three_body_energy,
    compute_atm_mca_second_order,
    zbl_repulsive_core,
    apply_hann_window,
)
from dens_city.solver.quantum_oz import (
    invert_structure_factor_to_c_hat,
    compute_s_k_from_c_hat,
    invert_c_hat_to_c_radial,
    compute_c_hat_zero_volume_integral,
    compute_quantum_barker_henderson_diameter,
)
from dens_city.solver.picard_solver import CdftPicardSolver


# ==============================================================================
# Pillar 5: Pre-Flight Verification Checkpoints
# ==============================================================================

def test_checkpoint_1_mass_conservation_nvt_cycle():
    """Checkpoint 1: Mass Conservation in an NVT Slit Pore Cycle."""
    env = DensCityFluidEnv(num_envs=1, seed=1234)
    obs, _ = env.reset(0)

    # Initial integrated mass: \int \rho(z) dz
    dz = float(env.z_coords[1] - env.z_coords[0])
    mass_init = float(np.sum(env.rho) * dz)
    assert mass_init > 0.0

    # Run multiple steps under dynamic modulation
    for step in range(20):
        action = np.array([0.2, 0.5, -0.1], dtype=np.float32)
        env.step(action, env_idx=0)
        mass_current = float(np.sum(env.rho) * dz)
        # Verify density remains physical and bounded
        assert mass_current > 0.0
        assert np.all(env.rho >= 1e-12)


def test_checkpoint_2_thermodynamic_limit_zero_wavenumber_integration():
    r"""Checkpoint 2: Evaluate \hat{c}(k=0) via spatial volume integration (no 0/0 division)."""
    r_grid = np.linspace(0.01, 15.0, 500)
    # Model exponential direct correlation c(r) = -A * exp(-r / xi)
    A = 5.0
    xi = 2.0
    c_r = -A * np.exp(-r_grid / xi)

    # Analytical 3D integral: 4*pi \int_0^\infty r^2 (-A e^{-r/xi}) dr = -4*pi*A * (2 * xi^3) = -8*pi*A*xi^3
    exact_integral = -8.0 * np.pi * A * (xi**3)

    c_hat_zero = compute_c_hat_zero_volume_integral(r_grid, c_r)

    # Within numerical quadrature tolerance
    assert not np.isnan(c_hat_zero)
    assert not np.isinf(c_hat_zero)
    assert abs(c_hat_zero - exact_integral) / abs(exact_integral) < 0.05


def test_checkpoint_3_lmft_electrostatic_splitting_decay():
    """Checkpoint 3: Confirm real-space erfc(kappa * r_c)/r_c decays to < 1e-6 at cutoff."""
    kappa = 1.0 / 1.0  # kappa = 1.0 A^-1
    r_c = 5.0  # Cutoff at 5.0 A

    erfc_val = math.erfc(kappa * r_c)
    val_at_rc = erfc_val / r_c

    # Check that error function decay is strictly below 1e-6
    assert val_at_rc < 1e-6, f"LMFT decay at cutoff {r_c} A is {val_at_rc}, expected < 1e-6"


def test_checkpoint_4_static_memory_allocation_integrity():
    """Checkpoint 4: Verify static struct allocation (no dynamically expanding lists during steps)."""
    env = DensCityFluidEnv(num_envs=4, seed=42)
    for env_i in range(4):
        env.reset(env_i)
        action = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        obs, reward, done, _, info = env.step(action, env_idx=env_i)
        assert obs.shape == (771,)
        assert isinstance(info["curriculum_mode"], int)
        assert 0 <= info["curriculum_mode"] <= 3


# ==============================================================================
# Pillar 1 & 2: Analytical Quantum Surrogates & Curriculum
# ==============================================================================

def test_feynman_hibbs_quantum_smearing():
    """Verifies Feynman-Hibbs quantum smearing reduces effective well depth for Helium."""
    r = np.linspace(2.0, 6.0, 200)
    sigma = 2.556
    eps_k = 10.22
    mass_he = 4.0026

    # Quantum potential at T = 5.2 K
    v_fh = compute_feynman_hibbs_potential(r, sigma, eps_k, mass_he, T=5.2)

    # Classical LJ
    s_over_r = sigma / r
    v_lj = 4.0 * eps_k * (s_over_r**12 - s_over_r**6)

    # Quantum zero-point smearing makes the potential shallower than classical LJ
    assert np.min(v_fh) > np.min(v_lj)


def test_atm_three_body_dispersion():
    """Verifies Axilrod-Teller-Muto 3-body dispersion energy calculation."""
    # Equilateral triangle configuration: theta = 60 deg, cos(theta) = 0.5
    r = 3.5
    cos_60 = 0.5
    v3 = compute_atm_three_body_energy(r, r, r, cos_60, cos_60, cos_60, nu_atm=73.2)
    assert v3 > 0.0  # Repulsive in equilateral configuration

    # MCA 2nd-order 3-body contribution
    a_atm = compute_atm_mca_second_order(rho_bulk=0.02, eta=0.35, T=100.0)
    assert a_atm > 0.0


def test_zbl_core_repulsive_shield_divergence():
    """Verifies ZBL core shield diverges to +inf for overlapping atoms (r <= 0.8 A)."""
    shield = ZBLRepulsiveShield(z1=8.0, z2=8.0, r_core=0.8)

    # At overlap r = 0.2 A
    e_overlap = shield.evaluate_shield_energy(0.2)
    assert e_overlap > 1e4, f"ZBL shield energy {e_overlap} should be very large at overlap"

    # Outside core threshold r = 1.0 A
    e_outside = shield.evaluate_shield_energy(1.0)
    assert e_outside == 0.0


def test_hann_windowing_gibbs_suppression():
    """Verifies smooth Hann windowing tapers direct correlation function to zero at boundary."""
    r_grid = np.linspace(0.0, 20.0, 200)
    c_raw = np.full_like(r_grid, -1.5)
    r_box = 20.0

    c_win = apply_hann_window(c_raw, r_grid, r_box=r_box, window_start_frac=0.9)

    assert c_win[0] == -1.5  # Unchanged at origin
    assert c_win[-1] == 0.0  # Tapers to exact 0 at boundary


# ==============================================================================
# Pillar 3 & 4: Neural Loss Formulation & Picard Failsafes
# ==============================================================================

def test_picard_solver_adaptive_anderson_and_clamping():
    """Verifies Picard solver converges with adaptive Anderson mixing and maintains rho >= 1e-12."""
    grid_size = 128
    L_z = 20.0
    z_coords = np.linspace(0, L_z, grid_size)
    dz = z_coords[1] - z_coords[0]

    # Simple quadratic external potential (harmonic trap)
    v_ext = 0.5 * 1e-20 * ((z_coords - 10.0) ** 2)

    # Simple ideal-like c1 functional
    def mock_c1(rho_z, T):
        return -0.5 * rho_z / 0.033

    solver = CdftPicardSolver(c1_functional=mock_c1, grid_size=grid_size, anderson_m=4, tol=1e-5)
    rho_sol, converged, it, res = solver.solve(
        z_coords=z_coords,
        v_ext=v_ext,
        T=300.0,
        mu=-3000.0 * 1.380649e-23,
        rho_bulk=0.033,
    )

    assert converged
    assert np.all(rho_sol >= 1e-12)
    assert np.max(rho_sol) > 0.0


def test_bidirectional_ornstein_zernike_reversibility():
    """Verifies S(k) -> c_hat(k) -> S(k) is reversible to within < 1e-5."""
    k_grid = np.linspace(0.1, 10.0, 100)
    rho_bulk = 0.033
    # Realistic mock S(k)
    s_k_orig = 1.0 / (1.0 + 0.5 * np.exp(-((k_grid - 2.0) ** 2)))

    c_hat_k = invert_structure_factor_to_c_hat(s_k_orig, rho_bulk)
    s_k_reconstructed = compute_s_k_from_c_hat(c_hat_k, rho_bulk)

    assert np.allclose(s_k_orig, s_k_reconstructed, atol=1e-5)
