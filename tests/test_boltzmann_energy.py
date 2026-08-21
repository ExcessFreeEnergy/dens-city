"""
Unit and physical validation tests for Microscopic Hamiltonian in dens_city.boltzmann.energy.
Verifies exact pairwise minimums, spherical cutoffs, PBC minimum image convention,
external wall steric barriers, Coulomb interactions, batch invariance, and autograd differentiability.
"""

import math
import numpy as np
import pytest
from tinygrad import Tensor
from dens_city.boltzmann.energy import MicroscopicEnergy, regularize_energy
from dens_city.materials import MaterialLoader


def test_argon_dimer_ground_truth_minimum():
    """
    Validates that two Argon atoms (sigma = 3.4 Å, epsilon = 119.8 K)
    at the Lennard-Jones potential minimum r = 2^(1/6) * sigma
    evaluate to exactly -119.8 K.
    """
    sigma = 3.4
    epsilon = 119.8
    r_min = (2.0 ** (1.0 / 6.0)) * sigma  # ~3.81637 Å

    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        charges=[0.0, 0.0],
        box_size=(30.0, 30.0, 40.0),
        r_cut=12.0,
    )

    # Place pair at r = r_min in the bulk slit center (z = 20 Å)
    pos = Tensor([[0.0, 0.0, 20.0], [r_min, 0.0, 20.0]])
    u_pair = energy_fn.compute_pair_energy(pos, shift=False)

    assert math.isclose(u_pair.item(), -epsilon, rel_tol=1e-5), (
        f"Argon dimer minimum energy {u_pair.item()} != {-epsilon} K"
    )


def test_spherical_cutoff_masking():
    """
    Validates that particle pairs separated by r > r_cut are strictly zeroed out by the spherical cutoff mask.
    """
    sigma = 3.4
    epsilon = 119.8
    r_cut = 12.0

    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(30.0, 30.0, 40.0),
        r_cut=r_cut,
    )

    # Separation r = 16.0 Å > r_cut
    pos = Tensor([[0.0, 0.0, 20.0], [16.0, 0.0, 20.0]])
    u_pair = energy_fn.compute_pair_energy(pos, shift=False)

    assert u_pair.item() == 0.0, f"Energy outside r_cut must be strictly zero, got {u_pair.item()}"


def test_continuous_boundary_shift():
    """
    Validates that shift=True smoothly zeroes the potential at r = r_cut to eliminate gradient singularities.
    """
    sigma = 3.4
    epsilon = 119.8
    r_cut = 12.0

    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(30.0, 30.0, 40.0),
        r_cut=r_cut,
    )

    # Position at r -> r_cut
    pos_at_cut = Tensor([[0.0, 0.0, 20.0], [r_cut - 1e-4, 0.0, 20.0]])
    u_pair_shifted = energy_fn.compute_pair_energy(pos_at_cut, shift=True)

    assert math.isclose(u_pair_shifted.item(), 0.0, abs_tol=1e-3), (
        f"Shifted potential at r_cut must smoothly vanish, got {u_pair_shifted.item()}"
    )


def test_minimum_image_convention_pbc():
    """
    Validates that periodic boundary wrapping across box edges in X and Y reproduces exact minimum image distances.
    """
    sigma = 3.4
    epsilon = 119.8
    r_min = (2.0 ** (1.0 / 6.0)) * sigma
    lx, ly = 30.0, 30.0

    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(lx, ly, 40.0),
        r_cut=12.0,
    )

    # Pair inside box
    pos_normal = Tensor([[5.0, 5.0, 20.0], [5.0 + r_min, 5.0, 20.0]])
    u_normal = energy_fn.compute_pair_energy(pos_normal, shift=False)

    # Pair wrapped across X boundary: atom 1 at x = 0.5, atom 2 at x = lx - (r_min - 0.5)
    pos_wrapped = Tensor([[0.5, 5.0, 20.0], [lx + 0.5 - r_min, 5.0, 20.0]])
    u_wrapped = energy_fn.compute_pair_energy(pos_wrapped, shift=False)

    assert math.isclose(u_normal.item(), u_wrapped.item(), rel_tol=1e-5), (
        f"PBC wrapped energy {u_wrapped.item()} != unwrapped energy {u_normal.item()}"
    )


def test_slit_wall_potential_in_z():
    """
    Validates that particles inside the steric wall core trigger the 10^6 K barrier
    and particles in the fluid channel receive finite physical wall potentials.
    """
    argon = MaterialLoader.load_material("argon")
    energy_fn = MicroscopicEnergy(
        material=argon,
        box_size=(30.0, 30.0, 40.0),
    )

    # 1. Atom inside left steric core (z = 1.0 <= 1.7 Å)
    pos_core = Tensor([[10.0, 10.0, 1.0]])
    u_wall_core = energy_fn.compute_wall_energy(pos_core)
    assert u_wall_core.item() >= 1e6, f"Steric core must have infinite potential barrier, got {u_wall_core.item()}"

    # 2. Atom in physical pore center well (z = 20.0 Å)
    pos_center = Tensor([[10.0, 10.0, 20.0]])
    u_wall_center = energy_fn.compute_wall_energy(pos_center)
    assert np.isfinite(u_wall_center.item()) and u_wall_center.item() < 0.0, (
        f"Pore center must have finite attractive well, got {u_wall_center.item()}"
    )


def test_coulomb_electrostatics():
    """
    Validates pairwise electrostatic Coulomb energy for opposite and like charge pairs.
    """
    r_dist = 5.0
    energy_fn = MicroscopicEnergy(
        sigmas=[3.0, 3.0],
        epsilons=[0.0, 0.0],  # Zero LJ to isolate Coulomb
        charges=[1.0, -1.0],  # Monovalent pair (Na+ and Cl-)
        box_size=(30.0, 30.0, 40.0),
        r_cut=12.0,
        dielectric_constant=1.0,
    )

    pos = Tensor([[10.0, 10.0, 20.0], [10.0 + r_dist, 10.0, 20.0]])
    u_coul_bare = energy_fn.compute_pair_energy(pos, shift=False)

    # Exact C_coul / r = -167101.0 / 5.0 = -33420.2 K
    exact_coul = -167101.0 / r_dist
    assert math.isclose(u_coul_bare.item(), exact_coul, rel_tol=1e-4), (
        f"Coulomb energy {u_coul_bare.item()} != exact {exact_coul} K"
    )


def test_batched_evaluation_and_autograd():
    """
    Validates that batched tensor evaluation (B, N, 3) matches individual item evaluations
    and autograd backward pass propagates finite gradients without NaNs.
    """
    sigma = 3.4
    eps = 119.8
    r_min = (2.0 ** (1.0 / 6.0)) * sigma

    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[eps, eps],
        box_size=(30.0, 30.0, 40.0),
        r_cut=12.0,
    )

    # 3 Configurations
    p1 = np.array([[10.0, 10.0, 20.0], [10.0 + r_min, 10.0, 20.0]], dtype=np.float32)
    p2 = np.array([[10.0, 10.0, 20.0], [10.0 + 2.0 * sigma, 10.0, 20.0]], dtype=np.float32)
    p3 = np.array([[10.0, 10.0, 20.0], [10.0 + 0.5 * r_min, 10.0, 20.0]], dtype=np.float32)

    batch_pos = Tensor(np.stack([p1, p2, p3], axis=0))
    batch_pos.requires_grad = True

    # Batched evaluation
    u_batch = energy_fn(batch_pos, shift=True)
    assert u_batch.shape == (3,)

    # Individual evaluations
    u1 = energy_fn(Tensor(p1), shift=True).item()
    u2 = energy_fn(Tensor(p2), shift=True).item()
    u3 = energy_fn(Tensor(p3), shift=True).item()

    u_batch_np = u_batch.numpy()
    assert math.isclose(u_batch_np[0], u1, rel_tol=1e-4)
    assert math.isclose(u_batch_np[1], u2, rel_tol=1e-4)
    assert math.isclose(u_batch_np[2], u3, rel_tol=1e-4)

    # Autograd test
    loss = u_batch.sum()
    loss.backward()
    grad = batch_pos.grad.numpy()

    assert np.all(np.isfinite(grad)), "Gradients must be finite without NaNs"
    assert not np.all(grad == 0.0), "Gradients must be non-zero"


def test_shifted_force_zero_energy_and_continuous_force_at_cutoff():
    """
    Validates that Shifted-Force (SF) ensures BOTH the potential energy U(r)
    AND the force magnitude F(r) = -dU/dr vanish continuously at r = r_cut.
    """
    sigma = 3.4
    epsilon = 119.8
    r_cut = 12.0

    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(30.0, 30.0, 40.0),
        r_cut=r_cut,
    )

    # Separation right at cutoff boundary r = r_cut - eps
    delta_r = 1e-4
    p_a = Tensor([[10.0, 10.0, 20.0], [10.0 + r_cut - delta_r, 10.0, 20.0]])
    p_a.requires_grad = True

    u_pair = energy_fn.compute_pair_energy(p_a, shift=True)
    # Energy must be O((delta_r)^2) ~ 0
    assert abs(u_pair.item()) < 1e-4, f"SF energy at cutoff {u_pair.item()} must be close to 0"

    # Evaluate force F = - dU/dx
    u_pair.backward()
    grad_a = p_a.grad.numpy()
    force_magnitude = np.abs(grad_a[1, 0])

    # In Shifted-Force, force at r_cut - delta_r is O(delta_r) ~ 0, unlike Shifted-Potential where force is O(1)
    assert force_magnitude < 1e-2, (
        f"Shifted-Force potential must have smoothly vanishing force at r_cut, got {force_magnitude}"
    )


def test_noe_energy_regularization_linear_and_log_regimes():
    """
    Validates Frank Noé's logarithmic energy regularization formula:
    E_reg = E for E < E_high
    E_reg = E_high + log(E - E_high + 1) for E_high <= E < E_max
    E_reg = E_high + log(E_max - E_high + 1) for E >= E_max
    and verifies that analytical derivatives match autograd gradients across all regimes.
    """
    e_high = 10000.0
    e_max = 1e20

    # 1. Linear regime (E < E_high): negative, zero, and intermediate positive energies
    e_linear = Tensor([-200.0, 0.0, 500.0, 9999.0])
    e_linear.requires_grad = True
    reg_linear = regularize_energy(e_linear, e_high=e_high, e_max=e_max)

    # In the linear regime, E_reg == E
    np.testing.assert_allclose(reg_linear.numpy(), e_linear.numpy(), rtol=1e-5)

    # Derivative d(E_reg)/dE == 1.0 everywhere in linear regime
    reg_linear.sum().backward()
    np.testing.assert_allclose(e_linear.grad.numpy(), [1.0, 1.0, 1.0, 1.0], rtol=1e-5)

    # 2. Boundary and Logarithmic regime (E_high <= E < E_max)
    test_energies = [10000.0, 10001.0, 20000.0, 100000.0, 1e6]
    e_log = Tensor(test_energies)
    e_log.requires_grad = True
    reg_log = regularize_energy(e_log, e_high=e_high, e_max=e_max)

    # Expected values E_high + log(E - E_high + 1)
    expected_vals = [e_high + math.log(val - e_high + 1.0) for val in test_energies]
    np.testing.assert_allclose(reg_log.numpy(), expected_vals, rtol=1e-4)

    # Expected derivatives 1 / (E - E_high + 1)
    reg_log.sum().backward()
    expected_grads = [1.0 / (val - e_high + 1.0) for val in test_energies]
    # Note: At exact boundary E = E_high, subgradient of maximum(0.0) is 0.5; for all E > E_high, exact 1/(E-E_high+1)
    np.testing.assert_allclose(e_log.grad.numpy()[1:], expected_grads[1:], rtol=1e-3)

    # 3. Clamped ceiling regime (E >= E_max)
    e_overflow = Tensor([1e21, 1e22])
    e_overflow.requires_grad = True
    reg_overflow = regularize_energy(e_overflow, e_high=e_high, e_max=e_max)
    expected_ceiling = e_high + math.log(e_max - e_high + 1.0)
    np.testing.assert_allclose(reg_overflow.numpy(), [expected_ceiling, expected_ceiling], rtol=1e-4)

    # Derivative in clamped ceiling regime must be 0.0
    reg_overflow.sum().backward()
    np.testing.assert_allclose(e_overflow.grad.numpy(), [0.0, 0.0], atol=1e-6)


def test_overlapping_atom_gradient_taming():
    """
    Validates that severely overlapping particles (e.g. during early flow training iterations)
    would produce catastrophic gradient explosions (> 10^13) under bare Lennard-Jones,
    while Noé energy regularization tames the gradient to a stable, gentle magnitude (O(10))
    directing overlapping atoms strictly apart without NaNs or numerical overflow.
    """
    sigma = 3.4
    epsilon = 119.8
    energy_fn = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(30.0, 30.0, 40.0),
        r_cut=12.0,
        e_high=10000.0,
    )

    # Place pair at severe overlap: r = 0.5 Å << sigma = 3.4 Å along x-axis
    p_unreg = Tensor([[15.0, 15.0, 20.0], [15.5, 15.0, 20.0]])
    p_unreg.requires_grad = True
    u_unreg = energy_fn(p_unreg, shift=True, regularize=False)

    # Unregularized energy is > 10^12 K
    assert u_unreg.item() > 1e12, f"Bare LJ energy {u_unreg.item()} must be astronomical"

    u_unreg.backward()
    unreg_grad = p_unreg.grad.numpy()
    unreg_force_mag = np.abs(unreg_grad[1, 0])
    # Unregularized force gradient explodes to > 10^13
    assert unreg_force_mag > 1e13, f"Bare gradient magnitude {unreg_force_mag} must be explosive"

    # Now evaluate with Noé regularization
    p_reg = Tensor([[15.0, 15.0, 20.0], [15.5, 15.0, 20.0]])
    p_reg.requires_grad = True
    u_reg = energy_fn(p_reg, shift=True, regularize=True)

    # Regularized energy is smoothly compressed around E_high + log(excess) ~ 10029 K
    assert 10000.0 < u_reg.item() < 10100.0, f"Regularized energy {u_reg.item()} must be gently compressed"

    u_reg.backward()
    reg_grad = p_reg.grad.numpy()
    reg_force_atom1_x = reg_grad[0, 0]
    reg_force_atom2_x = reg_grad[1, 0]

    # Gradients are finite and non-zero
    assert np.all(np.isfinite(reg_grad)), "Regularized gradients must be finite without NaNs"
    assert not np.all(reg_grad == 0.0), "Regularized gradients must be non-zero"

    # Gradient magnitude is tamed to gentle O(10) instead of 10^13
    assert abs(reg_force_atom2_x) < 100.0, f"Regularized force {reg_force_atom2_x} must be gentle"

    # Physical direction: negative on atom 2 (moves in +x direction to increase r), positive on atom 1
    # Because loss L = U(x), gradient dL/dx points toward higher energy, so -grad pushes atoms apart
    assert reg_force_atom1_x > 0.0, "Force gradient on atom 1 must push it toward -x"
    assert reg_force_atom2_x < 0.0, "Force gradient on atom 2 must push it toward +x"


def test_energy_fn_regularization_toggle():
    """
    Validates that MicroscopicEnergy respects regularize=False and e_high=None configurations.
    """
    sigma = 3.4
    epsilon = 119.8

    # 1. MicroscopicEnergy with e_high=1e4
    fn_reg = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(30.0, 30.0, 40.0),
        r_cut=12.0,
        e_high=10000.0,
    )

    pos_clash = Tensor([[15.0, 15.0, 20.0], [15.5, 15.0, 20.0]])
    u_regularized = fn_reg(pos_clash, regularize=True).item()
    u_unregularized = fn_reg(pos_clash, regularize=False).item()

    assert u_regularized < 20000.0, f"Regularized energy {u_regularized} must be compressed"
    assert u_unregularized > 1e12, f"Unregularized energy {u_unregularized} must be uncompressed"

    # 2. MicroscopicEnergy with e_high=None (regularization disabled at init)
    fn_noreg = MicroscopicEnergy(
        sigmas=[sigma, sigma],
        epsilons=[epsilon, epsilon],
        box_size=(30.0, 30.0, 40.0),
        r_cut=12.0,
        e_high=None,
    )

    u_none = fn_noreg(pos_clash, regularize=True).item()
    assert u_none == u_unregularized, f"When e_high=None, energy {u_none} must remain unregularized"


