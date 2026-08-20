"""
Unit and physical validation tests for Microscopic Hamiltonian in dens_city.boltzmann.energy.
Verifies exact pairwise minimums, spherical cutoffs, PBC minimum image convention,
external wall steric barriers, Coulomb interactions, batch invariance, and autograd differentiability.
"""

import math
import numpy as np
import pytest
from tinygrad import Tensor
from dens_city.boltzmann.energy import MicroscopicEnergy
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
