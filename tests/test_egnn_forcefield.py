"""
Unit and physical property tests for EGNN Machine Learned Force Field (MLFF).
Verifies:
1. Power-of-2 tensor shape alignment and 7-layer execution.
2. Exact E(n) rotational and translational invariance of predicted potential energy.
3. Equivariance of analytical autograd forces under 3D spatial rotations: F(R x) = R F(x).
4. Strict isolation and zero force propagation on padded dummy atoms.
5. Integration with EGNNMicroscopicEnergy and BoltzmannGenerator flow training.
"""

from __future__ import annotations

import math

import numpy as np
from tinygrad import Tensor, dtypes

from dens_city.boltzmann.bijectors import Base2CartesianFlow
from dens_city.boltzmann.egnn import EGNNForceField, EGNNLayer
from dens_city.boltzmann.energy import EGNNMicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.utils.materials import AtomSite, Material, MolecularBatch


def create_rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Creates a 3D rotation matrix in SO(3)."""
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def test_egnn_layer_dimensions_and_realization():
    """Verifies that EGNNLayer correctly processes power-of-2 (B, 128, 128) tensors."""
    B, N, F = 4, 128, 128
    layer = EGNNLayer(hidden_dim=F, edge_in_dim=F * 2 + 2)

    h = Tensor.randn(B, N, F)
    d_sq = Tensor.rand(B, N, N, 1)
    edge_mask = Tensor.ones(B, N, N, 1)
    atom_mask = Tensor.ones(B, N, 1)

    h_out = layer(h, d_sq, edge_mask, atom_mask)
    assert h_out.shape == (B, N, F)
    assert not np.isnan(h_out.numpy()).any()


def test_egnn_forward_shape_and_power_of_2_alignment():
    """Verifies that the full 7-layer EGNNForceField maps (B, 128, 3) coordinates to (B,) scalar energies."""
    for b_size in [1, 4, 16]:
        ff = EGNNForceField(num_layers=7, hidden_dim=128, n_particles=128)

        x = Tensor.randn(b_size, 128, 3)
        z = Tensor.full((b_size, 128), 6.0, dtype=dtypes.float32)
        atom_mask = Tensor.ones(b_size, 128, 1)
        mol_mask = Tensor.ones(b_size)

        u = ff.compute_energy(x, z, atom_mask, mol_mask)
        assert u.shape == (b_size,)
        assert not np.isnan(u.numpy()).any()


def test_egnn_e_n_rotational_and_translational_invariance():
    """
    Verifies that the EGNN scalar energy is strictly invariant under arbitrary 3D
    spatial rotations R in SO(3) and translations t in R^3:
    U(R x + t) == U(x)
    """
    ff = EGNNForceField(num_layers=7, hidden_dim=128, n_particles=128)

    # Synthetic ethanol molecule (9 atoms + 119 dummy atoms)
    np.random.seed(42)
    coords_base = np.random.randn(1, 128, 3).astype(np.float32) * 2.0
    z_base = np.zeros((1, 128), dtype=np.float32)
    z_base[0, :9] = [6, 6, 8, 1, 1, 1, 1, 1, 1]  # C2H5OH
    atom_mask_np = np.zeros((1, 128, 1), dtype=np.float32)
    atom_mask_np[0, :9, 0] = 1.0

    # 1. Base energy
    x_base = Tensor(coords_base)
    z_t = Tensor(z_base)
    mask_t = Tensor(atom_mask_np)
    u_base = ff.compute_energy(x_base, z_t, mask_t).item()

    # 2. Rotated and translated coordinates: x' = x @ R.T + t
    rot = create_rotation_matrix(0.7, 1.2, -0.4)
    translation = np.array([5.5, -3.2, 10.1], dtype=np.float32)

    coords_trans = (coords_base @ rot.T) + translation
    x_trans = Tensor(coords_trans)
    u_trans = ff.compute_energy(x_trans, z_t, mask_t).item()

    # Energy must match within numerical precision
    assert abs(u_base - u_trans) < 1e-4, f"E(n) invariance violated: base={u_base:.6f}, trans={u_trans:.6f}"


def test_egnn_autograd_forces_equivariance():
    """
    Verifies that conservative forces derived via reverse-mode autograd
    rotate equivariantly under 3D spatial rotation R:
    F(R x) == R F(x)
    """
    ff = EGNNForceField(num_layers=7, hidden_dim=128, n_particles=128)

    np.random.seed(123)
    coords_base = np.random.randn(1, 128, 3).astype(np.float32) * 1.5
    z_base = np.zeros((1, 128), dtype=np.float32)
    z_base[0, :5] = [6, 1, 1, 1, 1]  # Methane
    atom_mask_np = np.zeros((1, 128, 1), dtype=np.float32)
    atom_mask_np[0, :5, 0] = 1.0

    x_base = Tensor(coords_base)
    z_t = Tensor(z_base)
    mask_t = Tensor(atom_mask_np)

    # Base forces
    f_base = ff.compute_forces(x_base, z_t, mask_t).numpy()[0, :5, :]

    # Rotated coordinates
    rot = create_rotation_matrix(0.5, -0.8, 1.1)
    coords_rot = coords_base @ rot.T
    x_rot = Tensor(coords_rot)

    # Rotated forces
    f_rot = ff.compute_forces(x_rot, z_t, mask_t).numpy()[0, :5, :]

    # Expected: f_expected = f_base @ rot.T
    f_expected = f_base @ rot.T

    np.testing.assert_allclose(
        f_rot,
        f_expected,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Forces are not strictly equivariant under 3D rotation",
    )


def test_egnn_dummy_atom_isolation():
    """
    Verifies that padded dummy atoms (slots 10:128) have zero effect on the
    computed potential energy of real atoms and receive exactly zero force.
    """
    ff = EGNNForceField(num_layers=7, hidden_dim=128, n_particles=128)

    np.random.seed(99)
    coords = np.random.randn(1, 128, 3).astype(np.float32)
    z_base = np.zeros((1, 128), dtype=np.float32)
    z_base[0, :8] = [6, 6, 8, 8, 1, 1, 1, 1]  # 8 real atoms
    atom_mask_np = np.zeros((1, 128, 1), dtype=np.float32)
    atom_mask_np[0, :8, 0] = 1.0

    z_t = Tensor(z_base)
    mask_t = Tensor(atom_mask_np)

    # 1. Evaluate with initial dummy coordinates
    x1 = Tensor(coords)
    u1 = ff.compute_energy(x1, z_t, mask_t).item()
    f1 = ff.compute_forces(x1, z_t, mask_t).numpy()[0]

    # 2. Perturb dummy atom coordinates wildly
    coords_perturbed = coords.copy()
    coords_perturbed[0, 8:, :] += np.random.randn(120, 3).astype(np.float32) * 50.0

    x2 = Tensor(coords_perturbed)
    u2 = ff.compute_energy(x2, z_t, mask_t).item()
    f2 = ff.compute_forces(x2, z_t, mask_t).numpy()[0]

    # Energy must be completely unchanged
    assert abs(u1 - u2) < 1e-6, f"Dummy atoms affected energy: u1={u1}, u2={u2}"

    # Real atom forces must be identical
    np.testing.assert_allclose(f1[:8], f2[:8], atol=1e-6)

    # Dummy atom forces must be identically zero
    np.testing.assert_allclose(f1[8:], 0.0, atol=1e-7)
    np.testing.assert_allclose(f2[8:], 0.0, atol=1e-7)


def test_egnn_microscopic_energy_coupling():
    """Verifies EGNNMicroscopicEnergy evaluation and wall potential coupling."""
    site1 = AtomSite(
        site_name="C1",
        atom_type="c3",
        x=0.0,
        y=0.0,
        z=15.0,
        charge=0.0,
        sigma=3.4,
        epsilon_kcal=0.1,
        epsilon_k=120.0,
        mass=12.0,
        atomic_number=6,
    )
    site2 = AtomSite(
        site_name="O1",
        atom_type="oh",
        x=0.0,
        y=0.0,
        z=16.4,
        charge=0.0,
        sigma=3.1,
        epsilon_kcal=0.15,
        epsilon_k=150.0,
        mass=16.0,
        atomic_number=8,
    )
    mat = Material(name="CO", identifier="test", dimension_mode="3D_MOLECULAR", sites=[site1, site2])

    energy_fn = EGNNMicroscopicEnergy(material=mat, box_size=(30.0, 30.0, 40.0))

    # Single conformation
    pos = Tensor.randn(1, 128, 3) * 2.0 + Tensor([15.0, 15.0, 20.0])
    u = energy_fn(pos)
    assert u.shape == (1,)
    assert not np.isnan(u.numpy()).any()


def test_egnn_boltzmann_generator_training():
    """Verifies that BoltzmannGenerator can train flow parameters using the EGNN energy environment."""
    site = AtomSite(
        site_name="Ar",
        atom_type="ar",
        x=0.0,
        y=0.0,
        z=20.0,
        charge=0.0,
        sigma=3.4,
        epsilon_kcal=0.1,
        epsilon_k=120.0,
        mass=40.0,
        atomic_number=18,
    )
    mat = Material(name="Ar", identifier="argon", dimension_mode="1D_SPHERICAL", sites=[site])
    batch = MolecularBatch.create_batch([mat], batch_size=4, target_n_particles=128)

    energy_fn = EGNNMicroscopicEnergy(material=batch)
    flow = Base2CartesianFlow(n_atoms=128, n_layers=2, hidden_dim=32)

    bg = BoltzmannGenerator(flow=flow, energy_fn=energy_fn, batch_size=4, learning_rate=1e-3)

    losses = bg.train(steps=3, batch_size=4, verbose=False)
    assert len(losses) == 3
    assert not np.isnan(losses[-1])
