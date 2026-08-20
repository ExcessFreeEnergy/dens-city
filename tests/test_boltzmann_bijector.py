"""
Unit and physical validation tests for Differentiable ZMatrixBijector in dens_city.boltzmann.bijectors.
Verifies exact forward-inverse round-trip invertibility, analytical Jacobian log-determinants,
rigid-bond volume invariance, batched execution, and autograd differentiability.
"""

import math
import numpy as np
import pytest
from tinygrad import Tensor, nn
from dens_city.boltzmann.bijectors import ZMatrixBijector, AffineCouplingLayer, RealNVPFlow
from dens_city.materials import MaterialLoader


@pytest.mark.parametrize("n_atoms", [3, 4, 6, 8, 10])
def test_zmatrix_round_trip_invertibility(n_atoms: int):
    """
    Validates exact round-trip invertibility: IC -> Cartesian -> IC'
    Asserts max(abs(original_ic - reversed_ic)) < 1e-5 across bonds, angles, and torsions.
    """
    np.random.seed(42 + n_atoms)
    bijector = ZMatrixBijector(n_atoms=n_atoms)

    # Generate random physical internal coordinates
    bonds_np = np.random.uniform(1.2, 1.8, size=n_atoms - 1).astype(np.float32)
    angles_np = np.random.uniform(0.8, 2.3, size=max(0, n_atoms - 2)).astype(np.float32)
    torsions_np = np.random.uniform(-math.pi + 0.1, math.pi - 0.1, size=max(0, n_atoms - 3)).astype(np.float32)

    bonds = Tensor(bonds_np)
    angles = Tensor(angles_np) if n_atoms >= 3 else None
    torsions = Tensor(torsions_np) if n_atoms >= 4 else None

    # Forward pass: IC -> Cartesian (N, 3)
    coords, log_det_fwd = bijector.forward(bonds=bonds, angles=angles, torsions=torsions)
    assert coords.shape == (n_atoms, 3)

    # Reverse pass: Cartesian -> IC'
    ic_rec, log_det_inv = bijector.inverse(coords)

    # 1. Check bond lengths
    bonds_rec = ic_rec["bonds"].numpy()
    max_b_err = np.max(np.abs(bonds_np - bonds_rec))
    assert max_b_err < 1e-5, f"Bond length reconstruction error {max_b_err} >= 1e-5"

    # 2. Check bond angles (if N >= 3)
    if n_atoms >= 3:
        angles_rec = ic_rec["angles"].numpy()
        max_th_err = np.max(np.abs(angles_np - angles_rec))
        assert max_th_err < 1e-5, f"Bond angle reconstruction error {max_th_err} >= 1e-5"

    # 3. Check dihedral angles (if N >= 4)
    if n_atoms >= 4:
        torsions_rec = ic_rec["torsions"].numpy()
        max_phi_err = np.max(np.abs(torsions_np - torsions_rec))
        assert max_phi_err < 1e-5, f"Torsion angle reconstruction error {max_phi_err} >= 1e-5"

    # 4. Check Jacobian consistency: log_det(IC->X) + log_det(X->IC) == 0
    total_log_det = log_det_fwd.item() + log_det_inv.item()
    assert math.isclose(total_log_det, 0.0, abs_tol=1e-5), (
        f"Forward + inverse log-det sum {total_log_det} != 0"
    )


def test_analytical_jacobian_log_determinant_formula():
    """
    Validates that the computed log-determinant matches the analytical volume formula:
    log |det J| = sum_{i=2}^N log(b_i^2 * sin(theta_i))
    """
    n_atoms = 5
    bijector = ZMatrixBijector(n_atoms=n_atoms)

    bonds_np = np.array([1.54, 1.52, 1.53, 1.55], dtype=np.float32)
    angles_np = np.array([1.91, 1.95, 1.88], dtype=np.float32)
    torsions_np = np.array([1.04, -0.78], dtype=np.float32)

    bonds = Tensor(bonds_np)
    angles = Tensor(angles_np)
    torsions = Tensor(torsions_np)

    _, log_det = bijector.forward(bonds=bonds, angles=angles, torsions=torsions)

    # Analytical sum: i=2 is (bonds[1], angles[0]), i=3 is (bonds[2], angles[1]), i=4 is (bonds[3], angles[2])
    exact_log_det = sum(
        math.log((bonds_np[i - 1] ** 2) * math.sin(angles_np[i - 2]))
        for i in range(2, n_atoms)
    )

    assert math.isclose(log_det.item(), exact_log_det, rel_tol=1e-5), (
        f"Log |det J| {log_det.item()} != analytical {exact_log_det}"
    )


def test_rigid_bond_jacobian_invariance():
    """
    Validates that for rigid molecules where bonds and angles are fixed,
    the Jacobian determinant remains strictly invariant under arbitrary torsion changes.
    """
    n_atoms = 5
    bijector = ZMatrixBijector(n_atoms=n_atoms)

    bonds = Tensor([1.54, 1.52, 1.53, 1.55])
    angles = Tensor([1.91, 1.95, 1.88])

    _, log_det1 = bijector.forward(bonds=bonds, angles=angles, torsions=Tensor([0.0, 0.0]))
    _, log_det2 = bijector.forward(bonds=bonds, angles=angles, torsions=Tensor([1.5, -2.3]))
    _, log_det3 = bijector.forward(bonds=bonds, angles=angles, torsions=Tensor([-3.1, 0.8]))

    assert math.isclose(log_det1.item(), log_det2.item(), abs_tol=1e-6)
    assert math.isclose(log_det1.item(), log_det3.item(), abs_tol=1e-6)


def test_batched_bijector_and_autograd():
    """
    Validates batched coordinate transformation (B, N, 3) and autograd backpropagation through internal coordinates.
    """
    B, N = 4, 5
    bijector = ZMatrixBijector(n_atoms=N)

    bonds_np = np.random.uniform(1.2, 1.8, size=(B, N - 1)).astype(np.float32)
    angles_np = np.random.uniform(0.8, 2.3, size=(B, N - 2)).astype(np.float32)
    torsions_np = np.random.uniform(-math.pi, math.pi, size=(B, N - 3)).astype(np.float32)

    bonds = Tensor(bonds_np)
    angles = Tensor(angles_np)
    torsions = Tensor(torsions_np)

    bonds.requires_grad = True
    angles.requires_grad = True
    torsions.requires_grad = True

    # Batched forward pass
    coords, log_det = bijector.forward(bonds=bonds, angles=angles, torsions=torsions)
    assert coords.shape == (B, N, 3)
    assert log_det.shape == (B,)

    # Autograd test on coordinate loss
    loss = (coords * coords).sum() + log_det.sum()
    loss.backward()

    b_grad = bonds.grad.numpy()
    th_grad = angles.grad.numpy()
    phi_grad = torsions.grad.numpy()

    assert np.all(np.isfinite(b_grad)), "Bond gradients must be finite without NaNs"
    assert np.all(np.isfinite(th_grad)), "Angle gradients must be finite without NaNs"
    assert np.all(np.isfinite(phi_grad)), "Torsion gradients must be finite without NaNs"
    assert not np.all(b_grad == 0.0)


def test_real_material_water_round_trip():
    """
    Validates internal coordinate rotational invariance on real molecular geometry (water: 3 sites).
    """
    water = MaterialLoader.load_material("water")
    coords_np = np.array([[s.x, s.y, s.z] for s in water.sites], dtype=np.float32)
    coords = Tensor(coords_np)

    bijector = ZMatrixBijector(n_atoms=3)

    # 1. Invert real water from 3D Cartesian to internal coordinates
    ic_dict, log_det_inv = bijector.inverse(coords)
    bonds = ic_dict["bonds"]
    angles = ic_dict["angles"]

    # 2. Forward from IC to canonical Cartesian coordinates
    coords_canonical, log_det_fwd = bijector.forward(bonds=bonds, angles=angles, origin=coords[0])

    # 3. Re-invert canonical coordinates to ensure 100% internal coordinate conservation
    ic_rec, _ = bijector.inverse(coords_canonical)

    diff_b = np.max(np.abs(bonds.numpy() - ic_rec["bonds"].numpy()))
    diff_th = np.max(np.abs(angles.numpy() - ic_rec["angles"].numpy()))

    assert diff_b < 1e-5, f"Water bond reconstruction error {diff_b} >= 1e-5"
    assert diff_th < 1e-5, f"Water angle reconstruction error {diff_th} >= 1e-5"
    assert math.isclose(log_det_fwd.item() + log_det_inv.item(), 0.0, abs_tol=1e-5)


@pytest.mark.parametrize("swap", [False, True])
def test_affine_coupling_layer_invertibility(swap: bool):
    """
    Validates exact invertibility and Jacobian log-determinant sum = 0 for a single AffineCouplingLayer.
    """
    dim = 8
    layer = AffineCouplingLayer(dim=dim, hidden_dim=32, swap=swap)
    x = Tensor.randn(4, dim)

    y, log_det_fwd = layer.forward(x)
    x_rec, log_det_inv = layer.inverse(y)

    diff = (x - x_rec).abs().max().item()
    det_sum = (log_det_fwd + log_det_inv).abs().max().item()

    assert diff < 1e-5, f"Coupling layer reconstruction error {diff} >= 1e-5"
    assert det_sum < 1e-5, f"Coupling layer log-det sum {det_sum} != 0"


def test_realnvp_5_layer_stack_round_trip():
    """
    Validates that a 5-layer stacked RealNVP flow can be perfectly inverted
    and log |det J| + log |det J_inv| = 0.
    """
    dim = 12
    n_layers = 5
    flow = RealNVPFlow(dim=dim, n_layers=n_layers, hidden_dim=32)
    x = Tensor.randn(8, dim)

    # Forward flow: z -> x
    y, log_det_fwd = flow.forward(x)
    assert y.shape == (8, dim)
    assert log_det_fwd.shape == (8,)

    # Reverse flow: x -> z'
    x_rec, log_det_inv = flow.inverse(y)
    assert x_rec.shape == (8, dim)
    assert log_det_inv.shape == (8,)

    diff = (x - x_rec).abs().max().item()
    det_sum = (log_det_fwd + log_det_inv).abs().max().item()

    assert diff < 1e-5, f"5-layer RealNVP reconstruction error {diff} >= 1e-5"
    assert det_sum < 1e-5, f"5-layer RealNVP log-det sum {det_sum} != 0"


def test_realnvp_autograd_training_step():
    """
    Validates gradient propagation through the 5-layer RealNVPFlow network.
    """
    dim = 6
    flow = RealNVPFlow(dim=dim, n_layers=5, hidden_dim=32)
    x = Tensor.randn(4, dim)

    y, log_det = flow.forward(x)
    # Standard maximum likelihood loss proxy: 0.5 * ||y||^2 - log_det
    loss = (0.5 * (y * y).sum(axis=-1) - log_det).mean()

    # Get all layer parameters
    params = nn.state.get_parameters(flow)
    assert len(params) > 0, "Flow must have trainable parameters"

    loss.backward()

    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) == len(params), "All parameters must have gradients"
    Tensor.realize(*grads)

    for p in params:
        grad_np = p.grad.numpy()
        assert np.all(np.isfinite(grad_np)), "Parameter grad must be finite without NaNs"

