"""
Unit and physical validation tests for Torsional Loss Biasing (J_tor) in dens_city.boltzmann.
Verifies .mol2 bond graph quadruplet extraction, analytical rotamer minima/barriers,
mathematically safe collinear bond angle regularizations, and dual-path execution.
"""

import math
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from dens_city.materials import MaterialLoader
from dens_city.boltzmann.bijectors import (
    compute_cartesian_dihedrals,
    compute_cartesian_torsion_loss,
    compute_torsion_rotamer_loss,
    Base2CartesianFlow,
    CompositeFlow,
)
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator


def test_mol2_bond_graph_dihedral_quadruplet_extraction():
    """
    Validates that MaterialLoader parses @<TRIPOS>BOND and extracts authentic
    length-3 paths (a, b, c, d) across linear, branched, and ring materials.
    """
    for mat_name in ["5cb", "polyethylene", "n_decane", "sodium_dodecyl_sulfate"]:
        mat = MaterialLoader.load_material(mat_name)
        assert len(mat.bonds) > 0, f"Material {mat_name} must have parsed bonds from .mol2"
        assert len(mat.dihedral_quadruplets) > 0, (
            f"Material {mat_name} must have extracted dihedral quadruplets from bond graph"
        )

        # Build adjacency set from bonds to verify all quadruplets are connected
        bond_set = set()
        for a1, a2, _ in mat.bonds:
            bond_set.add((a1, a2))
            bond_set.add((a2, a1))

        for a, b, c, d in mat.dihedral_quadruplets:
            assert len({a, b, c, d}) == 4, f"Quadruplet {(a, b, c, d)} must have 4 distinct atoms"
            assert (a, b) in bond_set, f"Bond ({a}, {b}) must exist in {mat_name}"
            assert (b, c) in bond_set, f"Bond ({b}, {c}) must exist in {mat_name}"
            assert (c, d) in bond_set, f"Bond ({c}, {d}) must exist in {mat_name}"
            assert a < d, f"Quadruplet {(a, b, c, d)} must be canonicalized with a < d"


def test_rotamer_potential_analytical_minima_and_barriers():
    """
    Validates that J_tor(phi) = 1 + cos(3 * phi) evaluates to:
    - 0.0 at trans (pi, -pi) and gauche (+-pi/3)
    - 2.0 at eclipsed barriers (0, +-2pi/3)
    """
    # 1. Trans
    phi_trans = Tensor([math.pi, -math.pi])
    loss_trans = compute_torsion_rotamer_loss(phi_trans, periodicity=3)
    assert math.isclose(loss_trans.item(), 0.0, abs_tol=1e-5), f"Trans rotamer loss {loss_trans.item()} != 0.0"

    # 2. Gauche (+-60 deg)
    phi_gauche = Tensor([math.pi / 3.0, -math.pi / 3.0])
    loss_gauche = compute_torsion_rotamer_loss(phi_gauche, periodicity=3)
    assert math.isclose(loss_gauche.item(), 0.0, abs_tol=1e-5), f"Gauche rotamer loss {loss_gauche.item()} != 0.0"

    # 3. Eclipsed / clashing (0 deg, +-120 deg)
    phi_eclipsed = Tensor([0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0])
    loss_eclipsed = compute_torsion_rotamer_loss(phi_eclipsed, periodicity=3)
    assert math.isclose(loss_eclipsed.item(), 2.0, abs_tol=1e-5), (
        f"Eclipsed rotamer loss {loss_eclipsed.item()} != 2.0"
    )


def test_cartesian_dihedrals_trans_and_gauche_geometry():
    """
    Constructs canonical 4-atom butane-like geometries and verifies exact dihedral angle extraction.
    """
    # Trans conformation (staggered 180 deg) in local XY plane:
    # 0: (-1.5, 1.0, 0), 1: (-0.5, 0.0, 0), 2: (0.5, 0.0, 0), 3: (1.5, -1.0, 0)
    p_trans = np.array([
        [-1.5, 1.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.5, -1.0, 0.0],
    ], dtype=np.float32)

    quad = Tensor([[0, 1, 2, 3]])
    phi_calc = compute_cartesian_dihedrals(Tensor(p_trans), quad)
    assert math.isclose(abs(phi_calc.item()), math.pi, abs_tol=1e-4)

    # Gauche conformation (+60 deg): rotate atom 3 around X-axis from cis (+Y)
    th = math.pi / 3.0
    p_gauche = np.array([
        [-1.5, 1.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.5, math.cos(th), math.sin(th)],
    ], dtype=np.float32)

    phi_g = compute_cartesian_dihedrals(Tensor(p_gauche), quad)
    assert math.isclose(abs(phi_g.item()), th, abs_tol=1e-4)

    # Also test Chebyshev cartesian rotamer loss on trans and gauche
    j_trans = compute_cartesian_torsion_loss(Tensor(p_trans), quad)
    assert math.isclose(j_trans.item(), 0.0, abs_tol=1e-4)

    j_gauche = compute_cartesian_torsion_loss(Tensor(p_gauche), quad)
    assert math.isclose(j_gauche.item(), 0.0, abs_tol=1e-4)


def test_safe_norm_collinear_bond_angle_autograd():
    """
    Validates that perfectly straight 180-degree bond angles (collinear cross products = [0, 0, 0])
    do not trigger division-by-zero or NaN gradient poisoning in autograd backward passes.
    """
    # Atoms 0, 1, 2 perfectly collinear along X-axis -> v1 x v2 = [0, 0, 0]
    p_collinear = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ], dtype=np.float32)

    pos = Tensor(p_collinear)
    pos.requires_grad = True
    quad = Tensor([[0, 1, 2, 3]])

    loss = compute_cartesian_torsion_loss(pos, quad)
    loss.backward()

    grad_np = pos.grad.numpy()
    assert np.all(np.isfinite(grad_np)), "Collinear bond gradient must be finite without NaNs or Infs"


def test_composite_flow_direct_torsion_slicing_training():
    """
    Validates that CompositeFlow directly slices internal coordinate torsions
    and trains with w_torsion > 0 with zero NaN risk and decreasing loss.
    """
    n_atoms = 4  # 4 atoms gives 1 dihedral torsion

    def mock_energy(x: Tensor) -> Tensor:
        # Microscopic energy evaluator returns shape (B,)
        return 0.1 * (x * x).flatten(1).sum(axis=-1)

    flow = CompositeFlow(n_atoms=n_atoms, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=mock_energy,
        temperature_k=300.0,
        learning_rate=0.01,
        w_torsion=2.0,
    )

    losses = generator.train(steps=15, batch_size=16)
    assert len(losses) == 15
    assert np.all(np.isfinite(losses)), "CompositeFlow losses with w_torsion must be finite"
    assert losses[-1] < losses[0], "Loss must decrease during training"


def test_base2_cartesian_flow_torsion_biasing_on_polyethylene():
    """
    Validates end-to-end BoltzmannGenerator training with Base2CartesianFlow and
    w_torsion > 0 on polyethylene (62 atoms, 20-carbon backbone).
    """
    pe = MaterialLoader.load_material("polyethylene")
    assert len(pe.dihedral_quadruplets) > 10

    energy_fn = MicroscopicEnergy(
        material=pe,
        box_size=(40.0, 40.0, 50.0),
        r_cut=12.0,
        pad_to_power_of_2=True,
    )
    assert energy_fn.n_particles == 64  # Padded to power of 2

    flow = Base2CartesianFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=350.0,
        learning_rate=0.005,
        w_torsion=0.5,
        dihedral_quadruplets=pe.dihedral_quadruplets,
    )

    losses = generator.train(steps=15, batch_size=16)
    assert len(losses) == 15
    assert np.all(np.isfinite(losses)), "Polyethylene training losses with w_torsion must be finite"

    # Sample configurations
    samples = generator.sample(n_samples=4)
    assert samples.shape == (4, pe.num_sites, 3)
    assert np.all(np.isfinite(samples.numpy())), "Sampled 3D coordinates must be finite"
