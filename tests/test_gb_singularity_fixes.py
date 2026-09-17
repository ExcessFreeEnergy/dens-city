"""
Unit tests verifying Generalized Born dielectric singularity fixes,
conformer rigid-rotor bond preservation, and Boltzmann ensemble stability.
"""

from __future__ import annotations

import numpy as np
from tinygrad import Tensor

from dens_city.cdft.generalized_born import GeneralizedBornSolvation
from dens_city.utils.materials import generate_conformer_rotamer_diversity
from dens_city.utils.pipeline import (
    compute_conformer_internal_energy_diffs,
    get_global_egnn_model,
    get_global_gb_solver,
)


def test_rigid_rotor_rotamer_preserves_bonds():
    """Verify rigid-rotor dihedral sampling preserves 100% of covalent bond lengths."""
    # Build a simple butane-like chain: C1 - C2 - C3 - C4
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [2.05, 1.45, 0.0],
            [3.59, 1.45, 0.0],
        ],
        dtype=np.float32,
    )
    atomic_numbers = [6, 6, 6, 6]
    bonds = [(0, 1, "1"), (1, 2, "1"), (2, 3, "1")]

    conf = generate_conformer_rotamer_diversity(
        coords=coords,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
        n_rot=1,
        n_conf=8,
        seed=123,
    )

    assert conf.shape == (8, 4, 3)
    # Ground state matches exactly
    np.testing.assert_allclose(conf[0], coords, atol=1e-6)

    # All generated conformers must preserve bond lengths to machine precision
    for k in range(1, 8):
        for a1, a2, _ in bonds:
            d0 = np.linalg.norm(coords[a1] - coords[a2])
            dk = np.linalg.norm(conf[k, a1] - conf[k, a2])
            assert abs(dk - d0) < 1e-4, f"Conformer {k} distorted bond ({a1}, {a2}): d0={d0:.4f}, dk={dk:.4f}"


def test_distorted_conformer_internal_energy_penalty():
    """Verify compute_conformer_internal_energy_diffs assigns severe penalty (>=10000) to broken bonds."""

    class MockSite:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class MockMaterial:
        def __init__(self, coords, bonds):
            self.num_sites = len(coords)
            self.sites = [MockSite(*c) for c in coords]
            self.bonds = bonds

    coords0 = np.array([[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]], dtype=np.float32)
    bonds = [(0, 1, "1")]
    mat = MockMaterial(coords0, bonds)

    # Conformer 0: exact ground state
    # Conformer 1: valid bond (1.52 A)
    # Conformer 2: broken bond stretched to 2.50 A
    # Conformer 3: severe clash (0.50 A)
    conf_ensemble = np.array(
        [
            [[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.50, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.50, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    delta_e = compute_conformer_internal_energy_diffs(conf_ensemble, mat)
    assert delta_e[0] == 0.0
    assert delta_e[1] < 5.0  # Mild stretch
    assert delta_e[2] >= 10000.0  # Broken bond penalty
    assert delta_e[3] >= 10000.0  # Severe clash penalty


def test_boltzmann_weighting_masks_distorted_conformers():
    """Verify compute_ensembled_solvation_readouts masks out distorted conformers."""
    egnn = get_global_egnn_model()
    gb = get_global_gb_solver(dielectric_constant=78.4)

    B = 1
    s = 4
    N = 128
    x_ens = np.zeros((B, s, N, 3), dtype=np.float32)
    # Put water molecule in slot
    x_ens[0, 0, :3] = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    x_ens[0, 1, :3] = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    # Corrupted conformers with bond stretches
    x_ens[0, 2, :3] = [[0.0, 0.0, 0.0], [5.00, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    x_ens[0, 3, :3] = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [5.00, 0.0, 0.0]]

    z = np.zeros((B, N), dtype=np.float32)
    z[0, :3] = [8, 1, 1]
    mask = np.zeros((B, N, 1), dtype=np.float32)
    mask[0, :3, 0] = 1.0
    mol_mask = np.ones(B, dtype=np.float32)

    # Conformer internal energy: 0, 1 are valid; 2, 3 have severe penalty
    internal_e = np.array([[0.0, 0.5, 10000.0, 10000.0]], dtype=np.float32)

    tx = Tensor(x_ens)
    tz = Tensor(z)
    tm = Tensor(mask)
    tmol = Tensor(mol_mask)
    tie = Tensor(internal_e)

    q_mean, tot_mean, gb_mean = egnn.compute_ensembled_solvation_readouts(
        x_ensemble=tx,
        atomic_numbers=tz,
        atom_mask=tm,
        molecule_mask=tmol,
        total_charge=0.0,
        dielectric_constant=78.4,
        gb_solver=gb,
        detach_trunk=True,
        internal_energies=tie,
        temperature_k=298.15,
        return_global=False,
    )

    # Output free energy must be finite and physically reasonable for water
    tot_val = float(tot_mean.numpy()[0])
    gb_val = float(gb_mean.numpy()[0])
    assert not np.isnan(tot_val)
    assert not np.isinf(tot_val)
    assert -20.0 < gb_val < 0.0, f"Water Born energy out of physical range: {gb_val}"


def test_born_solvation_high_dielectric_stability():
    """Verify GeneralizedBornSolvation does not diverge in high-dielectric media."""
    gb = GeneralizedBornSolvation()
    # Formamide (eps=109.5) and N-methylacetamide (eps=191.3)
    coords = Tensor([[[0.0, 0.0, 0.0], [1.40, 0.0, 0.0], [0.0, 1.40, 0.0]]])
    z = Tensor([[6.0, 8.0, 7.0]])
    charges = Tensor([[-0.2, 0.4, -0.2]])
    mask = Tensor([[[1.0], [1.0], [1.0]]])

    for eps in [2.0, 10.0, 78.4, 109.5, 191.3]:
        born = gb.compute_solvation_free_energy(
            x=coords, charges=charges, atomic_numbers=z, atom_mask=mask, dielectric_constant=eps
        ).numpy()[0]
        assert not np.isnan(born)
        assert not np.isinf(born)
        assert -30.0 < born < 0.0, f"Born energy diverged at eps={eps}: {born}"
