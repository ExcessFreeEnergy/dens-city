"""
Unit and physical property tests for GeneralizedBornSolvation engine.
Verifies:
1. Zero-sync O(1) GPU Bondi radii gather accuracy.
2. Born radius volume descreening behavior (alpha_i >= rho_i).
3. Dielectric scaling: vacuum (epsilon=1 -> Delta G_GB = 0), water (epsilon=78.4), ethylene carbonate (epsilon=90).
4. Physical response to charge magnitude: Delta G_GB scales quadratically with charge.
5. Dynamic spatial shape scaling (N=16, 32, 128, 256).
"""

from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

from dens_city.cdft.generalized_born import (
    GeneralizedBornSolvation,
    get_bondi_radii_tensor,
)


def test_bondi_radii_gpu_gather():
    """Verifies that get_bondi_radii_tensor correctly maps Z to standard Bondi van der Waals radii."""
    # Test atomic numbers: H=1, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, I=53
    bondi_tensor = get_bondi_radii_tensor()
    z_test = Tensor([1, 6, 7, 8, 9, 15, 16, 17, 35, 53], dtype=dtypes.int32)
    radii = bondi_tensor[z_test].numpy()

    expected = np.array([1.20, 1.70, 1.55, 1.52, 1.47, 1.80, 1.80, 1.75, 1.85, 1.98], dtype=np.float32)
    np.testing.assert_allclose(radii, expected, atol=1e-5)


def test_born_radii_volume_descreening():
    """Verifies that effective Born radii alpha_i >= rho_i (intrinsic radius)."""
    gb = GeneralizedBornSolvation(dielectric_constant=78.4)

    # Water molecule (O at origin, two H atoms at ~0.96 Å)
    coords = Tensor([[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]], dtype=dtypes.float32)
    z = Tensor([[8, 1, 1]], dtype=dtypes.float32)
    atom_mask = Tensor.ones(1, 3, 1, dtype=dtypes.float32)

    alpha = gb.compute_born_radii(coords, z, atom_mask).numpy()[0, :, 0]

    # O radius ~ 1.52 - 0.09 = 1.43; descreening increases alpha > 1.43
    assert alpha[0] >= 1.43
    assert alpha[1] >= 1.11
    assert alpha[2] >= 1.11
    assert not np.isnan(alpha).any()


def test_generalized_born_dielectric_scaling():
    """Verifies dielectric scaling of solvation free energy across different solvent media."""
    # Monovalent ion (Na+ at origin)
    coords = Tensor([[[0.0, 0.0, 0.0]]], dtype=dtypes.float32)
    charges = Tensor([[1.0]], dtype=dtypes.float32)
    z = Tensor([[11]], dtype=dtypes.float32)
    atom_mask = Tensor.ones(1, 1, 1, dtype=dtypes.float32)

    gb_vacuum = GeneralizedBornSolvation(dielectric_constant=1.0)
    dg_vac = gb_vacuum.compute_solvation_free_energy(coords, charges, z, atom_mask).item()
    # In vacuum (epsilon=1.0), electrostatic solvation free energy must be identically zero
    assert abs(dg_vac) < 1e-5

    gb_water = GeneralizedBornSolvation(dielectric_constant=78.4)
    dg_water = gb_water.compute_solvation_free_energy(coords, charges, z, atom_mask).item()
    # In water (epsilon=78.4), ion hydration is strongly negative
    assert dg_water < -50.0

    gb_ec = GeneralizedBornSolvation(dielectric_constant=90.0)
    dg_ec = gb_ec.compute_solvation_free_energy(coords, charges, z, atom_mask).item()
    # In ethylene carbonate (epsilon=90.0), dielectric factor (1 - 1/90) is slightly stronger than water (1 - 1/78.4)
    assert dg_ec < dg_water


def test_generalized_born_charge_quadratic_scaling():
    """Verifies that Delta G_GB scales quadratically with charge magnitude (Born equation)."""
    gb = GeneralizedBornSolvation(dielectric_constant=78.4)
    coords = Tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]], dtype=dtypes.float32)
    z = Tensor([[11], [11]], dtype=dtypes.float32)
    atom_mask = Tensor.ones(2, 1, 1, dtype=dtypes.float32)

    # Molecule 0: q = 1.0, Molecule 1: q = 2.0 (e.g. Ca2+ vs Na+)
    charges = Tensor([[1.0], [2.0]], dtype=dtypes.float32)
    dg = gb.compute_solvation_free_energy(coords, charges, z, atom_mask).numpy()

    # dg[1] must be ~4x dg[0] (q^2 scaling)
    ratio = dg[1] / dg[0]
    np.testing.assert_allclose(ratio, 4.0, rtol=1e-3)


def test_generalized_born_dynamic_spatial_dimensions():
    """Verifies that GeneralizedBornSolvation executes on arbitrary spatial tensor sizes without 128-atom ceiling."""
    gb = GeneralizedBornSolvation(dielectric_constant=78.4)

    for n_atoms in [16, 32, 64, 128, 256]:
        coords = Tensor.randn(2, n_atoms, 3) * 3.0
        z = Tensor.full((2, n_atoms), 6.0, dtype=dtypes.float32)
        charges = Tensor.randn(2, n_atoms) * 0.2
        atom_mask = Tensor.ones(2, n_atoms, 1, dtype=dtypes.float32)

        dg = gb.compute_solvation_free_energy(coords, charges, z, atom_mask)
        assert dg.shape == (2,)
        assert not np.isnan(dg.numpy()).any()
