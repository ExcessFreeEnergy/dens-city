"""
Unit tests for first-principles RDKit solvent property derivation,
exact neat liquid self-association matching, and Sherman-Morrison LOOCV KRR prediction.
"""

from __future__ import annotations

import numpy as np
import pytest
from tinygrad.tensor import Tensor

from dens_city.boltzmann.train_charges import (
    D_PADDED_DIM,
    KRR_TOTAL_PADDED_DIM,
    PHYSICAL_DESCRIPTOR_WEIGHT,
    S_PADDED_DIM,
    SOLVENT_DESCRIPTOR_WEIGHT,
    Z_FEATURE_DIM,
    assemble_physical_descriptor_tensor,
    load_krr_device_tensors,
    predict_krr_residual_tensor,
)
from dens_city.utils.materials import MaterialLoader, compute_neat_liquid_self_association_correction
from dens_city.utils.solvents import (
    derive_solvent_properties_from_structure,
)


def test_rdkit_solvent_property_derivation():
    """Verifies that derive_solvent_properties_from_structure derives physical properties from SMILES."""
    # Ethanol: CCO
    ethanol_props = derive_solvent_properties_from_structure("CCO", density_g_cm3=0.789)
    assert ethanol_props.solvent_class == "polar_protic"
    assert ethanol_props.molecular_weight == pytest.approx(46.07, rel=1e-2)
    assert ethanol_props.refractive_index > 1.3
    assert ethanol_props.dielectric_constant > 5.0
    assert ethanol_props.abraham_alpha > 0.0  # Hydrogen bond donor

    # Benzene: c1ccccc1
    benzene_props = derive_solvent_properties_from_structure("c1ccccc1", density_g_cm3=0.876)
    assert benzene_props.solvent_class == "aromatic"
    assert benzene_props.molecular_weight == pytest.approx(78.11, rel=1e-2)
    assert benzene_props.abraham_alpha == 0.0  # Non-donor
    assert benzene_props.dielectric_constant == pytest.approx(2.28, rel=0.15)


def test_neat_liquid_self_association_exact_matching():
    """Verifies that neat liquid self-association uses exact canonical matching without false positives."""
    # Methanol (CO) in Methanol -> should match
    mat_methanol = MaterialLoader.from_smiles("CO", identifier="methanol")
    corr_match = compute_neat_liquid_self_association_correction(
        "methanol",
        "methanol",
        alpha_s=0.6,
        beta_s=0.6,
        packing_fraction=0.45,
        temp_k=298.15,
        solute=mat_methanol,
        solute_smiles="CO",
        solvent_smiles="CO",
    )
    assert corr_match < 0.0  # Negative association stabilization

    # 4-methoxyphenol in Methanol -> should NOT match (no substring false positive)
    mat_methoxy = MaterialLoader.from_smiles("COc1ccc(O)cc1", identifier="4-methoxyphenol")
    corr_nomatch = compute_neat_liquid_self_association_correction(
        "4-methoxyphenol",
        "methanol",
        alpha_s=0.6,
        beta_s=0.6,
        packing_fraction=0.45,
        temp_k=298.15,
        solute=mat_methoxy,
        solute_smiles="COc1ccc(O)cc1",
        solvent_smiles="CO",
    )
    assert corr_nomatch == 0.0


def test_krr_schema_constants_and_assembly():
    """Verifies physical descriptor assembly tensor shape and schema dimensions."""
    batch_size = 2
    atom_counts = [(5, 1, 0, 0), (6, 0, 1, 1)]  # (n_heavy, n_O, n_N, n_hal)
    gb_tensor = Tensor([[-2.5], [-4.0]])
    vdw_solv = [-1.2, -3.1]

    d_phys = assemble_physical_descriptor_tensor(atom_counts, gb_tensor, vdw_solv, batch_size)
    assert d_phys.shape == (batch_size, D_PADDED_DIM)
    arr = d_phys.numpy()

    # Check mol 0: [5, 1, 0, 0, -2.5, -1.2, 0, 0]
    np.testing.assert_allclose(arr[0], [5.0, 1.0, 0.0, 0.0, -2.5, -1.2, 0.0, 0.0], atol=1e-5)
    # Check mol 1: [6, 0, 1, 1, -4.0, -3.1, 0.0, 0.0]
    np.testing.assert_allclose(arr[1], [6.0, 0.0, 1.0, 1.0, -4.0, -3.1, 0.0, 0.0], atol=1e-5)

    assert Z_FEATURE_DIM == 384
    assert D_PADDED_DIM == 8
    assert S_PADDED_DIM == 8
    assert KRR_TOTAL_PADDED_DIM == 512


def test_sherman_morrison_loocv_subtraction():
    """
    Verifies that predict_krr_residual_tensor with eval_loocv=True
    subtracts alpha_j / [A^{-1}]_{jj} when query point matches training sample j.
    """
    dev = load_krr_device_tensors()
    if dev is None or "diag_a_inv" not in dev:
        pytest.skip("KRR device tensors or diag_a_inv not available.")

    # Reconstruct unstandardized query descriptors matching training sample 0
    z_train_0 = dev["z_train"][0:1]  # (1, 512)
    z_norm_0 = z_train_0[:, :384]
    d_norm_weighted_0 = z_train_0[:, 384:390]
    s_norm_weighted_0 = z_train_0[:, 392:399]

    z_mol_0 = z_norm_0 * dev["z_std"] + dev["z_mean"]
    d_phys_0 = (d_norm_weighted_0 / PHYSICAL_DESCRIPTOR_WEIGHT) * dev["d_std"] + dev["d_mean"]
    s_solv_0 = (s_norm_weighted_0 / SOLVENT_DESCRIPTOR_WEIGHT) * dev["s_std"] + dev["s_mean"]

    # Raw prediction:
    raw_pred, _ = predict_krr_residual_tensor(z_mol_0, d_phys_0, s_solv_0, eval_loocv=False)
    # LOOCV prediction:
    loocv_pred, _ = predict_krr_residual_tensor(z_mol_0, d_phys_0, s_solv_0, eval_loocv=True)

    raw_val = raw_pred.numpy()[0, 0]
    loocv_val = loocv_pred.numpy()[0, 0]

    alpha_0 = dev["alpha"].numpy()[0, 0]
    diag_inv_0 = dev["diag_a_inv"].numpy()[0]
    expected_correction = alpha_0 / diag_inv_0

    # The difference between raw and LOOCV must be exactly expected_correction:
    np.testing.assert_allclose(raw_val - loocv_val, expected_correction, atol=1e-4)
    assert abs(raw_val - loocv_val) > 1e-4
