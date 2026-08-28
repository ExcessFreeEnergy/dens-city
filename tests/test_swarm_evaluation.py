"""
Unit Tests for RL Stage Chemistry Analysis, SA Score, and Diversity Evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from dens_city.swarm import (
    compute_batch_diversity,
    evaluate_molecule_chemistry,
    run_spec_rl_stage,
)

SPECS_DIR = Path(__file__).resolve().parent / "data"
OLED_SPEC = SPECS_DIR / "conjugated_oled_semiconductors.yaml"


def test_evaluate_molecule_chemistry():
    """Verifies that evaluate_molecule_chemistry returns valid SMILES and realistic SA score."""
    # Test on Benzene
    mol_benzene = Chem.MolFromSmiles("c1ccccc1")
    res_benzene = evaluate_molecule_chemistry(mol_benzene)
    assert res_benzene["valid"] is True
    assert res_benzene["smiles"] == "c1ccccc1"
    assert res_benzene["sa_score"] is not None
    assert 1.0 <= res_benzene["sa_score"] <= 2.5

    # Test on Aspirin
    mol_aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    res_aspirin = evaluate_molecule_chemistry(mol_aspirin)
    assert res_aspirin["valid"] is True
    assert res_aspirin["sa_score"] is not None
    assert res_aspirin["sa_score"] < 3.0


def test_compute_batch_diversity():
    """Verifies that compute_batch_diversity calculates internal Tanimoto similarity and unique counts."""
    # Create 3 distinct molecules
    smi_list = ["c1ccccc1", "c1ccc(cc1)N", "c1ccc2c(c1)cccc2"]
    mols = [Chem.MolFromSmiles(s) for s in smi_list]

    mean_sim, diversity, unique_count = compute_batch_diversity(mols)
    assert 0.0 <= mean_sim <= 1.0
    assert 0.0 <= diversity <= 1.0
    assert unique_count == 3
    assert np.isclose(diversity, 1.0 - mean_sim)


def test_run_spec_rl_stage_quick(tmp_path):
    """Verifies that run_spec_rl_stage executes training, candidate sampling, and chemistry evaluation."""
    summary = run_spec_rl_stage(
        spec_path=OLED_SPEC,
        timesteps=256,
        num_candidates=10,
        num_envs=4,
        horizon=8,
        device="cpu",
        output_dir=tmp_path,
    )

    assert summary["spec_name"] == "conjugated_oled_semiconductors"
    assert summary["validity_rate_pct"] >= 90.0
    assert summary["sa_score_mean"] < 8.0
    assert summary["internal_diversity"] > 0.0
    assert len(summary["candidates"]) == 10
    assert (tmp_path / "conjugated_oled_semiconductors_evaluation.json").exists()
