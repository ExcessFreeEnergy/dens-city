"""
Unit and integration tests for the high-throughput molecular library generator in scripts/generate_molecular_library.py.
Validates arbitrary YAML spec parsing, combinatorial 2D generation with strict tensor limit enforcement,
3D conformer embedding, Gasteiger charge assignment, Tripos .mol2 export, and seamless dens-city cDFT ingestion.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from dens_city.utils.materials import MaterialLoader
from dens_city.utils.pipeline import MaterialPipelineTask, PipelineStatus, process_material_task
from scripts.generate_molecular_library import (
    embed_conformers,
    export_dataset,
    generate_library,
    load_spec,
)

SPECS_DIR = Path(__file__).resolve().parent / "data"
OLED_SPEC = SPECS_DIR / "conjugated_oled_semiconductors.yaml"
ELECTROLYTES_SPEC = SPECS_DIR / "fluorinated_battery_electrolytes.yaml"
DRUG_SPEC = SPECS_DIR / "sterically_hindered_drug_inhibitors.yaml"
SPONGE_SPEC = SPECS_DIR / "ultra_lightweight_aliphatic_sponges.yaml"
RESIN_SPEC = SPECS_DIR / "sacrificial_h_bond_toughness_resins.yaml"
ALL_SPECS = [OLED_SPEC, ELECTROLYTES_SPEC, DRUG_SPEC, SPONGE_SPEC, RESIN_SPEC]


def test_load_all_specs():
    """Validates that all 5 generic YAML specs load cleanly with proper structure."""
    for spec_path in ALL_SPECS:
        spec = load_spec(spec_path)
        assert "group_name" in spec
        assert "tensor_limits" in spec
        assert "scaffolds" in spec and len(spec["scaffolds"]) > 0
        assert "building_blocks" in spec and len(spec["building_blocks"]) > 0
        assert "generation_spec" in spec
        assert "rl_reward_targets" in spec


def test_2d_generation_oled():
    """Validates 2D combinatorial generation for conjugated OLED semiconductors."""
    spec = load_spec(OLED_SPEC)
    mols = generate_library(spec=spec, target_count=20, seed=42, verbose=False)

    assert len(mols) == 20
    smiles_set = set()

    for mol in mols:
        smi = Chem.MolToSmiles(mol, canonical=True)
        assert smi not in smiles_set
        smiles_set.add(smi)

        # 1. Total sites check (including H)
        num_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
        total_sites = mol.GetNumAtoms() + num_h
        assert total_sites <= spec["tensor_limits"]["max_sites"]

        # 2. Molecular weight check
        assert Descriptors.MolWt(mol) <= spec["tensor_limits"]["max_molecular_weight"]

        # 3. Allowed atomic numbers
        allowed_z = set(spec["tensor_limits"]["allowed_atomic_numbers"])
        for a in mol.GetAtoms():
            assert a.GetAtomicNum() in allowed_z

        # 4. Aromatic ring limits
        ring_info = mol.GetRingInfo()
        aromatic_rings = sum(1 for r in ring_info.AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
        assert (
            spec["tensor_limits"]["min_aromatic_rings"] <= aromatic_rings <= spec["tensor_limits"]["max_aromatic_rings"]
        )


def test_2d_generation_electrolytes():
    """Validates 2D combinatorial generation for fluorinated battery electrolytes."""
    spec = load_spec(ELECTROLYTES_SPEC)
    mols = generate_library(spec=spec, target_count=20, seed=42, verbose=False)

    assert len(mols) == 20
    allowed_z = set(spec["tensor_limits"]["allowed_atomic_numbers"])

    for mol in mols:
        # Fluorine count check
        f_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        assert spec["tensor_limits"]["min_fluorine_count"] <= f_count <= spec["tensor_limits"]["max_fluorine_count"]

        # MW check
        assert Descriptors.MolWt(mol) <= spec["tensor_limits"]["max_molecular_weight"]

        # Allowed elements
        for a in mol.GetAtoms():
            assert a.GetAtomicNum() in allowed_z


def test_2d_generation_drug_inhibitors():
    """Validates 2D combinatorial generation for sterically hindered drug inhibitors."""
    spec = load_spec(DRUG_SPEC)
    mols = generate_library(spec=spec, target_count=20, seed=2026, verbose=False)

    assert len(mols) == 20
    for mol in mols:
        # Rotatable bonds check
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        assert spec["tensor_limits"]["min_rotatable_bonds"] <= rot_bonds <= spec["tensor_limits"]["max_rotatable_bonds"]

        # MW check
        assert Descriptors.MolWt(mol) <= spec["tensor_limits"]["max_molecular_weight"]


def test_3d_embedding_and_charges():
    """Validates 3D conformer embedding, coordinates validity, and Gasteiger charge assignment."""
    spec = load_spec(ELECTROLYTES_SPEC)
    mols_2d = generate_library(spec=spec, target_count=5, seed=42, verbose=False)

    mols_3d = embed_conformers(mols_2d, num_conformations=1, seed=42, verbose=False)
    assert len(mols_3d) == 5

    for mol_h in mols_3d:
        assert mol_h.GetNumConformers() >= 1
        conf = mol_h.GetConformer(0)

        # Check non-zero coordinates
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())])
        assert np.all(np.isfinite(coords))
        assert np.ptp(coords) > 0.5  # Non-degenerate 3D structure

        # Check Gasteiger charges
        charges = []
        for a in mol_h.GetAtoms():
            if a.HasProp("_GasteigerCharge"):
                q = float(a.GetProp("_GasteigerCharge"))
                assert np.isfinite(q)
                charges.append(q)
        assert len(charges) == mol_h.GetNumAtoms()
        assert abs(sum(charges)) < 0.1  # Neutral or near-neutral net charge


def test_tripos_mol2_export_and_ingestion(tmp_path):
    """Validates full Tripos .mol2 export and round-trip ingestion into MaterialLoader."""
    spec = load_spec(ELECTROLYTES_SPEC)
    mols_2d = generate_library(spec=spec, target_count=3, seed=42, verbose=False)
    mols_3d = embed_conformers(mols_2d, num_conformations=1, seed=42, verbose=False)

    out_dir = export_dataset(mols_3d, spec, tmp_path, name_prefix="test_mol", verbose=False)
    assert (out_dir / "library_manifest.json").exists()
    assert (out_dir / "forcefield_parameters.json").exists()
    assert (out_dir / "gaff.dat").exists()

    mol2_files = list(out_dir.glob("*.mol2"))
    assert len(mol2_files) == 3

    # Load with MaterialLoader
    for f in mol2_files:
        mat = MaterialLoader.load_material(str(f))
        assert mat.name == f.stem
        assert mat.num_sites > 0
        assert mat.effective_sigma > 0.0
        assert mat.bulk_density_a3 > 0.0
        assert np.isfinite(mat.bulk_mu)


def test_cdft_pipeline_on_generated_molecule(tmp_path):
    """Validates running a generated molecule through dens-city cDFT screening pipeline."""
    spec = load_spec(ELECTROLYTES_SPEC)
    mols_2d = generate_library(spec=spec, target_count=1, seed=42, verbose=False)
    mols_3d = embed_conformers(mols_2d, num_conformations=1, seed=42, verbose=False)

    out_dir = export_dataset(mols_3d, spec, tmp_path / "data", name_prefix="e2e_mol", verbose=False)
    mol2_path = list(out_dir.glob("*.mol2"))[0]

    task = MaterialPipelineTask(
        material_path_or_name=str(mol2_path),
        out_dir=str(tmp_path / "runs"),
        temperature_k=300.0,
        pressure_bar=1.0,
        grid=32,
        cdft_steps=10,
        skip_bg=True,
    )

    res = process_material_task(task)
    assert res.status == PipelineStatus.SUCCESS_CDFT_ONLY.value
    assert res.num_sites == mols_3d[0].GetNumAtoms()
    assert res.bulk_density_a3 > 0.0
    assert np.isfinite(res.wall_pressure_bar)
    assert np.isfinite(res.cdft_final_loss)
