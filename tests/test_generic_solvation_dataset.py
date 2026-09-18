"""
Unit tests for GenericSolvationDataset and arbitrary dataset ingestion layer.
Verifies format parsing (.csv, .tsv, .jsonl, .sdf), conformer building,
dynamic material instantiation, and factory resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dens_city.utils.benchmark_dataset import get_benchmark_dataset
from dens_city.utils.generic_dataset import GenericSolvationDataset


def test_generic_dataset_csv_ingestion(tmp_path: Path):
    """Verifies that GenericSolvationDataset parses CSV files with alias matching."""
    csv_file = tmp_path / "custom_solvation.csv"
    csv_content = (
        "compound_id,compound_name,smiles,solvent,expt_dG,error\n"
        "comp_01,Ethanol,CCO,water,-5.05,0.15\n"
        "comp_02,Benzene,c1ccccc1,cyclohexane,-6.80,0.20\n"
        "comp_03,Acetone,CC(=O)C,octanol,-4.10,0.10\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    ds = GenericSolvationDataset(csv_file)
    entries = ds.load_entries()
    assert len(entries) == 3

    assert entries[0].solute_id == "comp_01"
    assert entries[0].solute_name == "Ethanol"
    assert entries[0].smiles == "CCO"
    assert entries[0].solvent_name == "WATER"
    assert entries[0].solvent_dielectric == pytest.approx(78.4, rel=1e-2)
    assert entries[0].expt_dG_solv == pytest.approx(-5.05)
    assert entries[0].expt_uncertainty == pytest.approx(0.15)

    assert entries[1].solute_id == "comp_02"
    assert entries[1].solvent_name == "CYCLOHEXANE"
    assert entries[1].expt_dG_solv == pytest.approx(-6.80)

    # Test get_unique_solutes and get_unique_solvents
    solutes = ds.get_unique_solutes()
    assert len(solutes) == 3
    assert "comp_01" in solutes
    assert "comp_02" in solutes

    solvents = ds.get_unique_solvents()
    assert "WATER" in solvents
    assert "CYCLOHEXANE" in solvents
    assert "OCTANOL" in solvents


def test_generic_dataset_jsonl_ingestion(tmp_path: Path):
    """Verifies that GenericSolvationDataset parses JSON Lines files."""
    jsonl_file = tmp_path / "custom_data.jsonl"
    records = [
        {
            "id": "mol_a",
            "name": "Methanol",
            "smiles": "CO",
            "solvent_name": "WATER",
            "dg_solv": -5.10,
        },
        {
            "id": "mol_b",
            "name": "Toluene",
            "smiles": "Cc1ccccc1",
            "solvent_name": "BENZENE",
            "dg_solv": -6.35,
        },
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds = GenericSolvationDataset(jsonl_file)
    entries = ds.load_entries()
    assert len(entries) == 2
    assert entries[0].solute_id == "mol_a"
    assert entries[0].expt_dG_solv == pytest.approx(-5.10)
    assert entries[1].solute_id == "mol_b"
    assert entries[1].solvent_name == "BENZENE"


def test_generic_dataset_material_instantiation_from_smiles(tmp_path: Path):
    """Verifies that get_material generates 3D conformers from SMILES automatically."""
    csv_file = tmp_path / "smiles_test.csv"
    csv_content = "id,name,smiles,expt\nethanol,Ethanol,CCO,-5.0\n"
    csv_file.write_text(csv_content, encoding="utf-8")

    ds = GenericSolvationDataset(csv_file)
    mat = ds.get_material("ethanol")
    assert mat is not None
    assert mat.num_sites > 0
    # Ethanol has 9 atoms (2 C, 1 O, 6 H)
    assert mat.num_sites == 9
    assert len(mat.sites) == 9
    assert isinstance(mat.sites[0].x, float)
    assert isinstance(mat.sites[0].y, float)
    assert isinstance(mat.sites[0].z, float)


def test_benchmark_dataset_factory_custom_file(tmp_path: Path):
    """Verifies get_benchmark_dataset returns GenericSolvationDataset for arbitrary paths."""
    tsv_file = tmp_path / "arbitrary.tsv"
    tsv_file.write_text("id\tname\tsmiles\texpt\nm1\tMol1\tCC\t1.5\n", encoding="utf-8")

    ds = get_benchmark_dataset(str(tsv_file))
    assert isinstance(ds, GenericSolvationDataset)
    assert len(ds.load_entries()) == 1
    assert ds.load_entries()[0].solute_id == "m1"
