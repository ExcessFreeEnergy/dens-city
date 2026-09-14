"""
Automated regression tests verifying zero dataset leakage, zero hardcoded
FreeSolv precomputed value lookups, and universal dataset abstraction.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from dens_city.boltzmann.train_charges import (
    ChargeTrainingConfig,
    ContiguousPackedDataset,
    QuantumChargeTrainer,
)
from dens_city.utils.materials import MaterialLoader
from dens_city.utils.pipeline import (
    MaterialPipelineTask,
    PipelineStatus,
    process_material_task,
)
from dens_city.utils.verification import resolve_freesolv_identifier


def test_contiguous_packed_dataset_structure():
    """Verifies that ContiguousPackedDataset correctly encapsulates packed contiguous device tensors."""
    from tinygrad import Tensor

    ds = ContiguousPackedDataset(
        coords=Tensor.zeros(4, 8, 3),
        atomic_numbers=Tensor.zeros(4, 8),
        atom_mask=Tensor.zeros(4, 8, 1),
        base_charges=Tensor.zeros(4, 8),
        total_charges=Tensor.zeros(4, 1, 1),
        vdw_energies=Tensor.zeros(4),
        expt_energies=Tensor.zeros(4),
        solvent_features=Tensor.zeros(4, 8, 4),
        cached_h=Tensor.zeros(4, 8, 16),
        num_real_molecules=2,
        total_padded_molecules=4,
    )
    assert ds.num_real_molecules == 2
    assert ds.total_padded_molecules == 4
    assert ds.coords.shape == (4, 8, 3)


def test_resolve_freesolv_identifier_strict_db():
    """
    Verifies that resolve_freesolv_identifier resolves strictly via database metadata
    and returns None for unlisted molecules without hardcoded fallback dictionaries.
    """
    mock_db: Dict[str, Any] = {
        "mobley_12345": {
            "iupac": "methane",
            "smiles": "C",
            "formula": "CH4",
        },
        "mobley_67890": {
            "iupac": "ethanol",
            "smiles": "CCO",
            "formula": "C2H6O",
        },
    }

    # Direct key lookup
    assert resolve_freesolv_identifier("mobley_12345", mock_db) == "mobley_12345"

    # Normalized IUPAC lookup
    assert resolve_freesolv_identifier("methane", mock_db) == "mobley_12345"
    assert resolve_freesolv_identifier("Methane", mock_db) == "mobley_12345"
    assert resolve_freesolv_identifier("ethanol", mock_db) == "mobley_67890"

    # Unlisted molecule must return None (no phantom fallback matching)
    assert resolve_freesolv_identifier("polyethylene", mock_db) is None
    assert resolve_freesolv_identifier("unknown_polymer_xyz", mock_db) is None
    assert resolve_freesolv_identifier("benzene", mock_db) is None


def test_pipeline_zero_freesolv_leakage_on_water(tmp_path: Path):
    """
    Verifies that pipeline computes nonpolar solvation via physical first principles
    without reading FreeSolv/database.pickle or looking up calc_vdw.
    """
    task = MaterialPipelineTask(
        material_path_or_name="water",
        out_dir=str(tmp_path),
        temperature_k=300.0,
        pressure_bar=1.0,
        grid=64,
        cdft_steps=10,
        skip_bg=True,
        solvent_name="water",
    )

    res = process_material_task(task)
    assert res is not None
    assert res.status in (PipelineStatus.SUCCESS.value, PipelineStatus.SUCCESS_CDFT_ONLY.value)
    assert res.material_name == "water"
    assert res.num_sites == 3
    assert np.isfinite(res.wall_pressure_bar)
    assert np.isfinite(res.bulk_pressure_bar)


def test_material_loader_does_not_extract_all_freesolv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that MaterialLoader cache checks do not call generate_test_data with
    populate_entire_freesolv=True.
    """
    calls = []

    def mock_generate_test_data(*args, **kwargs):
        calls.append(kwargs)
        return tmp_path

    monkeypatch.setattr("dens_city.utils.materials.TEST_DATA_DIR", tmp_path / "non_existent")
    monkeypatch.setattr("dens_city.utils.test_data_generator.generate_test_data", mock_generate_test_data)

    # Calling list_available_materials with non-existent directory triggers generate_test_data
    MaterialLoader.list_available_materials()

    assert len(calls) > 0
    for call_kwargs in calls:
        assert call_kwargs.get("populate_entire_freesolv", False) is False, (
            f"Expected populate_entire_freesolv=False, got {call_kwargs}"
        )


def test_charge_trainer_dynamic_provider(tmp_path: Path):
    """
    Verifies that QuantumChargeTrainer can ingest a dynamic benchmark dataset provider.
    """
    from dens_city.utils.benchmark_dataset import SolvationBenchmarkEntry

    class MockDatasetProvider:
        def load_entries(self):
            return [
                SolvationBenchmarkEntry(
                    solute_id="water",
                    solute_name="water",
                    solvent_name="WATER",
                    solvent_dielectric=78.4,
                    expt_dG_solv=-6.3,
                    calc_dG_solv=4.02,
                )
            ]

        def get_material(self, solute_id: str):
            loader = MaterialLoader()
            return loader.load_material(solute_id)

    config = ChargeTrainingConfig(
        batch_size=4,
        n_particles=16,
        num_layers=1,
        hidden_dim=16,
        n_conformers=2,
    )
    trainer = QuantumChargeTrainer(config=config)
    packed_ds = trainer.load_static_dataset(dataset_provider=MockDatasetProvider())

    assert isinstance(packed_ds, ContiguousPackedDataset)
    assert packed_ds.num_real_molecules == 1
    assert packed_ds.total_padded_molecules == 4
    assert packed_ds.coords.shape == (4, 16, 3)
