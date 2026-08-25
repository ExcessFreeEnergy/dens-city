"""
Unit and integration tests for dynamic physics engine routing and CLI feature-flagging.
Verifies:
1. CLI flag parsing for --energy-engine {classical, egnn}.
2. Graceful hardware throttling of batch size under EGNN mode.
3. Execution routing to MicroscopicEnergy vs EGNNMicroscopicEnergy in both single-task and batched pipelines.
"""

from __future__ import annotations

from unittest.mock import patch

from dens_city.boltzmann.energy import EGNNMicroscopicEnergy, MicroscopicEnergy
from dens_city.ui.cli import main
from dens_city.utils.materials import AtomSite, Material, MolecularBatch
from dens_city.utils.pipeline import (
    MaterialPipelineResult,
    MaterialPipelineTask,
    PreparedMolecularBatch,
    execute_prepared_batch,
    process_material_task,
)


def test_cli_energy_engine_flag_parsing(tmp_path):
    """Verifies that --energy-engine parses properly from command line arguments."""
    out_dir = str(tmp_path / "test_out")
    code = main(["--materials", "argon", "--energy-engine", "classical", "--skip-bg", "--out-dir", out_dir])
    assert code == 0


def test_cli_egnn_batch_throttling(tmp_path):
    """Verifies that selecting --energy-engine egnn auto-throttles default batch size to 128."""
    out_dir = str(tmp_path / "test_out_egnn")
    with patch("dens_city.ui.cli.execute_prepared_batch") as mock_exec:
        mock_exec.return_value = [
            MaterialPipelineResult(
                material_name="argon",
                status="SUCCESS",
                runtime_seconds=0.1,
            )
        ]
        code = main(["--materials", "argon", "--energy-engine", "egnn", "--out-dir", out_dir])
        assert code == 0
        assert mock_exec.called
        prepared_batch = mock_exec.call_args[1]["prepared_batch"]
        # Batch size should have been auto-throttled to 32
        assert prepared_batch.batch_size == 32


def test_cli_egnn_explicit_batch_size_preserved(tmp_path):
    """Verifies that an explicitly provided batch size (e.g. 64) is respected under EGNN."""
    out_dir = str(tmp_path / "test_out_explicit")
    with patch("dens_city.ui.cli.execute_prepared_batch") as mock_exec:
        mock_exec.return_value = [
            MaterialPipelineResult(
                material_name="argon",
                status="SUCCESS",
                runtime_seconds=0.1,
            )
        ]
        code = main(["--materials", "argon", "--energy-engine", "egnn", "--batch-size", "64", "--out-dir", out_dir])
        assert code == 0
        assert mock_exec.called
        prepared_batch = mock_exec.call_args[1]["prepared_batch"]
        assert prepared_batch.batch_size == 64


def test_process_material_task_routing_classical(tmp_path):
    """Verifies that process_material_task uses MicroscopicEnergy when energy_engine == 'classical'."""
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
    mat = Material(name="argon", identifier="ar", dimension_mode="1D_SPHERICAL", sites=[site])

    task = MaterialPipelineTask(
        material_path_or_name="argon",
        out_dir=str(tmp_path / "out_classical"),
        energy_engine="classical",
        cdft_steps=5,
        bg_steps=2,
        bg_batch_size=4,
    )

    with (
        patch("dens_city.utils.pipeline.MicroscopicEnergy", wraps=MicroscopicEnergy) as mock_classic,
        patch("dens_city.utils.pipeline.MaterialLoader.load_material", return_value=mat),
    ):
        process_material_task(task)
        assert mock_classic.called


def test_process_material_task_routing_egnn(tmp_path):
    """Verifies that process_material_task uses EGNNMicroscopicEnergy when energy_engine == 'egnn'."""
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
    mat = Material(name="argon", identifier="ar", dimension_mode="1D_SPHERICAL", sites=[site])

    task = MaterialPipelineTask(
        material_path_or_name="argon",
        out_dir=str(tmp_path / "out_egnn"),
        energy_engine="egnn",
        cdft_steps=5,
        bg_steps=2,
        bg_batch_size=4,
    )

    with (
        patch("dens_city.utils.pipeline.EGNNMicroscopicEnergy", wraps=EGNNMicroscopicEnergy) as mock_egnn,
        patch("dens_city.utils.pipeline.MaterialLoader.load_material", return_value=mat),
    ):
        process_material_task(task)
        assert mock_egnn.called


def test_execute_prepared_batch_routing_classical(tmp_path):
    """Verifies that execute_prepared_batch instantiates MicroscopicEnergy for classical tasks."""
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
    mat = Material(name="argon", identifier="ar", dimension_mode="1D_SPHERICAL", sites=[site])
    task = MaterialPipelineTask(
        material_path_or_name="argon",
        out_dir=str(tmp_path),
        energy_engine="classical",
        cdft_steps=2,
        bg_steps=2,
    )
    mol_batch = MolecularBatch.create_batch([mat], batch_size=4, target_n_particles=128)
    prepared = PreparedMolecularBatch(
        tasks=[task],
        batch_size=4,
        loaded_materials=[mat],
        task_indices=[0],
        results_map={},
        mol_batch=mol_batch,
    )

    with patch("dens_city.utils.pipeline.MicroscopicEnergy", wraps=MicroscopicEnergy) as mock_classic:
        execute_prepared_batch(prepared)
        assert mock_classic.called


def test_execute_prepared_batch_routing_egnn(tmp_path):
    """Verifies that execute_prepared_batch instantiates EGNNMicroscopicEnergy for egnn tasks."""
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
    mat = Material(name="argon", identifier="ar", dimension_mode="1D_SPHERICAL", sites=[site])
    task = MaterialPipelineTask(
        material_path_or_name="argon",
        out_dir=str(tmp_path),
        energy_engine="egnn",
        cdft_steps=2,
        bg_steps=2,
    )
    mol_batch = MolecularBatch.create_batch([mat], batch_size=4, target_n_particles=128)
    prepared = PreparedMolecularBatch(
        tasks=[task],
        batch_size=4,
        loaded_materials=[mat],
        task_indices=[0],
        results_map={},
        mol_batch=mol_batch,
    )

    with patch("dens_city.utils.pipeline.EGNNMicroscopicEnergy", wraps=EGNNMicroscopicEnergy) as mock_egnn:
        execute_prepared_batch(prepared)
        assert mock_egnn.called
