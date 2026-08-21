"""
Unit and integration tests for the high-throughput batch pipeline in dens_city.pipeline.
Verifies end-to-end cDFT screening, Boltzmann Generator generative handoff,
multi-frame XYZ trajectory export, state serialization, and process pool execution.
"""

import os
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from dens_city.pipeline import (
    MaterialPipelineTask,
    MaterialPipelineResult,
    PipelineStatus,
    process_material_task,
    write_xyz_trajectory,
    save_flow_weights,
)
from dens_city.materials import MaterialLoader


def test_single_material_full_pipeline(tmp_path):
    """
    Validates end-to-end execution of a single material (Water: 3 atoms) through the full pipeline:
    Thermodynamics -> cDFT -> Spatial Prior -> CompositeFlow -> Boltzmann Generator -> Artifacts.
    """
    task = MaterialPipelineTask(
        material_path_or_name="water",
        out_dir=str(tmp_path),
        temperature_k=300.0,
        pressure_bar=1.0,
        grid=64,
        cdft_steps=20,
        bg_steps=15,
        bg_batch_size=16,
        bg_samples=5,
        skip_bg=False,
    )

    res = process_material_task(task)

    assert res.status == PipelineStatus.SUCCESS.value
    assert res.num_sites == 3
    assert res.temperature_k == 300.0
    assert res.bulk_density_a3 > 0.0
    assert np.isfinite(res.wall_pressure_bar)
    assert np.isfinite(res.cdft_final_loss)
    assert res.bg_final_loss is not None and np.isfinite(res.bg_final_loss)

    # Check generated artifacts
    mat_dir = Path(tmp_path) / "water"
    assert mat_dir.exists()

    # 1. Density profiles
    npy_file = mat_dir / "density_profile.npy"
    csv_file = mat_dir / "density_profile.csv"
    assert npy_file.exists()
    assert csv_file.exists()
    rho = np.load(npy_file)
    assert len(rho) == 64
    assert np.all(np.isfinite(rho))

    # 2. cDFT summary
    summary_file = mat_dir / "cdft_summary.txt"
    assert summary_file.exists()
    summary_text = summary_file.read_text(encoding="utf-8")
    assert "Water" in summary_text or "water" in summary_text
    assert "Wall Contact Pressure" in summary_text

    # 3. XYZ Trajectory
    xyz_file = mat_dir / "trajectory.xyz"
    assert xyz_file.exists()
    xyz_lines = xyz_file.read_text(encoding="utf-8").strip().splitlines()
    # 5 frames * (1 atom count line + 1 header line + 3 atom lines) = 25 lines
    assert len(xyz_lines) == 5 * (2 + 3)
    assert xyz_lines[0] == "3"  # 3 atoms for water
    assert "Frame 0" in xyz_lines[1]
    assert "O" in xyz_lines[2]  # Oxygen site

    # 4. Flow weights
    weights_file = mat_dir / "flow_weights.npz"
    assert weights_file.exists()
    weights_data = np.load(weights_file)
    assert len(weights_data.files) > 0


def test_skip_bg_screening_mode(tmp_path):
    """
    Validates that setting skip_bg=True executes cDFT screening only and skips Boltzmann Generator.
    """
    task = MaterialPipelineTask(
        material_path_or_name="argon",
        out_dir=str(tmp_path),
        temperature_k=120.0,
        pressure_bar=1.0,
        grid=64,
        cdft_steps=15,
        skip_bg=True,
    )

    res = process_material_task(task)

    assert res.status == PipelineStatus.SUCCESS_CDFT_ONLY.value
    assert res.num_sites == 1
    assert res.bg_runtime_seconds == 0.0
    assert res.bg_final_loss is None

    mat_dir = Path(tmp_path) / "argon"
    assert (mat_dir / "density_profile.npy").exists()
    assert (mat_dir / "cdft_summary.txt").exists()
    # Generative files must NOT be generated when skip_bg is active
    assert not (mat_dir / "trajectory.xyz").exists()
    assert not (mat_dir / "flow_weights.npz").exists()


def test_error_handling_invalid_material(tmp_path):
    """
    Validates graceful degradation when given a non-existent or corrupted material input.
    """
    task = MaterialPipelineTask(
        material_path_or_name="non_existent_material_xyz_123.mol2",
        out_dir=str(tmp_path),
    )

    res = process_material_task(task)
    assert res.status in [PipelineStatus.FAILED_ERROR.value, PipelineStatus.SKIPPED_THERMO.value]
    assert res.error_message is not None


def test_write_xyz_trajectory(tmp_path):
    """
    Validates custom XYZ multi-frame trajectory writer.
    """
    out_file = str(tmp_path / "test.xyz")
    # 2 frames, 3 atoms
    coords = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]],
    ], dtype=np.float32)
    names = ["C1", "H2", "O3"]
    energies = [-100.5, -98.2]

    write_xyz_trajectory(out_file, coords, names, energies, material_name="TestMol")

    lines = Path(out_file).read_text().strip().splitlines()
    assert len(lines) == 2 * (2 + 3)
    assert lines[0] == "3"
    assert "Energy: -100.5000" in lines[1]
    assert lines[2].startswith("C")
    assert lines[3].startswith("H")
    assert lines[4].startswith("O")


def test_concurrent_multiprocessing_batch(tmp_path):
    """
    Validates parallel execution of multiple materials using concurrent.futures.ProcessPoolExecutor.
    """
    import multiprocessing as mp
    import concurrent.futures

    materials = ["argon", "water", "methane"]
    tasks = [
        MaterialPipelineTask(
            material_path_or_name=m,
            out_dir=str(tmp_path),
            temperature_k=300.0 if m != "argon" else 120.0,
            pressure_bar=1.0,
            grid=32,
            cdft_steps=10,
            bg_steps=10,
            bg_batch_size=8,
            bg_samples=3,
            skip_bg=False,
        )
        for m in materials
    ]

    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
        futures = [executor.submit(process_material_task, t) for t in tasks]
        results = [f.result(timeout=60.0) for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 3
    for r in results:
        assert r.status == PipelineStatus.SUCCESS.value
        mat_dir = Path(tmp_path) / r.material_name
        assert (mat_dir / "density_profile.npy").exists()
        assert (mat_dir / "trajectory.xyz").exists()
        assert (mat_dir / "flow_weights.npz").exists()
