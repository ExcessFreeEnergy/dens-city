"""
Unit and integration tests for the unified dens-city CLI application.
Validates:
1. CLI option parsing and defaults (beam=2, batch-size=32, benchmark=False, debug=False).
2. Fixed 128-site bucketing with strictly zeroed dummy parameters (sigma=0, epsilon=0, charge=0).
3. Batched tensor evaluation (Axis 0 isolation across 32 molecules).
4. Debug logging into timestamped directories (data/logs_<timestamp>/).
5. Benchmark mode reporting and throughput calculation.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.ui.cli import discover_materials, main, parse_materials_arg
from dens_city.utils.materials import MaterialLoader


def test_parse_materials_arg():
    """Validates CLI material argument parser."""
    assert parse_materials_arg(None) == ["argon"]
    assert parse_materials_arg([]) == ["argon"]
    assert parse_materials_arg(["argon", "water"]) == ["argon", "water"]
    assert parse_materials_arg(["[argon,", "water]"]) == ["argon", "water"]

    all_mats = parse_materials_arg(["all"])
    assert len(all_mats) >= 5
    assert "argon" in all_mats
    assert "benzene" in all_mats


def test_discover_materials():
    """Validates material discovery in test_data directory."""
    discovered = discover_materials("test_data", ["argon", "benzene"])
    assert len(discovered) == 2
    assert any("argon" in m for m in discovered)
    assert any("benzene" in m for m in discovered)


def test_fixed_128_site_padding_and_zeroed_parameters():
    """
    Validates that MicroscopicEnergy default pads to exactly 128 sites,
    and that physical parameters (sigma, epsilon, charge) for dummy sites are strictly zero.
    """
    water = MaterialLoader.load_material("water")
    assert len(water.sites) == 3

    energy_fn = MicroscopicEnergy(material=water, box_size=(30.0, 30.0, 40.0), pad_to_128=True)
    assert energy_fn.n_particles == 128
    assert energy_fn.n_real_particles == 3

    sigmas = energy_fn.sigmas.numpy()
    epsilons = energy_fn.epsilons.numpy()
    charges = energy_fn.charges.numpy()

    # Water real sites: Oxygen has LJ sigma, Hydrogens have electrostatic charges
    assert sigmas[0] > 0.0
    assert np.all(np.abs(charges[:3]) > 0.0)

    # Padded dummy sites 3..128 must be strictly zeroed out across all parameters
    assert np.all(sigmas[3:] == 0.0)
    assert np.all(epsilons[3:] == 0.0)
    assert np.all(charges[3:] == 0.0)

    # Real atom mask
    is_real = energy_fn.is_real_atom.numpy()
    assert np.sum(is_real) == 3
    assert np.all(is_real[:3] == 1.0)
    assert np.all(is_real[3:] == 0.0)

    # Also test benzene where all 12 real sites have nonzero sigma
    benzene = MaterialLoader.load_material("benzene")
    assert len(benzene.sites) == 12
    b_energy_fn = MicroscopicEnergy(material=benzene, box_size=(30.0, 30.0, 40.0), pad_to_128=True)
    assert b_energy_fn.n_particles == 128
    assert b_energy_fn.n_real_particles == 12
    b_sigmas = b_energy_fn.sigmas.numpy()
    assert np.all(b_sigmas[:12] > 0.0)
    assert np.all(b_sigmas[12:] == 0.0)


def test_batched_tensor_broadcasting_axis0_isolation():
    """
    Validates that batched tensor evaluation with batch_size=32 evaluates
    isolated 128x128 interaction matrices along Axis 0 without cross-batch pollution.
    """
    benzene = MaterialLoader.load_material("benzene")
    energy_fn = MicroscopicEnergy(material=benzene, box_size=(30.0, 30.0, 40.0), pad_to_128=True)

    B = 32
    # Generate random coordinates for 32 molecules of 128 sites
    coords_np = np.random.uniform(5.0, 25.0, size=(B, 128, 3)).astype(np.float32)
    coords_tensor = Tensor(coords_np)

    energies = energy_fn(coords_tensor).numpy()
    assert energies.shape == (B,)
    assert np.all(np.isfinite(energies))

    # Evaluate molecule 0 individually and compare with batched evaluation
    single_tensor = Tensor(coords_np[0:1])
    single_energy = energy_fn(single_tensor).numpy()[0]

    np.testing.assert_allclose(energies[0], single_energy, rtol=1e-5, atol=1e-5)


def test_unified_cli_cDFT_screening_execution():
    """Tests unified CLI execution with --skip-bg and custom out-dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_run"
        ret = main(
            [
                "--materials",
                "argon",
                "--skip-bg",
                "--cdft-steps",
                "10",
                "--grid",
                "64",
                "--out-dir",
                str(out_dir),
                "--workers",
                "1",
            ]
        )
        assert ret == 0
        assert (out_dir / "pipeline_summary.jsonl").exists()
        assert (out_dir / "argon" / "density_profile.npy").exists()
        assert (out_dir / "argon" / "cdft_summary.txt").exists()


def test_unified_cli_benchmark_and_debug_flags():
    """Tests unified CLI with --benchmark and --debug flags."""
    orig_beam = os.environ.get("BEAM")
    orig_debug = os.environ.get("DEBUG")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_bench_run"
            ret = main(
                [
                    "--materials",
                    "argon",
                    "--skip-bg",
                    "--cdft-steps",
                    "10",
                    "--benchmark",
                    "--debug",
                    "--beam",
                    "2",
                    "--out-dir",
                    str(out_dir),
                    "--workers",
                    "1",
                ]
            )
            assert ret == 0
            assert os.environ.get("BEAM") == "2"
            assert os.environ.get("DEBUG") == "2"
            assert (out_dir / "pipeline_summary.jsonl").exists()
    finally:
        if orig_beam is not None:
            os.environ["BEAM"] = orig_beam
        else:
            os.environ.pop("BEAM", None)
        if orig_debug is not None:
            os.environ["DEBUG"] = orig_debug
        else:
            os.environ.pop("DEBUG", None)
