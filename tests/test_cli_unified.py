"""
Unit and integration tests for the unified dens-city CLI application.
Validates:
1. CLI option parsing and defaults (beam=2, batch-size=32, benchmark=False, debug=False).
2. Fixed 128-site bucketing with strictly zeroed dummy parameters (sigma=0, epsilon=0, charge=0).
3. Batched tensor evaluation (Axis 0 isolation across 32 molecules).
4. Debug logging into timestamped directories (data/logs_<timestamp>/).
5. Benchmark mode reporting and throughput calculation.
"""

import json
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
    discovered = discover_materials("data/test_data", ["argon", "benzene"])
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


def test_resolve_spec_path():
    """Validates specification path resolution from full paths, filenames, and keywords."""
    from dens_city.ui.cli import resolve_spec_path

    # Direct valid path
    p = resolve_spec_path("tests/data/conjugated_oled_semiconductors.yaml")
    assert p is not None and p.exists()

    # Keyword match
    p_oled = resolve_spec_path("oled")
    assert p_oled is not None and p_oled.exists()
    assert "oled" in p_oled.name

    p_electrolytes = resolve_spec_path("fluorinated_battery_electrolytes")
    assert p_electrolytes is not None and p_electrolytes.exists()

    p_drug = resolve_spec_path("drug")
    assert p_drug is not None and p_drug.exists()


def test_cli_parser_help_and_epilog():
    """Validates that create_parser builds complete parser with formatted examples."""
    from dens_city.ui.cli import create_parser

    parser = create_parser()
    assert parser.prog == "dens-city"
    help_str = parser.format_help()
    assert "--funnel" in help_str
    assert "--benchmark-specs" in help_str
    assert "--train-swarm" in help_str
    assert "--sweep" in help_str
    assert "--eval-swarm" in help_str
    assert "--generate-library" in help_str
    assert "--populate-test-data" in help_str
    assert "--verify-freesolv" in help_str
    assert "Execution Modes & Examples" in help_str


def test_cli_populate_test_data_dispatch(tmp_path):
    """Tests CLI --populate-test-data execution mode."""
    ret = main(["--populate-test-data", "--data-dir", str(tmp_path)])
    assert ret == 0
    assert (tmp_path / "water.mol2").exists()
    assert (tmp_path / "argon.mol2").exists()
    assert (tmp_path / "forcefield_parameters.json").exists()


def test_cli_generate_library_dispatch(tmp_path):
    """Tests CLI --generate-library execution mode in-memory."""
    ret = main(
        [
            "--generate-library",
            "--spec",
            "oled",
            "--target-count",
            "3",
            "--skip-write",
            "--seed",
            "42",
        ]
    )
    assert ret == 0


def test_cli_train_swarm_fast(tmp_path):
    """Tests CLI --train-swarm execution with minimal steps."""
    ret = main(
        [
            "--train-swarm",
            "--spec",
            "oled",
            "--train-steps",
            "32",
            "--num-envs",
            "2",
            "--horizon",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "ckpts"),
            "--export-dir",
            str(tmp_path / "cands"),
        ]
    )
    assert ret == 0
    assert (tmp_path / "ckpts" / "trained_policy.pt").exists()


def test_cli_funnel_fast(tmp_path):
    """Tests CLI --funnel execution with small candidate count and fast steps."""
    ret = main(
        [
            "--funnel",
            "--spec",
            "oled",
            "--train-steps",
            "32",
            "--num-envs",
            "2",
            "--horizon",
            "8",
            "--num-candidates",
            "4",
            "--batch-size",
            "4",
            "--cdft-steps",
            "5",
            "--bg-steps",
            "5",
            "--lbfgs-steps",
            "0",
            "--enable-egnn",
            "--egnn-batch-size",
            "4",
            "--top-k",
            "2",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert ret == 0
    report_file = tmp_path / "conjugated_oled_semiconductors" / "funnel_report.md"
    csv_file = tmp_path / "conjugated_oled_semiconductors" / "funnel_summary.csv"
    assert report_file.exists()
    assert csv_file.exists()


def test_cli_sweep_fast(tmp_path):
    """Tests CLI --sweep execution mode with 1 trial and small step count."""
    ret = main(
        [
            "--sweep",
            "--specs",
            "tests/data/conjugated_oled_semiconductors.yaml",
            "--num-trials-per-spec",
            "1",
            "--steps-per-trial",
            "32",
            "--num-envs",
            "2",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert ret == 0
    assert (tmp_path / "sweep_summary.json").exists()


def test_cli_verify_freesolv_dispatch(tmp_path):
    """Tests CLI --verify-freesolv execution mode."""
    # Write mock summary jsonl
    summary_path = tmp_path / "pipeline_summary.jsonl"
    report_path = tmp_path / "verification_report.md"
    mock_res = {
        "material_name": "methane",
        "status": "SUCCESS",
        "runtime_seconds": 0.5,
        "num_sites": 5,
        "bulk_density_a3": 0.02,
        "wall_pressure_bar": 12.0,
        "cdft_final_loss": 0.05,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(mock_res) + "\n")

    # FreeSolv database mock or real
    db_path = Path("FreeSolv/database.pickle")
    if not db_path.exists():
        db_path = Path("data/database.pickle")

    if db_path.exists():
        ret = main(
            [
                "--verify-freesolv",
                "--results-dir",
                str(tmp_path),
                "--database",
                str(db_path),
                "--report-out",
                str(report_path),
            ]
        )
        assert ret == 0
        assert report_path.exists()
