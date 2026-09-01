"""
dens-city: Unified Molecular Classical Density Functional Theory, RL Swarm, & 3D Generative Platform.

Supports:
- Coupled cDFT screening -> Boltzmann Generator batch execution
- High-performance 3D interactive Raylib visualization
- 5-Stage Multi-Objective Generative Molecular Funnel & Pareto screening
- RL Swarm PPO policy training & Constellation curriculum sweeps
- Multi-specification chemical diversity & synthesizability diagnostics
- High-performance combinatorial molecular library generation in C & Python
- Test data & FreeSolv database population & statistical verification
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tinygrad.helpers import colored

from dens_city.utils.materials import MaterialLoader
from dens_city.utils.pipeline import (
    AsyncArtifactWriter,
    AsyncBatchPrefetcher,
    MaterialPipelineResult,
    MaterialPipelineTask,
    PipelineStatus,
    execute_prepared_batch,
    process_material_task,
)


def parse_materials_arg(mat_args: Optional[List[str]]) -> List[str]:
    """Parses material names from CLI arguments, handling commas, brackets, and 'all'."""
    all_materials = MaterialLoader.list_available_materials()

    if not mat_args:
        return ["argon"]

    joined = " ".join(mat_args).strip()
    if joined.lower() in ["all", "[all]", "all_materials", "*"]:
        return all_materials

    cleaned = re.sub(r"[\[\]\'\",]", " ", joined)
    requested = [m.strip().lower() for m in cleaned.split() if m.strip()]

    if "all" in requested:
        return all_materials

    return requested if requested else ["argon"]


def discover_materials(data_dir: str, requested_materials: Optional[List[str]] = None) -> List[str]:
    """Discovers .mol2 files in the specified directory and matches against requested materials."""
    p = Path(data_dir)
    discovered = []
    if p.exists() and p.is_dir():
        for f in sorted(p.glob("*.mol2")):
            discovered.append(str(f))

    if not discovered:
        discovered = MaterialLoader.list_available_materials()

    if requested_materials:
        req_clean = [m.lower().replace(".mol2", "") for m in requested_materials]
        if "all" in req_clean:
            return discovered
        filtered = []
        for item in discovered:
            stem = Path(item).stem.lower()
            if stem in req_clean or item in requested_materials:
                filtered.append(item)
        return filtered if filtered else requested_materials

    return discovered


def resolve_spec_path(spec_arg: Optional[str], default_dir: str = "tests/data") -> Optional[Path]:
    """Resolves a specification file path from a path, filename, or partial keyword match."""
    if not spec_arg:
        default_p = Path(default_dir) / "conjugated_oled_semiconductors.yaml"
        return default_p if default_p.exists() else None

    p = Path(spec_arg)
    if p.exists() and p.is_file():
        return p

    search_dirs = [Path(default_dir), Path("data"), Path(".")]
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        exact = s_dir / f"{spec_arg}.yaml"
        if exact.exists():
            return exact
        exact_direct = s_dir / spec_arg
        if exact_direct.exists() and exact_direct.is_file():
            return exact_direct

    query = spec_arg.lower().replace("_", "").replace("-", "")
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for yaml_file in sorted(s_dir.glob("*.yaml")):
            stem_clean = yaml_file.stem.lower().replace("_", "").replace("-", "")
            if query in stem_clean or stem_clean in query:
                return yaml_file

    return Path(spec_arg)


def _run_task_with_logging(task: MaterialPipelineTask, log_file: Optional[str] = None) -> MaterialPipelineResult:
    """Executes a single pipeline task, optionally capturing stdout/stderr into a per-material log file."""
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                return process_material_task(task)
    return process_material_task(task)


def print_banner(
    out_dir: str,
    num_materials: int,
    workers: int,
    temp_k: float,
    pressure_bar: Optional[float],
    batch_size: int,
    beam: int,
    skip_bg: bool,
    benchmark: bool,
    energy_engine: str = "classical",
) -> None:
    print("=" * 88)
    print(colored("  dens-city: Molecular Classical Density Functional Theory & Generative Platform", "cyan"))
    print("=" * 88)
    print(f"  Target Materials  : {num_materials} items")
    print(f"  Parallel Workers  : {workers} isolated processes (Batch Size = {batch_size} molecules/tensor)")
    print(f"  Compiler BEAM     : BEAM={beam}")
    print(
        f"  Physics Engine    : {colored('EGNN 7-Layer MLFF (Quantum E(n)-Invariant)', 'magenta') if energy_engine == 'egnn' else 'Classical (GAFF LJ + Coulomb)'}"
    )
    print(f"  Thermodynamics    : T = {temp_k:.1f} K" + (f", P = {pressure_bar:.2f} bar" if pressure_bar else ""))
    print(f"  Execution Mode    : {'cDFT Fast Screening (No BG)' if skip_bg else 'Coupled cDFT + Boltzmann Generator'}")
    if benchmark:
        print(f"  Benchmark Profiler: {colored('ACTIVE (Measuring per-material throughput)', 'green')}")
    print(f"  Artifact Output   : {out_dir}")
    print("=" * 88)
    print(f"{'Material':<20} | {'Sites':<5} | {'cDFT (s)':<8} | {'BG (s)':<8} | {'P_wall (bar)':<12} | {'Status':<16}")
    print("-" * 88)


def print_result_row(res: MaterialPipelineResult) -> None:
    status_colors = {
        PipelineStatus.SUCCESS.value: "green",
        PipelineStatus.SUCCESS_CDFT_ONLY.value: "green",
        PipelineStatus.SKIPPED_THERMO.value: "yellow",
        PipelineStatus.FAILED_TRAINING.value: "red",
        PipelineStatus.FAILED_TIMEOUT.value: "red",
        PipelineStatus.FAILED_ERROR.value: "red",
    }
    col = status_colors.get(res.status, "white")
    status_str = colored(res.status, col)
    cdft_t_str = f"{res.cdft_runtime_seconds:6.2f}" if res.cdft_runtime_seconds > 0 else "  --  "
    bg_t_str = f"{res.bg_runtime_seconds:6.2f}" if res.bg_runtime_seconds > 0 else "  --  "
    p_wall_str = (
        f"{res.wall_pressure_bar:10.2f}"
        if res.status in [PipelineStatus.SUCCESS.value, PipelineStatus.SUCCESS_CDFT_ONLY.value]
        else "    --    "
    )
    print(
        f"{res.material_name:<20} | {res.num_sites:<5} | {cdft_t_str:<8} | {bg_t_str:<8} | {p_wall_str:<12} | {status_str}"
    )


def print_benchmark_table(results: List[MaterialPipelineResult], total_time: float, batch_size: int) -> None:
    """Prints a detailed statistical benchmark table showing per-material and aggregate performance."""
    print("\n" + colored("=" * 96, "cyan"))
    print(colored("  dens-city Performance Benchmark Report", "cyan"))
    print(colored("=" * 96, "cyan"))
    print(
        f"{'Material':<22} | {'Sites (Real/Pad)':<16} | {'cDFT Time':<10} | {'BG Time':<10} | {'Total Time':<11} | {'Throughput':<14}"
    )
    print("-" * 96)

    total_samples = 0
    for r in results:
        pad_sites_str = f"{r.num_sites}/128"
        cdft_s = f"{r.cdft_runtime_seconds:6.3f} s" if r.cdft_runtime_seconds > 0 else "    --   "
        bg_s = f"{r.bg_runtime_seconds:6.3f} s" if r.bg_runtime_seconds > 0 else "    --   "
        tot_s = f"{r.runtime_seconds:6.3f} s"
        t_mat = max(1e-4, r.runtime_seconds)
        samples_count = 100 if r.bg_runtime_seconds > 0 else 0
        total_samples += samples_count
        tput_str = f"{samples_count / t_mat:6.1f} conf/s" if samples_count > 0 else f"{1.0 / t_mat:6.2f} mat/s"
        print(f"{r.material_name:<22} | {pad_sites_str:<16} | {cdft_s:<10} | {bg_s:<10} | {tot_s:<11} | {tput_str:<14}")

    print("-" * 96)
    print(f"  Total Wall Time   : {total_time:.2f} s across {len(results)} materials")
    print(
        f"  Average Rate      : {len(results) / max(1e-4, total_time):.2f} materials/s ({total_samples / max(1e-4, total_time):.1f} 3D conformations/s)"
    )
    print(colored("=" * 96, "cyan"))


def create_parser() -> argparse.ArgumentParser:
    """Constructs the comprehensive CLI argument parser for dens-city."""
    epilog_text = """
Execution Modes & Examples:
  # 1. Standard Coupled cDFT + Boltzmann Batch Screening (Default)
  uv run dens-city --materials argon water methane 5cb --batch-size 512
  uv run dens-city --materials all --benchmark

  # 2. 3D Interactive Raylib Visualizer
  uv run dens-city --interactive --materials argon

  # 3. 5-Stage Generative Molecular Funnel (Single Material Spec)
  uv run dens-city --funnel --spec tests/data/conjugated_oled_semiconductors.yaml --train-steps 25000 --num-candidates 512 --top-k 20
  uv run dens-city --funnel --spec fluorinated_battery_electrolytes --checkpoint runs/checkpoints/trained_policy.pt

  # 4. Cross-Material 5-Stage Funnel Benchmark (All Specs in tests/data/)
  uv run dens-city --benchmark-specs --train-steps 25000 --num-candidates 64 --batch-size 64

  # 5. Stage 1 RL Swarm PPO Policy Training
  uv run dens-city --train-swarm --spec conjugated_oled_semiconductors --total-timesteps 5000000 --num-envs 16

  # 6. Constellation Curriculum Hyperparameter Sweeps
  uv run dens-city --sweep --num-trials-per-spec 3 --steps-per-trial 10000

  # 7. Multi-Specification Chemical Diversity & Synthesizability Diagnostics
  uv run dens-city --eval-swarm --specs-dir tests/data --timesteps 10000 --num-candidates 50

  # 8. High-Performance Combinatorial Molecular Library Generation
  uv run dens-city --generate-library --spec tests/data/conjugated_oled_semiconductors.yaml --target-count 50000
  uv run dens-city --generate-library --spec fluorinated_battery_electrolytes --target-count 10000 --skip-3d

  # 9. Test Data & FreeSolv Dataset Population
  uv run dens-city --populate-test-data
  uv run dens-city --populate-test-data --all-freesolv

  # 10. FreeSolv Statistical Validation & Verification Report
  uv run dens-city --verify-freesolv --run-e2e
  uv run dens-city --verify-freesolv --results-dir runs/batch_20260828
"""

    parser = argparse.ArgumentParser(
        prog="dens-city",
        description="dens-city: High-Performance Classical Density Functional Theory, RL Swarm, & 3D Generative Molecular Platform.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Mode Action Flags
    # -------------------------------------------------------------------------
    mode_group = parser.add_argument_group("Execution Mode Selectors")
    mode_group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Launch the 3D Raylib molecular viewer for real-time visualization",
    )
    mode_group.add_argument(
        "--funnel",
        "--run-funnel",
        action="store_true",
        default=False,
        help="Execute the 5-Stage Generative Molecular Funnel (RL Swarm -> C Sampling -> cDFT/L-BFGS/BG -> EGNN -> Pareto Export)",
    )
    mode_group.add_argument(
        "--benchmark-specs",
        "--all-specs",
        action="store_true",
        default=False,
        help="Execute the 5-Stage Generative Funnel benchmark across all specification YAMLs in --specs-dir",
    )
    mode_group.add_argument(
        "--train-swarm",
        "--train-rl",
        action="store_true",
        default=False,
        help="Train the Stage 1 Molecular Swarm PPO policy on a target specification",
    )
    mode_group.add_argument(
        "--sweep",
        "--curriculum-sweep",
        action="store_true",
        default=False,
        help="Run PufferLib Constellation-compatible curriculum hyperparameter sweeps",
    )
    mode_group.add_argument(
        "--eval-swarm",
        "--evaluate-specs",
        action="store_true",
        default=False,
        help="Run chemical analysis, synthesizability (SA Score), and diversity diagnostics across all material specs",
    )
    mode_group.add_argument(
        "--generate-library",
        "--gen-library",
        action="store_true",
        default=False,
        help="Run combinatorial 2D/3D molecular library generation and parallel .mol2 file export",
    )
    mode_group.add_argument(
        "--populate-test-data",
        "--generate-test-data",
        action="store_true",
        default=False,
        help="Populate data/test_data/ with benchmark .mol2 files and force field parameter databases",
    )
    mode_group.add_argument(
        "--verify-freesolv",
        "--verify-e2e",
        action="store_true",
        default=False,
        help="Validate simulation results against FreeSolv experimental hydration thermodynamics and output report",
    )
    mode_group.add_argument(
        "--wikiskill-status",
        action="store_true",
        default=False,
        help="Display WikiSkill persistent knowledge index, patterns, and proposal audit trail",
    )
    mode_group.add_argument(
        "--wikiskill-init",
        action="store_true",
        default=False,
        help="Initialize WikiSkill three-layer directories and bootstrap foundational physics patterns",
    )
    mode_group.add_argument(
        "--wikiskill-consolidate",
        action="store_true",
        default=False,
        help="Analyze recent execution traces and consolidate root causes into wiki patterns",
    )
    mode_group.add_argument(
        "--wikiskill-audit",
        type=str,
        default=None,
        metavar="TARGET_SKILL",
        help="Audit a target skill name or proposed edit against past rejections and anti-patterns",
    )
    mode_group.add_argument(
        "--wikiskill-record",
        type=str,
        default=None,
        metavar="COMMAND",
        help="Execute a test/command and record an immutable trace into the WikiSkill raw layer",
    )
    mode_group.add_argument(
        "--train-charges",
        action="store_true",
        default=False,
        help="Train the EGNN dynamic quantum charge readout head directly against FreeSolv hydration free energies via end-to-end autograd",
    )

    # -------------------------------------------------------------------------
    # Material & Input Selection
    # -------------------------------------------------------------------------
    input_group = parser.add_argument_group("Material & Specification Inputs")
    input_group.add_argument(
        "--materials",
        "-m",
        nargs="+",
        default=None,
        help="Material names, .mol2 files, or 'all' (default: argon)",
    )
    input_group.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data/test_data",
        help="Directory containing .mol2 material files (default: data/test_data)",
    )
    input_group.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default=None,
        help="Destination path for structured artifacts, summaries, and logs",
    )
    input_group.add_argument(
        "--spec",
        "-s",
        type=str,
        default=None,
        help="Path or name of target specification YAML file (e.g. tests/data/conjugated_oled_semiconductors.yaml or 'oled')",
    )
    input_group.add_argument(
        "--specs",
        nargs="+",
        default=None,
        help="List of specification YAML files for curriculum sweeps",
    )
    input_group.add_argument(
        "--specs-dir",
        type=str,
        default="tests/data",
        help="Directory containing material specification YAML files (default: tests/data)",
    )

    # -------------------------------------------------------------------------
    # Thermodynamics & cDFT Parameters
    # -------------------------------------------------------------------------
    cdft_group = parser.add_argument_group("Thermodynamics & cDFT Solver Options")
    cdft_group.add_argument(
        "--temp",
        "-t",
        type=float,
        default=300.0,
        help="Thermodynamic reservoir temperature in Kelvin (default: 300.0 K)",
    )
    cdft_group.add_argument(
        "--pressure",
        "-p",
        type=float,
        default=1.0,
        help="Thermodynamic reservoir pressure in bar (default: 1.0 bar)",
    )
    cdft_group.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Optional chemical potential in k_B T (overrides bulk pressure EOS calculation)",
    )
    cdft_group.add_argument(
        "--grid",
        "-g",
        type=int,
        default=128,
        help="Spatial grid points for cDFT 1D pore discretization (default: 128)",
    )
    cdft_group.add_argument(
        "--cdft-steps",
        type=int,
        default=60,
        help="cDFT solver variational optimization steps (default: 60)",
    )
    cdft_group.add_argument(
        "--cdft-lr",
        type=float,
        default=0.02,
        help="cDFT solver learning rate (default: 0.02)",
    )
    cdft_group.add_argument(
        "--skip-bg",
        action="store_true",
        default=False,
        help="Skip Boltzmann Generator phase and halt after cDFT screening",
    )

    # -------------------------------------------------------------------------
    # Boltzmann Generator & Geometry Relaxation Options
    # -------------------------------------------------------------------------
    bg_group = parser.add_argument_group("Boltzmann Generator & Geometry Relaxation")
    bg_group.add_argument(
        "--bg-steps",
        type=int,
        default=40,
        help="Boltzmann Generator training steps (default: 40)",
    )
    bg_group.add_argument(
        "--bg-lr",
        type=float,
        default=0.01,
        help="Boltzmann Generator learning rate (default: 0.01)",
    )
    bg_group.add_argument(
        "--bg-samples",
        type=int,
        default=100,
        help="Number of 3D configurations to sample into .xyz trajectory (default: 100)",
    )
    bg_group.add_argument(
        "--bg-w-tor",
        type=float,
        default=0.0,
        help="Torsional rotamer loss biasing weight (default: 0.0)",
    )
    bg_group.add_argument(
        "--bg-mcmc-steps",
        type=int,
        default=0,
        help="Number of latent space Metropolis Monte Carlo relaxation steps per sample (default: 0)",
    )
    bg_group.add_argument(
        "--bg-mcmc-step-size",
        type=float,
        default=0.1,
        help="Step size for Gaussian perturbations in latent MCMC relaxation (default: 0.1)",
    )
    bg_group.add_argument(
        "--lbfgs-steps",
        type=int,
        default=50,
        help="Number of batched L-BFGS Quasi-Newton geometry relaxation steps on GPU (default: 50)",
    )
    bg_group.add_argument(
        "--lbfgs-tol",
        type=float,
        default=1e-3,
        help="RMS force convergence threshold for L-BFGS relaxation (default: 1e-3)",
    )

    # -------------------------------------------------------------------------
    # Quantum MLFF & EGNN Options
    # -------------------------------------------------------------------------
    egnn_group = parser.add_argument_group("EGNN Quantum Force Field Options")
    egnn_group.add_argument(
        "--energy-engine",
        choices=["classical", "electronegativity", "egnn", "auto"],
        default="classical",
        help="Microscopic Hamiltonian physics engine: 'classical' (GAFF LJ+Coulomb), 'electronegativity' (deterministic Pauling prior + GB), 'egnn' (trained 7-layer E(n)-equivariant MLFF + GB), or 'auto' (adaptive heuristic).",
    )
    egnn_group.add_argument(
        "--force-egnn",
        action="store_true",
        default=False,
        help="Force Stage 4 EGNN quantum surrogate evaluation across 100%% of batch slots, overriding speed heuristics.",
    )
    egnn_group.add_argument(
        "--enable-egnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Stage 4 E(n)-Equivariant Graph Neural Network (EGNN) quantum surrogate screening (default: True)",
    )
    egnn_group.add_argument(
        "--egnn-batch-size",
        type=int,
        default=32,
        help="Batch size for Stage 4 EGNN quantum force field evaluation (default: 32)",
    )
    egnn_group.add_argument(
        "--egnn-layers",
        type=int,
        default=7,
        help="Number of message-passing layers in the EGNN architecture (default: 7)",
    )
    egnn_group.add_argument(
        "--egnn-relax-steps",
        type=int,
        default=50,
        help="Number of unrolled GPU quantum geometry relaxation steps prior to EGNN evaluation (default: 50)",
    )
    egnn_group.add_argument(
        "--egnn-weights",
        type=str,
        default=None,
        help="Optional path to pretrained EGNN weights .npz archive",
    )
    egnn_group.add_argument(
        "--charge-epochs",
        type=int,
        default=150,
        help="Number of training epochs for --train-charges (default: 150)",
    )
    egnn_group.add_argument(
        "--charge-lr",
        type=float,
        default=8e-4,
        help="Learning rate for --train-charges (default: 8e-4)",
    )
    egnn_group.add_argument(
        "--charge-huber-delta",
        type=float,
        default=2.5,
        help="Huber loss transition delta threshold in kcal/mol (default: 2.5)",
    )
    egnn_group.add_argument(
        "--charge-lambda",
        type=float,
        default=0.02,
        help="L2 regularization penalty weight on (Δq)^2 neural charge perturbations (default: 0.02)",
    )
    egnn_group.add_argument(
        "--charge-weights-out",
        type=str,
        default="data/checkpoints/egnn_charges_trained.npz",
        help="Destination path for trained quantum charge checkpoint archive",
    )

    # -------------------------------------------------------------------------
    # RL Swarm & Generative Funnel Options
    # -------------------------------------------------------------------------
    rl_group = parser.add_argument_group("RL Swarm & Generative Funnel Options")
    rl_group.add_argument(
        "--train-steps",
        "--total-timesteps",
        dest="train_steps",
        type=int,
        default=5000000,
        help="RL training steps / timesteps (default: 5,000,000)",
    )
    rl_group.add_argument(
        "--num-candidates",
        type=int,
        default=512,
        help="Number of candidate molecules to generate from trained policy (default: 512)",
    )
    rl_group.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of final Pareto-refined candidates to export (default: 20)",
    )
    rl_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to existing policy checkpoint .pt file",
    )
    rl_group.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel C-FFI environment workers (default: 64)",
    )
    rl_group.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Rollout horizon per environment matching mean molecular growth path (default: 16)",
    )
    rl_group.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    rl_group.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Policy latent dimension (default: 256)",
    )
    rl_group.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="Use recurrent MinGRU backbone instead of MLP",
    )
    rl_group.add_argument(
        "--early-stopping-lookback",
        type=int,
        default=100000,
        help="Step lookback window for EMA reward flatline detection (default: 100,000)",
    )
    rl_group.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.05,
        help="EMA reward change threshold for early stopping (default: 0.05)",
    )
    rl_group.add_argument(
        "--no-early-stopping",
        action="store_true",
        default=False,
        help="Disable Dynamic EMA early stopping",
    )
    rl_group.add_argument(
        "--no-curriculum",
        action="store_true",
        default=False,
        help="Disable 3-stage curriculum scheduler",
    )
    rl_group.add_argument(
        "--no-sa-penalty",
        action="store_true",
        default=False,
        help="Disable in-the-loop batch SA score penalty",
    )
    rl_group.add_argument(
        "--sa-threshold",
        type=float,
        default=None,
        help="SA Score hinge threshold above which penalty is applied",
    )
    rl_group.add_argument(
        "--sa-penalty-slope",
        type=float,
        default=None,
        help="Slope multiplier for SA score excess penalty",
    )
    rl_group.add_argument(
        "--no-dynamic-entropy",
        action="store_true",
        default=False,
        help="Disable molecular-weight-scaling dynamic entropy coefficient",
    )
    rl_group.add_argument(
        "--max-sa-score",
        type=float,
        default=6.0,
        help="Maximum allowable RDKit Synthetic Accessibility (SA) Score ceiling (default: 6.0)",
    )
    rl_group.add_argument(
        "--disable-sa-filter",
        action="store_true",
        default=False,
        help="Disable Stage 5 RDKit synthesizability (SA Score) safety gate",
    )
    rl_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default="runs/checkpoints",
        help="Directory to save trained_policy.pt and periodic checkpoints (default: runs/checkpoints)",
    )
    rl_group.add_argument(
        "--export-dir",
        type=str,
        default="runs/candidates",
        help="Directory to save exported candidate .mol2 files (default: runs/candidates)",
    )

    # -------------------------------------------------------------------------
    # Curriculum Sweep Options
    # -------------------------------------------------------------------------
    sweep_group = parser.add_argument_group("Curriculum Sweep Options")
    sweep_group.add_argument(
        "--num-trials-per-spec",
        type=int,
        default=2,
        help="Number of random hyperparameter trials per material spec (default: 2)",
    )
    sweep_group.add_argument(
        "--steps-per-trial",
        type=int,
        default=5000,
        help="Timesteps to train each trial before evaluation (default: 5,000)",
    )

    # -------------------------------------------------------------------------
    # Molecular Library Generator Options
    # -------------------------------------------------------------------------
    lib_group = parser.add_argument_group("Combinatorial Library Generator Options")
    lib_group.add_argument(
        "--target-count",
        "-n",
        type=int,
        default=None,
        help="Target number of unique molecules to generate (default: from YAML target_molecules, e.g. 50,000)",
    )
    lib_group.add_argument(
        "--skip-3d",
        action="store_true",
        default=False,
        help="Skip 3D conformer embedding (2D combinatorial generation only)",
    )
    lib_group.add_argument(
        "--skip-write",
        action="store_true",
        default=False,
        help="Skip writing .mol2 files to disk (in-memory benchmark mode)",
    )
    lib_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic combinatorial sampling (default: 42)",
    )

    # -------------------------------------------------------------------------
    # FreeSolv & Test Data Options
    # -------------------------------------------------------------------------
    data_group = parser.add_argument_group("FreeSolv & Dataset Options")
    data_group.add_argument(
        "--all-freesolv",
        "--all-data",
        action="store_true",
        default=False,
        help="Extract and populate EVERY molecule from the FreeSolv database (642+ molecules) into data/test_data/",
    )
    data_group.add_argument(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Execute full end-to-end simulation across benchmark materials before verifying against FreeSolv",
    )
    data_group.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing pipeline_summary.jsonl for verification (defaults to latest in runs/)",
    )
    data_group.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to FreeSolv database.pickle (default: FreeSolv/database.pickle)",
    )
    data_group.add_argument(
        "--report-out",
        type=str,
        default="data/e2e_freesolv_verification_report.md",
        help="Destination path for FreeSolv verification Markdown report",
    )

    # -------------------------------------------------------------------------
    # Compiler & Execution Performance Options
    # -------------------------------------------------------------------------
    perf_group = parser.add_argument_group("Compiler & Execution Performance")
    perf_group.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=512,
        help="Molecule batch size for parallel tensor evaluation along Axis 0 (default: 512)",
    )
    perf_group.add_argument(
        "--workers",
        "-w",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=f"Number of concurrent worker processes (default: {min(4, os.cpu_count() or 1)})",
    )
    perf_group.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum execution timeout per material in seconds (default: 180s)",
    )
    perf_group.add_argument(
        "--beam",
        type=int,
        default=2,
        help="tinygrad compiler BEAM search optimization level (default: 2)",
    )
    perf_group.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Profile execution time per material and output comprehensive benchmark summary",
    )
    perf_group.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG=2 and write detailed per-material compiler logs to data/logs_<timestamp>/",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main unified CLI entrypoint for dens-city."""
    if argv is None:
        argv = sys.argv[1:]

    parser = create_parser()
    args = parser.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set tinygrad compiler optimization environment variables
    if args.beam:
        os.environ["BEAM"] = str(args.beam)
    if args.debug:
        os.environ["DEBUG"] = "2"

    # =========================================================================
    # MODE: WikiSkill Persistent Knowledge Base & Audit Modes
    # =========================================================================
    if args.wikiskill_init:
        from dens_city.wikiskill import RawTraceRecorder, WikiManager

        wiki = WikiManager()
        RawTraceRecorder()
        wiki.rebuild_index()
        patterns = wiki.list_patterns()
        print(colored("==========================================================================", "cyan"))
        print(colored("  WikiSkill: Persistent Knowledge Base Initialized                        ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"  Active Patterns   : {len(patterns)}")
        print(f"  Patterns Directory: {wiki.patterns_dir}")
        print(f"  Audit Tracker     : {wiki.impact_file}")
        print(f"  Index File        : {wiki.index_file}")
        print("-" * 74)
        for p in patterns:
            print(f"  - {p}")
        return 0

    if args.wikiskill_status:
        from dens_city.wikiskill import RawTraceRecorder, WikiManager

        wiki = WikiManager()
        recorder = RawTraceRecorder()
        patterns = wiki.list_patterns()
        traces = recorder.list_traces(limit=10)
        history = wiki.get_skill_impact_history()

        print(colored("==========================================================================", "cyan"))
        print(colored("  WikiSkill: Knowledge Evolution & Anti-Pattern Audit Status              ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"  Total Patterns    : {len(patterns)}")
        print(f"  Recorded Traces   : {len(traces)}")
        print("-" * 74)
        print(colored("Active Patterns in Catalog:", "green"))
        for p in patterns:
            print(f"  • {p}")
        print("-" * 74)
        print(colored("Recent Proposals & Outcomes (skill-impact.md):", "yellow"))
        # Print table lines from history
        table_lines = [
            tbl_line
            for tbl_line in history.splitlines()
            if tbl_line.startswith("|") and not tbl_line.startswith("| :---")
        ]
        for line in table_lines[-6:]:
            print(f"  {line}")
        print("=" * 74)
        return 0

    if args.wikiskill_record:
        from dens_city.wikiskill import RawTraceRecorder

        recorder = RawTraceRecorder()
        print(colored(f"[WIKISKILL] Executing and recording trace: `{args.wikiskill_record}`", "cyan"))
        trace = recorder.record_command(args.wikiskill_record, tags=["cli-recorded"])
        status_col = "green" if trace.passed else "red"
        print(colored(f"[WIKISKILL] Trace `{trace.trace_id}` saved: {trace.summary}", status_col))
        return 0 if trace.passed else 1

    if args.wikiskill_consolidate:
        from dens_city.wikiskill import RawTraceRecorder, WikiMaintainer, WikiManager

        wiki = WikiManager()
        recorder = RawTraceRecorder()
        maintainer = WikiMaintainer(wiki_manager=wiki, trace_recorder=recorder)
        report = maintainer.consolidate_traces()
        print(colored("==========================================================================", "cyan"))
        print(colored("  WikiSkill: Trace Consolidation Report                                   ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"  Summary           : {report.log_summary}")
        print(f"  Patterns Created  : {report.patterns_created or 'None'}")
        print(f"  Patterns Updated  : {report.patterns_updated or 'None'}")
        print(f"  Diagnosed Failures: {len(report.diagnosed_failures)}")
        return 0

    if args.wikiskill_audit:
        from dens_city.wikiskill import WikiManager

        wiki = WikiManager()
        target = args.wikiskill_audit
        rejected = wiki.is_proposal_previously_rejected(target, target)
        print(colored("==========================================================================", "cyan"))
        print(colored(f"  WikiSkill: Anti-Pattern & Rejection Audit: `{target}`", "cyan"))
        print(colored("==========================================================================", "cyan"))
        if rejected:
            print(colored(f"  ⚠️  WARNING: {rejected}", "red"))
        else:
            print(colored(f"  ✓ No previous rejections found for `{target}`.", "green"))
        return 0

    # =========================================================================
    # MODE: Train EGNN Dynamic Quantum Charges Mode
    # =========================================================================
    if args.train_charges:
        from dens_city.boltzmann.train_charges import run_train_charges

        run_train_charges(
            epochs=args.charge_epochs,
            lr=args.charge_lr,
            batch_size=args.batch_size if ("-b" in argv or "--batch-size" in argv) else 32,
            huber_delta=args.charge_huber_delta,
            lambda_l2=args.charge_lambda,
            weights_out=args.charge_weights_out,
        )
        return 0

    # =========================================================================
    # MODE 1: 3D Interactive Raylib Visualizer Mode
    # =========================================================================
    if args.interactive:
        from dens_city.ui.viewer import run_interactive_viewer

        materials = discover_materials(args.data_dir, args.materials)
        if not materials:
            print(colored(f"Error: No materials found in data directory: {args.data_dir}", "red"))
            return 1

        print(colored("==========================================================================", "cyan"))
        print(colored("  dens-city: 3D Interactive Raylib Molecular Visualizer                  ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"Target Materials   : {materials}")
        print("Controls           : Left Drag to Orbit | Scroll to Zoom | ← / → to Switch Materials")
        run_interactive_viewer(material_names=materials)
        return 0

    # =========================================================================
    # MODE 2: 5-Stage Generative Molecular Funnel Mode
    # =========================================================================
    if args.funnel:
        from dens_city.swarm.funnel import run_generative_funnel

        spec_file = resolve_spec_path(args.spec, default_dir=args.specs_dir)
        if not spec_file or not spec_file.exists():
            print(colored(f"Error: Target specification YAML not found: {args.spec}", "red"), file=sys.stderr)
            return 1

        out_dir = args.out_dir or "runs/funnel_results"
        run_generative_funnel(
            spec=spec_file,
            train_steps=args.train_steps,
            num_candidates=args.num_candidates,
            batch_size=args.batch_size,
            top_k=args.top_k,
            out_dir=out_dir,
            checkpoint=args.checkpoint,
            num_envs=args.num_envs,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            early_stopping_lookback=args.early_stopping_lookback,
            early_stopping_delta=args.early_stopping_delta,
            no_early_stopping=args.no_early_stopping,
            no_curriculum=args.no_curriculum,
            no_sa_penalty=args.no_sa_penalty,
            no_dynamic_entropy=args.no_dynamic_entropy,
            recurrent=args.recurrent,
            cdft_steps=args.cdft_steps,
            bg_steps=args.bg_steps,
            bg_samples=args.bg_samples,
            lbfgs_steps=args.lbfgs_steps,
            lbfgs_tol=args.lbfgs_tol,
            enable_egnn=args.enable_egnn,
            egnn_relax_steps=args.egnn_relax_steps,
            egnn_batch_size=args.egnn_batch_size,
            egnn_layers=args.egnn_layers,
            egnn_weights=args.egnn_weights,
            max_sa_score=args.max_sa_score,
            disable_sa_filter=args.disable_sa_filter,
            verbose=True,
        )
        return 0

    # =========================================================================
    # MODE 3: Cross-Material 5-Stage Funnel Benchmark Mode
    # =========================================================================
    if args.benchmark_specs:
        from dens_city.swarm.funnel import run_all_specs_funnel_benchmark

        out_dir = args.out_dir or f"runs/full_system_benchmark_{ts}"
        return run_all_specs_funnel_benchmark(
            specs_dir=args.specs_dir,
            train_steps=args.train_steps,
            num_envs=args.num_envs,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            early_stopping_lookback=args.early_stopping_lookback,
            early_stopping_delta=args.early_stopping_delta,
            num_candidates=args.num_candidates if "--num-candidates" in argv else 64,
            batch_size=args.batch_size if ("-b" in argv or "--batch-size" in argv) else None,
            egnn_batch_size=args.egnn_batch_size if "--egnn-batch-size" in argv else None,
            egnn_relax_steps=args.egnn_relax_steps,
            top_k=args.top_k if "--top-k" in argv else 10,
            out_dir=out_dir,
            max_sa_score=args.max_sa_score,
            enable_egnn=args.enable_egnn,
            recurrent=args.recurrent,
        )

    # =========================================================================
    # MODE 4: Stage 1 RL Swarm Training Mode
    # =========================================================================
    if args.train_swarm:
        from dens_city.swarm.trainer import train_swarm_policy

        spec_file = resolve_spec_path(args.spec, default_dir=args.specs_dir)
        if not spec_file or not spec_file.exists():
            print(colored(f"Error: Target specification YAML not found: {args.spec}", "red"), file=sys.stderr)
            return 1

        train_swarm_policy(
            spec=spec_file,
            total_timesteps=args.train_steps,
            num_envs=args.num_envs,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            recurrent=args.recurrent,
            no_curriculum=args.no_curriculum,
            no_early_stopping=args.no_early_stopping,
            early_stopping_lookback=args.early_stopping_lookback,
            early_stopping_delta=args.early_stopping_delta,
            no_sa_penalty=args.no_sa_penalty,
            sa_threshold=args.sa_threshold,
            sa_penalty_slope=args.sa_penalty_slope,
            no_dynamic_entropy=args.no_dynamic_entropy,
            checkpoint_dir=args.checkpoint_dir,
            export_dir=args.export_dir,
            seed=args.seed,
        )
        return 0

    # =========================================================================
    # MODE 5: Curriculum Sweep Mode
    # =========================================================================
    if args.sweep:
        from dens_city.swarm.sweep import run_curriculum_sweep

        out_dir = args.out_dir or "runs/constellation_sweeps"
        return run_curriculum_sweep(
            specs=args.specs,
            num_trials_per_spec=args.num_trials_per_spec,
            steps_per_trial=args.steps_per_trial,
            num_envs=args.num_envs if "--num-envs" in argv else 8,
            output_dir=out_dir,
            seed=args.seed,
        )

    # =========================================================================
    # MODE 6: Chemical Diversity & Synthesizability Diagnostics Mode
    # =========================================================================
    if args.eval_swarm:
        from dens_city.swarm.evaluator import evaluate_all_swarm_specs

        out_dir = args.out_dir or "runs/rl_stage_evaluation"
        return evaluate_all_swarm_specs(
            specs_dir=args.specs_dir,
            timesteps=args.train_steps if ("--train-steps" in argv or "--total-timesteps" in argv) else 10000,
            num_candidates=args.num_candidates if "--num-candidates" in argv else 50,
            num_envs=args.num_envs,
            out_dir=out_dir,
        )

    # =========================================================================
    # MODE 7: Combinatorial Molecular Library Generation Mode
    # =========================================================================
    if args.generate_library:
        from dens_city.utils.library_generator import run_library_generator

        spec_file = resolve_spec_path(args.spec, default_dir=args.specs_dir)
        if not spec_file or not spec_file.exists():
            print(colored(f"Error: Target specification YAML not found: {args.spec}", "red"), file=sys.stderr)
            return 1

        return run_library_generator(
            spec_path=spec_file,
            target_count=args.target_count,
            out_dir=args.out_dir,
            workers=args.workers,
            seed=args.seed,
            skip_3d=args.skip_3d,
            skip_write=args.skip_write,
        )

    # =========================================================================
    # MODE 8: Populate Test Data & FreeSolv Mode
    # =========================================================================
    if args.populate_test_data:
        from dens_city.utils.test_data_generator import generate_test_data

        target_dir = args.data_dir if "--data-dir" in argv or "-d" in argv else "data/test_data"
        generate_test_data(
            dest_dir=target_dir,
            populate_entire_freesolv=args.all_freesolv,
        )
        return 0

    # =========================================================================
    # MODE 9: FreeSolv Validation & Verification Mode
    # =========================================================================
    if args.verify_freesolv:
        from dens_city.utils.verification import verify_pipeline_against_freesolv

        return verify_pipeline_against_freesolv(
            results_dir=args.results_dir,
            database_path=args.database,
            report_out=args.report_out,
            run_e2e=args.run_e2e,
            populate_all_freesolv=args.all_freesolv,
            energy_engine=args.energy_engine,
            force_egnn=args.force_egnn,
            batch_size=args.batch_size if ("-b" in argv or "--batch-size" in argv) else None,
        )

    # =========================================================================
    # MODE 10: Standard Coupled cDFT Screening Mode (Default)
    # =========================================================================
    out_dir = args.out_dir or f"runs/batch_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    jsonl_log_path = os.path.join(out_dir, "pipeline_summary.jsonl")

    # Auto-throttle batch size for EGNN to prevent GPU VRAM exhaustion from (B, 128, 128, 128) tensors
    effective_engine = "egnn" if args.force_egnn else args.energy_engine
    if effective_engine in ("egnn", "auto") and "--batch-size" not in argv and "-b" not in argv:
        args.batch_size = 32
        print(
            colored(
                f"[ENGINE] Routing to {effective_engine.upper()} engine (auto-throttling batch size to 32 to optimize message passing VRAM)",
                "cyan",
            )
        )
    elif effective_engine in ("egnn", "auto"):
        print(colored(f"[ENGINE] Routing to {effective_engine.upper()} engine (batch size: {args.batch_size})", "cyan"))

    materials = discover_materials(args.data_dir, args.materials)
    if not materials:
        print(colored(f"Error: No materials found in data directory: {args.data_dir}", "red"))
        return 1

    debug_log_dir = None
    if args.debug:
        debug_log_dir = Path("data") / f"logs_{ts}"
        debug_log_dir.mkdir(parents=True, exist_ok=True)

    print_banner(
        out_dir=out_dir,
        num_materials=len(materials),
        workers=args.workers,
        temp_k=args.temp,
        pressure_bar=args.pressure,
        batch_size=args.batch_size,
        beam=args.beam,
        skip_bg=args.skip_bg,
        benchmark=args.benchmark,
        energy_engine=args.energy_engine,
    )

    tasks = [
        MaterialPipelineTask(
            material_path_or_name=m,
            out_dir=out_dir,
            temperature_k=args.temp,
            pressure_bar=args.pressure,
            chemical_potential_kbt=args.mu,
            grid=args.grid,
            cdft_steps=args.cdft_steps,
            cdft_lr=args.cdft_lr,
            bg_steps=args.bg_steps,
            bg_batch_size=args.batch_size,
            bg_lr=args.bg_lr,
            bg_samples=args.bg_samples,
            bg_w_tor=args.bg_w_tor,
            bg_mcmc_steps=args.bg_mcmc_steps,
            bg_mcmc_step_size=args.bg_mcmc_step_size,
            skip_bg=args.skip_bg,
            debug=args.debug,
            debug_log_path=str(debug_log_dir / f"{Path(m).stem}.log") if debug_log_dir else None,
            energy_engine=args.energy_engine,
            force_egnn=args.force_egnn,
        )
        for m in materials
    ]

    results: List[MaterialPipelineResult] = []
    t_batch_start = time.perf_counter()

    task_chunks = [tasks[i : i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    async_writer = AsyncArtifactWriter()
    prefetcher = AsyncBatchPrefetcher(
        task_chunks=task_chunks,
        batch_size=args.batch_size,
        prefetch_depth=2,
        energy_engine=args.energy_engine,
    ).start()

    try:
        with open(jsonl_log_path, "w", encoding="utf-8") as jsonl_file:
            for prepared_batch in prefetcher:
                chunk_results = execute_prepared_batch(
                    prepared_batch=prepared_batch,
                    async_writer=async_writer,
                )
                for res in chunk_results:
                    results.append(res)
                    print_result_row(res)
                    jsonl_file.write(json.dumps(res.to_dict()) + "\n")
                    jsonl_file.flush()
    finally:
        prefetcher.close()
        async_writer.flush()
        async_writer.close()

    t_batch_total = time.perf_counter() - t_batch_start

    n_success = sum(
        1 for r in results if r.status in [PipelineStatus.SUCCESS.value, PipelineStatus.SUCCESS_CDFT_ONLY.value]
    )
    n_skipped = sum(1 for r in results if r.status == PipelineStatus.SKIPPED_THERMO.value)
    n_failed = sum(
        1
        for r in results
        if r.status
        in [
            PipelineStatus.FAILED_TRAINING.value,
            PipelineStatus.FAILED_TIMEOUT.value,
            PipelineStatus.FAILED_ERROR.value,
        ]
    )

    if args.benchmark:
        print_benchmark_table(results, t_batch_total, args.batch_size)

    print("=" * 88)
    print(colored(f"  Batch Processing Completed in {t_batch_total:.2f} seconds", "cyan"))
    print(f"  Total Processed : {len(results)}")
    print(colored(f"  Successful      : {n_success}", "green"))
    if n_skipped > 0:
        print(colored(f"  Skipped (Thermo): {n_skipped}", "yellow"))
    if n_failed > 0:
        print(colored(f"  Failed          : {n_failed}", "red"))
    print(f"  Master Summary  : {jsonl_log_path}")
    if debug_log_dir:
        print(f"  Debug Logs Dir  : {debug_log_dir}")
    print("=" * 88)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
