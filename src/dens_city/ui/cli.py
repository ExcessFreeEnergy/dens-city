"""
dens-city: Unified Command-Line Interface.
Supports coupled cDFT screening -> Boltzmann Generator batch execution,
high-performance 3D interactive Raylib visualization, compiler BEAM search,
and high-throughput batched tensor parallelism across molecular datasets.

Usage:
    # 3D Interactive Raylib Visualizer
    uv run dens-city --interactive --materials argon
    uv run dens-city --interactive --materials water benzene 5cb

    # Standard High-Throughput Coupled Pipeline
    uv run dens-city --materials argon water --batch-size 32

    # Benchmark Mode with BEAM=2 Compiler Search
    uv run dens-city --materials all --benchmark --beam 2

    # Debug Mode with Detailed Compiler Traces
    uv run dens-city --materials argon benzene --debug
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
    MaterialPipelineResult,
    MaterialPipelineTask,
    PipelineStatus,
    process_batched_materials,
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
) -> None:
    print("=" * 88)
    print(colored("  dens-city: Molecular Classical Density Functional Theory & Generative Platform", "cyan"))
    print("=" * 88)
    print(f"  Target Materials  : {num_materials} items")
    print(f"  Parallel Workers  : {workers} isolated processes (Batch Size = {batch_size} molecules/tensor)")
    print(f"  Compiler BEAM     : BEAM={beam}")
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


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for dens-city."""
    if argv is None:
        argv = sys.argv[1:]

    default_data_dir = "data/test_data"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out_dir = f"runs/batch_{ts}"

    parser = argparse.ArgumentParser(
        prog="dens-city",
        description="dens-city: High-Performance Molecular Classical Density Functional Theory & 3D Generative Platform.",
    )
    parser.add_argument(
        "--materials",
        "-m",
        nargs="+",
        default=None,
        help="Material names, .mol2 files, or 'all' (default: all available)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=default_data_dir,
        help=f"Directory containing .mol2 material files (default: {default_data_dir})",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default=default_out_dir,
        help=f"Destination path for structured artifacts and logs (default: {default_out_dir})",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Launch the 3D Raylib molecular viewer for real-time visualization",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Profile execution time per material and output comprehensive benchmark summary",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=2,
        help="tinygrad compiler BEAM search optimization level (default: 2)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG=2 and write detailed per-material compiler logs to data/logs_<timestamp>/",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Molecule batch size for parallel tensor evaluation along Axis 0 (default: 32)",
    )
    parser.add_argument(
        "--temp",
        "-t",
        type=float,
        default=300.0,
        help="Thermodynamic reservoir temperature in Kelvin (default: 300.0)",
    )
    parser.add_argument(
        "--pressure",
        "-p",
        type=float,
        default=1.0,
        help="Thermodynamic reservoir pressure in bar (default: 1.0)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Optional chemical potential in k_B T (overrides bulk pressure calculation)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=f"Number of concurrent worker processes (default: {min(4, os.cpu_count() or 1)})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum execution timeout per material in seconds (default: 180s)",
    )
    parser.add_argument(
        "--grid",
        "-g",
        type=int,
        default=128,
        help="Spatial grid points for cDFT 1D pore discretization (default: 128)",
    )
    parser.add_argument(
        "--cdft-steps",
        type=int,
        default=60,
        help="cDFT solver variational optimization steps (default: 60)",
    )
    parser.add_argument(
        "--cdft-lr",
        type=float,
        default=0.02,
        help="cDFT solver learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--bg-steps",
        type=int,
        default=40,
        help="Boltzmann Generator training steps (default: 40)",
    )
    parser.add_argument(
        "--bg-lr",
        type=float,
        default=0.01,
        help="Boltzmann Generator learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--bg-samples",
        type=int,
        default=100,
        help="Number of 3D configurations to sample into .xyz trajectory (default: 100)",
    )
    parser.add_argument(
        "--bg-w-tor",
        type=float,
        default=0.0,
        help="Torsional rotamer loss biasing weight (default: 0.0)",
    )
    parser.add_argument(
        "--bg-mcmc-steps",
        type=int,
        default=0,
        help="Number of latent space Metropolis Monte Carlo relaxation steps per sample (default: 0)",
    )
    parser.add_argument(
        "--bg-mcmc-step-size",
        type=float,
        default=0.1,
        help="Step size for Gaussian perturbations in latent MCMC relaxation (default: 0.1)",
    )
    parser.add_argument(
        "--skip-bg",
        action="store_true",
        default=False,
        help="Skip Boltzmann Generator phase and halt after cDFT screening",
    )

    args = parser.parse_args(argv)

    # Configure tinygrad compiler environment
    if args.beam:
        os.environ["BEAM"] = str(args.beam)
    if args.debug:
        os.environ["DEBUG"] = "2"

    materials = discover_materials(args.data_dir, args.materials)
    if not materials:
        print(colored(f"Error: No materials found in data directory: {args.data_dir}", "red"))
        return 1

    # 1. Interactive 3D Raylib Visualizer Mode
    if args.interactive:
        from dens_city.ui.viewer import run_interactive_viewer

        print(colored("==========================================================================", "cyan"))
        print(colored("  dens-city: 3D Interactive Raylib Molecular Visualizer                  ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"Target Materials   : {materials}")
        print("Controls           : Left Drag to Orbit | Scroll to Zoom | ← / → to Switch Materials")
        run_interactive_viewer(material_names=materials)
        return 0

    # 2. Setup output and debug logging directories
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_log_path = os.path.join(args.out_dir, "pipeline_summary.jsonl")

    debug_log_dir = None
    if args.debug:
        debug_log_dir = Path("data") / f"logs_{ts}"
        debug_log_dir.mkdir(parents=True, exist_ok=True)

    print_banner(
        out_dir=args.out_dir,
        num_materials=len(materials),
        workers=args.workers,
        temp_k=args.temp,
        pressure_bar=args.pressure,
        batch_size=args.batch_size,
        beam=args.beam,
        skip_bg=args.skip_bg,
        benchmark=args.benchmark,
    )

    tasks = [
        MaterialPipelineTask(
            material_path_or_name=m,
            out_dir=args.out_dir,
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
        )
        for m in materials
    ]

    results: List[MaterialPipelineResult] = []
    t_batch_start = time.perf_counter()

    task_chunks = [tasks[i : i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
    async_writer = AsyncArtifactWriter()

    try:
        with open(jsonl_log_path, "w", encoding="utf-8") as jsonl_file:
            for chunk in task_chunks:
                chunk_results = process_batched_materials(
                    batch_tasks=chunk,
                    batch_size=args.batch_size,
                    async_writer=async_writer,
                )
                for res in chunk_results:
                    results.append(res)
                    print_result_row(res)
                    jsonl_file.write(json.dumps(res.to_dict()) + "\n")
                    jsonl_file.flush()
    finally:
        async_writer.flush()
        async_writer.close()

    t_batch_total = time.perf_counter() - t_batch_start

    # Final summary statistics
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
