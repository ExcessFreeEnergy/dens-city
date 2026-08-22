#!/usr/bin/env python3
"""
dens-city: High-Throughput Batch Processing Pipeline.
Executes the coupled cDFT Mean-Field Screening -> Spatial Prior -> Boltzmann Generator -> Trajectory Export
across directories of arbitrary .mol2 material files with process isolation, timeout handling, and JSONL logging.

Usage:
    # Run all materials in test_data/
    python scripts/run_batch_pipeline.py

    # Run on a custom data directory with 4 parallel worker processes
    python scripts/run_batch_pipeline.py --data-dir data/mol2files_gaff --workers 4 --out-dir runs/batch_gaff

    # Fast-screening cDFT only (skipping Boltzmann generator)
    python scripts/run_batch_pipeline.py --skip-bg --workers 4

    # Run specific materials at T=350 K, P=5.0 bar
    python scripts/run_batch_pipeline.py --materials water argon methane --temp 350.0 --pressure 5.0
"""

import argparse
import multiprocessing as mp
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure src/ is on PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from tinygrad.helpers import colored, getenv
from dens_city.materials import MaterialLoader
from dens_city.pipeline import (
    MaterialPipelineTask,
    MaterialPipelineResult,
    PipelineStatus,
    process_material_task,
)


def discover_materials(data_dir: str, requested_materials: Optional[List[str]] = None) -> List[str]:
    """
    Discovers .mol2 files in the specified directory and matches against requested materials.
    """
    p = Path(data_dir)
    discovered = []
    if p.exists() and p.is_dir():
        for f in sorted(p.glob("*.mol2")):
            discovered.append(str(f))

    if not discovered:
        # Fallback to registered material names if no mol2 files found in directory
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
        return filtered if filtered else discovered

    return discovered


def print_banner(out_dir: str, num_materials: int, workers: int, temp_k: float, pressure_bar: Optional[float], skip_bg: bool) -> None:
    print("=" * 80)
    print(colored("  dens-city: High-Throughput Isolated Batch Processor Pipeline", "cyan"))
    print("=" * 80)
    print(f"  Target Materials  : {num_materials} items")
    print(f"  Worker Processes  : {workers} isolated processes")
    print(f"  Thermodynamics    : T = {temp_k:.1f} K" + (f", P = {pressure_bar:.2f} bar" if pressure_bar else ""))
    print(f"  Pipeline Mode     : {'cDFT Fast Screening (No BG)' if skip_bg else 'Full Coupled (cDFT + Boltzmann Generator)'}")
    print(f"  Artifact Output   : {out_dir}")
    print("=" * 80)
    print(f"{'Material':<22} | {'Sites':<5} | {'cDFT (s)':<8} | {'BG (s)':<8} | {'P_wall (bar)':<12} | {'Status':<16}")
    print("-" * 80)


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
    p_wall_str = f"{res.wall_pressure_bar:10.2f}" if res.status in [PipelineStatus.SUCCESS.value, PipelineStatus.SUCCESS_CDFT_ONLY.value] else "    --    "
    print(f"{res.material_name:<22} | {res.num_sites:<5} | {cdft_t_str:<8} | {bg_t_str:<8} | {p_wall_str:<12} | {status_str}")


def main() -> int:
    default_data_dir = str(root_dir / "test_data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out_dir = str(root_dir / "runs" / f"batch_{timestamp}")

    parser = argparse.ArgumentParser(
        description="Execute high-throughput batch cDFT and Boltzmann Generator pipelines across material sets."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=default_data_dir,
        help=f"Directory containing .mol2 material files (default: {default_data_dir})",
    )
    parser.add_argument(
        "--materials",
        "-m",
        nargs="+",
        default=None,
        help="Optional specific materials to filter from data-dir (or 'all')",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default=default_out_dir,
        help=f"Destination path for structured artifacts and logs (default: {default_out_dir})",
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
        default=120.0,
        help="Maximum execution timeout per material in seconds (default: 120s)",
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
        help="Torsional rotamer loss biasing weight (default: 0.0, recommended 0.05 - 5.0 for long chains)",
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
        help="Skip Boltzmann Generator phase and halt after cDFT screening",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=128,
        help="Spatial grid points for cDFT 1D pore discretization (default: 128)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress ASCII terminal pore graphs in console",
    )

    args = parser.parse_args()

    materials = discover_materials(args.data_dir, args.materials)
    if not materials:
        print(colored(f"Error: No materials found in data directory: {args.data_dir}", "red"))
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_log_path = os.path.join(args.out_dir, "pipeline_summary.jsonl")

    print_banner(
        out_dir=args.out_dir,
        num_materials=len(materials),
        workers=args.workers,
        temp_k=args.temp,
        pressure_bar=args.pressure,
        skip_bg=args.skip_bg,
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
            bg_lr=args.bg_lr,
            bg_samples=args.bg_samples,
            bg_w_tor=args.bg_w_tor,
            bg_mcmc_steps=args.bg_mcmc_steps,
            bg_mcmc_step_size=args.bg_mcmc_step_size,
            skip_bg=args.skip_bg,
            no_plot=args.no_plot,
        )
        for m in materials
    ]

    results: List[MaterialPipelineResult] = []
    t_batch_start = time.perf_counter()

    with open(jsonl_log_path, "w", encoding="utf-8") as jsonl_file:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            future_to_task = {
                executor.submit(process_material_task, task): task
                for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                mat_name = Path(task.material_path_or_name).stem
                try:
                    res = future.result(timeout=args.timeout)
                except concurrent.futures.TimeoutError:
                    res = MaterialPipelineResult(
                        material_name=mat_name,
                        status=PipelineStatus.FAILED_TIMEOUT.value,
                        error_message=f"Task exceeded maximum timeout limit of {args.timeout} seconds.",
                        runtime_seconds=args.timeout,
                    )
                except Exception as e:
                    res = MaterialPipelineResult(
                        material_name=mat_name,
                        status=PipelineStatus.FAILED_ERROR.value,
                        error_message=f"Unhandled worker process crash: {str(e)}",
                    )

                results.append(res)
                print_result_row(res)

                # Append to streaming JSONL log
                jsonl_file.write(json.dumps(res.to_dict()) + "\n")
                jsonl_file.flush()

    t_batch_total = time.perf_counter() - t_batch_start

    # Final summary statistics
    n_success = sum(1 for r in results if r.status in [PipelineStatus.SUCCESS.value, PipelineStatus.SUCCESS_CDFT_ONLY.value])
    n_skipped = sum(1 for r in results if r.status == PipelineStatus.SKIPPED_THERMO.value)
    n_failed = sum(1 for r in results if r.status in [PipelineStatus.FAILED_TRAINING.value, PipelineStatus.FAILED_TIMEOUT.value, PipelineStatus.FAILED_ERROR.value])

    print("=" * 80)
    print(colored(f"Batch Processing Completed in {t_batch_total:.2f} seconds", "cyan"))
    print(f"  Total Processed : {len(results)}")
    print(colored(f"  Successful      : {n_success}", "green"))
    if n_skipped > 0:
        print(colored(f"  Skipped (Thermo): {n_skipped}", "yellow"))
    if n_failed > 0:
        print(colored(f"  Failed          : {n_failed}", "red"))
    print(f"  Master Summary  : {jsonl_log_path}")
    print("=" * 80)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
