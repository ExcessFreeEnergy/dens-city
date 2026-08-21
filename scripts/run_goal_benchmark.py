#!/usr/bin/env python3
"""
Goal Benchmark Runner: Executes all 20 materials with BEAM=2 DEBUG=2 for 3 iterations,
generating 20 detailed compiler and execution log files in data/logs/<material>.log.
"""

import os
import sys
import time
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple


def get_all_materials(data_dir: Path) -> List[str]:
    mol2_files = sorted(data_dir.glob("*.mol2"))
    return [f.stem for f in mol2_files]


def run_single_material(
    material: str,
    log_dir: Path,
    data_dir: Path,
    cdft_steps: int = 3,
    bg_steps: int = 3,
    bg_samples: int = 2,
    timeout: float = 300.0,
) -> Dict[str, any]:
    log_file_path = log_dir / f"{material}.log"
    out_dir = Path("/tmp/runs_goal") / material
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_batch_pipeline.py",
        "--data-dir", str(data_dir),
        "--materials", material,
        "--workers", "1",
        "--cdft-steps", str(cdft_steps),
        "--bg-steps", str(bg_steps),
        "--bg-samples", str(bg_samples),
        "--out-dir", str(out_dir),
        "--no-plot",
    ]

    env = os.environ.copy()
    env["BEAM"] = "2"
    env["DEBUG"] = "2"
    env["PYTHONPATH"] = f"{Path('src').resolve()}:{env.get('PYTHONPATH', '')}"

    t0 = time.perf_counter()
    status = "SUCCESS"
    err_msg = None

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                cwd=str(Path.cwd()),
            )
            if proc.returncode != 0:
                status = f"FAILED_EXIT_{proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        err_msg = f"Exceeded {timeout}s"
    except Exception as e:
        status = "ERROR"
        err_msg = str(e)

    elapsed = time.perf_counter() - t0
    log_size = log_file_path.stat().st_size if log_file_path.exists() else 0

    return {
        "material": material,
        "status": status,
        "elapsed": elapsed,
        "log_path": str(log_file_path),
        "log_size_kb": log_size / 1024.0,
        "error": err_msg,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "test_data"
    log_dir = repo_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    materials = get_all_materials(data_dir)
    print("=" * 80)
    print(f"  dens-city: BEAM=2 DEBUG=2 Benchmark for {len(materials)} Materials")
    print(f"  Target Log Directory: {log_dir}")
    print(f"  Iterations: 3 cDFT steps, 3 BG steps, 2 BG samples")
    print("=" * 80)

    n_workers = min(6, os.cpu_count() or 1)
    results = []
    t_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_mat = {
            executor.submit(
                run_single_material,
                mat,
                log_dir,
                data_dir,
                cdft_steps=3,
                bg_steps=3,
                bg_samples=2,
                timeout=300.0,
            ): mat
            for mat in materials
        }

        for idx, future in enumerate(concurrent.futures.as_completed(future_to_mat), 1):
            res = future.result()
            results.append(res)
            print(
                f"[{idx:02d}/{len(materials):02d}] {res['material']:<25} | "
                f"Status: {res['status']:<12} | "
                f"Time: {res['elapsed']:6.2f}s | "
                f"Log Size: {res['log_size_kb']:6.1f} KB"
            )

    total_time = time.perf_counter() - t_start
    n_success = sum(1 for r in results if r["status"] == "SUCCESS")

    print("=" * 80)
    print(f"Benchmark Complete: {n_success}/{len(materials)} Successful in {total_time:.2f}s")
    print(f"All logs saved to: {log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
