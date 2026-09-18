#!/usr/bin/env python3
"""
Smart Parallel Test Runner for dens-city.

Partitions the test suite into:
1. CPU-Bound Unit Tests: Run with high parallelism (max_workers = os.cpu_count()).
2. GPU/JIT Integration Tests: Run with capped parallelism (max_workers = 2) to
   prevent GPU memory fragmentation, OOM, and kernel contention.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

EXPLICIT_GPU_FILES = {
    "test_egnn_forcefield.py",
    "test_charge_trainer.py",
    "test_funnel_egnn_stage4.py",
    "test_generative_funnel.py",
    "test_boltzmann_jit_perf.py",
    "test_solvatum.py",
    "test_cli_unified.py",
    "test_batch_pipeline.py",
    "test_cdft_swarm.py",
    "test_swarm_policy.py",
    "test_batched_lbfgs.py",
    "test_boltzmann_generator.py",
    "test_audit_remediation_ingestion.py",
}


def is_gpu_test(file_path: Path) -> bool:
    if file_path.name in EXPLICIT_GPU_FILES:
        return True
    try:
        content = file_path.read_text(encoding="utf-8")
        return "pytest.mark.gpu" in content or "pytestmark = pytest.mark.gpu" in content
    except Exception:
        return False


def run_single_test_file(test_file: Path, extra_args: List[str]) -> Tuple[str, bool, float, str]:
    python_bin = REPO_ROOT / ".venv" / "bin" / "pytest"
    cmd = [str(python_bin) if python_bin.exists() else "pytest", str(test_file), "-q"] + extra_args
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    passed = res.returncode == 0
    output = res.stdout if passed else (res.stdout + "\n" + res.stderr)
    return test_file.name, passed, dt, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Parallel Test Runner for dens-city")
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 4, help="CPU test workers")
    parser.add_argument("--gpu-workers", type=int, default=2, help="Capped GPU test workers (default: 2)")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU-bound tests")
    parser.add_argument("--gpu-only", action="store_true", help="Run only GPU-bound tests")
    parser.add_argument("--pytest-args", type=str, default="", help="Extra arguments passed to pytest")
    args = parser.parse_args()

    extra_pytest_args = args.pytest_args.split() if args.pytest_args else []

    all_test_files = sorted(TESTS_DIR.glob("test_*.py"))
    cpu_files = [f for f in all_test_files if not is_gpu_test(f)]
    gpu_files = [f for f in all_test_files if is_gpu_test(f)]

    print("=" * 80)
    print("  dens-city Smart Parallel Test Runner")
    print("=" * 80)
    print(f"  Total Test Files : {len(all_test_files)}")
    print(f"  CPU Test Files   : {len(cpu_files)} (Workers: {args.cpu_workers})")
    print(f"  GPU Test Files   : {len(gpu_files)} (Workers: {args.gpu_workers}, OOM-protected)")
    print("=" * 80)

    t_global_start = time.perf_counter()
    failures: List[Tuple[str, str]] = []
    total_passed = 0
    total_failed = 0

    # 1. Run CPU Tests
    if not args.gpu_only and cpu_files:
        print(f"\n[PHASE 1/2] Running {len(cpu_files)} CPU-Bound Test Suites ({args.cpu_workers} workers)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.cpu_workers) as executor:
            futures = {executor.submit(run_single_test_file, f, extra_pytest_args): f for f in cpu_files}
            for fut in concurrent.futures.as_completed(futures):
                name, passed, dt, output = fut.result()
                if passed:
                    total_passed += 1
                    status_str = "\033[92mPASS\033[0m"
                else:
                    total_failed += 1
                    status_str = "\033[91mFAIL\033[0m"
                    failures.append((name, output))
                print(f"  [{status_str}] {name:<42} ({dt:5.2f}s)")

    # 2. Run GPU Tests
    if not args.cpu_only and gpu_files:
        print(f"\n[PHASE 2/2] Running {len(gpu_files)} GPU/JIT Integration Suites ({args.gpu_workers} workers)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.gpu_workers) as executor:
            futures = {executor.submit(run_single_test_file, f, extra_pytest_args): f for f in gpu_files}
            for fut in concurrent.futures.as_completed(futures):
                name, passed, dt, output = fut.result()
                if passed:
                    total_passed += 1
                    status_str = "\033[92mPASS\033[0m"
                else:
                    total_failed += 1
                    status_str = "\033[91mFAIL\033[0m"
                    failures.append((name, output))
                print(f"  [{status_str}] {name:<42} ({dt:5.2f}s)")

    t_global_total = time.perf_counter() - t_global_start

    print("\n" + "=" * 80)
    print("  Test Execution Summary")
    print("=" * 80)
    print(f"  Passed Test Files : {total_passed}")
    print(f"  Failed Test Files : {total_failed}")
    print(f"  Total Wall Time   : {t_global_total:.2f} seconds")
    print("=" * 80)

    if failures:
        print("\n\033[91mFAILED SUITES:\033[0m")
        for name, out in failures:
            print(f"\n--- Output for {name} ---")
            print(out)
        return 1

    print("\n\033[92mAll test suites passed successfully!\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
