#!/usr/bin/env python3
"""
Goal Benchmark Runner: Executes all 20 materials sorted by site count (1-site, 2-site, 3-site, etc.)
with BEAM=2 DEBUG=2 for 3 iterations, generating 20 detailed compiler and execution log files
in data/logs/<material>.log and noting BEAM recompilation / cache hit status.
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure dens_city package is in PYTHONPATH
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dens_city.utils.materials import MaterialLoader


def get_material_site_count(mat_name: str) -> int:
    try:
        mat = MaterialLoader.load_material(mat_name)
        return len(mat.sites) if mat.sites else 1
    except Exception:
        return 1


def get_power_of_2_padded_size(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def get_sorted_materials(data_dir: Path, selected: Optional[List[str]] = None) -> List[Tuple[str, int, int]]:
    mol2_files = sorted(data_dir.glob("*.mol2"))
    all_names = [f.stem for f in mol2_files]
    target_names = [m for m in selected if m in all_names] if selected else all_names

    mats_with_sites = [
        (m, get_material_site_count(m), get_power_of_2_padded_size(get_material_site_count(m))) for m in target_names
    ]
    # Sort strictly by power-of-2 bucket ascending, then by actual site count, then alphabetically
    return sorted(mats_with_sites, key=lambda item: (item[2], item[1], item[0]))


def analyze_compiler_log(log_content: str) -> Dict[str, any]:
    cache_misses = len(re.findall(r"CACHE MISS", log_content))
    beam_passes = len(re.findall(r"\d+\s*->\s*\d+\s*actions", log_content))
    beam_timeouts = len(re.findall(r"BEAM COMPILE TIMEOUT", log_content))

    if beam_passes > 0 or cache_misses > 0:
        beam_note = f"BEAM Compiled {beam_passes} kernels ({cache_misses} cache misses"
        if beam_timeouts > 0:
            beam_note += f", {beam_timeouts} compile timeouts)"
        else:
            beam_note += ")"
    else:
        beam_note = "BEAM Cache Hit (Reused cached kernels, 0 recompilations)"

    return {
        "cache_misses": cache_misses,
        "beam_passes": beam_passes,
        "beam_timeouts": beam_timeouts,
        "beam_note": beam_note,
    }


def run_single_material(
    material: str,
    site_count: int,
    pad_site_count: int,
    log_dir: Path,
    data_dir: Path,
    cdft_steps: int = 3,
    bg_steps: int = 3,
    bg_samples: int = 2,
    timeout: float = 500.0,
) -> Dict[str, any]:
    log_file_path = log_dir / f"{material}.log"
    out_dir = Path("/tmp/runs_goal") / material
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_batch_pipeline.py",
        "--data-dir",
        str(data_dir),
        "--materials",
        material,
        "--workers",
        "1",
        "--cdft-steps",
        str(cdft_steps),
        "--bg-steps",
        str(bg_steps),
        "--bg-samples",
        str(bg_samples),
        "--out-dir",
        str(out_dir),
        "--no-plot",
    ]

    env = os.environ.copy()
    env["BEAM"] = "2"
    env["DEBUG"] = "2"
    env["PYTHONPATH"] = f"{src_path.resolve()}:{env.get('PYTHONPATH', '')}"

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

    # Analyze compiler characteristics and append summary note to log
    log_text = log_file_path.read_text(encoding="utf-8", errors="ignore") if log_file_path.exists() else ""
    comp_info = analyze_compiler_log(log_text)

    if log_file_path.exists():
        with open(log_file_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n" + "=" * 80 + "\n")
            log_f.write(f"  Compiler & BEAM Summary Note: {comp_info['beam_note']}\n")
            log_f.write(
                f"  Material Sites: {site_count} (Padded: {pad_site_count}) | Status: {status} | Elapsed: {elapsed:.2f}s\n"
            )
            log_f.write("=" * 80 + "\n")

    return {
        "material": material,
        "sites": site_count,
        "pad_sites": pad_site_count,
        "status": status,
        "elapsed": elapsed,
        "log_path": str(log_file_path),
        "log_size_kb": log_size / 1024.0,
        "beam_note": comp_info["beam_note"],
        "error": err_msg,
    }


def main():
    parser = argparse.ArgumentParser(description="Power-of-2 Site-Grouped Goal Benchmark Runner for dens-city")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of concurrent worker processes (default: 1 for GPU BEAM safety)"
    )
    parser.add_argument("--timeout", type=float, default=500.0, help="Per-material timeout in seconds (default: 500.0)")
    parser.add_argument("--materials", nargs="+", default=None, help="Specific materials to run (default: all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "test_data"
    log_dir = repo_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    sorted_materials = get_sorted_materials(data_dir, args.materials)

    print("=" * 110)
    print(f"  dens-city: BEAM=2 DEBUG=2 Power-of-2 Site-Grouped Benchmark for {len(sorted_materials)} Materials")
    print(f"  Configuration: Timeout={args.timeout}s, Workers={args.workers}, Output={log_dir}")
    print("  Iterations: 3 cDFT steps, 3 BG steps, 2 BG samples")
    print("=" * 110)
    print(
        f"{'#':<3} | {'Bucket':<8} | {'Material':<24} | {'Sites (Pad)':<11} | {'Status':<12} | {'Time (s)':<8} | {'BEAM Compiler Note'}"
    )
    print("-" * 110)

    results = []
    t_start = time.perf_counter()
    current_bucket = None

    if args.workers <= 1:
        # Strictly sequential execution grouped by power-of-2 buckets
        for idx, (mat, sites, pad_sites) in enumerate(sorted_materials, 1):
            if current_bucket != pad_sites:
                current_bucket = pad_sites
                b_name = f"2^{int(pad_sites.bit_length() - 1)}"

            res = run_single_material(
                mat,
                sites,
                pad_sites,
                log_dir,
                data_dir,
                cdft_steps=3,
                bg_steps=3,
                bg_samples=2,
                timeout=args.timeout,
            )
            results.append(res)
            site_str = f"{sites} ({pad_sites})"
            print(
                f"[{idx:02d}] | {b_name:<8} | {mat:<24} | {site_str:<11} | {res['status']:<12} | {res['elapsed']:8.2f} | {res['beam_note']}"
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_mat = {
                executor.submit(
                    run_single_material,
                    mat,
                    sites,
                    pad_sites,
                    log_dir,
                    data_dir,
                    3,
                    3,
                    2,
                    args.timeout,
                ): (mat, sites, pad_sites)
                for mat, sites, pad_sites in sorted_materials
            }
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_mat), 1):
                mat, sites, pad_sites = future_to_mat[future]
                res = future.result()
                results.append(res)
                b_name = f"2^{int(pad_sites.bit_length() - 1)}"
                site_str = f"{sites} ({pad_sites})"
                print(
                    f"[{idx:02d}] | {b_name:<8} | {mat:<24} | {site_str:<11} | {res['status']:<12} | {res['elapsed']:8.2f} | {res['beam_note']}"
                )

    total_time = time.perf_counter() - t_start
    n_success = sum(1 for r in results if r["status"] == "SUCCESS")
    print("=" * 110)
    print(f"Power-of-2 Grouped Benchmark Complete: {n_success}/{len(results)} Successful in {total_time:.2f}s")
    print(f"All logs saved to: {log_dir}")
    print("=" * 110)


if __name__ == "__main__":
    main()
