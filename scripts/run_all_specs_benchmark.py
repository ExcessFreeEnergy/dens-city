"""
Comprehensive Cross-Material Benchmark Runner for dens-city.
Executes Stages 1-5 across all 10 material specification YAMLs in tests/data/,
collects end-to-end performance and physical metrics, isolates output directories,
and aggregates statistical variance, errors, strengths, and bottlenecks.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def run_benchmark():
    parser = argparse.ArgumentParser(description="Run full Stage 1-5 funnel across all material groups")
    parser.add_argument(
        "--train-steps",
        "--total-timesteps",
        dest="train_steps",
        type=int,
        default=5000000,
        help="RL training steps per group (default: 5,000,000 with dynamic early stopping)",
    )
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel C-FFI environments (default: 16)")
    parser.add_argument("--horizon", type=int, default=16, help="Rollout horizon per environment (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate (default: 3e-4)")
    parser.add_argument("--hidden-size", type=int, default=256, help="Policy latent dimension (default: 256)")
    parser.add_argument(
        "--early-stopping-lookback",
        type=int,
        default=500000,
        help="Lookback window for EMA reward flatline detection (default: 500,000)",
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.01,
        help="EMA delta threshold for early stopping (default: 0.01)",
    )
    parser.add_argument("--num-candidates", type=int, default=64, help="Candidates to sample per group (default: 64)")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU batch size for Stage 3 (default: 64)")
    parser.add_argument("--egnn-batch-size", type=int, default=32, help="EGNN batch size for Stage 4 (default: 32)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K candidates to export (default: 10)")
    parser.add_argument("--out-dir", type=str, default="runs/full_system_benchmark", help="Base output directory")
    parser.add_argument("--max-sa-score", type=float, default=6.0, help="Stage 5 SA Score ceiling (default: 6.0)")
    args = parser.parse_args()

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    spec_files = sorted(Path("tests/data").glob("*.yaml"))
    print(f"Found {len(spec_files)} material specification YAMLs in tests/data/")

    all_summaries: List[Dict[str, Any]] = []

    for i, spec_path in enumerate(spec_files):
        group_name = spec_path.stem
        spec_out_dir = base_out_dir / group_name
        spec_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"[{i + 1}/{len(spec_files)}] RUNNING 5-STAGE FUNNEL: {group_name}")
        print(f"Spec file: {spec_path}")
        print(f"Output directory: {spec_out_dir}")
        print("=" * 100)

        cmd = [
            sys.executable,
            "scripts/run_generative_funnel.py",
            "--spec",
            str(spec_path),
            "--train-steps",
            str(args.train_steps),
            "--num-envs",
            str(args.num_envs),
            "--horizon",
            str(args.horizon),
            "--learning-rate",
            str(args.learning_rate),
            "--hidden-size",
            str(args.hidden_size),
            "--early-stopping-lookback",
            str(args.early_stopping_lookback),
            "--early-stopping-delta",
            str(args.early_stopping_delta),
            "--num-candidates",
            str(args.num_candidates),
            "--batch-size",
            str(args.batch_size),
            "--egnn-batch-size",
            str(args.egnn_batch_size),
            "--top-k",
            str(args.top_k),
            "--max-sa-score",
            str(args.max_sa_score),
            "--out-dir",
            str(base_out_dir),
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        t_total = time.perf_counter() - t0

        stdout_log = spec_out_dir / "funnel_execution.log"
        stdout_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            print(f"ERROR: Execution failed for {group_name} with return code {proc.returncode}")
            print(proc.stderr[-1000:] if proc.stderr else proc.stdout[-1000:])
            all_summaries.append(
                {
                    "group": group_name,
                    "status": "FAILED",
                    "error": proc.stderr or proc.stdout,
                    "total_time_s": t_total,
                }
            )
            continue

        csv_path = spec_out_dir / "funnel_summary.csv"
        if csv_path.exists():
            rows: List[Dict[str, Any]] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)

            def get_float_list(key: str) -> List[float]:
                vals = []
                for r in rows:
                    v = r.get(key, "")
                    if v not in ("", None, "None", "nan"):
                        try:
                            vals.append(float(v))
                        except ValueError:
                            pass
                return vals

            scores = get_float_list("funnel_score")
            p_walls = get_float_list("wall_pressure_bar")
            sa_scores = get_float_list("sa_score")
            egnn_energies = get_float_list("egnn_energy")
            egnn_f_rms = get_float_list("egnn_force_rms")
            mws = get_float_list("molecular_weight")
            num_pareto = sum(1 for r in rows if r.get("is_pareto_optimal", "").lower() in ("true", "1", "yes"))

            all_summaries.append(
                {
                    "group": group_name,
                    "status": "SUCCESS",
                    "total_time_s": t_total,
                    "exported_candidates": len(rows),
                    "num_pareto": num_pareto,
                    "mean_funnel_score": float(sum(scores) / len(scores)) if scores else 0.0,
                    "mean_p_wall": float(sum(p_walls) / len(p_walls)) if p_walls else 0.0,
                    "mean_sa_score": float(sum(sa_scores) / len(sa_scores)) if sa_scores else 0.0,
                    "mean_egnn_energy": float(sum(egnn_energies) / len(egnn_energies)) if egnn_energies else 0.0,
                    "mean_egnn_f_rms": float(sum(egnn_f_rms) / len(egnn_f_rms)) if egnn_f_rms else 0.0,
                    "mean_mw": float(sum(mws) / len(mws)) if mws else 0.0,
                    "csv_path": str(csv_path),
                }
            )
            print(
                f"SUCCESS: {group_name} finished in {t_total:.2f}s ({len(rows)} candidates exported, {num_pareto} Pareto optimal)"
            )
        else:
            all_summaries.append(
                {
                    "group": group_name,
                    "status": "NO_CSV",
                    "total_time_s": t_total,
                }
            )

    # Save benchmark summary
    summary_json = base_out_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    # Generate aggregated markdown report
    summary_md = base_out_dir / "benchmark_report.md"
    lines = [
        "# End-to-End Generative Funnel Cross-Material Benchmark Report",
        "",
        f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Material Groups Tested**: {len(spec_files)}",
        f"**RL Steps / Group**: {args.train_steps}",
        f"**Candidates Screened / Group**: {args.num_candidates}",
        f"**GPU Batch Size**: {args.batch_size}",
        "",
        "## Summary Results Across All 10 Material Classes",
        "",
        "| Material Group | Status | Runtime (s) | Exported | Pareto | Mean Score | P_wall (bar) | Mean SA | Mean U_EGNN (K) | Mean F_RMS | Mean MW (amu) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in all_summaries:
        if s.get("status") == "SUCCESS":
            lines.append(
                f"| `{s['group']}` | {s['status']} | {s['total_time_s']:.1f}s | "
                f"{s['exported_candidates']} | {s['num_pareto']} | **{s['mean_funnel_score']:+.2f}** | "
                f"{s['mean_p_wall']:.1f} | {s['mean_sa_score']:.2f} | "
                f"{s['mean_egnn_energy']:.1f} | {s['mean_egnn_f_rms']:.2f} | {s['mean_mw']:.1f} |"
            )
        else:
            lines.append(
                f"| `{s.get('group', 'unknown')}` | **{s.get('status')}** | {s.get('total_time_s', 0):.1f}s | - | - | - | - | - | - | - | - |"
            )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBenchmark complete! Full report saved to: {summary_md}")


if __name__ == "__main__":
    run_benchmark()
