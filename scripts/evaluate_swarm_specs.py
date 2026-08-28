#!/usr/bin/env python3
"""
CLI entrypoint to run the Full RL Stage & Chemistry Analysis on all YAML specifications in tests/data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from dens_city.swarm.evaluator import run_spec_rl_stage


def main():
    parser = argparse.ArgumentParser(
        description="Run Full RL Stage & Chemistry Analysis on All Material Specifications."
    )
    parser.add_argument(
        "--specs-dir",
        type=str,
        default="tests/data",
        help="Directory containing material specification YAML files.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Training timesteps per specification.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Number of candidate molecules to generate and evaluate per specification.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel C environment workers.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/rl_stage_evaluation",
        help="Destination directory for evaluation summaries and candidate .mol2 artifacts.",
    )
    args = parser.parse_args()

    specs_dir = Path(args.specs_dir).resolve()
    spec_files = sorted(specs_dir.glob("*.yaml"))
    if not spec_files:
        print(f"Error: No specification YAML files found in {specs_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 88)
    print("  dens-city: Full RL Stage Evaluation & Chemistry Diagnostic Pipeline")
    print("=" * 88)
    print(f"  Target Specs Directory : {specs_dir} ({len(spec_files)} specifications)")
    print(f"  Training Budget / Spec : {args.timesteps:,} steps")
    print(f"  Evaluation Batch Size  : {args.num_candidates} molecules / spec")
    print(f"  Parallel C Workers     : {args.num_envs} envs (device={device})")
    print(f"  Output Artifacts Dir   : {out_dir}")
    print("=" * 88)

    all_summaries: List[Dict[str, Any]] = []
    t_global_start = time.time()

    for spec_path in spec_files:
        summary = run_spec_rl_stage(
            spec_path=spec_path,
            timesteps=args.timesteps,
            num_candidates=args.num_candidates,
            num_envs=args.num_envs,
            device=device,
            output_dir=out_dir,
        )
        all_summaries.append(summary)

    t_global_total = time.time() - t_global_start

    # Write Master Evaluation Summary JSON
    master_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": t_global_total,
        "num_specs_evaluated": len(all_summaries),
        "spec_summaries": all_summaries,
    }
    master_json_path = out_dir / "master_evaluation_summary.json"
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # Print Final Comparative Synthesis & Diversity Table
    print("\n" + "=" * 115)
    print("  FINAL MULTI-SPECIFICATION RL STAGE EVALUATION REPORT")
    print("=" * 115)
    print(
        f"{'Specification':<35} | {'Valid %':<8} | {'SA Score':<12} | {'Tanimoto (T)':<13} | {'Diversity (1-T)':<15} | {'Unique SMILES %':<15}"
    )
    print("-" * 115)

    for s in all_summaries:
        sa_str = f"{s['sa_score_mean']:.2f} ± {s['sa_score_std']:.2f}"
        print(
            f"{s['spec_name']:<35} | {s['validity_rate_pct']:6.1f}% | {sa_str:<12} | {s['mean_internal_tanimoto_similarity']:11.3f} | {s['internal_diversity']:13.3f} | {s['unique_smiles_ratio_pct']:13.1f}%"
        )

    print("-" * 115)
    avg_valid = float(np.mean([s["validity_rate_pct"] for s in all_summaries]))
    avg_sa = float(np.mean([s["sa_score_mean"] for s in all_summaries]))
    avg_div = float(np.mean([s["internal_diversity"] for s in all_summaries]))
    avg_uniq = float(np.mean([s["unique_smiles_ratio_pct"] for s in all_summaries]))

    print(
        f"{'OVERALL AVERAGE':<35} | {avg_valid:6.1f}% | {avg_sa:6.2f}       | {'--':<13} | {avg_div:13.3f} | {avg_uniq:13.1f}%"
    )
    print("=" * 115)
    print(f"\n[+] Master Summary saved to: {master_json_path}")
    print(f"[+] Total Pipeline Execution Time: {t_global_total:.1f} s")


if __name__ == "__main__":
    main()
