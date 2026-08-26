#!/usr/bin/env python3
"""
CLI entrypoint to launch PufferLib Constellation-compatible Curriculum Sweeps across material specifications.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from dens_city.swarm.sweep import CurriculumSweepRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PufferLib Constellation-compatible Curriculum Sweeps on Molecular Swarms."
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=[
            "tests/data/conjugated_oled_semiconductors.yaml",
            "tests/data/fluorinated_battery_electrolytes.yaml",
            "tests/data/sterically_hindered_drug_inhibitors.yaml",
            "tests/data/ultra_lightweight_aliphatic_sponges.yaml",
            "tests/data/sacrificial_h_bond_toughness_resins.yaml",
        ],
        help="List of specification YAML files to sweep.",
    )
    parser.add_argument(
        "--num-trials-per-spec", type=int, default=2, help="Number of random hyperparameter trials per material spec."
    )
    parser.add_argument(
        "--steps-per-trial", type=int, default=5000, help="Timesteps to train each trial before evaluation."
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel C environment workers per trial.")
    parser.add_argument(
        "--output-dir", type=str, default="runs/constellation_sweeps", help="Directory to save trial JSON records."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    spec_paths = [Path(p).resolve() for p in args.specs]
    for p in spec_paths:
        if not p.exists():
            print(f"Error: Specification file not found: {p}", file=sys.stderr)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Initializing Curriculum Sweep Runner ({len(spec_paths)} specs, device={device}) ===")
    runner = CurriculumSweepRunner(
        spec_yaml_paths=spec_paths,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    runner.run_sweep(
        num_trials_per_spec=args.num_trials_per_spec,
        steps_per_trial=args.steps_per_trial,
        num_envs=args.num_envs,
        device=device,
    )


if __name__ == "__main__":
    main()
