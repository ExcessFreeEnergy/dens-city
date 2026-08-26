#!/usr/bin/env python3
"""
CLI entrypoint to train the Molecular Swarm PPO policy on arbitrary material YAML specifications.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.trainer import SwarmPuffeRLTrainer, VectorizedSwarmEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PufferLib 4.0 Molecular Swarm RL Policy on Material YAML Specifications."
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="tests/data/conjugated_oled_semiconductors.yaml",
        help="Path to material specification YAML file.",
    )
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel C environment workers.")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total environment interaction steps.")
    parser.add_argument("--horizon", type=int, default=16, help="Rollout horizon per environment.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--hidden-size", type=int, default=256, help="Policy latent dimension.")
    parser.add_argument("--recurrent", action="store_true", help="Enable MinGRU recurrent backbone.")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable 3-stage curriculum scheduler.")
    parser.add_argument("--export-dir", type=str, default="runs/candidates", help="Directory to save mol2 candidates.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        print(f"Error: Specification file not found at {spec_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Loading Swarm Specification: {spec_path.name} ===")
    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)
    print(f"Target Material Objectives: {target_spec}")

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Vectorized C Environment Pool ({args.num_envs} envs, device={device})...")

    vec_env = VectorizedSwarmEnv(
        num_envs=args.num_envs,
        spec_yaml_path=spec_path,
        target_spec=target_spec,
        seed=args.seed,
    )

    policy = MolecularSwarmPolicy(
        obs_size=88,
        hidden_size=args.hidden_size,
        recurrent=args.recurrent,
    )

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_spec,
        total_timesteps=args.total_timesteps,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        use_curriculum=not args.no_curriculum,
        device=device,
    )

    print(f"Starting Swarm PPO Training for {args.total_timesteps} timesteps...")
    epochs = max(1, args.total_timesteps // (args.num_envs * args.horizon))

    try:
        for epoch in range(1, epochs + 1):
            metrics = trainer.train_epoch()

            if epoch % 5 == 0 or epoch == epochs:
                print(
                    f"Epoch {metrics['epoch']:3d}/{epochs:3d} | "
                    f"Steps: {metrics['global_step']:6d} | "
                    f"SPS: {metrics['SPS']:6.0f} | "
                    f"Stage: {metrics['curriculum_stage']} | "
                    f"Score: {metrics['env/score']:+6.2f} | "
                    f"Valid: {metrics['env/valid_rate'] * 100:4.1f}% | "
                    f"P_wall: {metrics['env/p_wall']:5.1f} bar | "
                    f"Best: {metrics['best_reward']:+6.2f}"
                )

        print(f"\nTraining Complete! Best Reward: {trainer.best_reward:.3f}")
        if trainer.best_mol2_str:
            out_mol2 = export_dir / f"{spec_path.stem}_best.mol2"
            out_mol2.write_text(trainer.best_mol2_str)
            print(f"Exported Pareto-optimal candidate to: {out_mol2}")

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
