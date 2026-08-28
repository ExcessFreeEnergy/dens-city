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
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5000000,
        help="Total environment interaction steps (scaled to 5M-10M steps).",
    )
    parser.add_argument("--horizon", type=int, default=16, help="Rollout horizon per environment.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--hidden-size", type=int, default=256, help="Policy latent dimension.")
    parser.add_argument("--recurrent", action="store_true", help="Enable MinGRU recurrent backbone.")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable 3-stage curriculum scheduler.")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable Dynamic EMA early stopping.")
    parser.add_argument(
        "--early-stopping-lookback",
        type=int,
        default=500000,
        help="Step lookback window for EMA reward flatline detection (default: 500,000).",
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.01,
        help="EMA reward change threshold for early stopping (default: 0.01).",
    )
    parser.add_argument("--no-sa-penalty", action="store_true", help="Disable in-the-loop batch SA score penalty.")
    parser.add_argument(
        "--sa-threshold",
        type=float,
        default=None,
        help="SA Score hinge threshold above which penalty is applied (default: dynamically derived from specification).",
    )
    parser.add_argument(
        "--sa-penalty-slope",
        type=float,
        default=None,
        help="Slope multiplier for SA score excess penalty (default: dynamically derived from specification).",
    )
    parser.add_argument(
        "--no-dynamic-entropy",
        action="store_true",
        help="Disable molecular-weight-scaling dynamic entropy coefficient.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="runs/checkpoints",
        help="Directory to save trained_policy.pt and periodic checkpoints.",
    )
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
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        early_stopping=not args.no_early_stopping,
        early_stopping_lookback=args.early_stopping_lookback,
        early_stopping_delta=args.early_stopping_delta,
        dynamic_entropy_scaling=not args.no_dynamic_entropy,
        sa_penalty=not args.no_sa_penalty,
        sa_threshold=args.sa_threshold,
        sa_penalty_slope=args.sa_penalty_slope,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    print(f"Starting Swarm PPO Training for {args.total_timesteps:,} timesteps (with C-level action masks)...")
    epochs = max(1, args.total_timesteps // (args.num_envs * args.horizon))

    try:
        for epoch in range(1, epochs + 1):
            metrics = trainer.train_epoch()

            if epoch % 5 == 0 or epoch == epochs or metrics.get("early_stopped", 0.0) > 0.5:
                print(
                    f"Epoch {metrics['epoch']:4d}/{epochs:4d} | "
                    f"Steps: {metrics['global_step']:8d} | "
                    f"SPS: {metrics['SPS']:5.0f} | "
                    f"Stage: {metrics['curriculum_stage']} | "
                    f"EMA: {metrics['env/reward_ema']:+5.2f} | "
                    f"Score: {metrics['env/score']:+5.2f} | "
                    f"Valid: {metrics['env/valid_rate'] * 100:4.1f}% | "
                    f"H(pi): {metrics['loss/entropy']:4.2f} | "
                    f"KL: {metrics['loss/approx_kl']:.4f} | "
                    f"SA: {metrics['env/sa_score']:4.2f} | "
                    f"P_wall: {metrics['env/p_wall']:4.1f} bar | "
                    f"Best: {metrics['best_reward']:+5.2f}"
                )

            if metrics.get("early_stopped", 0.0) > 0.5:
                print(f"\n[EARLY STOPPING HALT] {trainer.early_stop_reason}")
                break

        final_policy_path = checkpoint_dir / "trained_policy.pt"
        trainer.save_checkpoint(final_policy_path)
        print(f"\nTraining Complete! Dumped policy weights to: {final_policy_path}")
        print(f"Best Pareto Reward: {trainer.best_reward:.3f}")

        if trainer.best_mol2_str:
            out_mol2 = export_dir / f"{spec_path.stem}_best.mol2"
            out_mol2.write_text(trainer.best_mol2_str)
            print(f"Exported Pareto-optimal candidate to: {out_mol2}")

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
