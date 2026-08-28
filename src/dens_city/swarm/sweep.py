"""
PufferLib Constellation-Compatible Multi-Objective Curriculum Sweep Engine.
Runs hyperparameter and curriculum sweeps across the 5 material YAML specifications,
serializing trial metrics directly into the PufferLib Constellation JSON schema.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.trainer import SwarmPuffeRLTrainer, VectorizedSwarmEnv


class CurriculumSweepRunner:
    """
    Manages multi-trial hyperparameter and curriculum sweeps for the Molecular Swarm environment.
    Exports trial logs matching the pufferlib/constellation schema.
    """

    def __init__(
        self,
        spec_yaml_paths: List[str | Path],
        output_dir: str | Path = "runs/sweeps",
        seed: int = 42,
    ):
        self.spec_paths = [Path(p).resolve() for p in spec_yaml_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def sample_hyperparameters(self) -> Dict[str, Any]:
        """Samples a hyperparameter configuration for a sweep trial."""
        return {
            "learning_rate": float(10 ** random.uniform(-4.5, -2.5)),
            "ent_coef": float(10 ** random.uniform(-3.5, -1.0)),
            "gamma": float(random.choice([0.95, 0.98, 0.99, 0.995])),
            "gae_lambda": float(random.choice([0.90, 0.95, 0.98])),
            "clip_coef": float(random.choice([0.1, 0.2, 0.3])),
            "vf_coef": float(random.choice([0.25, 0.5, 1.0])),
            "max_grad_norm": float(random.choice([0.5, 1.0, 5.0])),
            "horizon": int(random.choice([8, 16, 24])),
            "minibatch_size": int(random.choice([32, 64, 128])),
            "hidden_size": int(random.choice([128, 256, 384])),
            "recurrent": bool(random.choice([False, True])),
            "use_curriculum": True,
        }

    def run_trial(
        self,
        trial_id: int,
        spec_path: Path,
        hypers: Dict[str, Any],
        total_timesteps: int = 10000,
        num_envs: int = 8,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Executes a single sweep trial and returns trial metrics formatted for Constellation."""
        trial_start = time.time()
        spec_data = SwarmSpecLoader.load_yaml(spec_path)
        target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

        vec_env = VectorizedSwarmEnv(
            num_envs=num_envs,
            spec_yaml_path=spec_path,
            target_spec=target_spec,
            seed=self.seed + trial_id * 100,
        )

        policy = MolecularSwarmPolicy(
            obs_size=88,
            hidden_size=hypers["hidden_size"],
            recurrent=hypers["recurrent"],
        )

        trainer = SwarmPuffeRLTrainer(
            vec_env=vec_env,
            policy=policy,
            final_target_spec=target_spec,
            total_timesteps=total_timesteps,
            horizon=hypers["horizon"],
            learning_rate=hypers["learning_rate"],
            gamma=hypers["gamma"],
            gae_lambda=hypers["gae_lambda"],
            clip_coef=hypers["clip_coef"],
            vf_coef=hypers["vf_coef"],
            ent_coef=hypers["ent_coef"],
            max_grad_norm=hypers["max_grad_norm"],
            minibatch_size=hypers["minibatch_size"],
            use_curriculum=hypers["use_curriculum"],
            device=device,
        )

        epochs = max(1, total_timesteps // (num_envs * hypers["horizon"]))

        # Time-series metric streams for Constellation
        metric_streams: Dict[str, List[float]] = {
            "SPS": [],
            "agent_steps": [],
            "uptime": [],
            "env/score": [],
            "env/perf": [],
            "env/valid_rate": [],
            "env/p_wall": [],
            "env/omega_solv": [],
            "loss/policy": [],
            "loss/value": [],
            "loss/entropy": [],
            "loss/approx_kl": [],
            "loss/clipfrac": [],
            "loss/explained_variance": [],
        }

        try:
            for epoch in range(1, epochs + 1):
                metrics = trainer.train_epoch()
                metric_streams["SPS"].append(float(metrics["SPS"]))
                metric_streams["agent_steps"].append(int(metrics["global_step"]))
                metric_streams["uptime"].append(float(time.time() - trial_start))
                metric_streams["env/score"].append(float(metrics["env/score"]))
                metric_streams["env/perf"].append(float(1.0 if metrics["env/score"] > 0 else 0.0))
                metric_streams["env/valid_rate"].append(float(metrics["env/valid_rate"]))
                metric_streams["env/p_wall"].append(float(metrics["env/p_wall"]))
                metric_streams["env/omega_solv"].append(float(metrics["env/omega_solv"]))
                metric_streams["loss/policy"].append(float(metrics["loss/policy"]))
                metric_streams["loss/value"].append(float(metrics["loss/value"]))
                metric_streams["loss/entropy"].append(float(metrics["loss/entropy"]))
                metric_streams["loss/approx_kl"].append(float(metrics.get("loss/approx_kl", 0.0)))
                metric_streams["loss/clipfrac"].append(float(metrics.get("loss/clipfrac", 0.0)))
                metric_streams["loss/explained_variance"].append(float(metrics.get("loss/explained_variance", 0.0)))
                if metrics.get("early_stopped", 0.0) > 0.5:
                    print(f"[EARLY STOPPING] Trial {trial_id} halted: {trainer.early_stop_reason}")
                    break
        finally:
            vec_env.close()

        # Build standard PufferLib Constellation experiment dictionary
        experiment_record = {
            "trial_id": trial_id,
            "spec_name": spec_path.stem,
            "train": {
                "learning_rate": hypers["learning_rate"],
                "ent_coef": hypers["ent_coef"],
                "gamma": hypers["gamma"],
                "gae_lambda": hypers["gae_lambda"],
                "vtrace_rho_clip": 1.0,
                "vtrace_c_clip": 1.0,
                "clip_coef": hypers["clip_coef"],
                "vf_clip_coef": 0.2,
                "vf_coef": hypers["vf_coef"],
                "max_grad_norm": hypers["max_grad_norm"],
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-5,
                "prio_alpha": 0.0,
                "prio_beta0": 0.0,
                "horizon": hypers["horizon"],
                "replay_ratio": 0.0,
                "minibatch_size": hypers["minibatch_size"],
                "total_timesteps": total_timesteps,
            },
            "policy": {
                "hidden_size": hypers["hidden_size"],
                "num_layers": 2,
                "recurrent": 1 if hypers["recurrent"] else 0,
            },
            "vec": {
                "total_agents": num_envs,
            },
            "args": {
                "env_name": "cdft_swarm",
                "train": {
                    "learning_rate": hypers["learning_rate"],
                    "ent_coef": hypers["ent_coef"],
                    "gamma": hypers["gamma"],
                    "gae_lambda": hypers["gae_lambda"],
                    "vtrace_rho_clip": 1.0,
                    "vtrace_c_clip": 1.0,
                    "clip_coef": hypers["clip_coef"],
                    "vf_clip_coef": 0.2,
                    "vf_coef": hypers["vf_coef"],
                    "max_grad_norm": hypers["max_grad_norm"],
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-5,
                    "prio_alpha": 0.0,
                    "prio_beta0": 0.0,
                    "horizon": hypers["horizon"],
                    "replay_ratio": 0.0,
                    "minibatch_size": hypers["minibatch_size"],
                    "total_timesteps": total_timesteps,
                },
                "policy": {
                    "hidden_size": hypers["hidden_size"],
                    "num_layers": 2,
                    "recurrent": 1 if hypers["recurrent"] else 0,
                },
                "vec": {
                    "total_agents": num_envs,
                },
            },
            "sweep": {
                "train": {
                    "learning_rate": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform"},
                    "ent_coef": {"min": 1e-4, "max": 1e-1, "distribution": "log_uniform"},
                    "gamma": {"min": 0.90, "max": 0.999, "distribution": "uniform"},
                    "gae_lambda": {"min": 0.90, "max": 0.99, "distribution": "uniform"},
                    "vtrace_rho_clip": {"min": 0.5, "max": 2.0, "distribution": "uniform"},
                    "vtrace_c_clip": {"min": 0.5, "max": 2.0, "distribution": "uniform"},
                    "clip_coef": {"min": 0.1, "max": 0.3, "distribution": "uniform"},
                    "vf_clip_coef": {"min": 0.1, "max": 0.5, "distribution": "uniform"},
                    "vf_coef": {"min": 0.25, "max": 1.0, "distribution": "uniform"},
                    "max_grad_norm": {"min": 0.5, "max": 5.0, "distribution": "uniform"},
                    "beta1": {"min": 0.8, "max": 0.99, "distribution": "uniform"},
                    "beta2": {"min": 0.9, "max": 0.9999, "distribution": "uniform"},
                    "eps": {"min": 1e-8, "max": 1e-4, "distribution": "log_uniform"},
                    "prio_alpha": {"min": 0.0, "max": 1.0, "distribution": "uniform"},
                    "prio_beta0": {"min": 0.0, "max": 1.0, "distribution": "uniform"},
                    "horizon": {"min": 8, "max": 24, "distribution": "uniform"},
                    "replay_ratio": {"min": 0.0, "max": 1.0, "distribution": "uniform"},
                    "minibatch_size": {"min": 32, "max": 128, "distribution": "uniform"},
                    "total_timesteps": {"min": 1000, "max": 100000, "distribution": "uniform"},
                },
                "policy": {
                    "hidden_size": {"min": 128, "max": 384, "distribution": "uniform"},
                    "num_layers": {"min": 1, "max": 4, "distribution": "uniform"},
                    "recurrent": {"min": 0, "max": 1, "distribution": "uniform"},
                },
                "vec": {
                    "total_agents": {"min": 2, "max": 32, "distribution": "uniform"},
                },
            },
            "metrics": metric_streams,
            "best_reward": trainer.best_reward,
            "best_mol2": trainer.best_mol2_str,
        }

        # Save trial log JSON
        trial_file = self.output_dir / f"trial_{trial_id:04d}_{spec_path.stem}.json"
        with open(trial_file, "w", encoding="utf-8") as f:
            json.dump(experiment_record, f, indent=2)

        return experiment_record

    def run_sweep(
        self,
        num_trials_per_spec: int = 3,
        steps_per_trial: int = 5000,
        num_envs: int = 8,
        device: str = "cpu",
    ) -> List[Dict[str, Any]]:
        """Executes full curriculum sweep across all target specifications."""
        all_trials = []
        trial_idx = 0

        for spec_path in self.spec_paths:
            for _ in range(num_trials_per_spec):
                trial_idx += 1
                hypers = self.sample_hyperparameters()
                print(
                    f"--- Launching Sweep Trial {trial_idx} on {spec_path.stem} (LR={hypers['learning_rate']:.2e}, H={hypers['hidden_size']}) ---"
                )
                record = self.run_trial(
                    trial_id=trial_idx,
                    spec_path=spec_path,
                    hypers=hypers,
                    total_timesteps=steps_per_trial,
                    num_envs=num_envs,
                    device=device,
                )
                all_trials.append(record)
                print(f"Trial {trial_idx} Finished | Best Score: {record['best_reward']:+.3f}")

        # Write merged index file
        summary_file = self.output_dir / "sweep_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_trials": len(all_trials),
                    "specs": [p.stem for p in self.spec_paths],
                    "trials": [
                        {
                            "id": t["trial_id"],
                            "spec": t["spec_name"],
                            "score": t["best_reward"],
                            "hypers": t["args"],
                        }
                        for t in all_trials
                    ],
                },
                f,
                indent=2,
            )

        print(f"=== Sweep Complete! Saved {len(all_trials)} trial records to {self.output_dir} ===")
        return all_trials
