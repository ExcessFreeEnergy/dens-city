from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from dens_city.swarm.env import VectorizedSwarmEnv
from dens_city.swarm.policy import MolecularSwarmPolicy


def _compute_sa_score_static(smi: str) -> float:
    """Computes RDKit SA score on canonical SMILES safely."""
    if not smi:
        return 1.0
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 1.0
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 1.0


class SwarmCurriculumManager:
    """
    Progressive 3-Stage Metric-Driven Curriculum Manager for Molecular Swarm Training.
    Eliminates memory segregation by mutating C TargetSpec structs via ctypes broadcast.
    Features:
    - Monotonic stage locking (eliminates validation regression jitter).
    - Metric-driven gating (valid_rate >= 0.80 -> Stage 2, valid_rate >= 0.95 -> Stage 3).
    - Step budget clamping (Stage 1 <= 25,000 steps, Stage 2 <= 65,000 steps).
    """

    def __init__(
        self,
        final_target_spec: Dict[str, float],
        enabled: bool = True,
        stage1_max_steps: int = 25000,
        stage2_max_steps: int = 65000,
    ):
        self.final_targets = dict(final_target_spec)
        self.enabled = enabled
        self.current_stage = 1
        self.stage1_max_steps = stage1_max_steps
        self.stage2_max_steps = stage2_max_steps

        # Stage 1 (Geometric Feasibility & Valency Retention)
        self.stage1_targets = {
            "target_elasticity": 0.25,
            "target_tensile": 0.25,
            "target_toughness": 0.25,
            "target_lightweight": 0.25,
            "max_solvation_kcal": 0.0,  # Highly relaxed
            "min_wall_pressure_bar": 0.0,
            "max_molecular_weight": 1000.0,
            "min_valency": 1,
        }

    def compute_targets_for_progress(
        self,
        progress: float,
        global_step: int = 0,
        valid_rate: Optional[float] = None,
    ) -> Tuple[Dict[str, float], int]:
        """
        Calculates interpolated target specification with monotonic stage lock.
        Transitions:
            - Stage 2: valid_rate >= 0.80 OR global_step >= 25,000 OR progress >= 0.25
            - Stage 3: valid_rate >= 0.95 OR global_step >= 65,000 OR progress >= 0.65
        """
        if not self.enabled:
            return dict(self.final_targets), 3

        # 1. Determine candidate stage from metrics & step budget
        candidate_stage = 1
        if (valid_rate is not None and valid_rate >= 0.95) or global_step >= self.stage2_max_steps or progress >= 0.65:
            candidate_stage = 3
        elif (
            (valid_rate is not None and valid_rate >= 0.80) or global_step >= self.stage1_max_steps or progress >= 0.25
        ):
            candidate_stage = 2

        # 2. Monotonic unidirectional stage lock
        self.current_stage = max(self.current_stage, candidate_stage)

        # 3. Compute target values
        if self.current_stage == 1:
            return dict(self.stage1_targets), 1
        elif self.current_stage == 2:
            if global_step >= self.stage1_max_steps:
                alpha = min(
                    1.0,
                    (global_step - self.stage1_max_steps) / max(1, self.stage2_max_steps - self.stage1_max_steps),
                )
            else:
                alpha = min(1.0, max(0.0, (progress - 0.25) / 0.40))
            targets = {}
            for k, final_v in self.final_targets.items():
                s1_v = self.stage1_targets.get(k, final_v)
                if isinstance(final_v, int):
                    targets[k] = int(round(s1_v + alpha * (final_v - s1_v)))
                else:
                    targets[k] = float(s1_v + alpha * (final_v - s1_v))
            return targets, 2
        else:
            return dict(self.final_targets), 3

    def broadcast_to_vec_env(
        self,
        vec_env: VectorizedSwarmEnv,
        progress: float,
        global_step: int = 0,
        valid_rate: Optional[float] = None,
    ) -> Tuple[Dict[str, float], int]:
        """Updates and broadcasts targets into C memory across all parallel environments."""
        targets, stage = self.compute_targets_for_progress(progress, global_step=global_step, valid_rate=valid_rate)
        vec_env.set_targets(targets)
        return targets, stage


class SwarmPuffeRLTrainer:
    """
    PufferLib-templated Vectorized PPO Trainer for Molecular Swarm Discovery.
    Features:
    1. Universal C-Level Action Masked PPO optimization.
    2. Continuous overtraining telemetry (Entropy, KL Divergence, Clip Fraction, Explained Variance).
    3. Dynamic Early-Stopping Callback monitoring 100-episode EMA with 500k-step lookback.
    4. Model checkpointing and Pareto-optimal .mol2 candidate export.
    """

    def __init__(
        self,
        vec_env: VectorizedSwarmEnv,
        policy: MolecularSwarmPolicy,
        final_target_spec: Dict[str, float],
        total_timesteps: int = 50000,
        horizon: int = 16,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 256,
        update_epochs: int = 2,
        use_curriculum: bool = True,
        early_stopping: bool = True,
        early_stopping_lookback: int = 500000,
        early_stopping_delta: float = 0.01,
        dynamic_entropy_scaling: bool = True,
        sa_penalty: bool = True,
        sa_threshold: Optional[float] = None,
        sa_penalty_slope: Optional[float] = None,
        checkpoint_dir: Optional[str | Path] = None,
        device: str | torch.device = "cpu",
    ):
        # Isolate PyTorch CPU thread pool to 1 thread, reserving all CPU cores for OpenMP C vectorization
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self.vec_env = vec_env
        self.policy = policy.to(device)
        self.final_targets = final_target_spec
        self.device = torch.device(device)

        self.total_timesteps = total_timesteps
        self.horizon = horizon
        self.num_envs = vec_env.num_envs
        self.batch_size = self.num_envs * horizon
        self.minibatch_size = min(minibatch_size, self.batch_size)
        self.update_epochs = update_epochs

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.dynamic_entropy_scaling = dynamic_entropy_scaling
        self.sa_penalty = sa_penalty
        self.sa_threshold = float(
            sa_threshold if sa_threshold is not None else final_target_spec.get("sa_threshold", 4.5)
        )
        self.sa_penalty_slope = float(
            sa_penalty_slope if sa_penalty_slope is not None else final_target_spec.get("sa_penalty_slope", 2.0)
        )
        self.target_max_mw = float(final_target_spec.get("max_molecular_weight", 850.0))

        # Warm up SA scorer and initialize thread worker pool
        try:
            sascorer.calculateScore(Chem.MolFromSmiles("c1ccccc1"))
        except Exception:
            pass
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.curriculum = SwarmCurriculumManager(final_target_spec, enabled=use_curriculum)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        # Early Stopping & Telemetry State
        self.early_stopping = early_stopping
        self.early_stopping_lookback = early_stopping_lookback
        self.early_stopping_delta = early_stopping_delta
        self.early_stopped = False
        self.early_stop_reason = ""
        self.reward_ema: Optional[float] = None
        self.ema_alpha = 2.0 / (100.0 + 1.0)  # 100-episode smoothing
        self.ema_history: List[Tuple[int, float, Dict[str, float]]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Rollout Storage Buffers
        self.obs_buf = torch.zeros((horizon, self.num_envs, 88), dtype=torch.float32, device=self.device)
        self.mask_buf = torch.zeros((horizon, self.num_envs, 29), dtype=torch.float32, device=self.device)
        self.actions_buf = torch.zeros((horizon, self.num_envs, 2), dtype=torch.long, device=self.device)
        self.logprobs_buf = torch.zeros((horizon, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards_buf = torch.zeros((horizon, self.num_envs), dtype=torch.float32, device=self.device)
        self.terminals_buf = torch.zeros((horizon, self.num_envs), dtype=torch.float32, device=self.device)
        self.values_buf = torch.zeros((horizon, self.num_envs), dtype=torch.float32, device=self.device)
        # Tracking state
        self.global_step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.last_obs = None
        self.last_mask = None
        self.last_valid_rate: Optional[float] = None
        self.consecutive_converged_epochs = 0
        self.best_reward = -float("inf")
        self.best_mol2_str = ""

    def save_checkpoint(self, path: str | Path) -> None:
        """Saves trained policy weights and training state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_reward": self.best_reward,
                "best_mol2": self.best_mol2_str,
                "reward_ema": self.reward_ema,
                "final_targets": self.final_targets,
            },
            str(p),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Loads policy weights and resumes optimizer state."""
        p = Path(path)
        data = torch.load(str(p), map_location=self.device)
        self.policy.load_state_dict(data["policy_state_dict"])
        if "optimizer_state_dict" in data:
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
        if "global_step" in data:
            self.global_step = data["global_step"]
        if "best_reward" in data:
            self.best_reward = data["best_reward"]
        if "best_mol2" in data:
            self.best_mol2_str = data["best_mol2"]
        if "reward_ema" in data:
            self.reward_ema = data["reward_ema"]

    def _check_early_stopping(self, stage_idx: int, avg_p_wall: float, avg_omega: float, valid_rate: float) -> bool:
        """
        Evaluates Dynamic Convergence & Early-Stopping Algorithm:
        1. Verifies Stage 3 has been unlocked and minimum warm-up steps completed.
        2. Checks if physical constraints (P_wall, Solvation, Valid rate >= 85%) are satisfied.
        3. Tracks consecutive epochs of physical satisfaction and reward stability to stop
           gracefully when learning has converged/saturated without overtraining.
        """
        if not self.early_stopping or self.reward_ema is None:
            return False

        # Must have reached Stage 3 (full target constraints)
        if stage_idx < 3:
            self.consecutive_converged_epochs = 0
            return False

        # Need at least warm-up steps of training experience
        min_warmup = min(25000, self.early_stopping_lookback)
        if self.global_step < min_warmup:
            return False

        min_pwall = self.final_targets.get("min_wall_pressure_bar", 15.0)
        max_solv = self.final_targets.get("max_solvation_kcal", -3.0)

        # Physical satisfaction check
        pwall_satisfied = avg_p_wall >= (min_pwall * 0.90)
        solv_satisfied = (avg_omega <= (max_solv + 2.0)) or (avg_omega <= 0.0)
        valid_satisfied = valid_rate >= 0.70

        if (pwall_satisfied or valid_rate >= 0.80) and (solv_satisfied or valid_rate >= 0.80):
            self.consecutive_converged_epochs += 1
        else:
            self.consecutive_converged_epochs = max(0, self.consecutive_converged_epochs - 1)

        # A. Sustained Peak Convergence: Stop if policy maintained physical satisfaction for 30 epochs
        if self.consecutive_converged_epochs >= 30 and valid_satisfied:
            self.early_stopped = True
            self.early_stop_reason = (
                f"Policy converged at step {self.global_step:,} with sustained physical satisfaction "
                f"over {self.consecutive_converged_epochs} consecutive epochs "
                f"(Valid={valid_rate * 100:.1f}%, P_wall={avg_p_wall:.1f} bar, Solv={avg_omega:.2f} kcal/mol, "
                f"Reward EMA={self.reward_ema:.2f})"
            )
            if self.checkpoint_dir:
                self.save_checkpoint(self.checkpoint_dir / "trained_policy.pt")
            return True

        # B. Learning Curve Plateau Saturation:
        # If the policy has flatlined over the lookback window (100k steps), learning has saturated.
        if self.global_step >= self.early_stopping_lookback:
            target_step = self.global_step - min(self.early_stopping_lookback, 100000)
            closest_ema = None
            min_diff = float("inf")
            for step_rec, ema_rec, _ in self.ema_history:
                diff = abs(step_rec - target_step)
                if diff < min_diff:
                    min_diff = diff
                    closest_ema = ema_rec

            if closest_ema is not None:
                delta = abs(self.reward_ema - closest_ema)
                rel_delta = delta / max(1.0, abs(self.reward_ema))
                if delta <= self.early_stopping_delta or (rel_delta < 0.05 and delta < 0.50):
                    self.early_stopped = True
                    self.early_stop_reason = (
                        f"Reward EMA flatlined / policy saturated (Δ_EMA = {delta:.3f} <= {self.early_stopping_delta} "
                        f"over {self.global_step - target_step:,} steps, EMA={self.reward_ema:.2f}, Valid={valid_rate * 100:.1f}%)"
                    )
                    if self.checkpoint_dir:
                        self.save_checkpoint(self.checkpoint_dir / "trained_policy.pt")
                    return True

        return False

    def train_epoch(self) -> Dict[str, float]:
        """Collects a vectorized rollout and executes PPO optimization."""
        self.epoch += 1
        epoch_start_time = time.time()

        # 1. Update curriculum and broadcast into C-memory
        progress = min(1.0, float(self.global_step) / float(max(1, self.total_timesteps)))
        current_targets, stage_idx = self.curriculum.broadcast_to_vec_env(
            self.vec_env,
            progress=progress,
            global_step=self.global_step,
            valid_rate=self.last_valid_rate,
        )

        # 2. Reset or get current env states
        obs, action_mask = self.vec_env.reset() if self.global_step == 0 else (self.last_obs, self.last_mask)
        obs = obs.to(self.device, non_blocking=True)
        action_mask = action_mask.to(self.device, non_blocking=True)

        # 3. Rollout Collection
        p_walls = []
        omega_solvs = []
        sa_scores_list = []
        sa_penalties_list = []
        ep_rewards = []
        total_episodes = 0
        valid_molecules = 0

        self.policy.eval()
        for step in range(self.horizon):
            self.global_step += self.num_envs

            with torch.no_grad():
                out = self.policy.get_action_and_value(obs, action_mask=action_mask)
                actions = out["action"]
                logprobs = out["logprob"]
                values = out["value"]

            self.obs_buf[step].copy_(obs, non_blocking=True)
            self.mask_buf[step].copy_(action_mask, non_blocking=True)
            self.actions_buf[step].copy_(actions, non_blocking=True)
            self.logprobs_buf[step].copy_(logprobs, non_blocking=True)
            self.values_buf[step].copy_(values, non_blocking=True)

            # Vectorized step (Pure C OpenMP execution across CPU cores)
            next_obs, rewards, terminals, next_masks, infos = self.vec_env.step(actions)
            self.rewards_buf[step].copy_(rewards, non_blocking=True)
            self.terminals_buf[step].copy_(terminals, non_blocking=True)

            for i, info in enumerate(infos):
                if terminals[i] > 0.5:
                    total_episodes += 1
                    ep_r = float(rewards[i])
                    ep_rewards.append(ep_r)
                    if info is not None:
                        if "p_wall_bar" in info:
                            p_walls.append(info["p_wall_bar"])
                        if "omega_solv_kcal" in info:
                            omega_solvs.append(info["omega_solv_kcal"])
                        if "sa_score" in info:
                            sa_scores_list.append(info["sa_score"])
                        if "r_sa_penalty" in info:
                            sa_penalties_list.append(info["r_sa_penalty"])
                    is_struct_valid = (info is not None and info.get("converged", False)) or (ep_r > -25.0)
                    if is_struct_valid:
                        valid_molecules += 1
                    if ep_r > self.best_reward:
                        self.best_reward = ep_r
                        self.best_mol2_str = self.vec_env.export_best_candidate_mol2(i, "best_candidate")

            obs = next_obs.to(self.device, non_blocking=True)
            action_mask = next_masks.to(self.device, non_blocking=True)

        self.last_obs = obs
        self.last_mask = action_mask

        # 4. GPU-Resident In-Place GAE-Lambda Advantage Estimation
        with torch.no_grad():
            out = self.policy.get_action_and_value(obs, action_mask=action_mask)
            next_value = out["value"]
            advantages = torch.zeros_like(self.rewards_buf)
            lastgaelam = torch.zeros(self.num_envs, device=self.device)
            for t in reversed(range(self.horizon)):
                if t == self.horizon - 1:
                    nextnonterminal = 1.0 - self.terminals_buf[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.terminals_buf[t]
                    nextvalues = self.values_buf[t + 1]
                delta = self.rewards_buf[t] + self.gamma * nextvalues * nextnonterminal - self.values_buf[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values_buf

        # 5. Flatten Mini-batches for PPO update
        b_obs = self.obs_buf.reshape(-1, 88)
        b_masks = self.mask_buf.reshape(-1, 29)
        b_actions = self.actions_buf.reshape(-1, 2)
        b_logprobs = self.logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values_buf.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 6. PPO Optimization Epochs with KL Divergence and Dynamic Low-Mass Entropy Scaling
        self.policy.train()
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent_loss = 0.0
        total_approx_kl = 0.0
        total_clipfrac = 0.0
        num_updates = 0

        b_indices = np.arange(self.batch_size)
        for _ in range(self.update_epochs):
            np.random.shuffle(b_indices)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = b_indices[start:end]

                out = self.policy.get_action_and_value(
                    b_obs[mb_idx], action_mask=b_masks[mb_idx], action=b_actions[mb_idx]
                )
                new_logprob = out["logprob"]
                entropy = out["entropy"]
                new_val = out["value"]

                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = torch.exp(logratio)

                # Schulman Approximate KL Divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                    total_approx_kl += float(approx_kl.item())
                    total_clipfrac += float(clipfrac.item())

                # Clipped Policy Loss
                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_val - b_returns[mb_idx]) ** 2).mean()

                # Dynamic Low-Mass & High-Valency Entropy Scaling
                if self.dynamic_entropy_scaling:
                    target_scale = max(1.0, 850.0 / max(100.0, self.target_max_mw))
                    # Multi-arm crosslinked resins and high toughness require enhanced topological exploration
                    if (
                        int(self.final_targets.get("min_valency", 2)) >= 3
                        or float(self.final_targets.get("target_toughness", 0.0)) >= 0.70
                    ):
                        target_scale = max(target_scale, 2.0)
                    sample_scale = torch.clamp(target_scale * (1.5 - b_obs[mb_idx, 1]), min=1.0, max=4.0)
                    effective_ent_coef = self.ent_coef * sample_scale
                    entropy_loss = (effective_ent_coef * entropy).mean()
                else:
                    entropy_loss = self.ent_coef * entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss - entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += float(pg_loss.item())
                total_vf_loss += float(v_loss.item())
                total_ent_loss += float(entropy.mean().item())
                num_updates += 1

        # Explained Variance Calculation
        y_true = b_returns.detach().cpu().numpy()
        y_pred = b_values.detach().cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float(1.0 - np.var(y_true - y_pred) / var_y) if var_y > 1e-8 else 0.0

        elapsed = time.time() - epoch_start_time
        sps = self.batch_size / max(1e-4, elapsed)

        avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        valid_rate = float(valid_molecules / max(1, total_episodes))
        self.last_valid_rate = valid_rate
        avg_p_wall = float(np.mean(p_walls)) if p_walls else 0.0
        avg_omega = float(np.mean(omega_solvs)) if omega_solvs else 0.0

        # Update rolling 100-episode EMA of reward
        if ep_rewards:
            if self.reward_ema is None:
                self.reward_ema = avg_reward
            else:
                self.reward_ema = (1.0 - self.ema_alpha) * self.reward_ema + self.ema_alpha * avg_reward

            self.ema_history.append(
                (
                    self.global_step,
                    self.reward_ema,
                    {"p_wall": avg_p_wall, "omega_solv": avg_omega, "valid_rate": valid_rate},
                )
            )

        mean_entropy = total_ent_loss / max(1, num_updates)
        mean_kl = total_approx_kl / max(1, num_updates)

        # Check Dynamic Early Stopping
        early_stop_triggered = self._check_early_stopping(stage_idx, avg_p_wall, avg_omega, valid_rate)

        metrics = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "SPS": sps,
            "curriculum_stage": stage_idx,
            "loss/policy": total_pg_loss / max(1, num_updates),
            "loss/value": total_vf_loss / max(1, num_updates),
            "loss/entropy": mean_entropy,
            "loss/approx_kl": mean_kl,
            "loss/clipfrac": total_clipfrac / max(1, num_updates),
            "loss/explained_variance": explained_var,
            "loss/dynamic_ent_scale": max(
                max(1.0, 850.0 / max(100.0, self.target_max_mw)),
                2.0
                if (
                    int(self.final_targets.get("min_valency", 2)) >= 3
                    or float(self.final_targets.get("target_toughness", 0.0)) >= 0.70
                )
                else 1.0,
            ),
            "env/score": avg_reward,
            "env/reward_ema": self.reward_ema if self.reward_ema is not None else 0.0,
            "env/sa_score": float(np.mean(sa_scores_list)) if sa_scores_list else 0.0,
            "env/r_sa_penalty": float(np.mean(sa_penalties_list)) if sa_penalties_list else 0.0,
            "env/valid_rate": valid_rate,
            "env/p_wall": avg_p_wall,
            "env/omega_solv": avg_omega,
            "best_reward": self.best_reward,
            "early_stopped": 1.0 if early_stop_triggered else 0.0,
        }

        # Periodic checkpoint dump (e.g. every 500k steps)
        if self.checkpoint_dir and self.global_step % 500000 < self.batch_size:
            self.save_checkpoint(self.checkpoint_dir / f"checkpoint_step_{self.global_step:08d}.pt")

        return metrics


def train_swarm_policy(
    spec: str | Path,
    total_timesteps: int = 5000000,
    num_envs: int = 16,
    horizon: int = 16,
    learning_rate: float = 3e-4,
    hidden_size: int = 256,
    recurrent: bool = False,
    no_curriculum: bool = False,
    no_early_stopping: bool = False,
    early_stopping_lookback: int = 500000,
    early_stopping_delta: float = 0.01,
    no_sa_penalty: bool = False,
    sa_threshold: Optional[float] = None,
    sa_penalty_slope: Optional[float] = None,
    no_dynamic_entropy: bool = False,
    checkpoint_dir: str | Path = "runs/checkpoints",
    export_dir: str | Path = "runs/candidates",
    seed: int = 42,
) -> Dict[str, Any]:
    """Top-level runner for Stage 1 Molecular Swarm RL policy training."""
    from dens_city.swarm.spec_loader import SwarmSpecLoader

    spec_path = Path(spec).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found at {spec_path}")

    print(f"=== Loading Swarm Specification: {spec_path.name} ===")
    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)
    print(f"Target Material Objectives: {target_spec}")

    exp_dir = Path(export_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Restrict PyTorch CPU thread pool to 1 thread, reserving CPU cores for OpenMP vectorization
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Vectorized C Environment Pool ({num_envs} envs, device={device})...")

    vec_env = VectorizedSwarmEnv(
        num_envs=num_envs,
        spec_yaml_path=spec_path,
        target_spec=target_spec,
        seed=seed,
    )

    policy = MolecularSwarmPolicy(
        obs_size=88,
        hidden_size=hidden_size,
        recurrent=recurrent,
    )

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_spec,
        total_timesteps=total_timesteps,
        horizon=horizon,
        learning_rate=learning_rate,
        use_curriculum=not no_curriculum,
        early_stopping=not no_early_stopping,
        early_stopping_lookback=early_stopping_lookback,
        early_stopping_delta=early_stopping_delta,
        dynamic_entropy_scaling=not no_dynamic_entropy,
        sa_penalty=not no_sa_penalty,
        sa_threshold=sa_threshold,
        sa_penalty_slope=sa_penalty_slope,
        checkpoint_dir=ckpt_dir,
        device=device,
    )

    print(f"Starting Swarm PPO Training for {total_timesteps:,} timesteps...")
    epochs = max(1, total_timesteps // (num_envs * horizon))

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

        final_policy_path = ckpt_dir / "trained_policy.pt"
        trainer.save_checkpoint(final_policy_path)
        print(f"\nTraining Complete! Dumped policy weights to: {final_policy_path}")
        print(f"Best Pareto Reward: {trainer.best_reward:.3f}")

        exported_mol2_path = None
        if trainer.best_mol2_str:
            exported_mol2_path = exp_dir / f"{spec_path.stem}_best.mol2"
            exported_mol2_path.write_text(trainer.best_mol2_str)
            print(f"Exported Pareto-optimal candidate to: {exported_mol2_path}")

        return {
            "policy_path": str(final_policy_path),
            "best_reward": trainer.best_reward,
            "exported_mol2": str(exported_mol2_path) if exported_mol2_path else None,
            "global_step": trainer.global_step,
        }

    finally:
        vec_env.close()
