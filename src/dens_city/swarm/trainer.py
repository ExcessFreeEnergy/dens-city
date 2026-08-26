"""
PufferLib Vectorized Swarm Environment & C-Memory Curriculum PPO Trainer.
Implements:
1. VectorizedSwarmEnv: High-performance parallel C-FFI environment pool.
2. SwarmCurriculumManager: 3-stage progressive difficulty scheduler with C-memory broadcast.
3. SwarmPuffeRLTrainer: Vectorized PPO engine with GAE-lambda and candidate .mol2 export.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dens_city.swarm.env import CDFTSwarmEnv
from dens_city.swarm.policy import MolecularSwarmPolicy


class VectorizedSwarmEnv:
    """
    Manages a pool of N parallel C-native CDFTSwarmEnv instances.
    Provides vectorized reset, step, action masking, and direct C-memory TargetSpec mutation.
    """

    def __init__(
        self,
        num_envs: int = 16,
        spec_yaml_path: Optional[str | Path] = None,
        target_spec: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.envs: List[CDFTSwarmEnv] = []
        for i in range(num_envs):
            env = CDFTSwarmEnv(
                spec_yaml_path=spec_yaml_path,
                target_spec=target_spec,
                seed=seed + i * 1000,
            )
            self.envs.append(env)

        self.obs_size = 88
        self.action_mask_size = 29
        self.num_atns = 2

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resets all parallel environments and returns stacked (obs, action_mask)."""
        obs_list = []
        mask_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            mask_list.append(info["action_mask"])
        obs_tensor = torch.from_numpy(np.stack(obs_list)).float()
        mask_tensor = torch.from_numpy(np.stack(mask_list)).float()
        return obs_tensor, mask_tensor

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Executes a parallel step across all N environments.
        Returns:
            obs: (N, 88)
            rewards: (N,)
            terminals: (N,)
            action_masks: (N, 29)
            infos: List of N info dicts
        """
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)

        obs_list = []
        reward_list = []
        term_list = []
        mask_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            a_port = int(actions_np[i, 0])
            a_frag = int(actions_np[i, 1])
            obs, reward, term, truncated, info = env.step((a_port, a_frag))

            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(1.0 if term else 0.0)
            mask_list.append(info["action_mask"])
            info_list.append(info)

        obs_tensor = torch.from_numpy(np.stack(obs_list)).float()
        reward_tensor = torch.from_numpy(np.array(reward_list, dtype=np.float32))
        term_tensor = torch.from_numpy(np.array(term_list, dtype=np.float32))
        mask_tensor = torch.from_numpy(np.stack(mask_list)).float()

        return obs_tensor, reward_tensor, term_tensor, mask_tensor, info_list

    def set_targets(self, targets: Dict[str, float]) -> None:
        """
        Directly broadcasts updated target parameters into the C TargetSpec memory of all N environments.
        """
        for env in self.envs:
            env.set_targets(targets)

    def export_best_candidate_mol2(self, env_idx: int, mol_name: str = "candidate") -> str:
        """Exports Tripos .mol2 string of the current molecule in environment env_idx."""
        return self.envs[env_idx].export_mol2_string(mol_name)

    def close(self) -> None:
        """Frees all C environment resources."""
        for env in self.envs:
            env.close()
        self.envs.clear()


class SwarmCurriculumManager:
    """
    Progressive 3-Stage Curriculum Manager for Molecular Swarm Training.
    Eliminates memory segregation by mutating C TargetSpec structs via ctypes broadcast.
    """

    def __init__(self, final_target_spec: Dict[str, float], enabled: bool = True):
        self.final_targets = dict(final_target_spec)
        self.enabled = enabled
        self.current_stage = 1

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

    def compute_targets_for_progress(self, progress: float) -> Tuple[Dict[str, float], int]:
        """
        Calculates interpolated target specification for normalized progress in [0, 1].
        Stages:
            [0.0, 0.25): Stage 1 (Feasibility)
            [0.25, 0.65): Stage 2 (Intermediate Shaping)
            [0.65, 1.00]: Stage 3 (Full Target Specification)
        """
        if not self.enabled:
            return dict(self.final_targets), 3

        if progress < 0.25:
            stage = 1
            return dict(self.stage1_targets), stage
        elif progress < 0.65:
            stage = 2
            # Linear interpolation factor within stage 2
            alpha = (progress - 0.25) / 0.40
            targets = {}
            for k, final_v in self.final_targets.items():
                s1_v = self.stage1_targets.get(k, final_v)
                if isinstance(final_v, int):
                    targets[k] = int(round(s1_v + alpha * (final_v - s1_v)))
                else:
                    targets[k] = float(s1_v + alpha * (final_v - s1_v))
            return targets, stage
        else:
            stage = 3
            return dict(self.final_targets), stage

    def broadcast_to_vec_env(self, vec_env: VectorizedSwarmEnv, progress: float) -> Tuple[Dict[str, float], int]:
        """Updates and broadcasts targets into C memory across all parallel environments."""
        targets, stage = self.compute_targets_for_progress(progress)
        self.current_stage = stage
        vec_env.set_targets(targets)
        return targets, stage


class SwarmPuffeRLTrainer:
    """
    PufferLib-templated Vectorized PPO Trainer for Molecular Swarm Discovery.
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
        minibatch_size: int = 64,
        update_epochs: int = 4,
        use_curriculum: bool = True,
        device: str | torch.device = "cpu",
    ):
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

        self.curriculum = SwarmCurriculumManager(final_target_spec, enabled=use_curriculum)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

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
        self.best_reward = -float("inf")
        self.best_mol2_str = ""

    def train_epoch(self) -> Dict[str, float]:
        """Collects a vectorized rollout and executes PPO optimization."""
        self.epoch += 1
        epoch_start_time = time.time()

        # 1. Update curriculum and broadcast into C-memory
        progress = min(1.0, float(self.global_step) / float(max(1, self.total_timesteps)))
        current_targets, stage_idx = self.curriculum.broadcast_to_vec_env(self.vec_env, progress)

        # 2. Reset or get current env states
        obs, action_mask = self.vec_env.reset() if self.global_step == 0 else (self.last_obs, self.last_mask)
        obs = obs.to(self.device)
        action_mask = action_mask.to(self.device)

        # 3. Rollout Collection
        ep_rewards = []
        p_walls = []
        omega_solvs = []
        valid_molecules = 0
        total_episodes = 0

        self.policy.eval()
        for step in range(self.horizon):
            self.global_step += self.num_envs

            with torch.no_grad():
                out = self.policy.get_action_and_value(obs, action_mask=action_mask)
                actions = out["action"]
                logprobs = out["logprob"]
                values = out["value"]

            self.obs_buf[step] = obs
            self.mask_buf[step] = action_mask
            self.actions_buf[step] = actions
            self.logprobs_buf[step] = logprobs
            self.values_buf[step] = values

            # Vectorized step
            next_obs, rewards, terminals, next_masks, infos = self.vec_env.step(actions)
            self.rewards_buf[step] = rewards.to(self.device)
            self.terminals_buf[step] = terminals.to(self.device)

            for i, info in enumerate(infos):
                if terminals[i] > 0.5:
                    total_episodes += 1
                    ep_r = float(rewards[i])
                    ep_rewards.append(ep_r)
                    if "p_wall_bar" in info:
                        p_walls.append(info["p_wall_bar"])
                    if "omega_solv_kcal" in info:
                        omega_solvs.append(info["omega_solv_kcal"])
                    if ep_r > 0.0:
                        valid_molecules += 1
                    if ep_r > self.best_reward:
                        self.best_reward = ep_r
                        self.best_mol2_str = self.vec_env.export_best_candidate_mol2(i, "best_candidate")

            obs = next_obs.to(self.device)
            action_mask = next_masks.to(self.device)

        self.last_obs = obs
        self.last_mask = action_mask

        # 4. GAE-Lambda Advantage Estimation
        with torch.no_grad():
            out = self.policy.get_action_and_value(obs, action_mask=action_mask)
            next_value = out["value"]
            advantages = torch.zeros_like(self.rewards_buf)
            lastgaelam = 0
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

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 6. PPO Optimization Epochs
        self.policy.train()
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent_loss = 0.0
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

                # Clipped Policy Loss
                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_val - b_returns[mb_idx]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += float(pg_loss.item())
                total_vf_loss += float(v_loss.item())
                total_ent_loss += float(entropy_loss.item())
                num_updates += 1

        elapsed = time.time() - epoch_start_time
        sps = self.batch_size / max(1e-4, elapsed)

        avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        valid_rate = float(valid_molecules / max(1, total_episodes))
        avg_p_wall = float(np.mean(p_walls)) if p_walls else 0.0
        avg_omega = float(np.mean(omega_solvs)) if omega_solvs else 0.0

        metrics = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "SPS": sps,
            "curriculum_stage": stage_idx,
            "loss/policy": total_pg_loss / max(1, num_updates),
            "loss/value": total_vf_loss / max(1, num_updates),
            "loss/entropy": total_ent_loss / max(1, num_updates),
            "env/score": avg_reward,
            "env/valid_rate": valid_rate,
            "env/p_wall": avg_p_wall,
            "env/omega_solv": avg_omega,
            "best_reward": self.best_reward,
        }
        return metrics
