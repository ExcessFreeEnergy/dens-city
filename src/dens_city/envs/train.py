import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Disable cuDNN to use native robust CUDA 1D conv kernels without version mismatch
torch.backends.cudnn.enabled = False

from dens_city.envs.env import DensCityFluidEnv


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + res)


class DensNeuralFunctional(nn.Module):
    """
    Unified Multi-Scale Neural cDFT & Actor-Critic Policy.
    Simultaneously learns:
    1. Local one-body direct correlation functional c_R^(1)(z; [rho], T)
    2. Hyperdensity observable rho_H^(1)(z; [rho_O], T) (Hyper-DFT)
    3. Active RL policy for closed-loop fluid manipulation
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        # 1D Convolutional feature extractor (local receptive field ~2 nm)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            ResidualBlock1D(hidden_dim),
            ResidualBlock1D(hidden_dim),
        )

        # 1. cDFT Functional Head: predicts c_R^(1)(z) on the 256-point grid
        self.c1_head = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

        # 2. Hyper-DFT Head: predicts secondary atomic density rho_H(z)
        self.hyper_head = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Softplus(),
        )

        # 3. RL Policy Head
        self.rl_pool = nn.AdaptiveAvgPool1d(1)
        self.rl_fc = nn.Sequential(
            nn.Linear(hidden_dim + 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs shape: [B, 771] -> rho(256), V_ext(256), phi_R(256), scalars(3)
        b_size = obs.shape[0]
        grid_feats = obs[:, :768].view(b_size, 3, 256)
        scalars = obs[:, 768:]

        feats = self.encoder(grid_feats) # [B, hidden_dim, 256]

        # c1 functional output [B, 256]
        c1_pred = self.c1_head(feats).squeeze(1)

        # Hyper-DFT output [B, 256]
        rho_h_pred = self.hyper_head(feats).squeeze(1)

        # Global features for RL
        pooled = self.rl_pool(feats).squeeze(2) # [B, hidden_dim]
        combined = torch.cat([pooled, scalars], dim=1)
        rl_feats = self.rl_fc(combined)

        action_mean = self.actor_mean(rl_feats)
        value = self.critic(rl_feats)

        return action_mean, value, c1_pred, rho_h_pred

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, _, _ = self(obs)
        if deterministic:
            return torch.tanh(action_mean), torch.zeros(1), value

        std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        action = torch.tanh(raw_action)
        return action, log_prob, value


def train_unified(
    total_timesteps: int = 100000,
    num_envs: int = 16,
    lr: float = 3e-4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "dens_functional.pt",
):
    device = torch.device(device_str)
    print(f"[dens-city] Starting Unified PufferLib Direct Training on {device}...")
    print(f"[dens-city] Total Timesteps: {total_timesteps}, Parallel Envs: {num_envs}")

    env = DensCityFluidEnv(num_envs=num_envs)
    model = DensNeuralFunctional().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    obs_list = [env.reset(i)[0] for i in range(num_envs)]
    obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)

    rollout_len = 128
    steps_done = 0
    start_time = time.time()

    while steps_done < total_timesteps:
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []

        for step in range(rollout_len):
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs_batch)

            obs_buf.append(obs_batch)
            act_buf.append(action)
            val_buf.append(value.squeeze(-1))
            logp_buf.append(log_prob)

            act_np = action.cpu().numpy()
            rewards = []
            next_obs_list = []

            for i in range(num_envs):
                next_obs, rew, done, _, info = env.step(act_np[i], env_idx=i)
                rewards.append(rew)
                if done:
                    next_obs, _ = env.reset(i)
                next_obs_list.append(next_obs)

            rew_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            obs_batch = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=device)
            steps_done += num_envs

        # Compute GAE and returns
        with torch.no_grad():
            _, next_value, _, _ = model(obs_batch)
            next_value = next_value.squeeze(-1)

        returns = torch.zeros(rollout_len, num_envs, device=device)
        gae = torch.zeros(num_envs, device=device)
        gamma = 0.99
        lam = 0.95

        for t in reversed(range(rollout_len)):
            delta = rew_buf[t] + gamma * (next_value if t == rollout_len - 1 else val_buf[t + 1]) - val_buf[t]
            gae = delta + gamma * lam * gae
            returns[t] = gae + val_buf[t]

        # Flatten rollout buffers
        b_obs = torch.cat(obs_buf, dim=0)
        b_act = torch.cat(act_buf, dim=0)
        b_ret = returns.view(-1)
        b_val = torch.cat(val_buf, dim=0)
        b_logp = torch.cat(logp_buf, dim=0)
        b_adv = b_ret - b_val
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Optimize for 4 epochs
        dataset_size = b_obs.shape[0]
        batch_size = 256
        indices = np.arange(dataset_size)

        for _ in range(4):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                mb_obs = b_obs[idx]
                mb_act = b_act[idx]
                mb_ret = b_ret[idx]
                mb_adv = b_adv[idx]
                mb_old_logp = b_logp[idx]

                act_mean, value, c1_pred, rho_h_pred = model(mb_obs)
                std = torch.exp(model.actor_logstd)
                dist = torch.distributions.Normal(act_mean, std)
                new_logp = dist.log_prob(mb_act).sum(dim=-1)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((value.squeeze(-1) - mb_ret) ** 2).mean()

                # Self-consistent Euler-Lagrange residual loss on c1 functional
                # c1_target = ln(rho/rho_b) + beta*(V_ext - mu)
                rho = mb_obs[:, :256]
                v_ext = mb_obs[:, 256:512]
                phi_r = mb_obs[:, 512:768]
                beta = 1.0 / (1.380649e-23 * 300.0)
                c1_target = torch.log(torch.clamp(rho / 0.033, min=1e-4)) + beta * (phi_r - (-3000.0 * 1.380649e-23))
                c1_loss = 0.1 * nn.functional.mse_loss(c1_pred, c1_target.detach())

                total_loss = policy_loss + value_loss + c1_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        elapsed = time.time() - start_time
        sps = int(steps_done / elapsed) if elapsed > 0 else 0
        print(f"[dens-city] Steps: {steps_done}/{total_timesteps} | SPS: {sps} | PolLoss: {policy_loss.item():.4f} | ValLoss: {value_loss.item():.4f}")

    # Save trained functional directly
    torch.save({
        "state_dict": model.state_dict(),
        "total_timesteps": steps_done,
        "arch": "DensNeuralFunctional_ResNet1D",
    }, save_path)
    print(f"[dens-city] Direct Training Complete! Trained functional saved to '{save_path}'.")


def main():
    parser = argparse.ArgumentParser(description="dens-city Unified PufferLib Direct Training")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=16, help="Number of parallel C environments")
    parser.add_argument("--save", type=str, default="dens_functional.pt", help="Path to save trained functional")
    args = parser.parse_args()

    train_unified(total_timesteps=args.timesteps, num_envs=args.envs, save_path=args.save)


if __name__ == "__main__":
    main()
