"""
PufferLib 4.0 Molecular Swarm Policy Architecture.
Implements:
1. MolecularPortEncoder: Entity-centric geometric encoder over 16 3D port orientation vectors.
2. MolecularActionDecoder: Dual-head MultiDiscrete [16, 13] decoder with integrated -1e9 action masking.
3. MLPBackbone & MinGRUBackbone: Modular backbones adhering to PufferLib forward_eval / forward_train contract.
4. MolecularSwarmPolicy: Unified PyTorch policy container.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Initializes linear layers using orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class MolecularPortEncoder(nn.Module):
    """
    Entity-centric geometric encoder for the cdft_swarm environment.
    Processes global graph/thermodynamic metrics (24 floats) and 16 3D port entities (64 floats).
    """

    def __init__(self, obs_size: int = 88, hidden_size: int = 256):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.num_ports = 16
        self.port_dim = 4  # (nx, ny, nz, state_empty)
        self.global_dim = 24  # 16 graph features + 8 target bounds

        # Factorized Port entity projection: W_g * global + W_p * port + b
        self.global_port_proj = layer_init(nn.Linear(self.global_dim, 128))
        self.port_proj = layer_init(nn.Linear(self.port_dim, 128, bias=False))
        self.port_mlp_out = nn.Sequential(
            nn.GELU(),
            layer_init(nn.Linear(128, hidden_size)),
            nn.GELU(),
        )

        # Global features MLP
        self.global_mlp = nn.Sequential(
            layer_init(nn.Linear(self.global_dim, 128)),
            nn.GELU(),
            layer_init(nn.Linear(128, hidden_size)),
            nn.GELU(),
        )

        # Fusion layer: Global + Pooled Max + Pooled Mean -> Unified Latent
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(hidden_size * 3, hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for observations of shape (..., 88).
        Returns:
            latent: (..., hidden_size)
            port_embs: (..., 16, hidden_size)
        """
        orig_shape = obs.shape[:-1]
        flat_obs = obs.reshape(-1, self.obs_size).float()
        batch_size = flat_obs.shape[0]

        # 1. Split observation into Global Context (24) and Ports (16 x 4 = 64)
        global_feats = flat_obs[:, : self.global_dim]
        ports_raw = flat_obs[:, self.global_dim : self.global_dim + self.num_ports * self.port_dim]
        ports = ports_raw.reshape(batch_size, self.num_ports, self.port_dim)

        # 2. Factorized Port Entity Embeddings (0 Cat allocations, 16x fewer global ops)
        g_proj = self.global_port_proj(global_feats).unsqueeze(1)
        p_proj = self.port_proj(ports)
        port_embs = self.port_mlp_out(g_proj + p_proj)  # (B, 16, H)

        # 3. Masked Geometric Pooling over active open ports
        empty_mask = ports[:, :, 3:4]  # 1.0 if empty, 0.0 if filled/inactive
        masked_port_embs = port_embs * empty_mask

        # Max pooling with -1e4 mask on filled ports (FP16/BF16 safe)
        fill_penalty = (1.0 - empty_mask) * -1e4
        pooled_max = (masked_port_embs + fill_penalty).max(dim=1)[0]
        # Guard against all ports being capped
        pooled_max = torch.where(empty_mask.sum(dim=1) > 0, pooled_max, torch.zeros_like(pooled_max))

        # Mean pooling
        mask_counts = torch.clamp(empty_mask.sum(dim=1), min=1.0)
        pooled_mean = masked_port_embs.sum(dim=1) / mask_counts

        # 4. Global Graph Embedding & Fusion
        global_emb = self.global_mlp(global_feats)
        latent = self.fusion(torch.cat([global_emb, pooled_max, pooled_mean], dim=-1))

        latent_out = latent.reshape(*orig_shape, self.hidden_size)
        port_embs_out = port_embs.reshape(*orig_shape, self.num_ports, self.hidden_size)
        return latent_out, port_embs_out


class MolecularActionDecoder(nn.Module):
    """
    Dual-head MultiDiscrete decoder for Port selection (16) and Fragment choice (13).
    Applies -1e4 action mask to prevent proposing physically invalid actions (safe for FP16/BF16).
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.num_ports = 16
        self.num_frags = 13  # 12 building blocks + 1 finalize

        self.port_head = layer_init(nn.Linear(hidden_size, self.num_ports), std=0.01)
        self.frag_head = layer_init(nn.Linear(hidden_size, self.num_frags), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(
        self, hidden: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Computes masked logits and state value.
        Args:
            hidden: (..., hidden_size)
            action_mask: Optional (..., 29) boolean/float tensor (16 ports + 13 frags)
        Returns:
            (port_logits, frag_logits): ((..., 16), (..., 13))
            values: (..., 1)
        """
        port_logits = self.port_head(hidden)
        frag_logits = self.frag_head(hidden)
        values = self.value_head(hidden)

        if action_mask is not None:
            # Flatten/align mask
            mask = action_mask.to(device=hidden.device, dtype=torch.float32)
            port_mask = mask[..., : self.num_ports]
            frag_mask = mask[..., self.num_ports : self.num_ports + self.num_frags]

            port_logits = torch.where(port_mask > 0.5, port_logits, -1e4)
            frag_logits = torch.where(frag_mask > 0.5, frag_logits, -1e4)

        return (port_logits, frag_logits), values


class MLPBackbone(nn.Module):
    """Standard modular MLP network for PufferLib policy container."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        layers = []
        for _ in range(num_layers):
            layers += [layer_init(nn.Linear(hidden_size, hidden_size)), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[()]:
        return ()

    def forward_eval(self, h: torch.Tensor, state: Tuple[()]) -> Tuple[torch.Tensor, Tuple[()]]:
        return self.net(h), state

    def forward_train(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class MinGRUBackbone(nn.Module):
    """Fast recurrent highway gate matching PufferLib's MinGRU."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)])

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _highway(self, x: torch.Tensor, out: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        g = proj.sigmoid()
        return g * out + (1.0 - g) * x

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor]:
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, h: torch.Tensor, state: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        st = state[0]
        h_in = h.unsqueeze(1)
        state_out = []
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h_in).chunk(3, dim=-1)
            out = torch.lerp(st[i : i + 1].transpose(0, 1), self._g(hidden), gate.sigmoid())
            h_in = self._highway(h_in, out, proj)
            state_out.append(out[:, -1:])
        return h_in.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, h: torch.Tensor) -> torch.Tensor:
        # For simplicity in feedforward batches, evaluate linearly
        h_out = h
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h_out).chunk(3, dim=-1)
            out = self._g(hidden) * gate.sigmoid()
            h_out = self._highway(h_out, out, proj)
        return h_out


def sample_logits(
    port_logits: torch.Tensor,
    frag_logits: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    High-performance fused tensor sampling matching PufferLib Ocean.
    Avoids torch.distributions overhead and computes log_prob, entropy,
    and actions via direct tensor math.
    """
    port_logsumexp = torch.logsumexp(port_logits, dim=-1, keepdim=True)
    port_logprobs = port_logits - port_logsumexp
    port_probs = torch.exp(port_logprobs)
    port_entropy = -(port_probs * port_logprobs).sum(dim=-1)

    frag_logsumexp = torch.logsumexp(frag_logits, dim=-1, keepdim=True)
    frag_logprobs = frag_logits - frag_logsumexp
    frag_probs = torch.exp(frag_logprobs)
    frag_entropy = -(frag_probs * frag_logprobs).sum(dim=-1)

    if action is None:
        if deterministic:
            a_port = port_logits.argmax(dim=-1)
            a_frag = frag_logits.argmax(dim=-1)
        else:
            a_port = torch.multinomial(port_probs, num_samples=1).squeeze(-1)
            a_frag = torch.multinomial(frag_probs, num_samples=1).squeeze(-1)
        action = torch.stack([a_port, a_frag], dim=-1)
    else:
        a_port = action[..., 0].long()
        a_frag = action[..., 1].long()

    port_lp = port_logprobs.gather(-1, a_port.unsqueeze(-1)).squeeze(-1)
    frag_lp = frag_logprobs.gather(-1, a_frag.unsqueeze(-1)).squeeze(-1)

    total_logprob = port_lp + frag_lp
    total_entropy = port_entropy + frag_entropy

    return action, total_logprob, total_entropy


class MolecularSwarmPolicy(nn.Module):
    """
    Unified PufferLib 4.0 Molecular Swarm Policy.
    Combines MolecularPortEncoder, Modular Backbone, and MolecularActionDecoder.
    """

    def __init__(
        self,
        obs_size: int = 88,
        hidden_size: int = 256,
        recurrent: bool = False,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.recurrent = recurrent

        self.encoder = MolecularPortEncoder(obs_size=obs_size, hidden_size=hidden_size)
        if recurrent:
            self.network = MinGRUBackbone(hidden_size=hidden_size, num_layers=1)
        else:
            self.network = MLPBackbone(hidden_size=hidden_size, num_layers=2)
        self.decoder = MolecularActionDecoder(hidden_size=hidden_size)

    def initial_state(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Any:
        return self.network.initial_state(batch_size, device)

    def forward_eval(
        self,
        obs: torch.Tensor,
        state: Any = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Any]:
        """Evaluation / rollout forward pass."""
        if state is None:
            state = self.initial_state(obs.shape[0], obs.device)
        h, _ = self.encoder(obs)
        h, next_state = self.network.forward_eval(h, state)
        logits, values = self.decoder(h, action_mask=action_mask)
        return logits, values, next_state

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Training forward pass."""
        B, TT = obs.shape[:2]
        flat_obs = obs.reshape(B * TT, -1)
        flat_mask = action_mask.reshape(B * TT, -1) if action_mask is not None else None

        h, _ = self.encoder(flat_obs)
        h = self.network.forward_train(h.reshape(B, TT, -1)).reshape(B * TT, -1)
        logits, values = self.decoder(h, action_mask=flat_mask)
        return logits, values.reshape(B, TT, 1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples or evaluates MultiDiscrete actions (port, fragment), calculating
        joint log-probabilities, entropy, and baseline values with fused tensor math.
        """
        h, _ = self.encoder(obs)
        h = self.network.forward_train(h)
        (port_logits, frag_logits), values = self.decoder(h, action_mask=action_mask)

        act, logprob, entropy = sample_logits(
            port_logits=port_logits,
            frag_logits=frag_logits,
            action=action,
            deterministic=deterministic,
        )

        return {
            "action": act,
            "logprob": logprob,
            "entropy": entropy,
            "value": values.squeeze(-1),
            "port_logits": port_logits,
            "frag_logits": frag_logits,
        }

    def sample_action(
        self,
        obs: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        hidden_state: Any = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Samples an action for evaluation or inference rollouts.
        Returns (actions, logprobs, values, next_hidden_state).
        """
        res = self.get_action_and_value(
            obs=obs,
            action_mask=action_masks,
            deterministic=deterministic,
        )
        if hidden_state is not None:
            return res["action"], res["logprob"], res["value"], hidden_state
        return res["action"], res["logprob"], res["value"]
