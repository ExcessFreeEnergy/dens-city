"""
Comprehensive Test Suite for PufferLib 4.0 Molecular Swarm Policy, Vectorized Trainer, and Curriculum Sweeps.
Validates:
1. MolecularPortEncoder geometric tensor transformations and shapes.
2. MolecularActionDecoder action masking (-1e9 penalty on invalid slots).
3. MolecularSwarmPolicy sampling validity and recurrent state transitions.
4. SwarmCurriculumManager C-memory broadcast across parallel environments.
5. SwarmPuffeRLTrainer vectorized PPO training step.
6. CurriculumSweepRunner JSON output schema matching pufferlib/constellation/cache_data.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dens_city.swarm.policy import (
    MolecularActionDecoder,
    MolecularPortEncoder,
    MolecularSwarmPolicy,
)
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.sweep import CurriculumSweepRunner
from dens_city.swarm.trainer import (
    SwarmCurriculumManager,
    SwarmPuffeRLTrainer,
    VectorizedSwarmEnv,
)

SPECS_DIR = Path(__file__).resolve().parent / "data"
OLED_SPEC = SPECS_DIR / "conjugated_oled_semiconductors.yaml"
SPONGE_SPEC = SPECS_DIR / "ultra_lightweight_aliphatic_sponges.yaml"


def test_molecular_port_encoder_forward():
    """Verifies that MolecularPortEncoder correctly transforms (B, 88) observation vectors."""
    encoder = MolecularPortEncoder(obs_size=88, hidden_size=128)
    dummy_obs = torch.randn(8, 88)
    # Set realistic empty mask flags on ports (indices 24 + 4*p + 3)
    for p in range(16):
        dummy_obs[:, 24 + 4 * p + 3] = 1.0 if p < 4 else 0.0

    latent, port_embs = encoder(dummy_obs)
    assert latent.shape == (8, 128), f"Latent shape mismatch: {latent.shape}"
    assert port_embs.shape == (8, 16, 128), f"Port embeddings shape mismatch: {port_embs.shape}"
    assert torch.isfinite(latent).all()


def test_molecular_action_decoder_masking():
    """Verifies that MolecularActionDecoder applies -1e9 mask to invalid ports and fragments."""
    decoder = MolecularActionDecoder(hidden_size=128)
    hidden = torch.randn(4, 128)

    # Mask: only port 0 and frag 3 are valid
    mask = torch.zeros(4, 29)
    mask[:, 0] = 1.0  # port 0
    mask[:, 16 + 3] = 1.0  # frag 3

    (port_logits, frag_logits), values = decoder(hidden, action_mask=mask)

    assert port_logits.shape == (4, 16)
    assert frag_logits.shape == (4, 13)
    assert values.shape == (4, 1)

    # Valid port 0 and frag 3 should have finite logits
    assert (port_logits[:, 0] > -1e8).all()
    assert (frag_logits[:, 3] > -1e8).all()

    # Invalid slots should have large negative penalty (-1e9)
    assert (port_logits[:, 1:] <= -1e8).all()
    assert (frag_logits[:, :3] <= -1e8).all()
    assert (frag_logits[:, 4:] <= -1e8).all()


def test_molecular_swarm_policy_sample_action():
    """Verifies that MolecularSwarmPolicy never samples masked actions."""
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=128, recurrent=False)
    dummy_obs = torch.randn(10, 88)

    # Mask allowing only port 2 and fragments 1, 5
    mask = torch.zeros(10, 29)
    mask[:, 2] = 1.0
    mask[:, 16 + 1] = 1.0
    mask[:, 16 + 5] = 1.0

    for _ in range(50):
        out = policy.get_action_and_value(dummy_obs, action_mask=mask)
        actions = out["action"]  # (10, 2)
        assert (actions[:, 0] == 2).all(), "Policy sampled an invalid port"
        assert ((actions[:, 1] == 1) | (actions[:, 1] == 5)).all(), "Policy sampled an invalid fragment"
        assert torch.isfinite(out["logprob"]).all()
        assert torch.isfinite(out["entropy"]).all()
        assert torch.isfinite(out["value"]).all()


def test_curriculum_c_memory_broadcast(tmp_path):
    """
    Verifies that SwarmCurriculumManager broadcasts updated TargetSpec directly into C memory
    across parallel environments, and that the C observation reflects the new targets.
    """
    spec_data = SwarmSpecLoader.load_yaml(OLED_SPEC)
    final_targets = SwarmSpecLoader.derive_target_spec(spec_data)

    vec_env = VectorizedSwarmEnv(num_envs=2, spec_yaml_path=OLED_SPEC, seed=42)
    curriculum = SwarmCurriculumManager(final_targets, enabled=True)

    try:
        # Progress 0.0 -> Stage 1 (Feasibility)
        t1, s1 = curriculum.broadcast_to_vec_env(vec_env, progress=0.0)
        assert s1 == 1
        obs, mask = vec_env.reset()
        # Observation indices [80..87] encode YAML targets
        # Index 86 is max_molecular_weight / 1000.0 (1000.0 / 1000.0 = 1.0 in Stage 1)
        assert np.isclose(obs[0, 86].item(), 1.0, atol=1e-2)

        # Progress 1.0 -> Stage 3 (Full Target Spec: OLED max MW = 850.0 / 1000.0 = 0.85)
        t3, s3 = curriculum.broadcast_to_vec_env(vec_env, progress=1.0)
        assert s3 == 3
        obs, mask = vec_env.reset()
        assert np.isclose(obs[0, 86].item(), 0.85, atol=1e-2)
    finally:
        vec_env.close()


def test_vectorized_pufferl_trainer_epoch():
    """Verifies that SwarmPuffeRLTrainer executes a full rollout and PPO update without error."""
    spec_data = SwarmSpecLoader.load_yaml(SPONGE_SPEC)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

    vec_env = VectorizedSwarmEnv(num_envs=4, spec_yaml_path=SPONGE_SPEC, seed=99)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False)

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_spec,
        total_timesteps=200,
        horizon=8,
        learning_rate=1e-3,
        minibatch_size=16,
        update_epochs=2,
        use_curriculum=True,
        device="cpu",
    )

    try:
        metrics = trainer.train_epoch()
        assert metrics["epoch"] == 1
        assert metrics["global_step"] == 4 * 8  # 32 steps
        assert "loss/policy" in metrics and np.isfinite(metrics["loss/policy"])
        assert "loss/value" in metrics and np.isfinite(metrics["loss/value"])
        assert "env/score" in metrics
    finally:
        vec_env.close()


def test_constellation_sweep_export_schema(tmp_path):
    """
    Verifies that CurriculumSweepRunner outputs JSON logs that adhere strictly to
    the PufferLib Constellation format expected by pufferlib/constellation/cache_data.py.
    """
    runner = CurriculumSweepRunner(
        spec_yaml_paths=[SPONGE_SPEC],
        output_dir=tmp_path,
        seed=123,
    )

    hypers = runner.sample_hyperparameters()
    hypers["horizon"] = 8
    hypers["minibatch_size"] = 16
    hypers["hidden_size"] = 64

    record = runner.run_trial(
        trial_id=1,
        spec_path=SPONGE_SPEC,
        hypers=hypers,
        total_timesteps=64,
        num_envs=2,
        device="cpu",
    )
    assert record["trial_id"] == 1

    trial_file = tmp_path / f"trial_0001_{SPONGE_SPEC.stem}.json"
    assert trial_file.exists()

    with open(trial_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate Constellation expected keys
    assert "args" in data
    assert "train" in data["args"]
    assert "learning_rate" in data["args"]["train"]
    assert "policy" in data["args"]
    assert "vec" in data["args"]

    assert "metrics" in data
    metrics = data["metrics"]
    for required_key in ["agent_steps", "uptime", "env/score", "env/perf", "loss/policy"]:
        assert required_key in metrics, f"Missing required Constellation metric: {required_key}"
        assert isinstance(metrics[required_key], list)
        assert len(metrics[required_key]) > 0
