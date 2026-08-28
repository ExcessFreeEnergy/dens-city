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

    metrics = data["metrics"]
    for required_key in ["agent_steps", "uptime", "env/score", "env/perf", "loss/policy", "loss/approx_kl"]:
        assert required_key in metrics, f"Missing required Constellation metric: {required_key}"
        assert isinstance(metrics[required_key], list)
        assert len(metrics[required_key]) > 0


def test_ppo_overtraining_telemetry():
    """
    Verifies that SwarmPuffeRLTrainer logs accurate PPO overtraining diagnostic metrics:
    approximate KL divergence, clip fraction, explained variance, and policy entropy.
    """
    spec_data = SwarmSpecLoader.load_yaml(OLED_SPEC)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

    vec_env = VectorizedSwarmEnv(num_envs=4, spec_yaml_path=OLED_SPEC, seed=42)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False)

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_spec,
        total_timesteps=128,
        horizon=8,
        learning_rate=1e-3,
        minibatch_size=16,
        update_epochs=2,
        use_curriculum=False,
        device="cpu",
    )

    try:
        metrics = trainer.train_epoch()
        assert "loss/approx_kl" in metrics and np.isfinite(metrics["loss/approx_kl"])
        assert "loss/clipfrac" in metrics and np.isfinite(metrics["loss/clipfrac"])
        assert "loss/entropy" in metrics and np.isfinite(metrics["loss/entropy"])
        assert "loss/explained_variance" in metrics and np.isfinite(metrics["loss/explained_variance"])
        assert metrics["loss/entropy"] > 0.0, "Policy entropy should be positive"
        assert metrics["loss/approx_kl"] >= 0.0, "Approximate KL divergence should be non-negative"
    finally:
        vec_env.close()


def test_dynamic_early_stopping_ema_algorithm(tmp_path):
    """
    Verifies that the Dynamic Early-Stopping Callback:
    1. Tracks a rolling 100-episode EMA of reward.
    2. Accurately compares the current EMA with the EMA from lookback steps ago.
    3. Halts execution and dumps trained_policy.pt when progress has flatlined (Δ_EMA < 0.01)
       and physical constraints are satisfied.
    """
    spec_data = SwarmSpecLoader.load_yaml(OLED_SPEC)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

    vec_env = VectorizedSwarmEnv(num_envs=2, spec_yaml_path=OLED_SPEC, seed=101)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False)

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_spec,
        total_timesteps=1000,
        horizon=4,
        learning_rate=1e-3,
        minibatch_size=8,
        update_epochs=1,
        use_curriculum=False,
        early_stopping=True,
        early_stopping_lookback=16,  # Short lookback for unit test
        early_stopping_delta=0.05,
        checkpoint_dir=tmp_path,
        device="cpu",
    )

    try:
        # Simulate populated history with flatlined EMA and satisfied constraints
        trainer.reward_ema = 5.50
        trainer.ema_history.append((16, 5.505, {"p_wall": 25.0, "omega_solv": -4.0, "valid_rate": 0.90}))
        trainer.global_step = 64

        # Test early stopping check helper
        halted = trainer._check_early_stopping(
            stage_idx=3,
            avg_p_wall=25.0,
            avg_omega=-4.0,
            valid_rate=0.90,
        )
        assert halted is True
        assert trainer.early_stopped is True
        assert "Reward EMA flatlined" in trainer.early_stop_reason

        # Verify trained_policy.pt was dumped
        ckpt_path = tmp_path / "trained_policy.pt"
        assert ckpt_path.exists()

        # Verify saved checkpoint can be reloaded
        trainer2 = SwarmPuffeRLTrainer(
            vec_env=vec_env,
            policy=MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False),
            final_target_spec=target_spec,
            device="cpu",
        )
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == 64
    finally:
        vec_env.close()


def test_dynamic_entropy_scaling_with_molecular_weight():
    """
    Verifies that SwarmPuffeRLTrainer dynamically inflates the policy exploration entropy
    coefficient when the molecular weight ceiling is low (e.g. Aliphatic Sponges <= 400 amu)
    compared to high-mass specs (850 amu).
    """
    spec_sponge = SwarmSpecLoader.load_yaml(SPONGE_SPEC)
    target_sponge = SwarmSpecLoader.derive_target_spec(spec_sponge)
    assert target_sponge["max_molecular_weight"] == 350.0

    vec_env = VectorizedSwarmEnv(num_envs=2, spec_yaml_path=SPONGE_SPEC, seed=42)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False)

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_sponge,
        total_timesteps=64,
        horizon=4,
        ent_coef=0.01,
        dynamic_entropy_scaling=True,
        device="cpu",
    )

    try:
        metrics = trainer.train_epoch()
        # With max_molecular_weight = 400 amu, dynamic entropy scale factor is 850 / 400 = 2.125
        assert metrics["loss/dynamic_ent_scale"] >= 2.0
        assert metrics["loss/entropy"] > 0.0
    finally:
        vec_env.close()


def test_post_rollout_batch_sa_penalty_injection():
    """
    Verifies that post-rollout batch SA calculation retroactively injects the soft hinge penalty
    R_SA = -slope * max(0, SA - threshold) into rewards_buf before GAE computation.
    """
    spec_oled = SwarmSpecLoader.load_yaml(OLED_SPEC)
    target_oled = SwarmSpecLoader.derive_target_spec(spec_oled)

    vec_env = VectorizedSwarmEnv(num_envs=2, spec_yaml_path=OLED_SPEC, seed=42)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False)

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_oled,
        total_timesteps=64,
        horizon=4,
        sa_penalty=True,
        sa_threshold=4.5,
        sa_penalty_slope=2.0,
        device="cpu",
    )

    try:
        metrics = trainer.train_epoch()
        assert "env/sa_score" in metrics
        assert "env/r_sa_penalty" in metrics
        assert np.isfinite(metrics["env/sa_score"])
        assert np.isfinite(metrics["env/r_sa_penalty"])
    finally:
        vec_env.close()


def test_dynamic_sa_threshold_derivation_across_specs():
    """
    Verifies that SwarmSpecLoader derives domain-specific SA thresholds and slopes:
    - H-Bond Resins -> SA thresh 5.5, slope 1.0 (dense polar donor/acceptor networks)
    - OLEDs -> SA thresh 5.0, slope 1.5 (extended pi-conjugation)
    - Battery Electrolytes -> SA thresh 5.2, slope 1.5 (fluorinated multi-clusters)
    - Aliphatic Sponges & Drug Inhibitors -> SA thresh 4.5, slope 2.0 (small molecule / steric bulk)
    """
    spec_dir = SPECS_DIR
    resin_target = SwarmSpecLoader.derive_target_spec(
        SwarmSpecLoader.load_yaml(spec_dir / "sacrificial_h_bond_toughness_resins.yaml")
    )
    oled_target = SwarmSpecLoader.derive_target_spec(
        SwarmSpecLoader.load_yaml(spec_dir / "conjugated_oled_semiconductors.yaml")
    )
    battery_target = SwarmSpecLoader.derive_target_spec(
        SwarmSpecLoader.load_yaml(spec_dir / "fluorinated_battery_electrolytes.yaml")
    )
    sponge_target = SwarmSpecLoader.derive_target_spec(
        SwarmSpecLoader.load_yaml(spec_dir / "ultra_lightweight_aliphatic_sponges.yaml")
    )
    drug_target = SwarmSpecLoader.derive_target_spec(
        SwarmSpecLoader.load_yaml(spec_dir / "sterically_hindered_drug_inhibitors.yaml")
    )

    assert resin_target["sa_threshold"] == 5.5 and resin_target["sa_penalty_slope"] == 1.0
    assert oled_target["sa_threshold"] == 5.0 and oled_target["sa_penalty_slope"] == 1.5
    assert battery_target["sa_threshold"] == 5.2 and battery_target["sa_penalty_slope"] == 1.5
    assert sponge_target["sa_threshold"] == 4.5 and sponge_target["sa_penalty_slope"] == 2.0
    assert drug_target["sa_threshold"] == 4.5 and drug_target["sa_penalty_slope"] == 2.0


def test_hbond_resin_diversity_restoration():
    """
    Verifies that Sacrificial H-Bond Resin training inherits the calibrated SA threshold (5.5)
    and enhanced high-valency exploration entropy scale (>= 2.0).
    """
    resin_path = SPECS_DIR / "sacrificial_h_bond_toughness_resins.yaml"
    target_resin = SwarmSpecLoader.derive_target_spec(SwarmSpecLoader.load_yaml(resin_path))

    vec_env = VectorizedSwarmEnv(num_envs=2, spec_yaml_path=resin_path, seed=77)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=64, recurrent=False)

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_resin,
        total_timesteps=64,
        horizon=4,
        sa_penalty=True,
        dynamic_entropy_scaling=True,
        device="cpu",
    )

    try:
        assert trainer.sa_threshold == 5.5
        assert trainer.sa_penalty_slope == 1.0
        metrics = trainer.train_epoch()
        assert metrics["loss/dynamic_ent_scale"] >= 2.0
        assert np.isfinite(metrics["env/score"])
    finally:
        vec_env.close()
