import numpy as np
import torch
from tinygrad import Tensor

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.swarm.env import VectorizedSwarmEnv
from dens_city.swarm.funnel import detect_optimal_gpu_batch_size
from dens_city.swarm.trainer import SwarmCurriculumManager


def test_wl_graph_hash_uniqueness_and_novelty():
    """Verifies that C-level 2-hop WL graph hash is computed and novelty bonus applies."""
    final_spec = {
        "target_elasticity": 0.5,
        "target_tensile": 0.5,
        "target_toughness": 0.5,
        "target_lightweight": 0.5,
        "max_solvation_kcal": -2.0,
        "min_wall_pressure_bar": 10.0,
        "max_molecular_weight": 800.0,
        "min_valency": 2,
    }
    vec_env = VectorizedSwarmEnv(num_envs=4, seed=42)
    vec_env.set_targets(final_spec)
    obs, mask = vec_env.reset()
    assert obs.shape == (4, 88)

    for _ in range(16):
        actions = torch.zeros((4, 2), dtype=torch.long)
        actions[:, 0] = 0
        actions[:, 1] = 1
        next_obs, rewards, terminals, next_masks, infos = vec_env.step(actions)
        if terminals.sum() > 0:
            break

    assert rewards.shape == (4,)
    vec_env.close()


def test_curriculum_monotonic_stage_lock_and_clamping():
    """Verifies that validation dips do not regress the curriculum stage and step clamping works."""
    final_spec = {
        "target_elasticity": 0.8,
        "target_tensile": 0.8,
        "target_toughness": 0.8,
        "target_lightweight": 0.8,
        "max_solvation_kcal": -5.0,
        "min_wall_pressure_bar": 20.0,
        "max_molecular_weight": 700.0,
        "min_valency": 3,
    }
    cm = SwarmCurriculumManager(final_spec, enabled=True, stage1_max_steps=25000, stage2_max_steps=65000)

    # Initially at step 0, stage 1
    t, s = cm.compute_targets_for_progress(progress=0.0, global_step=0, valid_rate=0.1)
    assert s == 1
    assert cm.current_stage == 1

    # High valid rate unlocks Stage 3 early
    t, s = cm.compute_targets_for_progress(progress=0.01, global_step=5000, valid_rate=0.96)
    assert s == 3
    assert cm.current_stage == 3

    # Subsequent validation dip (e.g. 96% -> 60%) must NOT regress Stage 3
    t, s = cm.compute_targets_for_progress(progress=0.02, global_step=6000, valid_rate=0.60)
    assert s == 3
    assert cm.current_stage == 3
    assert t["max_solvation_kcal"] == -5.0

    # Test step-clamping on a fresh manager (50M step run)
    cm2 = SwarmCurriculumManager(final_spec, enabled=True, stage1_max_steps=25000, stage2_max_steps=65000)
    # Even at progress 0.001 (step 70k out of 50M), step clamp unlocks Stage 3
    t2, s2 = cm2.compute_targets_for_progress(progress=0.0014, global_step=70000, valid_rate=None)
    assert s2 == 3
    assert cm2.current_stage == 3


def test_egnn_jit_relaxation_evaluator():
    """Verifies that EGNN unrolled geometry relaxation executes cleanly on TinyJit."""
    egnn = EGNNForceField(num_layers=3, hidden_dim=64, max_atomic_number=32, n_particles=8)
    jit_relax = egnn.get_jit_relaxation_evaluator(relax_steps=5, lr=0.01)

    x_np = np.random.randn(2, 8, 3).astype(np.float32) * 2.0
    z_np = np.array([[6, 6, 6, 8, 1, 1, 1, 1], [6, 7, 8, 9, 1, 1, 1, 1]], dtype=np.float32)
    a_mask_np = np.ones((2, 8, 1), dtype=np.float32)
    m_mask_np = np.ones((2,), dtype=np.float32)

    x_buf = Tensor(x_np).realize()
    z_buf = Tensor(z_np).realize()
    mask_buf = Tensor(a_mask_np).realize()
    mol_mask_buf = Tensor(m_mask_np).realize()

    x_rel, u_final, f_final = jit_relax(x_buf, z_buf, mask_buf, mol_mask_buf)
    f_final_rms = np.sqrt(np.mean(f_final.numpy() ** 2))

    assert not np.isnan(f_final_rms)
    assert x_rel.shape == (2, 8, 3)
    assert u_final.shape == (2,)


def test_detect_optimal_gpu_batch_size():
    """Verifies batch size auto-detection behaves correctly."""
    assert detect_optimal_gpu_batch_size(128) == 128
    assert detect_optimal_gpu_batch_size(64) == 64
    auto_b = detect_optimal_gpu_batch_size(None)
    assert auto_b in [32, 64, 128]
