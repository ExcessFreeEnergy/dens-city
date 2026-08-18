import numpy as np

from dens_city.envs.env import DensCityFluidEnv


def test_pufferlib_env_step_reset():
    env = DensCityFluidEnv(num_envs=4, seed=42)
    obs, info = env.reset(0)

    assert obs.shape == (771,)
    assert not np.isnan(obs).any()

    # Step environment
    action = np.array([0.5, 0.0, -0.2], dtype=np.float32)
    next_obs, rew, done, _, info = env.step(action, env_idx=0)

    assert next_obs.shape == (771,)
    assert isinstance(rew, float)
    assert not np.isnan(next_obs).any()
    assert 0.0 <= info["current_filling"] <= 2.0


def test_pufferlib_env_throughput():
    env = DensCityFluidEnv(num_envs=16, seed=123)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Run 1000 steps across 16 envs
    for _ in range(200):
        for i in range(16):
            env.step(action, env_idx=i)

    assert env.current_filling >= 0.0
