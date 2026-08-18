import os

import torch

from dens_city.envs.train import DensNeuralFunctional, train_unified


def test_neural_functional_forward():
    model = DensNeuralFunctional()
    dummy_obs = torch.randn(4, 771)
    action_mean, value, c1_pred, rho_h_pred = model(dummy_obs)

    assert action_mean.shape == (4, 3)
    assert value.shape == (4, 1)
    assert c1_pred.shape == (4, 256)
    assert rho_h_pred.shape == (4, 256)


def test_unified_training_quick(tmp_path):
    save_file = str(tmp_path / "test_func.pt")
    # Quick 1000-step training verification
    train_unified(total_timesteps=500, num_envs=2, save_path=save_file)

    assert os.path.exists(save_file)
    data = torch.load(save_file)
    assert "state_dict" in data
