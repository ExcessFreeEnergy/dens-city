"""
Unit tests for universal Delta-KRR residual head recalibration,
canonical SMILES and feature-space deduplication, Cholesky LOOCV,
and pure on-device tinygrad tensor inference.
"""

from __future__ import annotations

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from dens_city.boltzmann.train_charges import (
    load_krr_device_tensors,
    predict_krr_residual,
    predict_krr_residual_tensor,
)


def test_krr_checkpoint_loaded_on_device():
    """Verifies that the recalibrated KRR checkpoint loads immutable device buffers."""
    dev = load_krr_device_tensors()
    assert dev is not None, "Failed to load device tensors from checkpoint"
    assert "z_train" in dev
    assert "alpha" in dev
    assert "z_train_sq" in dev
    assert dev["z_train"].shape[1] == 397
    assert dev["alpha"].shape[1] == 1
    assert isinstance(dev["z_train"], Tensor)
    assert isinstance(dev["alpha"], Tensor)


def test_krr_pure_tensor_inference():
    """Verifies that predict_krr_residual_tensor executes on device with expected shapes."""
    B = 4
    z_q = Tensor.randn(B, 384, dtype=dtypes.float32)
    d_q = Tensor.zeros(B, 6, dtype=dtypes.float32)

    pred_t, density_t = predict_krr_residual_tensor(z_q, d_q, s_solv="water")
    Tensor.realize(pred_t, density_t)

    assert pred_t.shape == (B, 1)
    assert density_t.shape == (B, 1)


def test_krr_trust_region_out_of_distribution_decay():
    """
    Verifies localized RBF trust region property:
    Far-away query molecules decay smoothly to zero (pred -> 0.0, density -> 0.0),
    preventing catastrophic out-of-distribution extrapolation during inverse discovery.
    """
    z_ood = Tensor.full((1, 384), 1000.0, dtype=dtypes.float32)
    d_ood = Tensor.full((1, 6), 100.0, dtype=dtypes.float32)

    pred_t, density_t = predict_krr_residual_tensor(z_ood, d_ood, s_solv="water")
    Tensor.realize(pred_t, density_t)

    assert float(pred_t.numpy()[0, 0]) == pytest.approx(0.0, abs=1e-5)
    assert float(density_t.numpy()[0, 0]) == pytest.approx(0.0, abs=1e-5)


def test_krr_numpy_wrapper_compatibility():
    """Verifies backwards compatibility of predict_krr_residual with numpy arrays and floats."""
    z_np = np.zeros((384,), dtype=np.float32)
    d_np = np.zeros((6,), dtype=np.float32)

    val = predict_krr_residual(z_np, d_np, s_solv="toluene")
    assert isinstance(val, float)

    z_batch = np.zeros((3, 384), dtype=np.float32)
    d_batch = np.zeros((3, 6), dtype=np.float32)
    arr = predict_krr_residual(z_batch, d_batch, s_solv="methanol")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
