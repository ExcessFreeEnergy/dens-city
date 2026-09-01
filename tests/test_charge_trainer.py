"""
Unit tests for QuantumChargeTrainer, trunk-freezing rule, and autograd gradient flow.
"""

import numpy as np
from tinygrad import Tensor, nn

from dens_city.boltzmann.train_charges import PreprocessedBatch, QuantumChargeTrainer


def test_quantum_charge_trainer_trunk_freezing():
    """
    Verifies the Trunk-Freezing rule:
    Only charge_mlp parameters are registered in the optimizer;
    the 7-layer EGNN trunk and embedding layers remain frozen.
    """
    trainer = QuantumChargeTrainer()
    trainable_params = trainer.charge_params

    # Charge MLP has 2 Linear layers: Linear(128, 128) and Linear(128, 1)
    # Each has weight and bias -> 4 parameters total
    assert len(trainable_params) == 4, f"Expected 4 parameters in charge_mlp, got {len(trainable_params)}"

    all_params = nn.state.get_parameters(trainer.ff)
    # 7 layers * 10 params/layer + 2 embedding params + 4 readout params + 4 charge params = 80 params
    assert len(all_params) > 70
    assert len(trainable_params) < len(all_params)

    # Confirm that embedding and interaction layers are NOT in trainable_params
    embedding_params = set(nn.state.get_parameters(trainer.ff.embedding))
    for p in trainable_params:
        assert p not in embedding_params, "Trunk parameter leaked into trainable parameters!"


def test_quantum_charge_trainer_autograd_flow_and_loss_decrease():
    """
    Verifies that Huber loss + L2 regularization successfully backpropagates
    through the Generalized Born solver into charge_mlp, reducing loss over iterations.
    """
    trainer = QuantumChargeTrainer()
    B, N = 2, 32

    # Synthetic batch
    coords_np = np.zeros((B, N, 3), dtype=np.float32)
    coords_np[0, :3] = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    coords_np[1, :3] = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]

    z_np = np.zeros((B, N), dtype=np.float32)
    z_np[:, :3] = [8, 1, 1]

    mask_np = np.zeros((B, N, 1), dtype=np.float32)
    mask_np[:, :3, :] = 1.0

    bq_np = np.zeros((B, N), dtype=np.float32)
    bq_np[:, :3] = [-0.234, +0.117, +0.117]

    tot_q_np = np.zeros((B, 1, 1), dtype=np.float32)
    vdw_np = np.array([4.02, 4.02], dtype=np.float32)
    expt_np = np.array([-6.30, -6.30], dtype=np.float32)

    cached_h = Tensor.randn(B, N, 128).realize()
    cached_sf = Tensor.randn(B, N, 4).realize()

    batch = PreprocessedBatch(
        coords=Tensor(coords_np).realize(),
        atomic_numbers=Tensor(z_np).realize(),
        atom_mask=Tensor(mask_np).realize(),
        base_charges=Tensor(bq_np).realize(),
        total_charges=Tensor(tot_q_np).realize(),
        vdw_energies=Tensor(vdw_np).realize(),
        expt_energies=Tensor(expt_np).realize(),
        material_names=["water_1", "water_2"],
        num_real_atoms=[3, 3],
        cached_h=cached_h,
        cached_solvent_features=cached_sf,
    )

    losses = []
    orig_head_w = trainer.ff.charge_mlp[2].weight.numpy().copy()
    for step in range(5):
        loss, mae, max_dq = trainer.train_epoch([batch], lr_head=1e-2, lr_trunk=0.0, phase=1)
        losses.append(loss)
        assert max_dq <= 0.2501, f"Max delta_q exceeded tanh bound: {max_dq}"

    # Parameter weights in charge_mlp must have been updated
    assert np.any(trainer.ff.charge_mlp[2].weight.numpy() != orig_head_w)

    # Loss must strictly decrease
    assert losses[-1] < losses[0], f"Loss failed to decrease: {losses}"

    # Test Phase 2: Trunk Unfreezing & Cache Invalidation
    batch.cached_h = None
    orig_trunk_w = trainer.ff.embedding.weight.numpy().copy()
    losses_p2 = []
    for step in range(5):
        loss_p2, mae_p2, max_dq_p2 = trainer.train_epoch([batch], lr_head=1e-3, lr_trunk=1e-4, phase=2)
        losses_p2.append(loss_p2)
        assert max_dq_p2 <= 0.2501

    # In Phase 2, both loss decreases and trunk weights are updated end-to-end
    assert np.any(trainer.ff.embedding.weight.numpy() != orig_trunk_w)
    assert losses_p2[-1] < losses_p2[0], f"Phase 2 loss failed to decrease: {losses_p2}"


def test_dataset_static_shapes():
    """
    Verifies that all preprocessed batches have identical static shapes
    to prevent dynamic shape JIT kernel recompilation on the GPU.
    """
    trainer = QuantumChargeTrainer()
    batches = trainer.load_dataset()
    assert len(batches) > 0

    expected_B = trainer.config.batch_size
    expected_N = trainer.config.n_particles

    for i, b in enumerate(batches):
        assert b.coords.shape == (expected_B, expected_N, 3), f"Batch {i} coords shape mismatch: {b.coords.shape}"
        assert b.atomic_numbers.shape == (expected_B, expected_N), (
            f"Batch {i} Z shape mismatch: {b.atomic_numbers.shape}"
        )
        assert b.atom_mask.shape == (expected_B, expected_N, 1), f"Batch {i} mask shape mismatch: {b.atom_mask.shape}"
        assert b.cached_h.shape == (expected_B, expected_N, 128), (
            f"Batch {i} cached_h shape mismatch: {b.cached_h.shape}"
        )
        assert b.cached_solvent_features.shape == (expected_B, expected_N, 4), (
            f"Batch {i} sf shape mismatch: {b.cached_solvent_features.shape}"
        )


def test_static_dataset_and_sequential_jit_eval():
    """
    Verifies that StaticFreeSolvDataset packs all molecules contiguously into GPU memory
    and sequential JIT evaluation covers 100% of the dataset deterministically without shape errors.
    """
    trainer = QuantumChargeTrainer()
    static_ds = trainer.load_static_dataset()

    expected_B = trainer.config.batch_size
    expected_N = trainer.config.n_particles
    assert static_ds.num_real_molecules > 600
    assert static_ds.total_padded_molecules % expected_B == 0

    assert static_ds.coords.shape == (static_ds.total_padded_molecules, expected_N, 3)
    assert static_ds.atomic_numbers.shape == (static_ds.total_padded_molecules, expected_N)
    assert static_ds.atom_mask.shape == (static_ds.total_padded_molecules, expected_N, 1)
    assert static_ds.cached_h.shape == (static_ds.total_padded_molecules, expected_N, 128)
    assert static_ds.solvent_features.shape == (static_ds.total_padded_molecules, expected_N, 4)

    # Test sequential JIT evaluation
    trainer.dataset = static_ds
    mae, rmse, max_err, preds = trainer.evaluate()
    assert mae > 0.0, f"Expected positive MAE, got {mae}"
    assert rmse >= mae, f"Expected RMSE >= MAE, got RMSE={rmse}, MAE={mae}"
    assert len(preds) == static_ds.num_real_molecules, (
        f"Expected {static_ds.num_real_molecules} predictions, got {len(preds)}"
    )
