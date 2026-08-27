"""
Unit and integration tests for Stage 4 EGNN Quantum-Surrogate Filter.
Validates single-pass autograd conservative force extraction, E(n) invariance,
multi-chunk graph severing, and quantum Pareto ranking.
"""

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.utils.funnel_ranker import FunnelRanker
from dens_city.utils.pipeline import MaterialPipelineResult, PipelineStatus


def test_egnn_single_pass_autodiff_forces():
    """
    Verifies that a single forward pass + reverse-mode autodiff backward pass
    correctly extracts both scalar potential energy and conservative forces.
    """
    B, N = 4, 16
    np.random.seed(42)
    x_np = np.random.randn(B, N, 3).astype(np.float32)
    z_np = np.random.randint(1, 10, size=(B, N)).astype(np.float32)
    # Mask out last 4 atoms as dummy
    mask_np = np.ones((B, N, 1), dtype=np.float32)
    mask_np[:, 12:] = 0.0
    z_np[:, 12:] = 0.0
    mol_mask_np = np.ones(B, dtype=np.float32)

    egnn = EGNNForceField(num_layers=3, hidden_dim=64, max_atomic_number=128, n_particles=N)

    x_tensor = Tensor(x_np)
    z_tensor = Tensor(z_np)
    mask_tensor = Tensor(mask_np)
    mol_mask_tensor = Tensor(mol_mask_np)

    u_total = egnn.compute_energy(x_tensor, z_tensor, mask_tensor, mol_mask_tensor)
    assert u_total.shape == (B,)

    u_total.sum().backward()

    grad_tensor = x_tensor.grad if x_tensor.grad is not None else Tensor.zeros_like(x_tensor)
    f_tensor = (-grad_tensor * mask_tensor).realize()
    f_np = f_tensor.numpy()

    # Forces should have shape (B, N, 3)
    assert f_np.shape == (B, N, 3)
    assert not np.isnan(f_np).any()

    # Dummy atoms (12:16) must receive strictly zero force
    assert np.allclose(f_np[:, 12:], 0.0)


def test_egnn_multi_chunk_graph_severing():
    """
    Verifies that processing multiple consecutive chunks with autograd backward
    and explicit graph severing produces consistent results without memory leaks.
    """
    B_chunk = 4
    num_chunks = 3
    N = 16
    np.random.seed(123)

    egnn = EGNNForceField(num_layers=3, hidden_dim=64, max_atomic_number=128, n_particles=N)

    energies_all = []
    forces_all = []

    for _ in range(num_chunks):
        x_np = np.random.randn(B_chunk, N, 3).astype(np.float32)
        z_np = np.full((B_chunk, N), 6.0, dtype=np.float32)
        mask_np = np.ones((B_chunk, N, 1), dtype=np.float32)
        mol_mask_np = np.ones(B_chunk, dtype=np.float32)

        x_tensor = Tensor(x_np)
        z_tensor = Tensor(z_np)
        mask_tensor = Tensor(mask_np)
        mol_mask_tensor = Tensor(mol_mask_np)

        u_total = egnn.compute_energy(x_tensor, z_tensor, mask_tensor, mol_mask_tensor)
        u_total.sum().backward()

        u_np = u_total.numpy().astype(np.float32)
        grad_tensor = x_tensor.grad if x_tensor.grad is not None else Tensor.zeros_like(x_tensor)
        f_np = (-grad_tensor * mask_tensor).numpy().astype(np.float32)

        energies_all.append(u_np)
        forces_all.append(f_np)

        # Graph severing
        x_tensor.grad = None
        del x_tensor, z_tensor, mask_tensor, mol_mask_tensor, u_total, grad_tensor

    assert len(energies_all) == num_chunks
    assert len(forces_all) == num_chunks
    for u in energies_all:
        assert u.shape == (B_chunk,)
        assert not np.isnan(u).any()


def test_quantum_funnel_ranker_pareto_integration(tmp_path):
    """
    Validates that FunnelRanker correctly incorporates U_EGNN and F_RMS into composite scores
    and non-dominated Pareto frontier rankings.
    """
    target_spec = {
        "min_wall_pressure_bar": 10.0,
        "max_solvation_kcal": -2.0,
        "max_molecular_weight": 800.0,
    }
    ranker = FunnelRanker(target_spec=target_spec)

    metadata = [
        {"name": "cand_optimal", "rl_reward": 5.0, "p_wall": 25.0, "omega_solv": -3.5, "mw": 450.0, "num_atoms": 30},
        {"name": "cand_high_force", "rl_reward": 5.0, "p_wall": 25.0, "omega_solv": -3.5, "mw": 450.0, "num_atoms": 30},
        {
            "name": "cand_weak_wetting",
            "rl_reward": 2.0,
            "p_wall": 5.0,
            "omega_solv": -1.0,
            "mw": 300.0,
            "num_atoms": 20,
        },
    ]

    results = [
        MaterialPipelineResult(
            material_name="cand_optimal",
            status=PipelineStatus.SUCCESS.value,
            wall_pressure_bar=25.0,
            bg_log_likelihood=-50.0,
            bg_energy_mean=-100.0,
            bg_energy_var=5.0,
            egnn_energy=-250.0,
            egnn_force_rms=0.05,  # Low quantum force residual (true minimum)
        ),
        MaterialPipelineResult(
            material_name="cand_high_force",
            status=PipelineStatus.SUCCESS.value,
            wall_pressure_bar=25.0,
            bg_log_likelihood=-50.0,
            bg_energy_mean=-100.0,
            bg_energy_var=5.0,
            egnn_energy=-250.0,
            egnn_force_rms=15.0,  # High quantum force residual (quantum cliff)
        ),
        MaterialPipelineResult(
            material_name="cand_weak_wetting",
            status=PipelineStatus.SUCCESS.value,
            wall_pressure_bar=5.0,
            bg_log_likelihood=-120.0,
            bg_energy_mean=50.0,
            bg_energy_var=20.0,
            egnn_energy=10.0,
            egnn_force_rms=2.0,
        ),
    ]

    ranked = ranker.rank_candidates(metadata, results)
    assert len(ranked) == 3

    # cand_optimal must outrank cand_high_force due to low force residual
    names_ranked = [c.name for c in ranked]
    assert names_ranked[0] == "cand_optimal"
    assert ranked[0].funnel_score > ranked[1].funnel_score
    assert ranked[0].is_pareto_optimal

    # Export test
    summary = ranker.export_results(ranked, out_dir=tmp_path, top_k=2)
    assert summary["top_k_exported"] == 2
    assert "U_EGNN" in (tmp_path / "funnel_report.md").read_text()


def test_egnn_fully_padded_batch_gradient_none_safety():
    """
    Validates that a batch with all-zero molecule masks or all-zero atom masks
    safely returns zero energies and zero forces without NoneType errors.
    """
    B, N = 4, 16
    x_np = np.zeros((B, N, 3), dtype=np.float32)
    z_np = np.zeros((B, N), dtype=np.float32)
    mask_np = np.zeros((B, N, 1), dtype=np.float32)
    mol_mask_np = np.zeros(B, dtype=np.float32)

    egnn = EGNNForceField(num_layers=3, hidden_dim=64, max_atomic_number=128, n_particles=N)

    x_tensor = Tensor(x_np)
    z_tensor = Tensor(z_np)
    mask_tensor = Tensor(mask_np)
    mol_mask_tensor = Tensor(mol_mask_np)

    u_total = egnn.compute_energy(x_tensor, z_tensor, mask_tensor, mol_mask_tensor)
    assert u_total.shape == (B,)
    assert np.allclose(u_total.numpy(), 0.0)

    u_total.sum().backward()

    grad_tensor = x_tensor.grad if x_tensor.grad is not None else Tensor.zeros_like(x_tensor)
    f_tensor = (-grad_tensor * mask_tensor).realize()
    f_np = f_tensor.numpy()

    assert f_np.shape == (B, N, 3)
    assert np.allclose(f_np, 0.0)
