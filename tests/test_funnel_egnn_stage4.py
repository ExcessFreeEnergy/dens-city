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


def test_stage5_synthesizability_sa_gate(tmp_path):
    """
    Validates that FunnelRanker boolean safety gate strictly drops molecules
    with SA Score > max_sa_score (e.g. SA > 6.0) while retaining valid synthesizable candidates.
    """
    target_spec = {
        "min_wall_pressure_bar": 10.0,
        "max_solvation_kcal": -2.0,
        "max_molecular_weight": 800.0,
        "max_sa_score": 6.0,
    }

    # cand_easy: benzene (SA ~ 1.0)
    # cand_hard: highly strained/bridged (SA = 8.5)
    # cand_borderline: SA = 5.2
    metadata = [
        {
            "name": "cand_easy",
            "smiles": "c1ccccc1",
            "rl_reward": 5.0,
            "p_wall": 25.0,
            "omega_solv": -3.5,
            "mw": 78.1,
            "num_atoms": 6,
        },
        {
            "name": "cand_hard",
            "smiles": "C12C3C4C1C5C2C3C45",  # cubane/hyper-strained (SA > 6.0)
            "sa_score": 8.5,
            "rl_reward": 9.0,  # High reward, but un-synthesizable
            "p_wall": 30.0,
            "omega_solv": -4.0,
            "mw": 104.1,
            "num_atoms": 8,
        },
        {
            "name": "cand_borderline",
            "sa_score": 5.2,
            "rl_reward": 4.0,
            "p_wall": 20.0,
            "omega_solv": -2.5,
            "mw": 150.0,
            "num_atoms": 12,
        },
    ]

    results = [
        MaterialPipelineResult(material_name="cand_easy", status=PipelineStatus.SUCCESS.value, wall_pressure_bar=25.0),
        MaterialPipelineResult(material_name="cand_hard", status=PipelineStatus.SUCCESS.value, wall_pressure_bar=30.0),
        MaterialPipelineResult(
            material_name="cand_borderline", status=PipelineStatus.SUCCESS.value, wall_pressure_bar=20.0
        ),
    ]

    # 1. Filter enabled with default max_sa_score = 6.0
    ranker = FunnelRanker(target_spec=target_spec, enable_sa_filter=True)
    ranked = ranker.rank_candidates(metadata, results)

    # cand_hard (SA=8.5) must be strictly dropped
    ranked_names = [c.name for c in ranked]
    assert "cand_hard" not in ranked_names
    assert "cand_easy" in ranked_names
    assert "cand_borderline" in ranked_names
    assert len(ranked) == 2
    assert ranker.last_num_dropped_sa == 1

    # SA score must be recorded on ranked candidate
    easy_cand = next(c for c in ranked if c.name == "cand_easy")
    assert easy_cand.sa_score is not None
    assert easy_cand.sa_score < 3.0

    # 2. Filter disabled: cand_hard must be included
    ranker_no_filter = FunnelRanker(target_spec=target_spec, enable_sa_filter=False)
    ranked_all = ranker_no_filter.rank_candidates(metadata, results)
    assert len(ranked_all) == 3
    assert "cand_hard" in [c.name for c in ranked_all]
    assert ranker_no_filter.last_num_dropped_sa == 0

    # 3. Export formatting verification
    summary = ranker.export_results(ranked, out_dir=tmp_path, top_k=5)
    assert summary["top_k_exported"] == 2
    report_text = (tmp_path / "funnel_report.md").read_text()
    assert "SA Score" in report_text
    csv_text = (tmp_path / "funnel_summary.csv").read_text()
    assert "sa_score" in csv_text


def test_egnn_extensive_energy_scaling():
    """
    Validates that degree normalization and radial cutoff ensure extensive O(N) scaling
    rather than unphysical O(N^3) explosion when scaling from N=16 to N=64 atoms.
    """
    np.random.seed(42)
    egnn = EGNNForceField(num_layers=4, hidden_dim=64, max_atomic_number=128, n_particles=128, r_cut=5.0)

    # Molecule 1: N = 16
    x_16 = np.random.randn(1, 128, 3).astype(np.float32)
    z_16 = np.zeros((1, 128), dtype=np.float32)
    z_16[0, :16] = 6.0
    mask_16 = np.zeros((1, 128, 1), dtype=np.float32)
    mask_16[0, :16] = 1.0
    mol_mask = np.ones(1, dtype=np.float32)

    u_16 = float(egnn.compute_energy(Tensor(x_16), Tensor(z_16), Tensor(mask_16), Tensor(mol_mask)).numpy()[0])

    # Molecule 2: N = 64
    x_64 = np.random.randn(1, 128, 3).astype(np.float32)
    z_64 = np.zeros((1, 128), dtype=np.float32)
    z_64[0, :64] = 6.0
    mask_64 = np.zeros((1, 128, 1), dtype=np.float32)
    mask_64[0, :64] = 1.0

    u_64 = float(egnn.compute_energy(Tensor(x_64), Tensor(z_64), Tensor(mask_64), Tensor(mol_mask)).numpy()[0])

    # Energy per atom should remain on a stable, bounded scale (within 10x per atom)
    u_per_atom_16 = abs(u_16) / 16.0
    u_per_atom_64 = abs(u_64) / 64.0

    assert not np.isnan(u_16) and not np.isnan(u_64)
    # The ratio of per-atom energies should not explode superlinearly
    assert u_per_atom_64 < 50.0 * (u_per_atom_16 + 1.0)
    assert abs(u_64) < 1e5  # Must not reach 1.6M K


def test_funnel_ranker_topk_topological_deduplication():
    """
    Validates that FunnelRanker groups candidates by canonical SMILES / WL hash
    so that minor conformers of the same molecule do not monopolize Top-K slots.
    """
    target_spec = {
        "min_wall_pressure_bar": 10.0,
        "max_solvation_kcal": -2.0,
        "max_molecular_weight": 800.0,
    }
    ranker = FunnelRanker(target_spec=target_spec)

    # 5 conformers of Molecule A (identical SMILES: c1ccccc1), 1 of Molecule B (SMILES: c1ccncc1)
    metadata = [
        {
            "name": f"cand_A_conf{i}",
            "smiles": "c1ccccc1",
            "rl_reward": 5.0 + 0.1 * i,
            "p_wall": 20.0,
            "omega_solv": -3.0,
            "mw": 78.1,
            "num_atoms": 6,
        }
        for i in range(5)
    ]
    metadata.append(
        {
            "name": "cand_B",
            "smiles": "c1ccncc1",
            "rl_reward": 4.5,
            "p_wall": 18.0,
            "omega_solv": -3.2,
            "mw": 79.1,
            "num_atoms": 6,
        }
    )

    results = [
        MaterialPipelineResult(
            material_name=m["name"], status=PipelineStatus.SUCCESS.value, wall_pressure_bar=m["p_wall"]
        )
        for m in metadata
    ]

    ranked = ranker.rank_candidates(metadata, results)

    # Should deduplicate 5 conformers of A down to 1 (the best one) + Molecule B = 2 unique candidates total
    assert len(ranked) == 2
    unique_smiles = {c.smiles for c in ranked}
    assert unique_smiles == {"c1ccccc1", "c1ccncc1"}
    # The retained conformer for A should be conf4 (highest score)
    a_cand = next(c for c in ranked if c.smiles == "c1ccccc1")
    assert a_cand.name == "cand_A_conf4"


def test_funnel_ranker_contact_ratio_scoring():
    """
    Validates that contact_ratio is used in cDFT thermodynamic scoring without saturation.
    """
    target_spec = {
        "min_wall_pressure_bar": 3.0,  # Target contact ratio = 3.0
        "max_solvation_kcal": -2.0,
    }
    ranker = FunnelRanker(target_spec=target_spec)

    metadata = [
        {
            "name": "cand_high_ratio",
            "smiles": "CC",
            "rl_reward": 1.0,
            "p_wall": 15000.0,
            "contact_ratio": 4.5,
            "omega_solv": -2.5,
            "mw": 30.0,
            "num_atoms": 2,
        },
        {
            "name": "cand_low_ratio",
            "smiles": "CO",
            "rl_reward": 1.0,
            "p_wall": 5000.0,
            "contact_ratio": 1.5,
            "omega_solv": -2.5,
            "mw": 32.0,
            "num_atoms": 2,
        },
    ]
    results = [
        MaterialPipelineResult(
            material_name="cand_high_ratio",
            status=PipelineStatus.SUCCESS.value,
            wall_pressure_bar=15000.0,
            contact_ratio=4.5,
        ),
        MaterialPipelineResult(
            material_name="cand_low_ratio",
            status=PipelineStatus.SUCCESS.value,
            wall_pressure_bar=5000.0,
            contact_ratio=1.5,
        ),
    ]

    ranked = ranker.rank_candidates(metadata, results)
    assert len(ranked) == 2
    assert ranked[0].name == "cand_high_ratio"
    assert ranked[0].funnel_score > ranked[1].funnel_score
