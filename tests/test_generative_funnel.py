"""
End-to-end integration test for the 3-stage Generative Molecular Funnel Pipeline.
"""

from pathlib import Path

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.bijectors import Base2CartesianFlow
from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.boltzmann.lbfgs import BatchedLBFGS
from dens_city.cdft.cdft import BatchedTinyCDFT
from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.sampler import SwarmCandidateSampler
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.utils.funnel_ranker import FunnelRanker
from dens_city.utils.pipeline import MaterialPipelineResult, PipelineStatus


def test_generative_funnel_e2e(tmp_path):
    spec_path = Path(__file__).resolve().parent / "data" / "conjugated_oled_semiconductors.yaml"
    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

    # 1. Sampler
    policy = MolecularSwarmPolicy(hidden_size=128, recurrent=False)
    sampler = SwarmCandidateSampler(
        policy=policy,
        spec_path=spec_path,
        num_envs=8,
        device="cpu",
    )

    candidate_batch = sampler.sample_candidates(
        total_candidates=8,
        max_rollout_steps=3000,
        temperature=1.0,
    )

    assert candidate_batch.num_candidates > 0
    mol_batch = candidate_batch.slice_molecular_batch(start_idx=0, count=8, batch_size=8)

    # 2. cDFT
    batched_cdft = BatchedTinyCDFT(batch=mol_batch, n_grid=64, learning_rate=0.05)
    cdft_losses = batched_cdft.solve(steps=5, verbose=False)
    p_walls = batched_cdft.get_wall_contact_pressures()

    # 3. L-BFGS Relaxation & Boltzmann Flow
    energy_fn = MicroscopicEnergy(material=mol_batch, pad_to_128=True)
    lbfgs = BatchedLBFGS(m=4, max_iter=10, grad_tol=1e-3, lr=1.0, verbose=False)
    coords_chunk = np.zeros((8, candidate_batch.n_particles, 3), dtype=np.float32)
    coords_chunk[:8] = candidate_batch.coords[:8]
    lbfgs_res = lbfgs.minimize(energy_fn, coords_chunk, atom_mask=mol_batch.atom_mask)
    assert len(lbfgs_res.final_energies) == 8

    flow = Base2CartesianFlow(n_atoms=128, n_layers=2, hidden_dim=32)
    generator = BoltzmannGenerator(flow=flow, energy_fn=energy_fn, batch_size=8)
    generator.train(steps=5, batch_size=8, verbose=False)

    stats = generator.evaluate_conformer_ensemble(n_samples=8)
    assert len(stats["mean_energy"]) == 8
    assert len(stats["mean_log_px"]) == 8

    # 4. Stage 4: EGNN Quantum-Surrogate Filter
    egnn = EGNNForceField(num_layers=3, hidden_dim=64, max_atomic_number=128, n_particles=candidate_batch.n_particles)
    x_tensor = Tensor(lbfgs_res.x_relaxed)
    z_tensor = Tensor(candidate_batch.atomic_numbers[:8])
    mask_tensor = Tensor(candidate_batch.atom_mask[:8]).reshape(8, candidate_batch.n_particles, 1)
    mol_mask_tensor = Tensor.ones(8)

    u_total = egnn.compute_energy(x_tensor, z_tensor, mask_tensor, mol_mask_tensor)
    u_total.sum().backward()

    u_np = u_total.numpy().astype(np.float32)
    grad_tensor = x_tensor.grad if x_tensor.grad is not None else Tensor.zeros_like(x_tensor)
    f_np = (-grad_tensor * mask_tensor).numpy().astype(np.float32)
    num_real = np.maximum(1.0, candidate_batch.atom_mask[:8].reshape(8, -1).sum(axis=1))
    f_rms_np = np.sqrt(np.sum(f_np**2, axis=(1, 2)) / num_real).astype(np.float32)

    x_tensor.grad = None
    del x_tensor, z_tensor, mask_tensor, mol_mask_tensor, u_total, grad_tensor

    # 5. Pipeline Results
    pipeline_results = []
    for i in range(candidate_batch.num_candidates):
        meta = candidate_batch.metadata[i]
        res = MaterialPipelineResult(
            material_name=meta["name"],
            status=PipelineStatus.SUCCESS.value,
            runtime_seconds=0.1,
            num_sites=meta["num_atoms"],
            wall_pressure_bar=float(p_walls[i]),
            cdft_final_loss=float(cdft_losses[-1]),
            bg_log_likelihood=float(stats["mean_log_px"][i]),
            bg_energy_mean=float(stats["mean_energy"][i]),
            bg_energy_var=float(stats["var_energy"][i]),
            egnn_energy=float(u_np[i]),
            egnn_force_rms=float(f_rms_np[i]),
            solvation_free_energy_kcal_mol=float(meta["omega_solv"]),
        )
        pipeline_results.append(res)

    # 6. Funnel Ranker & Pareto Export
    ranker = FunnelRanker(target_spec=target_spec)
    ranked = ranker.rank_candidates(candidate_batch.metadata, pipeline_results)
    assert 0 < len(ranked) <= candidate_batch.num_candidates

    summary = ranker.export_results(ranked, out_dir=tmp_path, top_k=5)
    assert summary["top_k_exported"] <= 5
    assert Path(summary["csv_path"]).exists()
    assert Path(summary["report_path"]).exists()
