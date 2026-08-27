#!/usr/bin/env python3
"""
High-Throughput 3-Stage Generative Molecular Funnel Pipeline for dens-city.
Couples:
1. Stage 1: PufferLib RL Swarm Training & C-Native Vectorized Candidate Sampling
2. Stage 2: In-Memory Contiguous Array Streaming & Zero-Copy GPU Batching
3. Stage 3: Coupled tinygrad cDFT Thermodynamics + Boltzmann Generator Refinement + Pareto Selection
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tinygrad import Tensor

from dens_city.boltzmann.bijectors import Base2CartesianFlow
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.boltzmann.lbfgs import BatchedLBFGS
from dens_city.cdft.cdft import BatchedTinyCDFT
from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.sampler import SwarmCandidateSampler
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.trainer import SwarmPuffeRLTrainer, VectorizedSwarmEnv
from dens_city.utils.funnel_ranker import FunnelRanker
from dens_city.utils.pipeline import MaterialPipelineResult, PipelineStatus


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-Stage Generative Molecular Funnel (RL Swarm -> Contiguous GPU Batching -> Boltzmann Refinement)"
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="tests/data/conjugated_oled_semiconductors.yaml",
        help="Path to material specification YAML",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=25000,
        help="Number of RL curriculum training steps before sampling (default: 25000)",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=512,
        help="Number of candidates to generate from the trained policy (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for GPU cDFT and Boltzmann Generator screening (default: 512)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of final refined candidates to export (default: 20)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/funnel_results",
        help="Output directory for final exported candidates and reports",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to existing policy checkpoint .pt file",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel C-FFI environments (default: 16)",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Use recurrent MinGRU backbone instead of MLP",
    )
    parser.add_argument(
        "--cdft-steps",
        type=int,
        default=50,
        help="Number of variational cDFT solver steps per batch (default: 50)",
    )
    parser.add_argument(
        "--bg-steps",
        type=int,
        default=30,
        help="Number of Boltzmann Generator training steps per batch (default: 30)",
    )
    parser.add_argument(
        "--bg-samples",
        type=int,
        default=32,
        help="Number of 3D conformer samples to evaluate per candidate (default: 32)",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=50,
        help="Number of batched L-BFGS Quasi-Newton geometry relaxation steps on GPU (default: 50)",
    )
    parser.add_argument(
        "--lbfgs-tol",
        type=float,
        default=1e-3,
        help="RMS force convergence threshold for L-BFGS relaxation (default: 1e-3)",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Specification YAML not found at {spec_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir) / spec_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

    print("=" * 80)
    print(f"=== 3-STAGE GENERATIVE MOLECULAR FUNNEL: {spec_data.get('group_name', spec_path.stem)} ===")
    print(f"Target Wall Pressure: >= {target_spec.get('min_wall_pressure_bar', 15.0):.1f} bar")
    print(f"Target Max Solvation: <= {target_spec.get('max_solvation_kcal', -3.0):.1f} kcal/mol")
    print(f"Target Max Weight:   <= {target_spec.get('max_molecular_weight', 850.0):.1f} amu")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------
    # STAGE 1: Policy Training (or Checkpoint Load)
    # -------------------------------------------------------------
    policy = MolecularSwarmPolicy(
        hidden_size=256,
        recurrent=args.recurrent,
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\n[Stage 1] Loading pre-trained policy checkpoint from {args.checkpoint}...")
        policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"\n[Stage 1] Training RL Molecular Swarm Policy ({args.train_steps} timesteps, {args.num_envs} envs)...")
        vec_env = VectorizedSwarmEnv(
            num_envs=args.num_envs,
            target_spec=target_spec,
            seed=42,
        )
        trainer = SwarmPuffeRLTrainer(
            vec_env=vec_env,
            policy=policy,
            final_target_spec=target_spec,
            total_timesteps=args.train_steps,
            horizon=16,
            minibatch_size=64,
            learning_rate=3e-4,
            device=device,
        )

        epochs = max(1, args.train_steps // (args.num_envs * 16))
        for epoch in range(1, epochs + 1):
            metrics = trainer.train_epoch()
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(
                    f"  Epoch {epoch:03d}/{epochs:03d} | "
                    f"Step {int(metrics['global_step']):6d} | "
                    f"Score: {metrics['env/score']:+6.2f} | "
                    f"Valid Rate: {metrics['env/valid_rate'] * 100:5.1f}% | "
                    f"P_wall: {metrics['env/p_wall']:5.1f} bar | "
                    f"SPS: {metrics['SPS']:5.0f}"
                )
        vec_env.close()

        ckpt_path = out_dir / "trained_policy.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"  Saved trained policy checkpoint to {ckpt_path}")

    # -------------------------------------------------------------
    # STAGE 2: High-Throughput C-Native Candidate Sampling
    # -------------------------------------------------------------
    print(f"\n[Stage 2] Sampling {args.num_candidates} likely candidates via C-FFI engine...")
    t0_sample = time.perf_counter()
    sampler = SwarmCandidateSampler(
        policy=policy,
        spec_path=spec_path,
        num_envs=args.num_envs,
        device=device,
    )
    candidate_batch = sampler.sample_candidates(
        total_candidates=args.num_candidates,
        max_rollout_steps=max(50000, args.num_candidates * 50),
        temperature=1.0,
    )
    t_sample = time.perf_counter() - t0_sample
    print(
        f"  Generated {candidate_batch.num_candidates} valid candidates in {t_sample:.2f}s "
        f"({candidate_batch.num_candidates / max(1e-3, t_sample):.1f} molecules/s) directly in contiguous host memory!"
    )

    if candidate_batch.num_candidates == 0:
        print("Error: No valid candidate molecules were generated. Check specification parameters.")
        sys.exit(1)

    # -------------------------------------------------------------
    # STAGE 3: High-Throughput cDFT & Boltzmann Generative Screening
    # -------------------------------------------------------------
    print(
        f"\n[Stage 3] Screening {candidate_batch.num_candidates} candidates on GPU via Batched cDFT + Boltzmann Generator..."
    )
    t0_gpu = time.perf_counter()

    batch_size = args.batch_size
    num_chunks = max(1, math.ceil(candidate_batch.num_candidates / batch_size))
    pipeline_results: List[MaterialPipelineResult] = []

    for chunk_idx in range(num_chunks):
        start_i = chunk_idx * batch_size
        count_i = min(batch_size, candidate_batch.num_candidates - start_i)
        end_i = start_i + count_i
        mol_batch = candidate_batch.slice_molecular_batch(start_idx=start_i, count=count_i, batch_size=batch_size)

        # 1. Batched 1D cDFT Solver
        batched_cdft = BatchedTinyCDFT(
            batch=mol_batch,
            n_grid=128,
            learning_rate=0.02,
        )
        cdft_losses = batched_cdft.solve(steps=args.cdft_steps, verbose=False)
        cdft_pressures = batched_cdft.get_wall_contact_pressures()
        cdft_gammas = batched_cdft.get_excess_adsorptions()

        # 2. Potential Energy Surface Setup & Batched L-BFGS Quasi-Newton Geometry Relaxation
        energy_fn = MicroscopicEnergy(material=mol_batch, pad_to_128=True)
        coords_chunk = np.zeros((batch_size, candidate_batch.n_particles, 3), dtype=np.float32)
        coords_chunk[:count_i] = candidate_batch.coords[start_i:end_i]

        if args.lbfgs_steps > 0:
            lbfgs = BatchedLBFGS(
                max_iter=args.lbfgs_steps,
                grad_tol=args.lbfgs_tol,
                m=6,
                lr=1.0,
                verbose=False,
            )
            lbfgs_res = lbfgs.minimize(
                energy_fn=energy_fn,
                x_init=coords_chunk,
                atom_mask=mol_batch.atom_mask,
            )
            u_evaluated = lbfgs_res.final_energies
        else:
            u_evaluated = energy_fn(Tensor(coords_chunk)).numpy()

        # 3. Batched Boltzmann Generator & Normalizing Flow
        flow = Base2CartesianFlow(n_atoms=128, n_layers=4, hidden_dim=64)
        generator = BoltzmannGenerator(
            flow=flow,
            energy_fn=energy_fn,
            prior=None,
            batch_size=batch_size,
        )
        generator.train(steps=args.bg_steps, batch_size=batch_size, verbose=False)

        # 4. Evaluate exact Normalizing Flow log-likelihood & conformer ensemble
        conformer_stats = generator.evaluate_conformer_ensemble(n_samples=args.bg_samples)
        var_energies = conformer_stats["var_energy"]
        mean_log_pxs = conformer_stats["mean_log_px"]

        for local_i in range(count_i):
            global_i = start_i + local_i
            meta = candidate_batch.metadata[global_i]
            res = MaterialPipelineResult(
                material_name=meta["name"],
                status=PipelineStatus.SUCCESS.value,
                runtime_seconds=0.0,
                num_sites=meta["num_atoms"],
                wall_pressure_bar=float(cdft_pressures[local_i]),
                excess_adsorption_a2=float(cdft_gammas[local_i]),
                cdft_final_loss=float(cdft_losses[-1]) if cdft_losses else 0.0,
                bg_log_likelihood=float(mean_log_pxs[local_i]),
                bg_energy_mean=float(u_evaluated[local_i]),
                bg_energy_var=float(var_energies[local_i]),
                solvation_free_energy_kcal_mol=float(meta["omega_solv"]),
            )
            pipeline_results.append(res)

        print(f"  Processed Batch Chunk {chunk_idx + 1}/{num_chunks} ({count_i} molecules on device)")

    t_gpu = time.perf_counter() - t0_gpu
    print(f"  GPU screening finished in {t_gpu:.2f}s ({candidate_batch.num_candidates / max(1e-3, t_gpu):.1f} mol/s)")

    # -------------------------------------------------------------
    # STAGE 4: Multi-Objective Funnel Ranking & Pareto Export
    # -------------------------------------------------------------
    print(f"\n[Stage 4] Ranking and sorting Pareto frontier (Top {args.top_k} selection)...")
    ranker = FunnelRanker(target_spec=target_spec)
    ranked_candidates = ranker.rank_candidates(
        candidate_metadata=candidate_batch.metadata,
        pipeline_results=pipeline_results,
    )

    export_summary = ranker.export_results(
        ranked_candidates=ranked_candidates,
        out_dir=out_dir,
        top_k=args.top_k,
    )

    print("\n" + "=" * 95)
    print(f"=== TOP {args.top_k} REFINED MOLECULAR CANDIDATES ===")
    print("=" * 95)
    print(
        f"{'Rank':<5} {'Candidate Name':<32} {'Score':>8} {'P_wall (bar)':>14} {'ln p(x)':>10} {'<U_3D> (K)':>12} {'MW (amu)':>10} {'Pareto':>8}"
    )
    print("-" * 95)
    for cand in ranked_candidates[: args.top_k]:
        pareto_str = "YES" if cand.is_pareto_optimal else "no"
        print(
            f"{cand.rank:<5} {cand.name:<32} {cand.funnel_score:>+8.3f} "
            f"{cand.wall_pressure_bar:>14.1f} {cand.bg_log_likelihood:>+10.2f} "
            f"{cand.bg_energy_mean:>12.1f} {cand.molecular_weight:>10.1f} {pareto_str:>8}"
        )
    print("=" * 95)
    print(f"\nSuccessfully exported top candidates to: {export_summary['mol2_dir']}")
    print(f"Summary Report: {export_summary['report_path']}")
    print(f"Summary CSV:    {export_summary['csv_path']}")
    print("=== Generative Molecular Funnel Complete ===")


if __name__ == "__main__":
    import math

    main()
