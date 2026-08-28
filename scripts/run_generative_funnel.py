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
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tinygrad import Tensor, dtypes

from dens_city.boltzmann.bijectors import Base2CartesianFlow
from dens_city.boltzmann.egnn import EGNNForceField
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
        "--total-timesteps",
        dest="train_steps",
        type=int,
        default=5000000,
        help="Number of RL curriculum training steps before sampling (scaled to 5M-10M steps, default: 5000000)",
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
        "--horizon",
        type=int,
        default=16,
        help="Rollout horizon per environment matching mean molecular growth path (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Policy latent dimension (default: 256)",
    )
    parser.add_argument(
        "--early-stopping-lookback",
        type=int,
        default=500000,
        help="Step lookback window for EMA reward flatline detection (default: 500000)",
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.01,
        help="EMA reward change threshold for early stopping (default: 0.01)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable Dynamic EMA early stopping",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable 3-stage curriculum scheduler",
    )
    parser.add_argument(
        "--no-sa-penalty",
        action="store_true",
        help="Disable in-the-loop batch SA score penalty",
    )
    parser.add_argument(
        "--no-dynamic-entropy",
        action="store_true",
        help="Disable molecular-weight-scaling dynamic entropy coefficient",
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
    parser.add_argument(
        "--enable-egnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Stage 4 E(n)-Equivariant Graph Neural Network (EGNN) quantum surrogate screening (default: True)",
    )
    parser.add_argument(
        "--egnn-batch-size",
        type=int,
        default=32,
        help="Batch size for Stage 4 EGNN quantum force field evaluation (default: 32)",
    )
    parser.add_argument(
        "--egnn-layers",
        type=int,
        default=7,
        help="Number of message-passing layers in the EGNN architecture (default: 7)",
    )
    parser.add_argument(
        "--egnn-weights",
        type=str,
        default=None,
        help="Optional path to pretrained EGNN weights .npz archive",
    )
    parser.add_argument(
        "--max-sa-score",
        type=float,
        default=6.0,
        help="Maximum allowable RDKit Synthetic Accessibility (SA) Score (default: 6.0)",
    )
    parser.add_argument(
        "--disable-sa-filter",
        action="store_true",
        help="Disable Stage 5 RDKit synthesizability (SA Score) safety gate",
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
    print(f"=== 5-STAGE GENERATIVE MOLECULAR FUNNEL: {spec_data.get('group_name', spec_path.stem)} ===")
    print(f"Target Wall Pressure: >= {target_spec.get('min_wall_pressure_bar', 15.0):.1f} bar")
    print(f"Target Max Solvation: <= {target_spec.get('max_solvation_kcal', -3.0):.1f} kcal/mol")
    print(f"Target Max Weight:   <= {target_spec.get('max_molecular_weight', 850.0):.1f} amu")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------
    # STAGE 1: Policy Training (or Checkpoint Load)
    # -------------------------------------------------------------
    policy = MolecularSwarmPolicy(
        hidden_size=args.hidden_size,
        recurrent=args.recurrent,
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\n[Stage 1] Loading pre-trained policy checkpoint from {args.checkpoint}...")
        policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(
            f"\n[Stage 1] Training RL Molecular Swarm Policy ({args.train_steps:,} timesteps, {args.num_envs} envs, horizon={args.horizon})..."
        )
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
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            minibatch_size=64,
            update_epochs=4,
            use_curriculum=not args.no_curriculum,
            early_stopping=not args.no_early_stopping,
            early_stopping_lookback=args.early_stopping_lookback,
            early_stopping_delta=args.early_stopping_delta,
            dynamic_entropy_scaling=not args.no_dynamic_entropy,
            sa_penalty=not args.no_sa_penalty,
            device=device,
        )

        batch_steps = args.num_envs * args.horizon
        epochs = max(1, args.train_steps // batch_steps)
        log_interval = max(1, min(epochs, 25000 // batch_steps))

        for epoch in range(1, epochs + 1):
            metrics = trainer.train_epoch()
            is_early_stopped = metrics.get("early_stopped", 0.0) > 0.5

            if epoch % log_interval == 0 or epoch == epochs or is_early_stopped:
                print(
                    f"  Epoch {epoch:05d}/{epochs:05d} | "
                    f"Step {int(metrics['global_step']):8,d} | "
                    f"Stage {int(metrics.get('curriculum_stage', 3))} | "
                    f"Score: {metrics['env/score']:+6.2f} | "
                    f"EMA: {metrics.get('env/reward_ema', 0.0):+6.2f} | "
                    f"Valid Rate: {metrics['env/valid_rate'] * 100:5.1f}% | "
                    f"P_wall: {metrics['env/p_wall']:5.1f} bar | "
                    f"SPS: {metrics['SPS']:5.0f}"
                )

            if is_early_stopped:
                print(f"\n  >> Early stopping triggered at step {int(metrics['global_step']):,}!")
                print(f"     Reason: {trainer.early_stop_reason}")
                break

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
    coords_relaxed_all = np.zeros_like(candidate_batch.coords)

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
            coords_relaxed_all[start_i:end_i] = lbfgs_res.x_relaxed[:count_i]
            u_evaluated = lbfgs_res.final_energies
        else:
            coords_relaxed_all[start_i:end_i] = candidate_batch.coords[start_i:end_i]
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
    # STAGE 4: EGNN Quantum-Surrogate Filter
    # -------------------------------------------------------------
    if args.enable_egnn:
        print(
            f"\n[Stage 4] Screening {candidate_batch.num_candidates} relaxed candidates via 7-Layer Invariant EGNN MLFF (B={args.egnn_batch_size})..."
        )
        t0_egnn = time.perf_counter()
        egnn = EGNNForceField(
            num_layers=args.egnn_layers,
            hidden_dim=128,
            max_atomic_number=128,
            n_particles=candidate_batch.n_particles,
        )

        jit_eval = egnn.get_jit_evaluator()
        egnn_batch_size = args.egnn_batch_size
        num_egnn_chunks = max(1, math.ceil(candidate_batch.num_candidates / egnn_batch_size))

        # Persistent static GPU buffers for TinyJit execution across all batch chunks
        x_buf = Tensor.zeros(egnn_batch_size, candidate_batch.n_particles, 3, dtype=dtypes.float32).realize()
        z_buf = Tensor.zeros(egnn_batch_size, candidate_batch.n_particles, dtype=dtypes.float32).realize()
        mask_buf = Tensor.zeros(egnn_batch_size, candidate_batch.n_particles, 1, dtype=dtypes.float32).realize()
        mol_mask_buf = Tensor.zeros(egnn_batch_size, dtype=dtypes.float32).realize()

        for chunk_idx in range(num_egnn_chunks):
            start_i = chunk_idx * egnn_batch_size
            count_i = min(egnn_batch_size, candidate_batch.num_candidates - start_i)
            end_i = start_i + count_i

            # Zero-padded host arrays ensure static (egnn_batch_size, ...) shapes on all chunks
            x_chunk = np.zeros((egnn_batch_size, candidate_batch.n_particles, 3), dtype=np.float32)
            z_chunk = np.zeros((egnn_batch_size, candidate_batch.n_particles), dtype=np.float32)
            mask_chunk = np.zeros((egnn_batch_size, candidate_batch.n_particles, 1), dtype=np.float32)
            mol_mask_chunk = np.zeros(egnn_batch_size, dtype=np.float32)

            x_chunk[:count_i] = coords_relaxed_all[start_i:end_i]
            z_chunk[:count_i] = candidate_batch.atomic_numbers[start_i:end_i]
            mask_chunk[:count_i] = candidate_batch.atom_mask[start_i:end_i].reshape(
                count_i, candidate_batch.n_particles, 1
            )
            mol_mask_chunk[:count_i] = 1.0

            # 1. Update static device buffers in-place via .assign() with clean grad state
            x_buf.grad = None
            x_buf.assign(Tensor(x_chunk)).realize()
            x_buf.requires_grad = True
            z_buf.assign(Tensor(z_chunk)).realize()
            mask_buf.assign(Tensor(mask_chunk)).realize()
            mol_mask_buf.assign(Tensor(mol_mask_chunk)).realize()

            # 2. JIT-compiled single-pass energy & conservative forces
            u_tensor, f_tensor = jit_eval(x_buf, z_buf, mask_buf, mol_mask_buf)

            # 3. Extraction to host NumPy
            u_np = u_tensor.numpy().astype(np.float32)
            f_np = f_tensor.numpy().astype(np.float32)

            # 4. Quantum Force RMS Stability
            num_real = np.maximum(1.0, mask_chunk.reshape(egnn_batch_size, -1).sum(axis=1))
            f_rms_np = np.sqrt(np.sum(f_np**2, axis=(1, 2)) / num_real).astype(np.float32)

            # 5. Clean up gradient reference on static leaf tensor
            x_buf.grad = None

            for local_i in range(count_i):
                global_i = start_i + local_i
                pipeline_results[global_i].egnn_energy = float(u_np[local_i])
                pipeline_results[global_i].egnn_force_rms = float(f_rms_np[local_i])

        t_egnn = time.perf_counter() - t0_egnn
        print(
            f"  EGNN quantum screening finished in {t_egnn:.2f}s ({candidate_batch.num_candidates / max(1e-3, t_egnn):.1f} mol/s)"
        )

    # -------------------------------------------------------------
    # STAGE 5: Multi-Objective Funnel Ranking & Pareto Export
    # -------------------------------------------------------------
    print(f"\n[Stage 5] Ranking and sorting Pareto frontier (Top {args.top_k} selection)...")
    ranker = FunnelRanker(
        target_spec=target_spec,
        max_sa_score=args.max_sa_score,
        enable_sa_filter=not args.disable_sa_filter,
    )
    ranked_candidates = ranker.rank_candidates(
        candidate_metadata=candidate_batch.metadata,
        pipeline_results=pipeline_results,
    )

    if ranker.last_num_dropped_sa > 0:
        print(
            f"  Synthesizability safety gate dropped {ranker.last_num_dropped_sa} candidate(s) with SA Score > {args.max_sa_score:.1f}"
        )

    export_summary = ranker.export_results(
        ranked_candidates=ranked_candidates,
        out_dir=out_dir,
        top_k=args.top_k,
    )

    print("\n" + "=" * 130)
    print(f"=== TOP {args.top_k} REFINED MOLECULAR CANDIDATES (QUANTUM-REFURBISHED & SA-GATED) ===")
    print("=" * 130)
    print(
        f"{'Rank':<5} {'Candidate Name':<32} {'Score':>8} {'SA':>6} {'P_wall (bar)':>14} {'ln p(x)':>10} {'<U_3D> (K)':>12} {'U_EGNN (K)':>12} {'F_RMS':>10} {'MW (amu)':>10} {'Pareto':>8}"
    )
    print("-" * 130)
    for cand in ranked_candidates[: args.top_k]:
        pareto_str = "YES" if cand.is_pareto_optimal else "no"
        sa_str = f"{cand.sa_score:.2f}" if cand.sa_score is not None else "N/A"
        print(
            f"{cand.rank:<5} {cand.name:<32} {cand.funnel_score:>+8.3f} "
            f"{sa_str:>6} "
            f"{cand.wall_pressure_bar:>14.1f} {cand.bg_log_likelihood:>+10.2f} "
            f"{cand.bg_energy_mean:>12.1f} {cand.egnn_energy:>12.1f} {cand.egnn_force_rms:>10.2f} "
            f"{cand.molecular_weight:>10.1f} {pareto_str:>8}"
        )
    print("=" * 130)
    print(f"\nSuccessfully exported top candidates to: {export_summary['mol2_dir']}")
    print(f"Summary Report: {export_summary['report_path']}")
    print(f"Summary CSV:    {export_summary['csv_path']}")
    print("=== Generative Molecular Funnel Complete ===")


if __name__ == "__main__":
    import math

    main()
