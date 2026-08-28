"""
High-Throughput 5-Stage Generative Molecular Funnel Pipeline.
Couples:
1. Stage 1: PufferLib RL Swarm Training & C-Native Vectorized Candidate Sampling
2. Stage 2: In-Memory Contiguous Array Streaming & Zero-Copy GPU Batching
3. Stage 3: Coupled tinygrad cDFT Thermodynamics + Batched L-BFGS Relaxation + Boltzmann Generator Refinement
4. Stage 4: TinyJit 7-Layer Invariant EGNN MLFF Quantum Surrogate Screening
5. Stage 5: Multi-Objective Pareto Frontier Ranking & Artifact Export
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def run_generative_funnel(
    spec: str | Path,
    train_steps: int = 5000000,
    num_candidates: int = 512,
    batch_size: int = 512,
    top_k: int = 20,
    out_dir: str | Path = "runs/funnel_results",
    checkpoint: Optional[str | Path] = None,
    num_envs: int = 16,
    horizon: int = 16,
    learning_rate: float = 3e-4,
    hidden_size: int = 256,
    early_stopping_lookback: int = 500000,
    early_stopping_delta: float = 0.01,
    no_early_stopping: bool = False,
    no_curriculum: bool = False,
    no_sa_penalty: bool = False,
    no_dynamic_entropy: bool = False,
    recurrent: bool = False,
    cdft_steps: int = 50,
    bg_steps: int = 30,
    bg_samples: int = 32,
    lbfgs_steps: int = 50,
    lbfgs_tol: float = 1e-3,
    enable_egnn: bool = True,
    egnn_batch_size: int = 32,
    egnn_layers: int = 7,
    egnn_weights: Optional[str | Path] = None,
    max_sa_score: float = 6.0,
    disable_sa_filter: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete 5-stage generative molecular funnel on a single material specification.
    """
    spec_path = Path(spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification YAML not found at {spec_path}")

    target_out_dir = Path(out_dir) / spec_path.stem
    target_out_dir.mkdir(parents=True, exist_ok=True)

    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

    if verbose:
        print("=" * 80)
        print(f"=== 5-STAGE GENERATIVE MOLECULAR FUNNEL: {spec_data.get('group_name', spec_path.stem)} ===")
        print(f"Target Wall Pressure: >= {target_spec.get('min_wall_pressure_bar', 15.0):.1f} bar")
        print(f"Target Max Solvation: <= {target_spec.get('max_solvation_kcal', -3.0):.1f} kcal/mol")
        print(f"Target Max Weight:   <= {target_spec.get('max_molecular_weight', 850.0):.1f} amu")
        print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # STAGE 1: Policy Training or Checkpoint Load
    policy = MolecularSwarmPolicy(
        hidden_size=hidden_size,
        recurrent=recurrent,
    ).to(device)

    if checkpoint and Path(checkpoint).exists():
        if verbose:
            print(f"\n[Stage 1] Loading pre-trained policy checkpoint from {checkpoint}...")
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        if verbose:
            print(
                f"\n[Stage 1] Training RL Molecular Swarm Policy ({train_steps:,} timesteps, {num_envs} envs, horizon={horizon})..."
            )
        vec_env = VectorizedSwarmEnv(
            num_envs=num_envs,
            target_spec=target_spec,
            seed=42,
        )
        trainer = SwarmPuffeRLTrainer(
            vec_env=vec_env,
            policy=policy,
            final_target_spec=target_spec,
            total_timesteps=train_steps,
            horizon=horizon,
            learning_rate=learning_rate,
            minibatch_size=64,
            update_epochs=4,
            use_curriculum=not no_curriculum,
            early_stopping=not no_early_stopping,
            early_stopping_lookback=early_stopping_lookback,
            early_stopping_delta=early_stopping_delta,
            dynamic_entropy_scaling=not no_dynamic_entropy,
            sa_penalty=not no_sa_penalty,
            device=device,
        )

        batch_steps = num_envs * horizon
        epochs = max(1, train_steps // batch_steps)
        log_interval = max(1, min(epochs, 25000 // batch_steps))

        for epoch in range(1, epochs + 1):
            metrics = trainer.train_epoch()
            is_early_stopped = metrics.get("early_stopped", 0.0) > 0.5

            if verbose and (epoch % log_interval == 0 or epoch == epochs or is_early_stopped):
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
                if verbose:
                    print(f"\n  >> Early stopping triggered at step {int(metrics['global_step']):,}!")
                    print(f"     Reason: {trainer.early_stop_reason}")
                break

        vec_env.close()

        ckpt_path = target_out_dir / "trained_policy.pt"
        torch.save(policy.state_dict(), ckpt_path)
        if verbose:
            print(f"  Saved trained policy checkpoint to {ckpt_path}")

    # STAGE 2: C-Native Candidate Sampling
    if verbose:
        print(f"\n[Stage 2] Sampling {num_candidates} likely candidates via C-FFI engine...")
    t0_sample = time.perf_counter()
    sampler = SwarmCandidateSampler(
        policy=policy,
        spec_path=spec_path,
        num_envs=num_envs,
        device=device,
    )
    candidate_batch = sampler.sample_candidates(
        total_candidates=num_candidates,
        max_rollout_steps=max(50000, num_candidates * 50),
        temperature=1.0,
    )
    t_sample = time.perf_counter() - t0_sample
    if verbose:
        print(
            f"  Generated {candidate_batch.num_candidates} valid candidates in {t_sample:.2f}s "
            f"({candidate_batch.num_candidates / max(1e-3, t_sample):.1f} molecules/s) directly in contiguous host memory!"
        )

    if candidate_batch.num_candidates == 0:
        raise RuntimeError("No valid candidate molecules were generated. Check specification parameters.")

    # STAGE 3: Batched cDFT + L-BFGS + Boltzmann Screening
    if verbose:
        print(
            f"\n[Stage 3] Screening {candidate_batch.num_candidates} candidates on GPU via Batched cDFT + Boltzmann Generator..."
        )
    t0_gpu = time.perf_counter()

    num_chunks = max(1, math.ceil(candidate_batch.num_candidates / batch_size))
    pipeline_results: List[MaterialPipelineResult] = []
    coords_relaxed_all = np.zeros_like(candidate_batch.coords)

    for chunk_idx in range(num_chunks):
        start_i = chunk_idx * batch_size
        count_i = min(batch_size, candidate_batch.num_candidates - start_i)
        end_i = start_i + count_i
        mol_batch = candidate_batch.slice_molecular_batch(start_idx=start_i, count=count_i, batch_size=batch_size)

        batched_cdft = BatchedTinyCDFT(
            batch=mol_batch,
            n_grid=128,
            learning_rate=0.02,
        )
        cdft_losses = batched_cdft.solve(steps=cdft_steps, verbose=False)
        cdft_pressures = batched_cdft.get_wall_contact_pressures()
        cdft_gammas = batched_cdft.get_excess_adsorptions()

        energy_fn = MicroscopicEnergy(material=mol_batch, pad_to_128=True)
        coords_chunk = np.zeros((batch_size, candidate_batch.n_particles, 3), dtype=np.float32)
        coords_chunk[:count_i] = candidate_batch.coords[start_i:end_i]

        if lbfgs_steps > 0:
            lbfgs = BatchedLBFGS(
                max_iter=lbfgs_steps,
                grad_tol=lbfgs_tol,
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

        flow = Base2CartesianFlow(n_atoms=128, n_layers=4, hidden_dim=64)
        generator = BoltzmannGenerator(
            flow=flow,
            energy_fn=energy_fn,
            prior=None,
            batch_size=batch_size,
        )
        generator.train(steps=bg_steps, batch_size=batch_size, verbose=False)

        conformer_stats = generator.evaluate_conformer_ensemble(n_samples=bg_samples)
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

        if verbose:
            print(f"  Processed Batch Chunk {chunk_idx + 1}/{num_chunks} ({count_i} molecules on device)")

    t_gpu = time.perf_counter() - t0_gpu
    if verbose:
        print(
            f"  GPU screening finished in {t_gpu:.2f}s ({candidate_batch.num_candidates / max(1e-3, t_gpu):.1f} mol/s)"
        )

    # STAGE 4: EGNN Quantum-Surrogate Filter
    if enable_egnn:
        if verbose:
            print(
                f"\n[Stage 4] Screening {candidate_batch.num_candidates} relaxed candidates via 7-Layer Invariant EGNN MLFF (B={egnn_batch_size})..."
            )
        t0_egnn = time.perf_counter()
        egnn = EGNNForceField(
            num_layers=egnn_layers,
            hidden_dim=128,
            max_atomic_number=128,
            n_particles=candidate_batch.n_particles,
        )

        jit_eval = egnn.get_jit_evaluator()
        num_egnn_chunks = max(1, math.ceil(candidate_batch.num_candidates / egnn_batch_size))

        x_buf = Tensor.zeros(egnn_batch_size, candidate_batch.n_particles, 3, dtype=dtypes.float32).realize()
        z_buf = Tensor.zeros(egnn_batch_size, candidate_batch.n_particles, dtype=dtypes.float32).realize()
        mask_buf = Tensor.zeros(egnn_batch_size, candidate_batch.n_particles, 1, dtype=dtypes.float32).realize()
        mol_mask_buf = Tensor.zeros(egnn_batch_size, dtype=dtypes.float32).realize()

        for chunk_idx in range(num_egnn_chunks):
            start_i = chunk_idx * egnn_batch_size
            count_i = min(egnn_batch_size, candidate_batch.num_candidates - start_i)
            end_i = start_i + count_i

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

            x_buf.grad = None
            x_buf.assign(Tensor(x_chunk)).realize()
            x_buf.requires_grad = True
            z_buf.assign(Tensor(z_chunk)).realize()
            mask_buf.assign(Tensor(mask_chunk)).realize()
            mol_mask_buf.assign(Tensor(mol_mask_chunk)).realize()

            u_tensor, f_tensor = jit_eval(x_buf, z_buf, mask_buf, mol_mask_buf)
            u_np = u_tensor.numpy().astype(np.float32)
            f_np = f_tensor.numpy().astype(np.float32)

            num_real = np.maximum(1.0, mask_chunk.reshape(egnn_batch_size, -1).sum(axis=1))
            f_rms_np = np.sqrt(np.sum(f_np**2, axis=(1, 2)) / num_real).astype(np.float32)

            x_buf.grad = None

            for local_i in range(count_i):
                global_i = start_i + local_i
                pipeline_results[global_i].egnn_energy = float(u_np[local_i])
                pipeline_results[global_i].egnn_force_rms = float(f_rms_np[local_i])

        t_egnn = time.perf_counter() - t0_egnn
        if verbose:
            print(
                f"  EGNN quantum screening finished in {t_egnn:.2f}s ({candidate_batch.num_candidates / max(1e-3, t_egnn):.1f} mol/s)"
            )

    # STAGE 5: Multi-Objective Funnel Ranking & Pareto Export
    if verbose:
        print(f"\n[Stage 5] Ranking and sorting Pareto frontier (Top {top_k} selection)...")
    ranker = FunnelRanker(
        target_spec=target_spec,
        max_sa_score=max_sa_score,
        enable_sa_filter=not disable_sa_filter,
    )
    ranked_candidates = ranker.rank_candidates(
        candidate_metadata=candidate_batch.metadata,
        pipeline_results=pipeline_results,
    )

    if verbose and ranker.last_num_dropped_sa > 0:
        print(
            f"  Synthesizability safety gate dropped {ranker.last_num_dropped_sa} candidate(s) with SA Score > {max_sa_score:.1f}"
        )

    export_summary = ranker.export_results(
        ranked_candidates=ranked_candidates,
        out_dir=target_out_dir,
        top_k=top_k,
    )

    if verbose:
        print("\n" + "=" * 130)
        print(f"=== TOP {top_k} REFINED MOLECULAR CANDIDATES (QUANTUM-REFURBISHED & SA-GATED) ===")
        print("=" * 130)
        print(
            f"{'Rank':<5} {'Candidate Name':<32} {'Score':>8} {'SA':>6} {'P_wall (bar)':>14} {'ln p(x)':>10} {'<U_3D> (K)':>12} {'U_EGNN (K)':>12} {'F_RMS':>10} {'MW (amu)':>10} {'Pareto':>8}"
        )
        print("-" * 130)
        for cand in ranked_candidates[:top_k]:
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

    return {
        "group_name": spec_data.get("group_name", spec_path.stem),
        "spec_path": str(spec_path),
        "out_dir": str(target_out_dir),
        "num_candidates_generated": candidate_batch.num_candidates,
        "ranked_candidates": ranked_candidates,
        "export_summary": export_summary,
    }


def run_all_specs_funnel_benchmark(
    specs_dir: str | Path = "tests/data",
    train_steps: int = 5000000,
    num_envs: int = 16,
    horizon: int = 16,
    learning_rate: float = 3e-4,
    hidden_size: int = 256,
    early_stopping_lookback: int = 500000,
    early_stopping_delta: float = 0.01,
    num_candidates: int = 64,
    batch_size: int = 64,
    egnn_batch_size: int = 32,
    top_k: int = 10,
    out_dir: str | Path = "runs/full_system_benchmark",
    max_sa_score: float = 6.0,
    enable_egnn: bool = True,
    recurrent: bool = False,
) -> int:
    """Executes the full 5-stage funnel across all specification YAMLs in specs_dir."""
    base_out_dir = Path(out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    spec_files = sorted(Path(specs_dir).glob("*.yaml"))
    if not spec_files:
        print(f"Error: No YAML specifications found in {specs_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(spec_files)} material specification YAMLs in {specs_dir}")
    all_summaries: List[Dict[str, Any]] = []

    for i, spec_path in enumerate(spec_files):
        group_name = spec_path.stem
        spec_out_dir = base_out_dir / group_name
        spec_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"[{i + 1}/{len(spec_files)}] RUNNING 5-STAGE FUNNEL: {group_name}")
        print(f"Spec file: {spec_path}")
        print(f"Output directory: {spec_out_dir}")
        print("=" * 100)

        t0 = time.perf_counter()
        try:
            run_generative_funnel(
                spec=spec_path,
                train_steps=train_steps,
                num_candidates=num_candidates,
                batch_size=batch_size,
                top_k=top_k,
                out_dir=base_out_dir,
                num_envs=num_envs,
                horizon=horizon,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                early_stopping_lookback=early_stopping_lookback,
                early_stopping_delta=early_stopping_delta,
                recurrent=recurrent,
                enable_egnn=enable_egnn,
                egnn_batch_size=egnn_batch_size,
                max_sa_score=max_sa_score,
                verbose=True,
            )
            t_total = time.perf_counter() - t0
            csv_path = spec_out_dir / "funnel_summary.csv"

            rows: List[Dict[str, Any]] = []
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)

            def get_float_list(key: str) -> List[float]:
                vals = []
                for r in rows:
                    v = r.get(key, "")
                    if v not in ("", None, "None", "nan"):
                        try:
                            vals.append(float(v))
                        except ValueError:
                            pass
                return vals

            scores = get_float_list("funnel_score")
            p_walls = get_float_list("wall_pressure_bar")
            sa_scores = get_float_list("sa_score")
            egnn_energies = get_float_list("egnn_energy")
            egnn_f_rms = get_float_list("egnn_force_rms")
            mws = get_float_list("molecular_weight")
            num_pareto = sum(1 for r in rows if r.get("is_pareto_optimal", "").lower() in ("true", "1", "yes"))

            all_summaries.append(
                {
                    "group": group_name,
                    "status": "SUCCESS",
                    "total_time_s": t_total,
                    "exported_candidates": len(rows),
                    "num_pareto": num_pareto,
                    "mean_funnel_score": float(sum(scores) / len(scores)) if scores else 0.0,
                    "mean_p_wall": float(sum(p_walls) / len(p_walls)) if p_walls else 0.0,
                    "mean_sa_score": float(sum(sa_scores) / len(sa_scores)) if sa_scores else 0.0,
                    "mean_egnn_energy": float(sum(egnn_energies) / len(egnn_energies)) if egnn_energies else 0.0,
                    "mean_egnn_f_rms": float(sum(egnn_f_rms) / len(egnn_f_rms)) if egnn_f_rms else 0.0,
                    "mean_mw": float(sum(mws) / len(mws)) if mws else 0.0,
                    "csv_path": str(csv_path),
                }
            )
            print(
                f"SUCCESS: {group_name} finished in {t_total:.2f}s ({len(rows)} candidates exported, {num_pareto} Pareto optimal)"
            )
        except Exception as e:
            t_total = time.perf_counter() - t0
            print(f"ERROR: Execution failed for {group_name}: {e}", file=sys.stderr)
            all_summaries.append(
                {
                    "group": group_name,
                    "status": "FAILED",
                    "error": str(e),
                    "total_time_s": t_total,
                }
            )

    summary_json = base_out_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    summary_md = base_out_dir / "benchmark_report.md"
    lines = [
        "# End-to-End Generative Funnel Cross-Material Benchmark Report",
        "",
        f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Material Groups Tested**: {len(spec_files)}",
        f"**RL Steps / Group**: {train_steps:,}",
        f"**Candidates Screened / Group**: {num_candidates}",
        f"**GPU Batch Size**: {batch_size}",
        "",
        "## Summary Results Across All Material Classes",
        "",
        "| Material Group | Status | Runtime (s) | Exported | Pareto | Mean Score | P_wall (bar) | Mean SA | Mean U_EGNN (K) | Mean F_RMS | Mean MW (amu) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in all_summaries:
        if s.get("status") == "SUCCESS":
            lines.append(
                f"| `{s['group']}` | {s['status']} | {s['total_time_s']:.1f}s | "
                f"{s['exported_candidates']} | {s['num_pareto']} | **{s['mean_funnel_score']:+.2f}** | "
                f"{s['mean_p_wall']:.1f} | {s['mean_sa_score']:.2f} | "
                f"{s['mean_egnn_energy']:.1f} | {s['mean_egnn_f_rms']:.2f} | {s['mean_mw']:.1f} |"
            )
        else:
            lines.append(
                f"| `{s.get('group', 'unknown')}` | **{s.get('status')}** | {s.get('total_time_s', 0):.1f}s | - | - | - | - | - | - | - | - |"
            )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBenchmark complete! Full report saved to: {summary_md}")
    return 0
