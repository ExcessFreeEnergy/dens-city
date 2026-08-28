"""
Chemistry Analysis, Synthesizability (SA Score), and Diversity Evaluation for Molecular Swarm Candidates.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Contrib.SA_Score import sascorer

from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.sampler import SwarmCandidateSampler
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.trainer import SwarmPuffeRLTrainer, VectorizedSwarmEnv


def evaluate_molecule_chemistry(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Rigorously analyzes an RDKit Mol object for chemical validity,
    canonical SMILES, SA Score, and molecular descriptors.
    """
    res = {
        "valid": False,
        "smiles": "",
        "sa_score": None,
        "mw": 0.0,
        "num_heavy_atoms": 0,
        "num_rotatable_bonds": 0,
        "num_aromatic_rings": 0,
        "hbd": 0,
        "hba": 0,
        "tpsa": 0.0,
    }

    if mol is None:
        return res

    try:
        # Full RDKit Sanitization check
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        # Re-parse from SMILES to ensure round-trip chemical validity
        mol_from_smi = Chem.MolFromSmiles(smiles)
        if mol_from_smi is None:
            return res

        res["valid"] = True
        res["smiles"] = smiles
        res["mw"] = float(Descriptors.MolWt(mol))
        res["num_heavy_atoms"] = int(mol.GetNumHeavyAtoms())
        res["num_rotatable_bonds"] = int(Descriptors.NumRotatableBonds(mol))
        res["num_aromatic_rings"] = int(Descriptors.NumAromaticRings(mol))
        res["hbd"] = int(Descriptors.NumHDonors(mol))
        res["hba"] = int(Descriptors.NumHAcceptors(mol))
        res["tpsa"] = float(Descriptors.TPSA(mol))

        # Synthetic Accessibility (SA) Score (1 = easy to synthesize, 10 = impossible)
        sa = sascorer.calculateScore(mol)
        res["sa_score"] = float(sa)
    except Exception:
        res["valid"] = False

    return res


def compute_batch_diversity(mols: List[Chem.Mol]) -> Tuple[float, float, int]:
    """
    Computes mean internal pairwise Tanimoto similarity, internal diversity (1 - mean_sim),
    and unique SMILES count across valid molecules using 2048-bit Morgan Fingerprints (radius=2).
    """
    valid_mols = [m for m in mols if m is not None]
    if len(valid_mols) < 2:
        return 0.0, 1.0, len(valid_mols)

    fps = []
    smiles_set = set()
    for m in valid_mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
            fps.append(fp)
            smi = Chem.MolToSmiles(m, canonical=True)
            smiles_set.add(smi)
        except Exception:
            continue

    n = len(fps)
    if n < 2:
        return 0.0, 1.0, len(smiles_set)

    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    diversity = 1.0 - mean_sim
    unique_count = len(smiles_set)

    return mean_sim, diversity, unique_count


def run_spec_rl_stage(
    spec_path: Path,
    timesteps: int = 15000,
    num_candidates: int = 100,
    num_envs: int = 16,
    horizon: int = 16,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes full RL stage for a single material YAML specification:
    1. Loads target bounds from YAML.
    2. Trains PPO MolecularSwarmPolicy with universal C-level action masks.
    3. Samples candidate molecules from the trained policy.
    4. Computes validity, SA Score, Tanimoto diversity, and target compliance.
    """
    t_start = time.time()
    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    target_spec = SwarmSpecLoader.derive_target_spec(spec_data)
    spec_name = spec_path.stem

    print(f"\n{'=' * 88}")
    print(f"  Evaluating Specification: {spec_name}")
    print(f"{'=' * 88}")
    print("  Target Objectives & Constraints:")
    for k, v in target_spec.items():
        print(f"    - {k:<25}: {v}")

    # 1. Initialize Vectorized Environment and Policy
    vec_env = VectorizedSwarmEnv(
        num_envs=num_envs,
        spec_yaml_path=spec_path,
        target_spec=target_spec,
        seed=42,
    )

    policy = MolecularSwarmPolicy(
        obs_size=88,
        hidden_size=256,
        recurrent=False,
    )

    trainer = SwarmPuffeRLTrainer(
        vec_env=vec_env,
        policy=policy,
        final_target_spec=target_spec,
        total_timesteps=timesteps,
        horizon=horizon,
        learning_rate=3e-4,
        use_curriculum=True,
        early_stopping=True,
        early_stopping_lookback=max(2000, timesteps // 2),
        early_stopping_delta=0.01,
        checkpoint_dir=output_dir / "checkpoints" if output_dir else None,
        device=device,
    )

    print(f"\n[*] Training Molecular Swarm Policy ({timesteps:,} steps on {num_envs} envs)...")
    epochs = max(1, timesteps // (num_envs * horizon))
    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch()
        if epoch % max(1, epochs // 4) == 0 or epoch == epochs or metrics.get("early_stopped", 0.0) > 0.5:
            print(
                f"    Epoch {metrics['epoch']:3d}/{epochs:3d} | "
                f"Steps: {metrics['global_step']:6d} | "
                f"Stage: {metrics['curriculum_stage']} | "
                f"EMA: {metrics['env/reward_ema']:+5.2f} | "
                f"Score: {metrics['env/score']:+5.2f} | "
                f"Valid: {metrics['env/valid_rate'] * 100:4.1f}% | "
                f"H(pi): {metrics['loss/entropy']:4.2f} | "
                f"KL: {metrics['loss/approx_kl']:.4f} | "
                f"SA: {metrics['env/sa_score']:4.2f} | "
                f"P_wall: {metrics['env/p_wall']:4.1f} bar"
            )
        if metrics.get("early_stopped", 0.0) > 0.5:
            print(f"    [EARLY STOPPING HALT] {trainer.early_stop_reason}")
            break

    vec_env.close()

    # 2. Sample candidate batch using trained policy
    print(f"\n[*] Sampling {num_candidates} candidate molecules from trained policy...")
    sampler = SwarmCandidateSampler(
        policy=policy,
        spec_path=spec_path,
        num_envs=num_envs,
        device=device,
        target_n_particles=128,
    )

    candidate_batch = sampler.sample_candidates(
        total_candidates=num_candidates,
        max_rollout_steps=max(50000, num_candidates * 50),
        temperature=1.0,
    )

    # 3. Chemical Evaluation & Sanitization via RDKit
    print("[*] Running RDKit chemical sanitization, SA Score, and Tanimoto diversity analysis...")
    mols: List[Chem.Mol] = []
    results_list: List[Dict[str, Any]] = []
    valid_count = 0
    sa_scores: List[float] = []
    mw_list: List[float] = []
    p_wall_list: List[float] = []
    omega_solv_list: List[float] = []
    pmi_list: List[float] = []
    aromatic_list: List[float] = []
    rotatable_list: List[float] = []

    for meta in candidate_batch.metadata:
        smi = meta.get("smiles", "")
        if smi:
            mol = Chem.MolFromSmiles(smi)
        else:
            mol2_str = meta["mol2"]
            mol = Chem.MolFromMol2Block(mol2_str, removeHs=False)

        chem_info = evaluate_molecule_chemistry(mol)

        if chem_info["valid"]:
            valid_count += 1
            mols.append(mol)
            if chem_info["sa_score"] is not None:
                sa_scores.append(chem_info["sa_score"])
            mw_list.append(chem_info["mw"])
            p_wall_list.append(meta.get("p_wall", 0.0))
            omega_solv_list.append(meta.get("omega_solv", 0.0))
            pmi_list.append(meta.get("pmi_linearity", 0.0))
            aromatic_list.append(meta.get("aromatic_density", 0.0))
            rotatable_list.append(meta.get("rotatable_fraction", 0.0))

        entry = {
            "name": meta["name"],
            "valid": chem_info["valid"],
            "smiles": chem_info["smiles"],
            "sa_score": chem_info["sa_score"],
            "mw": chem_info["mw"],
            "p_wall_bar": meta.get("p_wall", 0.0),
            "omega_solv_kcal": meta.get("omega_solv", 0.0),
            "pmi_linearity": meta.get("pmi_linearity", 0.0),
            "aromatic_density": meta.get("aromatic_density", 0.0),
            "rotatable_fraction": meta.get("rotatable_fraction", 0.0),
            "rl_reward": meta.get("rl_reward", 0.0),
        }
        results_list.append(entry)

    total_sampled = len(candidate_batch.metadata)
    validity_rate = (valid_count / max(1, total_sampled)) * 100.0
    mean_sa = float(np.mean(sa_scores)) if sa_scores else float("nan")
    std_sa = float(np.std(sa_scores)) if sa_scores else 0.0
    min_sa = float(np.min(sa_scores)) if sa_scores else float("nan")
    max_sa = float(np.max(sa_scores)) if sa_scores else float("nan")

    mean_sim, diversity, unique_smiles = compute_batch_diversity(mols)
    unique_ratio = (unique_smiles / max(1, valid_count)) * 100.0
    t_elapsed = time.time() - t_start

    summary = {
        "spec_name": spec_name,
        "spec_path": str(spec_path),
        "target_spec": target_spec,
        "timesteps_trained": trainer.global_step,
        "best_rl_reward": trainer.best_reward,
        "num_candidates_generated": total_sampled,
        "num_valid_candidates": valid_count,
        "validity_rate_pct": validity_rate,
        "sa_score_mean": mean_sa,
        "sa_score_std": std_sa,
        "sa_score_min": min_sa,
        "sa_score_max": max_sa,
        "mean_internal_tanimoto_similarity": mean_sim,
        "internal_diversity": diversity,
        "num_unique_smiles": unique_smiles,
        "unique_smiles_ratio_pct": unique_ratio,
        "mean_mw": float(np.mean(mw_list)) if mw_list else 0.0,
        "mean_p_wall_bar": float(np.mean(p_wall_list)) if p_wall_list else 0.0,
        "mean_omega_solv_kcal": float(np.mean(omega_solv_list)) if omega_solv_list else 0.0,
        "mean_pmi_linearity": float(np.mean(pmi_list)) if pmi_list else 0.0,
        "mean_aromatic_density": float(np.mean(aromatic_list)) if aromatic_list else 0.0,
        "mean_rotatable_fraction": float(np.mean(rotatable_list)) if rotatable_list else 0.0,
        "runtime_seconds": t_elapsed,
        "candidates": results_list,
    }

    # Save per-spec artifacts
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_out_json = output_dir / f"{spec_name}_evaluation.json"
        with open(spec_out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if trainer.best_mol2_str:
            best_mol2_path = output_dir / f"{spec_name}_best_candidate.mol2"
            best_mol2_path.write_text(trainer.best_mol2_str)

    print(f"\n[+] Results for {spec_name}:")
    print(f"    - Chemical Validity Rate     : {validity_rate:5.1f}% ({valid_count}/{total_sampled})")
    print(f"    - Synthesizability (SA Score): {mean_sa:.2f} ± {std_sa:.2f} [Range: {min_sa:.2f} - {max_sa:.2f}]")
    print(f"    - Internal Tanimoto Sim (T)  : {mean_sim:.3f} (Diversity 1 - T = {diversity:.3f})")
    print(f"    - Unique SMILES Ratio        : {unique_ratio:5.1f}% ({unique_smiles} distinct topologies)")
    print(
        f"    - Mean Contact Pressure P_wall: {summary['mean_p_wall_bar']:5.1f} bar (Target >= {target_spec.get('min_wall_pressure_bar', 0):.1f})"
    )
    print(
        f"    - Mean Solvation Energy      : {summary['mean_omega_solv_kcal']:5.2f} kcal/mol (Target <= {target_spec.get('max_solvation_kcal', 0):.1f})"
    )
    print(f"    - Elapsed Time               : {t_elapsed:.1f} s")

    return summary


def evaluate_all_swarm_specs(
    specs_dir: str | Path = "tests/data",
    timesteps: int = 10000,
    num_candidates: int = 50,
    num_envs: int = 16,
    out_dir: str | Path = "runs/rl_stage_evaluation",
) -> int:
    """Evaluates all YAML specifications in specs_dir and outputs master summary JSON & diversity report."""
    target_specs_dir = Path(specs_dir).resolve()
    spec_files = sorted(target_specs_dir.glob("*.yaml"))
    if not spec_files:
        print(f"Error: No specification YAML files found in {target_specs_dir}", file=sys.stderr)
        return 1

    target_out_dir = Path(out_dir)
    target_out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 88)
    print("  dens-city: Full RL Stage Evaluation & Chemistry Diagnostic Pipeline")
    print("=" * 88)
    print(f"  Target Specs Directory : {target_specs_dir} ({len(spec_files)} specifications)")
    print(f"  Training Budget / Spec : {timesteps:,} steps")
    print(f"  Evaluation Batch Size  : {num_candidates} molecules / spec")
    print(f"  Parallel C Workers     : {num_envs} envs (device={device})")
    print(f"  Output Artifacts Dir   : {target_out_dir}")
    print("=" * 88)

    all_summaries: List[Dict[str, Any]] = []
    t_global_start = time.time()

    for spec_path in spec_files:
        summary = run_spec_rl_stage(
            spec_path=spec_path,
            timesteps=timesteps,
            num_candidates=num_candidates,
            num_envs=num_envs,
            device=device,
            output_dir=target_out_dir,
        )
        all_summaries.append(summary)

    t_global_total = time.time() - t_global_start

    # Write Master Evaluation Summary JSON
    master_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": t_global_total,
        "num_specs_evaluated": len(all_summaries),
        "spec_summaries": all_summaries,
    }
    master_json_path = target_out_dir / "master_evaluation_summary.json"
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # Print Final Comparative Synthesis & Diversity Table
    print("\n" + "=" * 115)
    print("  FINAL MULTI-SPECIFICATION RL STAGE EVALUATION REPORT")
    print("=" * 115)
    print(
        f"{'Specification':<35} | {'Valid %':<8} | {'SA Score':<12} | {'Tanimoto (T)':<13} | {'Diversity (1-T)':<15} | {'Unique SMILES %':<15}"
    )
    print("-" * 115)

    for s in all_summaries:
        sa_str = f"{s['sa_score_mean']:.2f} ± {s['sa_score_std']:.2f}"
        print(
            f"{s['spec_name']:<35} | {s['validity_rate_pct']:6.1f}% | {sa_str:<12} | {s['mean_internal_tanimoto_similarity']:11.3f} | {s['internal_diversity']:13.3f} | {s['unique_smiles_ratio_pct']:13.1f}%"
        )

    print("-" * 115)
    avg_valid = float(np.mean([s["validity_rate_pct"] for s in all_summaries]))
    avg_sa = float(np.mean([s["sa_score_mean"] for s in all_summaries]))
    avg_div = float(np.mean([s["internal_diversity"] for s in all_summaries]))
    avg_uniq = float(np.mean([s["unique_smiles_ratio_pct"] for s in all_summaries]))

    print(
        f"{'OVERALL AVERAGE':<35} | {avg_valid:6.1f}% | {avg_sa:6.2f}       | {'--':<13} | {avg_div:13.3f} | {avg_uniq:13.1f}%"
    )
    print("=" * 115)
    print(f"\n[+] Master Summary saved to: {master_json_path}")
    print(f"[+] Total Pipeline Execution Time: {t_global_total:.1f} s")
    return 0
