"""
Multi-Objective Funnel Ranker for dens-city.
Filters and ranks thousands of RL-generated molecular candidates against
coupled cDFT thermodynamics, Normalizing Flow exact log-likelihood,
3D microscopic internal energy stability, and target material specifications.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from dens_city.utils.pipeline import MaterialPipelineResult


@dataclass
class RankedCandidate:
    """
    Encapsulates a fully evaluated and scored molecular candidate across all 4 funnel stages.
    """

    rank: int
    name: str
    funnel_score: float
    rl_reward: float
    wall_pressure_bar: float
    target_wall_pressure_bar: float
    solvation_free_energy_kcal_mol: float
    target_solvation_kcal: float
    bg_log_likelihood: float
    bg_energy_mean: float
    bg_energy_var: float
    egnn_energy: float
    egnn_force_rms: float
    molecular_weight: float
    num_sites: int
    pmi_linearity: float
    aromatic_density: float
    rotatable_fraction: float
    mol2_content: str
    is_pareto_optimal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("mol2_content", None)
        return d


class FunnelRanker:
    """
    Ranks candidate molecules using coupled cDFT + Boltzmann Normalizing Flow + EGNN MLFF observables.
    """

    def __init__(
        self,
        target_spec: Dict[str, Any],
        w_rl: float = 0.30,
        w_cdft: float = 0.30,
        w_boltzmann: float = 0.20,
        w_egnn: float = 0.20,
        alpha_energy: float = 0.001,
        beta_logp: float = 0.05,
        gamma_var: float = 0.0001,
        alpha_egnn_energy: float = 0.001,
        beta_egnn_force: float = 0.01,
    ):
        self.target_spec = target_spec
        self.w_rl = w_rl
        self.w_cdft = w_cdft
        self.w_boltzmann = w_boltzmann
        self.w_egnn = w_egnn
        self.alpha_energy = alpha_energy
        self.beta_logp = beta_logp
        self.gamma_var = gamma_var
        self.alpha_egnn_energy = alpha_egnn_energy
        self.beta_egnn_force = beta_egnn_force

        self.min_p_wall = float(target_spec.get("min_wall_pressure_bar", 15.0))
        self.max_solv = float(target_spec.get("max_solvation_kcal", -3.0))
        self.max_mw = float(target_spec.get("max_molecular_weight", 850.0))

    def rank_candidates(
        self,
        candidate_metadata: List[Dict[str, Any]],
        pipeline_results: List[MaterialPipelineResult],
    ) -> List[RankedCandidate]:
        """
        Combines Stage 1 RL swarm metrics with Stage 2/3 cDFT, Boltzmann Generator,
        and Stage 4 EGNN quantum observables into a unified Pareto-ranked candidate list.
        """
        results_by_name = {r.material_name: r for r in pipeline_results}

        evaluated: List[RankedCandidate] = []

        for meta in candidate_metadata:
            name = meta["name"]
            res = results_by_name.get(name)

            rl_reward = float(meta.get("rl_reward", 0.0))
            p_w = float(res.wall_pressure_bar) if res else float(meta.get("p_wall", 0.0))
            solv_e = (
                float(res.solvation_free_energy_kcal_mol)
                if (res and res.solvation_free_energy_kcal_mol is not None)
                else float(meta.get("omega_solv", 0.0))
            )
            bg_logp = float(res.bg_log_likelihood) if (res and res.bg_log_likelihood is not None) else 0.0
            bg_u_mean = float(res.bg_energy_mean) if (res and res.bg_energy_mean is not None) else 0.0
            bg_u_var = float(res.bg_energy_var) if (res and res.bg_energy_var is not None) else 0.0
            egnn_u = float(res.egnn_energy) if (res and res.egnn_energy is not None) else 0.0
            egnn_f_rms = float(res.egnn_force_rms) if (res and res.egnn_force_rms is not None) else 0.0

            # 1. RL Score Component
            s_rl = max(0.0, rl_reward)

            # 2. cDFT Thermodynamic Score Component
            # Reward wall pressure exceeding threshold, penalize failing threshold
            p_ratio = p_w / max(1.0, self.min_p_wall)
            s_cdft_p = min(3.0, p_ratio)
            s_cdft_solv = 1.0 if solv_e <= self.max_solv else max(0.0, 1.0 - (solv_e - self.max_solv) / 5.0)
            s_cdft = 0.7 * s_cdft_p + 0.3 * s_cdft_solv

            # 3. Boltzmann Normalizing Flow PDF & Structural Energy Component
            # Higher log p(x) -> more accessible from prior without steric clash
            # Lower <U_3D> -> lower energy ground state
            # Lower Var(U) -> rigid, stable conformer ensemble
            s_bg = (
                self.beta_logp * bg_logp
                - self.alpha_energy * max(-10000.0, min(10000.0, bg_u_mean))
                - self.gamma_var * min(50000.0, bg_u_var)
            )

            # 4. EGNN Quantum Force Field Component
            # Lower U_EGNN -> quantum ground state energetic minimum
            # Lower ||F_EGNN||_RMS -> true stationary quantum minimum (not a quantum cliff)
            s_egnn = -self.alpha_egnn_energy * max(-10000.0, min(50000.0, egnn_u)) - self.beta_egnn_force * min(
                1000.0, egnn_f_rms
            )

            # Total Composite Score
            total_score = self.w_rl * s_rl + self.w_cdft * s_cdft + self.w_boltzmann * s_bg + self.w_egnn * s_egnn

            cand = RankedCandidate(
                rank=0,
                name=name,
                funnel_score=float(total_score),
                rl_reward=rl_reward,
                wall_pressure_bar=p_w,
                target_wall_pressure_bar=self.min_p_wall,
                solvation_free_energy_kcal_mol=solv_e,
                target_solvation_kcal=self.max_solv,
                bg_log_likelihood=bg_logp,
                bg_energy_mean=bg_u_mean,
                bg_energy_var=bg_u_var,
                egnn_energy=egnn_u,
                egnn_force_rms=egnn_f_rms,
                molecular_weight=float(meta.get("mw", 0.0)),
                num_sites=int(meta.get("num_atoms", 0)),
                pmi_linearity=float(meta.get("pmi_linearity", 0.0)),
                aromatic_density=float(meta.get("aromatic_density", 0.0)),
                rotatable_fraction=float(meta.get("rotatable_fraction", 0.0)),
                mol2_content=meta.get("mol2", ""),
            )
            evaluated.append(cand)

        # Sort by total score descending
        evaluated.sort(key=lambda c: c.funnel_score, reverse=True)

        # Compute Pareto optimality across (P_wall, bg_log_likelihood, -egnn_energy, -egnn_force_rms, rl_reward)
        for i, c1 in enumerate(evaluated):
            c1.rank = i + 1
            is_dominated = False
            # If EGNN energy is populated, use quantum energy, else fall back to classical
            u1 = c1.egnn_energy if c1.egnn_energy != 0.0 else c1.bg_energy_mean
            f1 = c1.egnn_force_rms

            for j, c2 in enumerate(evaluated):
                if i != j:
                    u2 = c2.egnn_energy if c2.egnn_energy != 0.0 else c2.bg_energy_mean
                    f2 = c2.egnn_force_rms

                    if (
                        c2.wall_pressure_bar >= c1.wall_pressure_bar
                        and c2.bg_log_likelihood >= c1.bg_log_likelihood
                        and u2 <= u1
                        and f2 <= f1
                        and c2.rl_reward >= c1.rl_reward
                        and (
                            c2.wall_pressure_bar > c1.wall_pressure_bar
                            or c2.bg_log_likelihood > c1.bg_log_likelihood
                            or u2 < u1
                            or f2 < f1
                            or c2.rl_reward > c1.rl_reward
                        )
                    ):
                        is_dominated = True
                        break
            c1.is_pareto_optimal = not is_dominated

        return evaluated

    def export_results(
        self,
        ranked_candidates: List[RankedCandidate],
        out_dir: str | Path,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Exports top K candidates to .mol2 files, CSV table, and Markdown report.
        """
        out_path = Path(out_dir)
        mol2_dir = out_path / "top_candidates_mol2"
        mol2_dir.mkdir(parents=True, exist_ok=True)

        top_list = ranked_candidates[:top_k]

        # 1. Export top .mol2 structures
        for cand in top_list:
            if cand.mol2_content:
                mol2_file = mol2_dir / f"rank_{cand.rank:02d}_{cand.name}.mol2"
                mol2_file.write_text(cand.mol2_content, encoding="utf-8")

        # 2. Export CSV summary
        csv_file = out_path / "funnel_summary.csv"
        if top_list:
            fieldnames = list(top_list[0].to_dict().keys())
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for cand in top_list:
                    writer.writerow(cand.to_dict())

        # 3. Export Markdown summary report
        md_file = out_path / "funnel_report.md"
        lines = [
            "# Generative Molecular Funnel Report (Quantum-Refined)",
            "",
            f"**Total Candidates Screened**: {len(ranked_candidates)}",
            f"**Top Candidates Exported**: {len(top_list)}",
            "",
            "| Rank | Candidate | Score | P_wall (bar) | ln p(x) | <U_3D> (K) | U_EGNN (K) | F_RMS | MW (amu) | Sites | Pareto |",
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for cand in top_list:
            pareto_str = "Yes" if cand.is_pareto_optimal else "No"
            lines.append(
                f"| {cand.rank} | `{cand.name}` | **{cand.funnel_score:+.3f}** | "
                f"{cand.wall_pressure_bar:.1f} / {cand.target_wall_pressure_bar:.1f} | "
                f"{cand.bg_log_likelihood:+.2f} | {cand.bg_energy_mean:.1f} | "
                f"{cand.egnn_energy:.1f} | {cand.egnn_force_rms:.2f} | "
                f"{cand.molecular_weight:.1f} | {cand.num_sites} | {pareto_str} |"
            )

        md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {
            "total_candidates": len(ranked_candidates),
            "top_k_exported": len(top_list),
            "csv_path": str(csv_file),
            "report_path": str(md_file),
            "mol2_dir": str(mol2_dir),
            "top_candidates": [c.to_dict() for c in top_list],
        }
