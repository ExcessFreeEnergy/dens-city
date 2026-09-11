"""
Artifact Pool Store & Lifecycle Manager for dens-city server.
Handles persistent on-disk storage of molecular pools across stage boundaries,
intelligent power-of-2 chunking for 10,000+ molecules, and disk creep garbage collection.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ArtifactPoolStore:
    """
    Manages structured, stateless molecular pools on disk under `runs/pools/{pool_id}/`.
    """

    def __init__(self, root_dir: str | Path = "runs/pools"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _generate_pool_id(self, prefix: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rand = uuid.uuid4().hex[:6]
        return f"{prefix}_{ts}_{rand}"

    def create_candidate_pool(
        self,
        candidate_metadata: List[Dict[str, Any]],
        spec_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Stores a candidate pool (Stage 2 output).
        """
        pool_id = self._generate_pool_id("pool_cand")
        pool_dir = self.root_dir / pool_id
        mol2_dir = pool_dir / "mol2"
        mol2_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "pool_id": pool_id,
            "pool_type": "candidate_pool",
            "created_at": time.time(),
            "count": len(candidate_metadata),
            "spec_data": spec_data or {},
        }
        (pool_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        csv_path = pool_dir / "molecules.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "name", "smiles", "num_atoms", "mw", "sa_score", "rl_reward"])
            for m in candidate_metadata:
                writer.writerow(
                    [
                        m.get("index", 0),
                        m.get("name", ""),
                        m.get("smiles", ""),
                        m.get("num_atoms", 0),
                        m.get("mw", 0.0),
                        m.get("sa_score", 0.0),
                        m.get("rl_reward", 0.0),
                    ]
                )
                mol2_str = m.get("mol2", "")
                if mol2_str:
                    cand_file = mol2_dir / f"{m.get('name', 'cand')}.mol2"
                    cand_file.write_text(mol2_str, encoding="utf-8")

        return pool_id

    def create_thermo_pool(
        self,
        pipeline_results: List[Any],
        candidate_metadata: List[Dict[str, Any]],
        solvent_id: str = "water",
        parent_pool_id: Optional[str] = None,
    ) -> str:
        """
        Stores relaxed thermodynamics pool (Stage 3 output).
        """
        pool_id = self._generate_pool_id("pool_thermo")
        pool_dir = self.root_dir / pool_id
        pool_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "pool_id": pool_id,
            "pool_type": "thermo_pool",
            "parent_pool_id": parent_pool_id,
            "solvent_id": solvent_id,
            "created_at": time.time(),
            "count": len(pipeline_results),
        }
        (pool_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        results_file = pool_dir / "thermo_results.jsonl"
        with open(results_file, "w", encoding="utf-8") as f:
            for r, meta in zip(pipeline_results, candidate_metadata):
                rec = {
                    "material_name": r.material_name,
                    "smiles": meta.get("smiles", ""),
                    "wall_pressure_bar": float(r.wall_pressure_bar),
                    "excess_adsorption_a2": float(r.excess_adsorption_a2),
                    "cdft_final_loss": float(r.cdft_final_loss),
                    "bg_log_likelihood": float(r.bg_log_likelihood),
                    "bg_energy_mean": float(r.bg_energy_mean),
                    "bg_energy_var": float(r.bg_energy_var),
                    "solvation_free_energy_kcal_mol": float(r.solvation_free_energy_kcal_mol),
                    "metadata": meta,
                }
                f.write(json.dumps(rec) + "\n")

        return pool_id

    def create_egnn_scored_pool(
        self,
        pipeline_results: List[Any],
        candidate_metadata: List[Dict[str, Any]],
        parent_pool_id: Optional[str] = None,
    ) -> str:
        """
        Stores EGNN quantum-scored pool (Stage 4 output).
        """
        pool_id = self._generate_pool_id("pool_egnn")
        pool_dir = self.root_dir / pool_id
        pool_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "pool_id": pool_id,
            "pool_type": "egnn_scored_pool",
            "parent_pool_id": parent_pool_id,
            "created_at": time.time(),
            "count": len(pipeline_results),
        }
        (pool_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        results_file = pool_dir / "scored_results.jsonl"
        with open(results_file, "w", encoding="utf-8") as f:
            for r, meta in zip(pipeline_results, candidate_metadata):
                rec = {
                    "material_name": r.material_name,
                    "smiles": meta.get("smiles", ""),
                    "wall_pressure_bar": float(r.wall_pressure_bar),
                    "excess_adsorption_a2": float(r.excess_adsorption_a2),
                    "bg_log_likelihood": float(r.bg_log_likelihood),
                    "bg_energy_mean": float(r.bg_energy_mean),
                    "bg_energy_var": float(r.bg_energy_var),
                    "solvation_free_energy_kcal_mol": float(r.solvation_free_energy_kcal_mol),
                    "egnn_energy": float(r.egnn_energy) if hasattr(r, "egnn_energy") else 0.0,
                    "egnn_force_rms": float(r.egnn_force_rms) if hasattr(r, "egnn_force_rms") else 0.0,
                    "metadata": meta,
                }
                f.write(json.dumps(rec) + "\n")

        return pool_id

    def load_pool_manifest(self, pool_id: str) -> Dict[str, Any]:
        """Loads the pool manifest or raises FileNotFoundError."""
        manifest_path = self.root_dir / pool_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pool not found: {pool_id}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_pool_candidates(self, pool_id: str) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
        """
        Loads metadata and pipeline results from a pool.
        """
        pool_dir = self.root_dir / pool_id
        if not pool_dir.exists():
            raise FileNotFoundError(f"Pool directory does not exist: {pool_id}")

        manifest = self.load_pool_manifest(pool_id)
        p_type = manifest.get("pool_type", "")

        metadata: List[Dict[str, Any]] = []
        pipeline_results: Optional[List[Any]] = None

        if p_type == "candidate_pool":
            csv_path = pool_dir / "molecules.csv"
            mol2_dir = pool_dir / "mol2"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("name", "")
                        mol2_file = mol2_dir / f"{name}.mol2"
                        mol2_content = mol2_file.read_text(encoding="utf-8") if mol2_file.exists() else ""
                        metadata.append(
                            {
                                "index": int(row.get("index", 0)),
                                "name": name,
                                "smiles": row.get("smiles", ""),
                                "num_atoms": int(row.get("num_atoms", 0)),
                                "mw": float(row.get("mw", 0.0)),
                                "sa_score": float(row.get("sa_score", 0.0)) if row.get("sa_score") else None,
                                "rl_reward": float(row.get("rl_reward", 0.0)),
                                "mol2": mol2_content,
                            }
                        )

        elif p_type in ("thermo_pool", "egnn_scored_pool"):
            jsonl_file = (
                pool_dir / "scored_results.jsonl" if p_type == "egnn_scored_pool" else pool_dir / "thermo_results.jsonl"
            )
            from dens_city.utils.pipeline import MaterialPipelineResult, PipelineStatus

            pipeline_results = []
            if jsonl_file.exists():
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        meta = d.get("metadata", {})
                        metadata.append(meta)

                        res = MaterialPipelineResult(
                            material_name=d["material_name"],
                            status=PipelineStatus.SUCCESS.value,
                            runtime_seconds=0.0,
                            num_sites=meta.get("num_atoms", 0),
                            wall_pressure_bar=d.get("wall_pressure_bar", 0.0),
                            excess_adsorption_a2=d.get("excess_adsorption_a2", 0.0),
                            cdft_final_loss=d.get("cdft_final_loss", 0.0),
                            bg_log_likelihood=d.get("bg_log_likelihood", 0.0),
                            bg_energy_mean=d.get("bg_energy_mean", 0.0),
                            bg_energy_var=d.get("bg_energy_var", 0.0),
                            solvation_free_energy_kcal_mol=d.get("solvation_free_energy_kcal_mol", 0.0),
                        )
                        if "egnn_energy" in d:
                            res.egnn_energy = d["egnn_energy"]
                        if "egnn_force_rms" in d:
                            res.egnn_force_rms = d["egnn_force_rms"]
                        pipeline_results.append(res)

        return metadata, pipeline_results

    @staticmethod
    def chunk_molecules(items: List[Any], chunk_size: int = 1024) -> List[List[Any]]:
        """
        Splits a large list of molecules into static power-of-2 chunks (e.g. 1024 or 512)
        to prevent GPU out-of-memory or CPU thread exhaustion when handling 10,000+ items.
        """
        # Ensure chunk_size is clamped to reasonable power-of-2
        valid_chunks = [32, 64, 128, 256, 512, 1024, 2048]
        eff_chunk = min(valid_chunks, key=lambda c: abs(c - chunk_size))
        return [items[i : i + eff_chunk] for i in range(0, len(items), eff_chunk)]

    def cleanup_pool(self, pool_id: str) -> int:
        """Deletes a single pool directory, returning freed bytes."""
        pool_dir = self.root_dir / pool_id
        if not pool_dir.exists():
            return 0
        total_size = sum(f.stat().st_size for f in pool_dir.glob("**/*") if f.is_file())
        shutil.rmtree(pool_dir, ignore_errors=True)
        return total_size

    def prune_intermediate_pools(self, keep_pareto: bool = True) -> Tuple[List[str], int]:
        """
        Prunes intermediate candidate, thermo, and egnn pools while retaining final Pareto exports.
        """
        deleted: List[str] = []
        freed = 0
        for p in self.root_dir.iterdir():
            if not p.is_dir():
                continue
            manifest_path = p / "manifest.json"
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                    p_type = m.get("pool_type", "")
                    if keep_pareto and p_type in ("candidate_pool", "thermo_pool", "egnn_scored_pool"):
                        freed += self.cleanup_pool(p.name)
                        deleted.append(p.name)
                except Exception:
                    pass
        return deleted, freed

    def prune_older_than(self, hours: float) -> Tuple[List[str], int]:
        """
        Deletes pools with creation timestamp older than `hours`.
        """
        cutoff = time.time() - (hours * 3600.0)
        deleted: List[str] = []
        freed = 0
        for p in self.root_dir.iterdir():
            if not p.is_dir():
                continue
            manifest_path = p / "manifest.json"
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if m.get("created_at", 0) < cutoff:
                        freed += self.cleanup_pool(p.name)
                        deleted.append(p.name)
                except Exception:
                    pass
            elif p.stat().st_mtime < cutoff:
                freed += self.cleanup_pool(p.name)
                deleted.append(p.name)
        return deleted, freed

    def get_storage_stats(self) -> Dict[str, Any]:
        """Returns statistics on active pools and disk usage."""
        pools: List[Dict[str, Any]] = []
        total_bytes = 0
        for p in self.root_dir.iterdir():
            if p.is_dir():
                p_size = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
                total_bytes += p_size
                m_type = "unknown"
                m_path = p / "manifest.json"
                if m_path.exists():
                    try:
                        m_type = json.loads(m_path.read_text(encoding="utf-8")).get("pool_type", "unknown")
                    except Exception:
                        pass
                pools.append({"pool_id": p.name, "pool_type": m_type, "size_mb": p_size / (1024 * 1024)})

        return {
            "root_dir": str(self.root_dir),
            "pool_count": len(pools),
            "total_disk_mb": total_bytes / (1024 * 1024),
            "pools": pools,
        }
