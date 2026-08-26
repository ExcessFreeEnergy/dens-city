"""
Swarm Specification Loader & 3D Local Fragment Resolver.
Translates macroscopic material requirement YAMLs into C Swarm TargetSpecs and 3D fragment definitions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class SwarmSpecLoader:
    """Loads and translates YAML material specifications for the PufferLib RL Swarm."""

    @staticmethod
    def load_yaml(yaml_path: str | Path) -> Dict[str, Any]:
        """Loads and sanitizes an arbitrary YAML molecular specification file."""
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Specification YAML file not found: {p}")
        content = p.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(content)
        except yaml.scanner.ScannerError:
            sanitized = re.sub(r"\"([^\"]*)\"", lambda m: '"' + m.group(1).replace("\\", "\\\\") + '"', content)
            data = yaml.safe_load(sanitized)
        return data

    @classmethod
    def derive_target_spec(cls, spec_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Translates macroscopic YAML target limits into normalized multi-objective reward weights and bounds.
        """
        limits = spec_data.get("tensor_limits", {})
        default_max_mw = float(limits.get("max_molecular_weight", 850.0))

        if "rl_reward_targets" in spec_data:
            targets = spec_data["rl_reward_targets"]
            return {
                "target_elasticity": float(targets.get("target_elasticity", 0.25)),
                "target_tensile": float(targets.get("target_tensile", 0.25)),
                "target_toughness": float(targets.get("target_toughness", 0.25)),
                "target_lightweight": float(targets.get("target_lightweight", 0.25)),
                "max_solvation_kcal": float(targets.get("max_solvation_kcal", -3.0)),
                "min_wall_pressure_bar": float(targets.get("min_wall_pressure_bar", 15.0)),
                "max_molecular_weight": float(targets.get("max_molecular_weight", default_max_mw)),
                "min_valency": int(targets.get("min_valency", 2)),
            }

        max_mw = default_max_mw
        min_rings = float(limits.get("min_aromatic_rings", 0))
        min_rot = float(limits.get("min_rotatable_bonds", 0))
        min_f = float(limits.get("min_fluorine_count", 0))

        # Default balanced targets
        target_elasticity = 0.25
        target_tensile = 0.25
        target_toughness = 0.25
        target_lightweight = 0.25

        # Heuristic biasing based on YAML goals
        if min_rot > 2:
            target_elasticity = 0.45
            target_tensile = 0.20
        elif min_rings >= 3:
            target_tensile = 0.50
            target_elasticity = 0.15
        elif min_f > 0:
            target_toughness = 0.40
            target_lightweight = 0.30

        # Normalize sum of weights
        w_sum = target_elasticity + target_tensile + target_toughness + target_lightweight
        target_elasticity /= w_sum
        target_tensile /= w_sum
        target_toughness /= w_sum
        target_lightweight /= w_sum

        return {
            "target_elasticity": float(target_elasticity),
            "target_tensile": float(target_tensile),
            "target_toughness": float(target_toughness),
            "target_lightweight": float(target_lightweight),
            "max_solvation_kcal": -3.0,
            "min_wall_pressure_bar": 15.0,
            "max_molecular_weight": max_mw,
            "min_valency": 2,
        }

    @classmethod
    def generate_canonical_3d_fragment(
        cls, smiles_or_smarts: str, name: str = "fragment", seed: int = 42
    ) -> Optional[Dict[str, Any]]:
        """
        Generates canonical local 3D coordinates and outward attachment normals for a building block.
        """
        mol = Chem.MolFromSmiles(smiles_or_smarts)
        if mol is None:
            mol = Chem.MolFromSmarts(smiles_or_smarts)
        if mol is None:
            return None

        mol_h = Chem.AddHs(mol)
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=1, randomSeed=seed))
        if not cids:
            cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=1, useRandomCoords=True, randomSeed=seed))
            if cids:
                try:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=50)
                except Exception:
                    pass

        if not cids:
            return None

        conf = mol_h.GetConformer(0)
        atoms = []
        dummy_indices = []

        for i, atom in enumerate(mol_h.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            z = atom.GetAtomicNum()
            if z == 0:
                dummy_indices.append(i)
            atoms.append(
                {
                    "index": i,
                    "atomic_number": z,
                    "pos": (float(pos.x), float(pos.y), float(pos.z)),
                    "is_aromatic": atom.GetIsAromatic(),
                    "mass": float(atom.GetMass()),
                }
            )

        # Calculate attachment normal vectors from neighbor atoms to dummy atoms
        ports = []
        for d_idx in dummy_indices:
            dummy_atom = mol_h.GetAtomWithIdx(d_idx)
            nbrs = dummy_atom.GetNeighbors()
            if nbrs:
                nbr_idx = nbrs[0].GetIdx()
                p_dummy = np.array(atoms[d_idx]["pos"])
                p_nbr = np.array(atoms[nbr_idx]["pos"])
                normal = p_dummy - p_nbr
                norm = np.linalg.norm(normal)
                normal = (normal / max(1e-6, norm)).tolist()
                ports.append(
                    {
                        "origin_atom": nbr_idx,
                        "pos": atoms[nbr_idx]["pos"],
                        "normal": normal,
                    }
                )

        return {
            "name": name,
            "smiles": smiles_or_smarts,
            "num_atoms": len(atoms),
            "atoms": atoms,
            "ports": ports,
            "molecular_weight": float(Descriptors.MolWt(mol_h)),
        }
