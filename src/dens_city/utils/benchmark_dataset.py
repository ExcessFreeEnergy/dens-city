"""
Universal Solvation Benchmark Dataset Abstraction Layer.
Provides standardized protocols and dataset implementations for FreeSolv (aqueous single-solvent)
and Solv@TUM / Solvatum (non-aqueous multi-solvent ~6,200 pairs), supporting arbitrary molecular
benchmarking, dynamic .mol2 generation, and coupled simulation verification.
"""

from __future__ import annotations

import pickle
import re
import shutil
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dens_city.utils.materials import Material, MaterialLoader
from dens_city.utils.solvents import get_solvent_dielectric, normalize_solvent_name

# Standard gas constant and standard state temperature conversion:
# Delta G_solv = -ln(10) * R * T * log10(K)
# At T = 298.15 K and R = 1.98720425864083e-3 kcal/(mol*K):
# Prefactor = ln(10) * R * T approx 1.364233 kcal/mol
RT_LN10_KCAL_MOL: float = 1.36423297


@dataclass
class SolvationBenchmarkEntry:
    """Represents a single experimental or calculated solute-solvent solvation measurement."""

    solute_id: str
    solute_name: str
    solvent_name: str
    solvent_dielectric: float
    expt_dG_solv: float  # kcal/mol
    expt_uncertainty: Optional[float] = None  # kcal/mol
    calc_dG_solv: Optional[float] = None  # Baseline computational free energy (e.g. GAFF)
    smiles: Optional[str] = None
    formula: Optional[str] = None
    charge: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)


class BenchmarkDataset(ABC):
    """Abstract Base Class for universal solvation datasets."""

    def __init__(self, name: str, db_path: Optional[Path] = None, repo_root: Optional[Path] = None):
        self.name = name.lower()
        self.repo_root = repo_root or Path.cwd()
        self.db_path = db_path

    @abstractmethod
    def load_entries(self) -> List[SolvationBenchmarkEntry]:
        """Loads and returns all solute-solvent benchmark entries."""
        pass

    @abstractmethod
    def get_unique_solutes(self) -> Dict[str, Dict[str, Any]]:
        """Returns map of unique solute_id -> metadata dictionary."""
        pass

    @abstractmethod
    def get_unique_solvents(self) -> List[str]:
        """Returns list of unique solvent names evaluated in this dataset."""
        pass

    @abstractmethod
    def get_material(self, solute_id: str) -> Optional[Material]:
        """Resolves and instantiates a dens-city Material for the given solute_id."""
        pass

    @abstractmethod
    def populate_test_data(
        self,
        dest_dir: Path,
        max_molecules: Optional[int] = None,
    ) -> int:
        """Extracts/writes canonical .mol2 files into target directory for simulation."""
        pass


class FreeSolvDataset(BenchmarkDataset):
    """FreeSolv database: 642 small organic solutes in aqueous liquid (epsilon_r = 78.4)."""

    def __init__(self, db_path: Optional[Path] = None, repo_root: Optional[Path] = None):
        root = repo_root or Path.cwd()
        default_db = root / "FreeSolv" / "database.pickle"
        if not default_db.exists():
            alt = root / "data" / "database.pickle"
            if alt.exists():
                default_db = alt
        super().__init__(name="freesolv", db_path=db_path or default_db, repo_root=root)
        self._cache: Optional[Dict[str, Any]] = None
        self._entries: Optional[List[SolvationBenchmarkEntry]] = None

    def _load_raw_db(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if self.db_path is None or not self.db_path.exists():
            raise FileNotFoundError(f"FreeSolv database not found at {self.db_path}")
        with open(self.db_path, "rb") as f:
            self._cache = pickle.load(f)
        return self._cache

    def load_entries(self) -> List[SolvationBenchmarkEntry]:
        if self._entries is not None:
            return self._entries

        raw = self._load_raw_db()
        entries = []
        for solute_id, data in raw.items():
            expt = float(data.get("expt", 0.0))
            calc = float(data.get("calc", 0.0)) if "calc" in data else None
            dexpt = float(data.get("dcalc", data.get("dexpt", 0.0))) if "dcalc" in data or "dexpt" in data else None
            entry = SolvationBenchmarkEntry(
                solute_id=solute_id,
                solute_name=data.get("iupac", solute_id),
                solvent_name="WATER",
                solvent_dielectric=78.4,
                expt_dG_solv=expt,
                expt_uncertainty=dexpt,
                calc_dG_solv=calc,
                smiles=data.get("smiles"),
                formula=data.get("formula"),
                charge=0.0,
                properties=data,
            )
            entries.append(entry)
        self._entries = entries
        return entries

    def get_unique_solutes(self) -> Dict[str, Dict[str, Any]]:
        return self._load_raw_db()

    def get_unique_solvents(self) -> List[str]:
        return ["WATER"]

    def get_material(self, solute_id: str) -> Optional[Material]:
        # Look for .mol2 in test_data or FreeSolv mol2files_gaff
        candidates = [
            self.repo_root / "data" / "test_data" / f"{solute_id}.mol2",
            self.repo_root / "FreeSolv" / "mol2files_gaff" / f"{solute_id}.mol2",
            self.repo_root / "data" / "mol2files_gaff" / f"{solute_id}.mol2",
        ]
        for p in candidates:
            if p.exists():
                return MaterialLoader.from_mol2_file(p, identifier=solute_id)
        return None

    def populate_test_data(
        self,
        dest_dir: Path,
        max_molecules: Optional[int] = None,
    ) -> int:
        dest_dir.mkdir(parents=True, exist_ok=True)
        freesolv_dir = self.repo_root / "FreeSolv"
        mol2_gaff_dir = freesolv_dir / "mol2files_gaff"

        if not (mol2_gaff_dir.exists() and any(mol2_gaff_dir.glob("*.mol2"))):
            tar_paths = [
                freesolv_dir / "mol2files_gaff.tar.gz",
                self.repo_root / "data" / "mol2files_gaff.tar.gz",
            ]
            for tp in tar_paths:
                if tp.exists():
                    with tarfile.open(tp, "r:gz") as tar:
                        if hasattr(tarfile, "data_filter"):
                            tar.extractall(path=self.repo_root / "data", filter="data")
                        else:
                            tar.extractall(path=self.repo_root / "data")
                    mol2_gaff_dir = self.repo_root / "data" / "mol2files_gaff"
                    break

        count = 0
        if mol2_gaff_dir.exists():
            files = sorted(list(mol2_gaff_dir.glob("*.mol2")))
            if max_molecules is not None:
                files = files[:max_molecules]
            for f in files:
                dst = dest_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                count += 1
        return count


class SolvatumDataset(BenchmarkDataset):
    """Solv@TUM (Solvatum) database: 658 solutes across 146 non-aqueous liquid solvents (~5,952 pairs)."""

    def __init__(self, db_path: Optional[Path] = None, repo_root: Optional[Path] = None):
        root = repo_root or Path.cwd()
        default_db = root / "Solvatum" / "solvatum" / "data" / "solvatum.sdf"
        if not default_db.exists():
            alt = root / "solvatum" / "data" / "solvatum.sdf"
            if alt.exists():
                default_db = alt
        super().__init__(name="solvatum", db_path=db_path or default_db, repo_root=root)
        self._mols: Optional[List[Any]] = None
        self._entries: Optional[List[SolvationBenchmarkEntry]] = None
        self._solutes_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._mol2_cache: Dict[str, str] = {}

    def _load_sdf(self) -> List[Any]:
        if self._mols is not None:
            return self._mols
        from rdkit import Chem

        if self.db_path is None or not self.db_path.exists():
            raise FileNotFoundError(f"Solvatum SDF database not found at {self.db_path}")

        suppl = Chem.SDMolSupplier(str(self.db_path), removeHs=False)
        self._mols = [m for m in suppl if m is not None]
        return self._mols

    @staticmethod
    def _rdkit_mol_to_tripos_mol2(mol: Any, name: str) -> str:
        """
        Converts an RDKit 3D molecule directly to Tripos .mol2 string with
        physically verified Sybyl/GAFF atom types adhering to pattern_tripos_mol2_forcefield_derivation.
        """
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        lines = [
            "@<TRIPOS>MOLECULE",
            name,
            f"{num_atoms:5d} {num_bonds:5d}     0     0     0",
            "SMALL",
            "NO_CHARGES",
            "",
            "",
            "@<TRIPOS>ATOM",
        ]

        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            sym = atom.GetSymbol()
            # Resolve Sybyl / GAFF atom type
            if sym == "C":
                atype = (
                    "ca"
                    if atom.GetIsAromatic()
                    else (
                        "c3"
                        if str(atom.GetHybridization()) == "SP3"
                        else ("c2" if str(atom.GetHybridization()) == "SP2" else "c1")
                    )
                )
            elif sym == "N":
                atype = "na" if atom.GetIsAromatic() else ("n3" if str(atom.GetHybridization()) == "SP3" else "n2")
            elif sym == "O":
                if atom.GetIsAromatic():
                    atype = "oa"
                elif any(nbr.GetSymbol() == "H" for nbr in atom.GetNeighbors()):
                    atype = "oh"
                elif len(atom.GetNeighbors()) >= 2:
                    atype = "os"
                else:
                    atype = "o"
            elif sym == "H":
                nbrs = atom.GetNeighbors()
                if nbrs and nbrs[0].GetIsAromatic():
                    atype = "ha"
                elif nbrs and nbrs[0].GetSymbol() in ("O", "N", "S"):
                    atype = "ho" if nbrs[0].GetSymbol() == "O" else ("hn" if nbrs[0].GetSymbol() == "N" else "hs")
                else:
                    atype = "hc"
            elif sym in ("F", "Cl", "Br", "I"):
                atype = sym.lower()
            elif sym == "S":
                atype = "ss" if atom.GetIsAromatic() else "s"
            else:
                atype = sym

            lines.append(
                f"{i + 1:5d} {sym}{i + 1:<4} {pos.x:10.4f} {pos.y:10.4f} {pos.z:10.4f} {atype:<4} 1 <1> 0.0000"
            )

        lines.append("@<TRIPOS>BOND")
        for i, bond in enumerate(mol.GetBonds()):
            b_ar = bond.GetIsAromatic()
            b_type = "ar" if b_ar else str(int(bond.GetBondTypeAsDouble()))
            lines.append(f"{i + 1:5d} {bond.GetBeginAtomIdx() + 1:5d} {bond.GetEndAtomIdx() + 1:5d} {b_type}")

        return "\n".join(lines) + "\n"

    def load_entries(self) -> List[SolvationBenchmarkEntry]:
        if self._entries is not None:
            return self._entries

        mols = self._load_sdf()
        entries = []

        for mol in mols:
            solute_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"solvatum_{len(entries)}"
            solute_name = mol.GetProp("Name") if mol.HasProp("Name") else solute_id
            smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else None
            formula = mol.GetProp("Molecular Formula") if mol.HasProp("Molecular Formula") else None
            try:
                charge = float(mol.GetProp("Charge")) if mol.HasProp("Charge") else 0.0
            except ValueError:
                charge = 0.0

            for prop in mol.GetPropNames():
                m = re.match(r"^logK \((.*)\)$", prop)
                if not m:
                    continue
                raw_solvent = m.group(1)
                canon_solvent = normalize_solvent_name(raw_solvent)
                eps_solvent = get_solvent_dielectric(canon_solvent)

                try:
                    log_k = float(mol.GetProp(prop))
                except ValueError:
                    continue

                # Thermodynamic conversion: Delta G_solv = -ln(10) * R * T * log10(K)
                dG_solv_expt = -RT_LN10_KCAL_MOL * log_k

                entry = SolvationBenchmarkEntry(
                    solute_id=solute_id,
                    solute_name=solute_name,
                    solvent_name=canon_solvent,
                    solvent_dielectric=eps_solvent,
                    expt_dG_solv=dG_solv_expt,
                    expt_uncertainty=None,
                    calc_dG_solv=None,
                    smiles=smiles,
                    formula=formula,
                    charge=charge,
                    properties={
                        "raw_solvent": raw_solvent,
                        "log_k": log_k,
                        "mean_polarizability": mol.GetProp("mean polarizability")
                        if mol.HasProp("mean polarizability")
                        else None,
                        "dipole_moment": mol.GetProp("dipole moment") if mol.HasProp("dipole moment") else None,
                    },
                )
                entries.append(entry)

        self._entries = entries
        return entries

    def get_unique_solutes(self) -> Dict[str, Dict[str, Any]]:
        if self._solutes_map is not None:
            return self._solutes_map

        mols = self._load_sdf()
        solutes = {}
        for mol in mols:
            solute_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"solvatum_{len(solutes)}"
            solute_name = mol.GetProp("Name") if mol.HasProp("Name") else solute_id
            solutes[solute_id] = {
                "id": solute_id,
                "name": solute_name,
                "smiles": mol.GetProp("SMILES") if mol.HasProp("SMILES") else "",
                "formula": mol.GetProp("Molecular Formula") if mol.HasProp("Molecular Formula") else "",
                "num_atoms": mol.GetNumAtoms(),
            }
        self._solutes_map = solutes
        return solutes

    def get_unique_solvents(self) -> List[str]:
        entries = self.load_entries()
        return sorted(list(set(e.solvent_name for e in entries)))

    def get_mol2_text(self, solute_id: str) -> Optional[str]:
        if solute_id in self._mol2_cache:
            return self._mol2_cache[solute_id]

        mols = self._load_sdf()
        target_mol = None
        for mol in mols:
            m_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            m_name = mol.GetProp("Name") if mol.HasProp("Name") else ""
            if m_id == solute_id or m_name.upper() == solute_id.upper():
                target_mol = mol
                break

        if target_mol is None:
            return None

        name = target_mol.GetProp("Name") if target_mol.HasProp("Name") else solute_id
        mol2_txt = self._rdkit_mol_to_tripos_mol2(target_mol, name)
        self._mol2_cache[solute_id] = mol2_txt
        return mol2_txt

    def get_material(self, solute_id: str) -> Optional[Material]:
        mols = self._load_sdf()
        target_mol = None
        for mol in mols:
            m_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            m_name = mol.GetProp("Name") if mol.HasProp("Name") else ""
            if (
                m_id == solute_id
                or m_name.upper() == solute_id.upper()
                or f"solvatum_{m_id}".lower() == solute_id.lower()
            ):
                target_mol = mol
                break

        if target_mol is not None:
            raw_name = target_mol.GetProp("Name") if target_mol.HasProp("Name") else solute_id
            m_id = target_mol.GetProp("_Name") if target_mol.HasProp("_Name") else solute_id
            clean_name = re.sub(r"[^\w\-]", "_", raw_name).strip("_").lower()
            mat_id = f"solvatum_{m_id}_{clean_name}"
        else:
            mat_id = solute_id

        # Check if exists as file on disk first
        candidates = [
            self.repo_root / "data" / "test_data" / f"{mat_id}.mol2",
            self.repo_root / "data" / "test_data" / f"{solute_id}.mol2",
            self.repo_root / "data" / "test_data" / "solvatum" / f"{solute_id}.mol2",
        ]
        for p in candidates:
            if p.exists():
                return MaterialLoader.from_mol2_file(p, identifier=mat_id)

        mol2_txt = self.get_mol2_text(solute_id)
        if mol2_txt:
            return MaterialLoader.from_mol2_string(mol2_txt, identifier=mat_id)
        return None

    def populate_test_data(
        self,
        dest_dir: Path,
        max_molecules: Optional[int] = None,
    ) -> int:
        dest_dir.mkdir(parents=True, exist_ok=True)
        mols = self._load_sdf()
        if max_molecules is not None:
            mols = mols[:max_molecules]

        count = 0
        for mol in mols:
            solute_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"solvatum_{count}"
            name = mol.GetProp("Name") if mol.HasProp("Name") else solute_id
            clean_name = re.sub(r"[^\w\-]", "_", name).strip("_").lower()
            out_filename = f"solvatum_{solute_id}_{clean_name}.mol2"
            out_path = dest_dir / out_filename

            if not out_path.exists():
                mol2_txt = self._rdkit_mol_to_tripos_mol2(mol, name)
                out_path.write_text(mol2_txt, encoding="utf-8")
            count += 1
        return count


def get_benchmark_dataset(
    dataset_name: str,
    db_path: Optional[Path | str] = None,
    repo_root: Optional[Path] = None,
) -> BenchmarkDataset:
    """Factory function returning instantiated BenchmarkDataset."""
    root = repo_root or Path.cwd()
    d_path = Path(db_path) if db_path else None
    name_clean = dataset_name.lower().strip()

    if name_clean in ("solvatum", "solv@tum", "multi_solvent"):
        return SolvatumDataset(db_path=d_path, repo_root=root)
    elif name_clean in ("freesolv", "aqueous", "default"):
        return FreeSolvDataset(db_path=d_path, repo_root=root)
    else:
        raise ValueError(f"Unknown benchmark dataset: '{dataset_name}'. Must be 'freesolv' or 'solvatum'.")
