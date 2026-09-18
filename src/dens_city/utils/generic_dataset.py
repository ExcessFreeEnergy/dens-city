"""
Universal Solvation Dataset Abstraction for Arbitrary Datasets.
Supports loading arbitrary .sdf, .csv, .tsv, .jsonl, and .json files into standard
SolvationBenchmarkEntry and Material objects, decoupling dens-city from fixed benchmarks.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dens_city.utils.benchmark_dataset import (
    BenchmarkDataset,
    SolvationBenchmarkEntry,
    rdkit_mol_to_tripos_mol2,
)
from dens_city.utils.materials import Material, MaterialLoader
from dens_city.utils.solvents import get_solvent_dielectric, normalize_solvent_name

logger = logging.getLogger(__name__)

# Column / property alias definitions for flexible schema matching:
SOLUTE_ID_ALIASES = [
    "solute_id",
    "id",
    "compound_id",
    "mol_id",
    "molecule_id",
    "cid",
    "_name",
    "name",
    "compound",
    "solute",
]

SOLUTE_NAME_ALIASES = [
    "solute_name",
    "compound_name",
    "molecule_name",
    "mol_name",
    "iupac",
    "iupac_name",
    "common_name",
    "name",
    "compound",
    "title",
]

SMILES_ALIASES = [
    "smiles",
    "canonical_smiles",
    "smiles_str",
    "structure",
    "smiles_string",
    "isosmiles",
    "mol",
]

SOLVENT_ALIASES = [
    "solvent",
    "solvent_name",
    "solvent_id",
    "solv",
    "solvent_formula",
]

DIELECTRIC_ALIASES = [
    "solvent_dielectric",
    "dielectric",
    "epsilon",
    "epsilon_r",
    "eps",
    "dielectric_constant",
]

EXPT_DG_ALIASES = [
    "expt",
    "expt_dg_solv",
    "expt_dg",
    "dg",
    "dg_solv",
    "delta_g",
    "dG_solv",
    "dG_solv_expt",
    "experimental",
    "expt_val",
    "calc_dg_solv_expt",
    "f_expt",
]

UNCERTAINTY_ALIASES = [
    "expt_uncertainty",
    "uncertainty",
    "dexpt",
    "dcalc",
    "error",
    "std",
    "sem",
    "sigma",
]

CALC_DG_ALIASES = [
    "calc",
    "calc_dg_solv",
    "calc_dg",
    "baseline",
    "gaff",
    "calc_dG_solv",
]


def _match_alias(keys: Dict[str, Any], aliases: List[str]) -> Optional[Any]:
    """Helper to find the first matching key from aliases in a case-insensitive dictionary."""
    lower_map = {k.lower().strip(): v for k, v in keys.items()}
    for alias in aliases:
        if alias.lower() in lower_map:
            val = lower_map[alias.lower()]
            if val is not None and str(val).strip() != "":
                return val
    return None


class GenericSolvationDataset(BenchmarkDataset):
    """
    Ingests arbitrary chemical datasets (.sdf, .csv, .tsv, .jsonl, .json).
    Decouples dens-city from fixed benchmark conventions, providing universal
    conformer building, solvent property resolution, and benchmark entry serialization.
    """

    def __init__(
        self,
        file_path: Path | str,
        repo_root: Optional[Path] = None,
        default_solvent: str = "WATER",
    ):
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        super().__init__(name=path.stem, db_path=path, repo_root=repo_root or Path.cwd())
        self.file_path = path
        self.default_solvent = default_solvent

        self._entries: Optional[List[SolvationBenchmarkEntry]] = None
        self._solutes_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._mols_by_id: Dict[str, Any] = {}
        self._mol2_cache: Dict[str, str] = {}

    def load_entries(self) -> List[SolvationBenchmarkEntry]:
        if self._entries is not None:
            return self._entries

        suffix = self.file_path.suffix.lower()
        if suffix in (".sdf", ".sd"):
            entries = self._load_sdf()
        elif suffix in (".csv", ".tsv", ".tab", ".txt"):
            entries = self._load_delimited()
        elif suffix in (".jsonl", ".json"):
            entries = self._load_json()
        else:
            # Probe file content:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
            if first_line.strip().startswith("{") or first_line.strip().startswith("["):
                entries = self._load_json()
            elif "\t" in first_line:
                entries = self._load_delimited(delimiter="\t")
            else:
                entries = self._load_delimited(delimiter=",")

        self._entries = entries
        return entries

    def _load_sdf(self) -> List[SolvationBenchmarkEntry]:
        try:
            from rdkit import Chem
        except ImportError as err:
            raise ImportError("RDKit is required to load SDF files into GenericSolvationDataset.") from err

        suppl = Chem.SDMolSupplier(str(self.file_path), removeHs=False)
        entries: List[SolvationBenchmarkEntry] = []

        for idx, mol in enumerate(suppl):
            if mol is None:
                continue

            props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
            if mol.HasProp("_Name"):
                props["_name"] = mol.GetProp("_Name")

            # Extract solute ID & Name
            solute_id_val = _match_alias(props, SOLUTE_ID_ALIASES)
            solute_id = str(solute_id_val).strip() if solute_id_val is not None else f"mol_{idx + 1}"
            solute_name_val = _match_alias(props, SOLUTE_NAME_ALIASES)
            solute_name = str(solute_name_val).strip() if solute_name_val is not None else solute_id

            # Extract SMILES
            smiles_val = _match_alias(props, SMILES_ALIASES)
            if smiles_val:
                smiles = str(smiles_val).strip()
            else:
                try:
                    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                except Exception:
                    smiles = None

            # Extract solvent & dielectric
            solv_val = _match_alias(props, SOLVENT_ALIASES)
            solvent_name = normalize_solvent_name(str(solv_val) if solv_val is not None else self.default_solvent)
            diel_val = _match_alias(props, DIELECTRIC_ALIASES)
            try:
                dielectric = float(diel_val) if diel_val is not None else get_solvent_dielectric(solvent_name)
            except (ValueError, TypeError):
                dielectric = get_solvent_dielectric(solvent_name)

            # Extract expt & baseline
            expt_val = _match_alias(props, EXPT_DG_ALIASES)
            try:
                expt_dG = float(expt_val) if expt_val is not None else 0.0
            except (ValueError, TypeError):
                expt_dG = 0.0

            unc_val = _match_alias(props, UNCERTAINTY_ALIASES)
            try:
                unc = float(unc_val) if unc_val is not None else None
            except (ValueError, TypeError):
                unc = None

            calc_val = _match_alias(props, CALC_DG_ALIASES)
            try:
                calc_dG = float(calc_val) if calc_val is not None else None
            except (ValueError, TypeError):
                calc_dG = None

            # Formula
            try:
                from rdkit.Chem.rdMolDescriptors import CalcMolFormula

                formula = CalcMolFormula(mol)
            except Exception:
                formula = None

            entry = SolvationBenchmarkEntry(
                solute_id=solute_id,
                solute_name=solute_name,
                solvent_name=solvent_name,
                solvent_dielectric=dielectric,
                expt_dG_solv=expt_dG,
                expt_uncertainty=unc,
                calc_dG_solv=calc_dG,
                smiles=smiles,
                formula=formula,
                charge=0.0,
                properties=props,
            )
            entries.append(entry)
            self._mols_by_id[solute_id] = mol

        return entries

    def _load_delimited(self, delimiter: Optional[str] = None) -> List[SolvationBenchmarkEntry]:
        entries: List[SolvationBenchmarkEntry] = []
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            if delimiter is None:
                sample = f.read(4096)
                f.seek(0)
                if "\t" in sample and "," not in sample:
                    delimiter = "\t"
                elif "," in sample:
                    delimiter = ","
                else:
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        delimiter = dialect.delimiter
                    except Exception:
                        delimiter = ","

            reader = csv.DictReader(f, delimiter=delimiter)
            for idx, row in enumerate(reader):
                if not row or all(v is None or v.strip() == "" for v in row.values()):
                    continue

                solute_id_val = _match_alias(row, SOLUTE_ID_ALIASES)
                solute_id = str(solute_id_val).strip() if solute_id_val is not None else f"row_{idx + 1}"
                solute_name_val = _match_alias(row, SOLUTE_NAME_ALIASES)
                solute_name = str(solute_name_val).strip() if solute_name_val is not None else solute_id

                smiles_val = _match_alias(row, SMILES_ALIASES)
                smiles = str(smiles_val).strip() if smiles_val is not None else None

                solv_val = _match_alias(row, SOLVENT_ALIASES)
                solvent_name = normalize_solvent_name(str(solv_val) if solv_val is not None else self.default_solvent)

                diel_val = _match_alias(row, DIELECTRIC_ALIASES)
                try:
                    dielectric = float(diel_val) if diel_val is not None else get_solvent_dielectric(solvent_name)
                except (ValueError, TypeError):
                    dielectric = get_solvent_dielectric(solvent_name)

                expt_val = _match_alias(row, EXPT_DG_ALIASES)
                try:
                    expt_dG = float(expt_val) if expt_val is not None else 0.0
                except (ValueError, TypeError):
                    expt_dG = 0.0

                unc_val = _match_alias(row, UNCERTAINTY_ALIASES)
                try:
                    unc = float(unc_val) if unc_val is not None else None
                except (ValueError, TypeError):
                    unc = None

                calc_val = _match_alias(row, CALC_DG_ALIASES)
                try:
                    calc_dG = float(calc_val) if calc_val is not None else None
                except (ValueError, TypeError):
                    calc_dG = None

                entry = SolvationBenchmarkEntry(
                    solute_id=solute_id,
                    solute_name=solute_name,
                    solvent_name=solvent_name,
                    solvent_dielectric=dielectric,
                    expt_dG_solv=expt_dG,
                    expt_uncertainty=unc,
                    calc_dG_solv=calc_dG,
                    smiles=smiles,
                    charge=0.0,
                    properties=dict(row),
                )
                entries.append(entry)

        return entries

    def _load_json(self) -> List[SolvationBenchmarkEntry]:
        entries: List[SolvationBenchmarkEntry] = []
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()

        raw_records = []
        if content.startswith("["):
            raw_records = json.loads(content)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        for idx, rec in enumerate(raw_records):
            if not isinstance(rec, dict):
                continue

            solute_id_val = _match_alias(rec, SOLUTE_ID_ALIASES)
            solute_id = str(solute_id_val).strip() if solute_id_val is not None else f"item_{idx + 1}"
            solute_name_val = _match_alias(rec, SOLUTE_NAME_ALIASES)
            solute_name = str(solute_name_val).strip() if solute_name_val is not None else solute_id

            smiles_val = _match_alias(rec, SMILES_ALIASES)
            smiles = str(smiles_val).strip() if smiles_val is not None else None

            solv_val = _match_alias(rec, SOLVENT_ALIASES)
            solvent_name = normalize_solvent_name(str(solv_val) if solv_val is not None else self.default_solvent)

            diel_val = _match_alias(rec, DIELECTRIC_ALIASES)
            try:
                dielectric = float(diel_val) if diel_val is not None else get_solvent_dielectric(solvent_name)
            except (ValueError, TypeError):
                dielectric = get_solvent_dielectric(solvent_name)

            expt_val = _match_alias(rec, EXPT_DG_ALIASES)
            try:
                expt_dG = float(expt_val) if expt_val is not None else 0.0
            except (ValueError, TypeError):
                expt_dG = 0.0

            unc_val = _match_alias(rec, UNCERTAINTY_ALIASES)
            try:
                unc = float(unc_val) if unc_val is not None else None
            except (ValueError, TypeError):
                unc = None

            calc_val = _match_alias(rec, CALC_DG_ALIASES)
            try:
                calc_dG = float(calc_val) if calc_val is not None else None
            except (ValueError, TypeError):
                calc_dG = None

            entry = SolvationBenchmarkEntry(
                solute_id=solute_id,
                solute_name=solute_name,
                solvent_name=solvent_name,
                solvent_dielectric=dielectric,
                expt_dG_solv=expt_dG,
                expt_uncertainty=unc,
                calc_dG_solv=calc_dG,
                smiles=smiles,
                charge=0.0,
                properties=rec,
            )
            entries.append(entry)

        return entries

    def get_unique_solutes(self) -> Dict[str, Dict[str, Any]]:
        if self._solutes_map is not None:
            return self._solutes_map

        entries = self.load_entries()
        solutes: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            if e.solute_id not in solutes:
                solutes[e.solute_id] = {
                    "id": e.solute_id,
                    "name": e.solute_name,
                    "smiles": e.smiles,
                    "formula": e.formula,
                    "properties": e.properties,
                }
        self._solutes_map = solutes
        return solutes

    def get_unique_solvents(self) -> List[str]:
        entries = self.load_entries()
        return sorted(list(set(e.solvent_name for e in entries)))

    def get_material(self, solute_id: str) -> Optional[Material]:
        # 1. Check if candidate .mol2 already exists on disk
        candidates = [
            self.repo_root / "data" / "test_data" / f"{solute_id}.mol2",
            self.repo_root / "data" / "test_data" / "solvatum" / f"{solute_id}.mol2",
            self.repo_root / "FreeSolv" / "mol2files_gaff" / f"{solute_id}.mol2",
        ]
        for p in candidates:
            if p.exists():
                return MaterialLoader.from_mol2_file(p, identifier=solute_id)

        # 2. Check if we already have an RDKit Mol in memory
        mol = self._mols_by_id.get(solute_id)
        if mol is not None:
            return MaterialLoader.from_rdkit_mol(mol, identifier=solute_id)

        # 3. Check if we have an entry with SMILES
        entries = self.load_entries()
        matched_entry = next((e for e in entries if e.solute_id == solute_id), None)
        if matched_entry and matched_entry.smiles:
            try:
                from rdkit import Chem

                m = Chem.MolFromSmiles(matched_entry.smiles)
                if m is not None:
                    self._mols_by_id[solute_id] = m
                    return MaterialLoader.from_rdkit_mol(m, identifier=solute_id)
            except Exception as e:
                logger.warning("Failed to construct RDKit Mol from SMILES %s: %s", matched_entry.smiles, e)

        return None

    def populate_test_data(
        self,
        dest_dir: Path,
        max_molecules: Optional[int] = None,
    ) -> int:
        dest_dir.mkdir(parents=True, exist_ok=True)
        solutes = self.get_unique_solutes()
        solute_ids = list(solutes.keys())
        if max_molecules is not None:
            solute_ids = solute_ids[:max_molecules]

        count = 0
        for sid in solute_ids:
            out_path = dest_dir / f"{sid}.mol2"
            if out_path.exists():
                count += 1
                continue

            mol = self._mols_by_id.get(sid)
            if mol is None:
                meta = solutes[sid]
                smiles = meta.get("smiles")
                if smiles:
                    try:
                        from rdkit import Chem

                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            self._mols_by_id[sid] = mol
                    except Exception:
                        pass

            if mol is not None:
                # Ensure 3D conformer exists for .mol2 serialization
                try:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    mol_3d = Chem.Mol(mol)
                    if mol_3d.GetNumConformers() == 0:
                        mol_3d = Chem.AddHs(mol_3d)
                        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=100)
                    mol2_txt = rdkit_mol_to_tripos_mol2(mol_3d, solutes[sid]["name"])
                    out_path.write_text(mol2_txt, encoding="utf-8")
                    count += 1
                except Exception as e:
                    logger.warning("Failed to generate .mol2 for %s: %s", sid, e)

        return count
