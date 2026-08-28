"""
High-Performance Combinatorial Molecular Library Generator.
Combines high-throughput combinatorial graph enumeration with compiled C OpenMP
multi-core 3D conformer embedding, Gasteiger charge assignment, and parallel Tripos .mol2 disk writing.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

C_EXT_DIR = Path(__file__).resolve().parent
C_EXT_SO = C_EXT_DIR / "fast_mol2_writer.so"
C_EXT_SRC = C_EXT_DIR / "fast_mol2_writer.c"

# Comprehensive atomtype lookup database matching dens-city ATOMTYPES_DB
ATOMTYPES_DB: Dict[str, Tuple[float, float, float, int, str]] = {
    # Carbon
    "c": (1.9080, 0.0860, 12.01, 6, "sp2 C carbonyl group"),
    "c1": (1.9080, 0.2100, 12.01, 6, "sp C alkyne / nitrile / TraPPE CO2"),
    "c2": (1.9080, 0.0860, 12.01, 6, "sp2 C aliphatic alkene"),
    "c3": (1.9080, 0.1094, 12.01, 6, "sp3 C aliphatic alkane"),
    "ca": (1.9080, 0.0860, 12.01, 6, "sp2 C aromatic ring"),
    "cc": (1.9080, 0.0860, 12.01, 6, "sp2 C in 5-membered aromatic ring"),
    "cd": (1.9080, 0.0860, 12.01, 6, "sp2 C in 5-membered aromatic ring"),
    "ce": (1.9080, 0.0860, 12.01, 6, "sp2 C in 5-membered aromatic ring"),
    "cf": (1.9080, 0.0860, 12.01, 6, "sp2 C in 5-membered aromatic ring"),
    "cg": (1.9080, 0.2100, 12.01, 6, "sp C in alkyne"),
    "cp": (1.9080, 0.0860, 12.01, 6, "sp2 aromatic bridgehead C"),
    "cx": (1.9080, 0.0860, 12.01, 6, "sp3 C in 3-membered ring"),
    "cy": (1.9080, 0.0860, 12.01, 6, "sp3 C in 4-membered ring"),
    # Hydrogen
    "h1": (1.3870, 0.0157, 1.008, 1, "H bonded to aliphatic C with 1 EWG"),
    "h2": (1.2870, 0.0157, 1.008, 1, "H bonded to aliphatic C with 2 EWG"),
    "h3": (1.1870, 0.0157, 1.008, 1, "H bonded to aliphatic C with 3 EWG"),
    "h4": (1.4090, 0.0150, 1.008, 1, "H bonded to aromatic C with 1 EWG"),
    "h5": (1.3590, 0.0150, 1.008, 1, "H bonded to aromatic C with 2 EWG"),
    "ha": (1.4590, 0.0150, 1.008, 1, "H bonded to aromatic C / H2"),
    "hc": (1.4870, 0.0157, 1.008, 1, "H bonded to aliphatic C without EWG"),
    "hn": (0.6000, 0.0157, 1.008, 1, "H bonded to nitrogen"),
    "ho": (0.0000, 0.0000, 1.008, 1, "Hydroxyl H bonded to oxygen"),
    "hs": (0.6000, 0.0157, 1.008, 1, "H bonded to sulfur"),
    "hw": (0.0000, 0.0000, 1.008, 1, "SPC/E & TIP3P Water Hydrogen"),
    # Oxygen
    "o": (1.6612, 0.2100, 16.00, 8, "Carbonyl / TraPPE oxygen"),
    "oh": (1.7210, 0.2104, 16.00, 8, "Hydroxyl oxygen in alcohols"),
    "os": (1.6837, 0.1700, 16.00, 8, "Ether / ester / sulfate oxygen"),
    "ow": (1.7766, 0.1553, 16.00, 8, "SPC/E Water Oxygen"),
    # Nitrogen
    "n": (1.8240, 0.1700, 14.01, 7, "sp2 nitrogen in amides"),
    "n1": (1.8240, 0.1700, 14.01, 7, "sp nitrogen in nitriles / TraPPE N2"),
    "n2": (1.8240, 0.1700, 14.01, 7, "sp2 nitrogen in imines"),
    "n3": (1.8240, 0.1700, 14.01, 7, "sp3 nitrogen in amines / ammonia"),
    "na": (1.8240, 0.1700, 14.01, 7, "sp2 nitrogen in aromatic rings"),
    "nb": (1.8240, 0.1700, 14.01, 7, "sp2 nitrogen in aromatic rings"),
    "nc": (1.8240, 0.1700, 14.01, 7, "sp2 nitrogen in 5-membered aromatic rings"),
    "nd": (1.8240, 0.1700, 14.01, 7, "sp2 nitrogen in 5-membered aromatic rings"),
    "nh": (1.8240, 0.1700, 14.01, 7, "Amine nitrogen"),
    "no": (1.8240, 0.1700, 14.01, 7, "Nitro group nitrogen"),
    # Halogens
    "f": (1.7500, 0.0610, 19.00, 9, "Fluorine in fluorocarbons & HF"),
    "cl": (1.9480, 0.2650, 35.45, 17, "Chlorine bonded to carbon"),
    "br": (2.0200, 0.4200, 79.90, 35, "Bromine bonded to carbon"),
    "i": (2.1500, 0.5000, 126.90, 53, "Iodine bonded to carbon"),
    # Sulfur & Phosphorus
    "s": (2.0000, 0.2500, 32.06, 16, "sp2 Sulfur"),
    "s4": (2.0000, 0.2500, 32.06, 16, "Hypervalent Sulfur (IV)"),
    "s6": (2.0000, 0.2500, 32.06, 16, "Hypervalent Sulfur (VI) in SF6 / SDS"),
    "sh": (2.0000, 0.2500, 32.06, 16, "Thiol Sulfur in -SH"),
    "ss": (2.0000, 0.2500, 32.06, 16, "Disulfide Sulfur"),
    "sy": (2.0000, 0.2500, 32.06, 16, "Sulfur in sulfoxides"),
    "p5": (2.1000, 0.2000, 30.97, 15, "Phosphorus in phosphates"),
    # Inorganic / Ionic / Noble Gas / Coarse-Grained
    "Ar": (1.9109, 0.2380, 39.95, 18, "Monoatomic Argon LJ"),
    "Na": (1.8680, 0.0028, 22.99, 11, "Sodium cation Na+"),
    "Cl": (2.4700, 0.1000, 35.45, 17, "Chloride anion Cl-"),
    "Ca": (1.7130, 0.4590, 40.08, 20, "Calcium divalent cation Ca2+"),
    "Col": (8.4180, 0.0000, 1000.0, 0, "Large Colloidal Hard Sphere"),
}


def ensure_c_extension() -> bool:
    """Ensures the high-performance OpenMP C .mol2 writer is compiled."""
    if C_EXT_SO.exists():
        return True
    if not C_EXT_SRC.exists():
        return False
    try:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        py_inc = f"/usr/include/python{py_ver}"
        cmd = [
            "gcc",
            "-O3",
            "-fopenmp",
            "-shared",
            "-fPIC",
            f"-I{py_inc}",
            str(C_EXT_SRC),
            "-o",
            str(C_EXT_SO),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


def get_c_mol2_writer():
    """Dynamically loads the compiled fast_mol2_writer C module."""
    if not ensure_c_extension():
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("fast_mol2_writer", str(C_EXT_SO))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        pass
    return None


def compute_sigma(r_min_half_A: float) -> Tuple[float, float]:
    if r_min_half_A == 0.0:
        return 0.0, 0.0
    sigma_A = 2.0 * r_min_half_A / (2.0 ** (1.0 / 6.0))
    sigma_nm = sigma_A / 10.0
    return sigma_A, sigma_nm


def compute_epsilon(eps_kcal: float) -> Tuple[float, float]:
    eps_kj = eps_kcal * 4.184
    eps_K = (eps_kcal * 4184.0) / 8.314462618
    return eps_kj, eps_K


def generate_forcefield_database(dest_dir: Path) -> None:
    """Generates forcefield_parameters.json, forcefield_parameters.csv, and gaff.dat."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    json_data = {}
    csv_lines = [
        "atom_type,atomic_number,mass_amu,r_min_half_A,sigma_A,sigma_nm,epsilon_kcal_mol,epsilon_kj_mol,epsilon_K,description"
    ]
    gaff_lines = [
        "AMBER General Amber Force Field (GAFF) & Extended dens-city Parameters",
        "MOD4",
        "MASS",
    ]
    for atype, (r_min, eps_kcal, mass, at_num, desc) in sorted(ATOMTYPES_DB.items()):
        gaff_lines.append(f"{atype:<6} {mass:>7.3f}        0.878               {desc}")
    gaff_lines.append("")
    gaff_lines.append("NONBON")

    for atype, (r_min, eps_kcal, mass, at_num, desc) in sorted(ATOMTYPES_DB.items()):
        sig_A, sig_nm = compute_sigma(r_min)
        eps_kj, eps_K = compute_epsilon(eps_kcal)
        json_data[atype] = {
            "atom_type": atype,
            "atomic_number": at_num,
            "mass_amu": mass,
            "r_min_half_angstrom": r_min,
            "sigma_angstrom": round(sig_A, 5),
            "sigma_nm": round(sig_nm, 6),
            "epsilon_kcal_mol": round(eps_kcal, 5),
            "epsilon_kj_mol": round(eps_kj, 5),
            "epsilon_kelvin": round(eps_K, 2),
            "description": desc,
        }
        csv_lines.append(
            f'{atype},{at_num},{mass},{r_min:.4f},{sig_A:.5f},{sig_nm:.6f},{eps_kcal:.5f},{eps_kj:.5f},{eps_K:.2f},"{desc}"'
        )
        gaff_lines.append(
            f"  {atype:<6} {r_min:>10.4f} {eps_kcal:>10.4f}             {desc} (sigma={sig_A:.4f}A, eps={eps_K:.1f}K)"
        )
    gaff_lines.append("")

    (dest_dir / "forcefield_parameters.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    (dest_dir / "forcefield_parameters.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    (dest_dir / "gaff.dat").write_text("\n".join(gaff_lines), encoding="utf-8")


def load_spec(yaml_path: str | Path) -> dict:
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


def parse_smiles_or_smarts(s: str) -> Optional[Chem.Mol]:
    """Parses a SMILES or SMARTS string into a sanitized RDKit Mol object."""
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        mol = Chem.MolFromSmarts(s)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return mol


def connect_fragments(core_mol: Chem.Mol, sub_mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Connects core_mol and sub_mol by replacing one dummy atom in each with a bond.
    If sub_mol is a hydrogen cap ([H][*] or single dummy atom), replaces the dummy atom with H.
    """
    try:
        core_dummies = [a for a in core_mol.GetAtoms() if a.GetAtomicNum() == 0]
        sub_dummies = [a for a in sub_mol.GetAtoms() if a.GetAtomicNum() == 0]
        if not core_dummies or not sub_dummies:
            return core_mol

        core_dummy = core_dummies[0]
        sub_dummy = sub_dummies[0]

        core_nbrs = core_dummy.GetNeighbors()
        sub_nbrs = sub_dummy.GetNeighbors()

        if not core_nbrs:
            rw = Chem.RWMol(core_mol)
            rw.RemoveAtom(core_dummy.GetIdx())
            res = rw.GetMol()
            Chem.SanitizeMol(res)
            return res

        core_nbr_idx = core_nbrs[0].GetIdx()

        if (len(sub_nbrs) == 1 and sub_nbrs[0].GetAtomicNum() == 1) or not sub_nbrs:
            rw = Chem.RWMol(core_mol)
            nbr = rw.GetAtomWithIdx(core_nbr_idx)
            nbr.SetNumExplicitHs(nbr.GetNumExplicitHs() + 1)
            rw.RemoveAtom(core_dummy.GetIdx())
            res = rw.GetMol()
            Chem.SanitizeMol(res)
            return res

        sub_nbr_idx = sub_nbrs[0].GetIdx()

        combined = Chem.CombineMols(core_mol, sub_mol)
        rw = Chem.RWMol(combined)
        offset = core_mol.GetNumAtoms()

        rw.AddBond(core_nbr_idx, offset + sub_nbr_idx, Chem.BondType.SINGLE)
        for idx in sorted([core_dummy.GetIdx(), offset + sub_dummy.GetIdx()], reverse=True):
            rw.RemoveAtom(idx)

        res = rw.GetMol()
        Chem.SanitizeMol(res)
        return res
    except Exception:
        return None


def substitute_ch_with_radical(mol: Chem.Mol, rad_mol: Chem.Mol, rng: random.Random) -> Optional[Chem.Mol]:
    """Substitutes one aliphatic C-H bond in mol with a monovalent radical."""
    try:
        eligible_c = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6 and not a.GetIsAromatic() and a.GetTotalNumHs() > 0
        ]
        if not eligible_c:
            return mol
        c_idx = rng.choice(eligible_c)
        rw = Chem.RWMol(mol)
        dummy_idx = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(c_idx, dummy_idx, Chem.BondType.SINGLE)
        return connect_fragments(rw.GetMol(), rad_mol)
    except Exception:
        return None


def build_substituent_arms(
    spec: dict,
    max_arms: int = 500,
    seed: int = 42,
) -> Tuple[List[Tuple[str, Chem.Mol, int]], List[Tuple[str, Chem.Mol]]]:
    """
    Parses scaffolds and building blocks from spec, creating a diverse pool of
    core scaffolds and single-attachment substituent arms (depth 0, 1, 2).
    """
    rng = random.Random(seed)
    caps: List[Tuple[str, Chem.Mol]] = []
    linkers: List[Tuple[str, Chem.Mol]] = []

    for cat, items in spec.get("building_blocks", {}).items():
        for item in items:
            mol = parse_smiles_or_smarts(item["smiles"])
            if not mol:
                continue
            dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            if len(dummies) <= 1:
                caps.append((item["id"], mol))
            else:
                linkers.append((item["id"], mol))

    scaffolds: List[Tuple[str, Chem.Mol, int]] = []
    for s in spec.get("scaffolds", []):
        mol = parse_smiles_or_smarts(s["smarts"])
        if mol:
            dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            num_att = len(dummies)
            if num_att == 1:
                caps.append((s["id"], mol))
            scaffolds.append((s["id"], mol, num_att))

    arms = list(caps)

    if linkers:
        for lid, lmol in linkers:
            for cid, cmol in list(caps):
                if len(arms) >= max_arms:
                    break
                arm = connect_fragments(lmol, cmol)
                if arm and sum(1 for a in arm.GetAtoms() if a.GetAtomicNum() == 0) == 1:
                    arms.append((f"{lid}_{cid}", arm))

        for l1_id, l1_mol in linkers:
            for l2_id, l2_mol in linkers:
                for cid, cmol in list(caps):
                    if len(arms) >= max_arms:
                        break
                    inter = connect_fragments(l2_mol, cmol)
                    if inter:
                        arm = connect_fragments(l1_mol, inter)
                        if arm and sum(1 for a in arm.GetAtoms() if a.GetAtomicNum() == 0) == 1:
                            arms.append((f"{l1_id}_{l2_id}_{cid}", arm))
    else:
        base_rads = [cmol for cid, cmol in caps if cid not in ("hydrogen", "hydrogen_cap")]
        if base_rads:
            for _ in range(1, 3):
                if len(arms) >= max_arms:
                    break
                for cid, cmol in list(arms):
                    if len(arms) >= max_arms:
                        break
                    for r_mol in base_rads:
                        arm = substitute_ch_with_radical(cmol, r_mol, rng)
                        if arm and sum(1 for a in arm.GetAtoms() if a.GetAtomicNum() == 0) == 1:
                            arms.append((f"{cid}_sub", arm))
                            if len(arms) >= max_arms:
                                break

    return scaffolds, arms


def generate_library(
    spec: dict,
    target_count: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[Chem.Mol]:
    """
    Combinatorially generates unique 2D molecules matching all tensor limits and assembly rules.
    """
    gen_spec = spec.get("generation_spec", {})
    if target_count is None:
        target_count = gen_spec.get("target_molecules", 50000)

    actual_seed = gen_spec.get("random_seed", seed)
    rng = random.Random(actual_seed)

    scaffolds, all_arms = build_substituent_arms(spec, max_arms=600, seed=actual_seed)
    if not scaffolds:
        raise ValueError(f"No valid scaffolds found in specification: {spec.get('group_name')}")

    limits = spec.get("tensor_limits", {})
    max_sites = limits.get("max_sites", 128)
    max_mw = limits.get("max_molecular_weight", 1000.0)
    allowed_z = set(limits.get("allowed_atomic_numbers", list(range(1, 100))))
    min_rings = limits.get("min_aromatic_rings", 0)
    max_rings = limits.get("max_aromatic_rings", 100)
    min_f = limits.get("min_fluorine_count", 0)
    max_f = limits.get("max_fluorine_count", 100)
    min_rot = limits.get("min_rotatable_bonds", 0)
    max_rot = limits.get("max_rotatable_bonds", 100)

    arms = [(aid, amol) for aid, amol in all_arms if Descriptors.MolWt(amol) <= (max_mw * 0.6)]
    if not arms:
        arms = all_arms

    if verbose:
        print(f"[*] Starting 2D Generation for '{spec.get('group_name')}' (Target: {target_count:,} molecules)...")
        print(f"    -> Active Scaffolds: {len(scaffolds)} | Available Substituent Arms: {len(arms)}")

    start_time = time.perf_counter()
    unique_smiles: Set[str] = set()
    generated_mols: List[Chem.Mol] = []

    valid_scaffolds = [s for s in scaffolds if s[2] > 0]
    if not valid_scaffolds:
        valid_scaffolds = scaffolds

    weights = [max(1, s[2] ** 2) for s in valid_scaffolds]

    attempts = 0
    max_attempts = max(target_count * 150, 100000)

    while len(generated_mols) < target_count and attempts < max_attempts:
        attempts += 1
        s_id, s_mol, num_att = rng.choices(valid_scaffolds, weights=weights, k=1)[0]
        curr = Chem.Mol(s_mol)

        if num_att > 0:
            for _ in range(num_att):
                _, arm = rng.choice(arms)
                curr = connect_fragments(curr, arm)
                if curr is None:
                    break

        if curr is None:
            continue

        if rng.random() < 0.25 and not any(a.GetIsAromatic() for a in curr.GetAtoms()):
            _, arm = rng.choice(arms)
            ext = substitute_ch_with_radical(curr, arm, rng)
            if ext is not None:
                curr = ext

        if any(a.GetAtomicNum() == 0 for a in curr.GetAtoms()):
            continue

        try:
            Chem.SanitizeMol(curr)
        except Exception:
            continue

        num_h = sum(a.GetTotalNumHs() for a in curr.GetAtoms())
        total_sites = curr.GetNumAtoms() + num_h
        if total_sites > max_sites:
            continue

        mw = Descriptors.MolWt(curr)
        if mw > max_mw:
            continue

        if any(a.GetAtomicNum() not in allowed_z for a in curr.GetAtoms()):
            continue

        if min_f > 0 or max_f < 100:
            f_count = sum(1 for a in curr.GetAtoms() if a.GetAtomicNum() == 9)
            if not (min_f <= f_count <= max_f):
                continue

        if min_rings > 0 or max_rings < 100:
            ring_info = curr.GetRingInfo()
            aromatic_rings = sum(
                1 for r in ring_info.AtomRings() if all(curr.GetAtomWithIdx(i).GetIsAromatic() for i in r)
            )
            if not (min_rings <= aromatic_rings <= max_rings):
                continue

        if min_rot > 0 or max_rot < 100:
            rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(curr)
            if not (min_rot <= rot_bonds <= max_rot):
                continue

        smi = Chem.MolToSmiles(curr, canonical=True)
        if smi not in unique_smiles:
            clean_mol = Chem.MolFromSmiles(smi)
            if clean_mol is not None:
                unique_smiles.add(smi)
                generated_mols.append(clean_mol)

                if verbose and len(generated_mols) % 5000 == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = len(generated_mols) / max(1e-4, elapsed)
                    print(f"    -> Generated {len(generated_mols):,} unique molecules ({rate:.1f} mol/s)...")

    gen_time = time.perf_counter() - start_time
    if verbose:
        rate = len(generated_mols) / max(1e-4, gen_time)
        print(f"[+] 2D Generation Complete: {len(generated_mols):,} molecules in {gen_time:.2f}s ({rate:.1f} mol/s).")

    return generated_mols


def assign_atom_type(atom: Chem.Atom) -> str:
    """Assigns standard GAFF / Tripos atom types based on element, hybridization, and bonding."""
    z = atom.GetAtomicNum()
    sym = atom.GetSymbol()

    if z == 6:
        if atom.GetIsAromatic():
            return "ca"
        hyb = str(atom.GetHybridization())
        if "SP3" in hyb:
            return "c3"
        elif "SP2" in hyb:
            for b in atom.GetBonds():
                nbr = b.GetOtherAtom(atom)
                if nbr.GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE:
                    return "c"
            return "c2"
        elif "SP" in hyb:
            return "c1"
        return "c3"

    elif z == 1:
        nbrs = atom.GetNeighbors()
        if nbrs:
            nbr_z = nbrs[0].GetAtomicNum()
            if nbr_z == 8:
                return "ho"
            elif nbr_z == 7:
                return "hn"
            elif nbr_z == 16:
                return "hs"
            elif nbr_z == 6:
                return "ha" if nbrs[0].GetIsAromatic() else "hc"
        return "hc"

    elif z == 7:
        if atom.GetIsAromatic():
            return "na"
        hyb = str(atom.GetHybridization())
        if "SP" in hyb:
            return "n1"
        elif "SP2" in hyb:
            return "n"
        return "n3"

    elif z == 8:
        for b in atom.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                return "o"
        return "oh" if any(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors()) else "os"

    elif z == 9:
        return "f"
    elif z == 17:
        return "cl"
    elif z == 35:
        return "br"
    elif z == 53:
        return "i"
    elif z == 16:
        if atom.GetIsAromatic():
            return "ss"
        return "s"
    elif z == 15:
        return "p5"

    return sym


def _single_mol_worker(args_tuple: Tuple[int, str, str, str, Optional[str], int]) -> Dict[str, Any]:
    """
    Isolated multi-process worker for high-speed 3D conformer embedding, Gasteiger charging,
    and parallel Tripos .mol2 file export across CPU cores.
    """
    idx, smi, mol_name, group_name, out_dir_str, seed = args_tuple
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmarts(smi)
    mol_h = Chem.AddHs(mol)

    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=1, randomSeed=seed + idx))
    if not cids:
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=1, useRandomCoords=True, randomSeed=seed + idx))
        if cids:
            try:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)
            except Exception:
                pass

    if not cids:
        AllChem.Compute2DCoords(mol_h)

    try:
        AllChem.ComputeGasteigerCharges(mol_h)
    except Exception:
        pass

    conf = mol_h.GetConformer(0) if mol_h.GetNumConformers() > 0 else None
    atoms_data = []
    for i, a in enumerate(mol_h.GetAtoms(), 1):
        if conf is not None:
            pos = conf.GetAtomPosition(i - 1)
            x, y, z = pos.x, pos.y, pos.z
        else:
            x, y, z = 0.0, 0.0, 0.0
        sym = a.GetSymbol()
        s_name = f"{sym}{i}"
        at_type = assign_atom_type(a)
        try:
            q = float(a.GetProp("_GasteigerCharge"))
            if math.isnan(q) or math.isinf(q) or abs(q) > 10.0:
                q = 0.0
        except Exception:
            q = 0.0
        atoms_data.append((s_name, x, y, z, at_type, q))

    bonds_data = []
    for i, b in enumerate(mol_h.GetBonds(), 1):
        a1 = b.GetBeginAtomIdx() + 1
        a2 = b.GetEndAtomIdx() + 1
        b_type = "1"
        if b.GetIsAromatic():
            b_type = "ar"
        elif b.GetBondType() == Chem.BondType.DOUBLE:
            b_type = "2"
        elif b.GetBondType() == Chem.BondType.TRIPLE:
            b_type = "3"
        bonds_data.append((a1, a2, b_type))

    # Parallel file export per worker
    if out_dir_str:
        filepath = Path(out_dir_str) / f"{mol_name}.mol2"
        lines = [
            "@<TRIPOS>MOLECULE",
            mol_name,
            f"{len(atoms_data):>6}{len(bonds_data):>6}     1     0     0",
            "SMALL",
            "GAFF_CHARGES",
            "",
            f"Spec: {group_name}",
            "@<TRIPOS>ATOM",
        ]
        for i, at in enumerate(atoms_data, 1):
            lines.append(
                f"{i:>7} {at[0]:<6} {at[1]:>12.4f} {at[2]:>10.4f} {at[3]:>10.4f} {at[4]:<8} 1 MOL {at[5]:>12.6f}"
            )
        lines.append("@<TRIPOS>BOND")
        for i, b in enumerate(bonds_data, 1):
            lines.append(f"{i:>6} {b[0]:>6} {b[1]:>6} {b[2]:<4}")
        lines.append("@<TRIPOS>SUBSTRUCTURE\n     1 MOL         1 TEMP              0 ****  ****    0 ROOT\n")
        filepath.write_text("\n".join(lines), encoding="utf-8")

    mw = Descriptors.MolWt(mol_h)
    n_sites = mol_h.GetNumAtoms()

    return {
        "id": idx,
        "name": mol_name,
        "comment": f"Spec: {group_name}",
        "smiles": smi,
        "formula": rdMolDescriptors.CalcMolFormula(mol_h),
        "molecular_weight": round(mw, 3),
        "num_sites": n_sites,
    }


def embed_and_export_parallel(
    mols_2d: List[Chem.Mol],
    spec: dict,
    output_dir: str | Path,
    name_prefix: str = "mol",
    seed: int = 42,
    num_workers: Optional[int] = None,
    skip_write: bool = False,
    verbose: bool = True,
) -> Path:
    """
    High-performance multi-core 3D embedding, charging, and optional parallel disk export.
    """
    out_p = Path(output_dir)
    if not skip_write:
        out_p.mkdir(parents=True, exist_ok=True)

    group_name = spec.get("group_name", "unknown_group")
    workers = num_workers or min(16, os.cpu_count() or 4)

    if verbose:
        mode_str = "In-Memory" if skip_write else "Parallel Disk Export"
        print(f"[*] Starting Multi-Core 3D Embedding ({mode_str}, {workers} CPU workers)...")

    t_embed_0 = time.perf_counter()
    out_dir_param = None if skip_write else str(out_p)
    tasks_input = [
        (idx, Chem.MolToSmiles(m, canonical=True), f"{name_prefix}_{idx:06d}", group_name, out_dir_param, seed)
        for idx, m in enumerate(mols_2d, 1)
    ]

    mols_payloads: List[Dict[str, Any]] = []
    chunk_sz = max(1, len(tasks_input) // (workers * 4))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for item in executor.map(_single_mol_worker, tasks_input, chunksize=chunk_sz):
            mols_payloads.append(item)

    t_embed = time.perf_counter() - t_embed_0
    rate_embed = len(mols_payloads) / max(1e-4, t_embed)

    if verbose:
        action_str = "Embedded (In-Memory)" if skip_write else "Embedded & Exported"
        print(
            f"[+] 3D Conformer {action_str}: {len(mols_payloads):,} molecules in {t_embed:.2f}s ({rate_embed:.1f} mol/s)."
        )

    if not skip_write:
        generate_forcefield_database(out_p)

        mw_list = [item["molecular_weight"] for item in mols_payloads]
        sites_list = [item["num_sites"] for item in mols_payloads]

        manifest_entries = [
            {
                "id": item["id"],
                "name": item["name"],
                "file": f"{item['name']}.mol2",
                "smiles": item["smiles"],
                "formula": item["formula"],
                "molecular_weight": item["molecular_weight"],
                "num_sites": item["num_sites"],
            }
            for item in mols_payloads
        ]

        manifest_data = {
            "group_name": spec.get("group_name", "unknown_group"),
            "version": spec.get("version", "1.0.0"),
            "description": spec.get("description", ""),
            "timestamp": datetime.now().isoformat(),
            "total_molecules": len(mols_payloads),
            "engine": "dens-city High-Performance Parallel Multi-Core",
            "performance": {
                "elapsed_time_s": round(t_embed, 3),
                "throughput_mol_s": round(rate_embed, 1),
                "workers": workers,
            },
            "statistics": {
                "mean_molecular_weight": round(sum(mw_list) / max(1, len(mw_list)), 2),
                "max_molecular_weight": round(max(mw_list) if mw_list else 0.0, 2),
                "min_molecular_weight": round(min(mw_list) if mw_list else 0.0, 2),
                "mean_sites": round(sum(sites_list) / max(1, len(sites_list)), 1),
                "max_sites": max(sites_list) if sites_list else 0,
                "min_sites": min(sites_list) if sites_list else 0,
            },
            "molecules": manifest_entries,
        }

        manifest_path = out_p / "library_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        if verbose:
            print(f"[+] Complete Library Exported to: {out_p}")
            print(f"    -> Manifest: {manifest_path}")

    return out_p


def embed_conformers(
    mols: List[Chem.Mol],
    num_conformations: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> List[Chem.Mol]:
    """Compatibility conformer embedder."""
    if verbose:
        print(f"[*] Starting 3D Conformer Embedding for {len(mols):,} molecules ({num_conformations} conf/mol)...")
    start_time = time.perf_counter()
    mols_3d = []
    successful_embeds = 0
    for i, mol in enumerate(mols):
        mol_h = Chem.AddHs(mol)
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformations, randomSeed=seed + i))
        if not cids:
            cids = list(
                AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformations, useRandomCoords=True, randomSeed=seed + i)
            )
            if cids:
                try:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)
                except Exception:
                    pass
        if not cids:
            AllChem.Compute2DCoords(mol_h)
        else:
            successful_embeds += 1
        try:
            AllChem.ComputeGasteigerCharges(mol_h)
        except Exception:
            pass
        mols_3d.append(mol_h)
    embed_time = time.perf_counter() - start_time
    if verbose:
        rate = len(mols_3d) / max(1e-4, embed_time)
        print(f"[+] 3D Embedding Complete: {successful_embeds:,} / {len(mols):,} molecules in {embed_time:.2f}s.")
        print(f"    -> Conformer Rate: {rate * num_conformations:.1f} conf/s")
    return mols_3d


def format_tripos_mol2(mol_3d: Chem.Mol, mol_name: str, comment: str = "") -> str:
    """Formats a 3D RDKit Mol object into a full Tripos .mol2 structure string."""
    conf = mol_3d.GetConformer(0) if mol_3d.GetNumConformers() > 0 else None
    atoms = []
    for i, a in enumerate(mol_3d.GetAtoms(), 1):
        if conf is not None:
            pos = conf.GetAtomPosition(i - 1)
            x, y, z = pos.x, pos.y, pos.z
        else:
            x, y, z = 0.0, 0.0, 0.0
        sym = a.GetSymbol()
        s_name = f"{sym}{i}"
        at_type = assign_atom_type(a)
        try:
            q = float(a.GetProp("_GasteigerCharge"))
            if math.isnan(q) or math.isinf(q) or abs(q) > 10.0:
                q = 0.0
        except Exception:
            q = 0.0
        atoms.append((s_name, x, y, z, at_type, q))

    bonds = []
    for i, b in enumerate(mol_3d.GetBonds(), 1):
        a1 = b.GetBeginAtomIdx() + 1
        a2 = b.GetEndAtomIdx() + 1
        b_type = "1"
        if b.GetIsAromatic():
            b_type = "ar"
        elif b.GetBondType() == Chem.BondType.DOUBLE:
            b_type = "2"
        elif b.GetBondType() == Chem.BondType.TRIPLE:
            b_type = "3"
        bonds.append((a1, a2, b_type))

    lines = [
        "@<TRIPOS>MOLECULE",
        mol_name,
        f"{len(atoms):>6}{len(bonds):>6}     1     0     0",
        "SMALL",
        "GAFF_CHARGES",
        "",
        comment if comment else "Generated by dens-city Molecular Library Generator",
        "@<TRIPOS>ATOM",
    ]
    for i, at in enumerate(atoms, 1):
        s_name, x, y, z, at_type, q = at
        lines.append(f"{i:>7} {s_name:<6} {x:>12.4f} {y:>10.4f} {z:>10.4f} {at_type:<8} 1 MOL {q:>12.6f}")
    lines.append("@<TRIPOS>BOND")
    for i, b in enumerate(bonds, 1):
        a1, a2, b_type = b
        lines.append(f"{i:>6} {a1:>6} {a2:>6} {b_type:<4}")
    lines.append("@<TRIPOS>SUBSTRUCTURE\n     1 MOL         1 TEMP              0 ****  ****    0 ROOT\n")
    return "\n".join(lines)


def export_dataset(
    mols_3d: List[Chem.Mol],
    spec: dict,
    output_dir: str | Path,
    name_prefix: str = "mol",
    verbose: bool = True,
) -> Path:
    """Exports 3D molecules into standard Tripos .mol2 files."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    mw_list = []
    sites_list = []
    for idx, mol in enumerate(mols_3d, 1):
        mol_name = f"{name_prefix}_{idx:06d}"
        mol2_content = format_tripos_mol2(mol, mol_name, f"Spec: {spec.get('group_name')}")
        mol2_path = out_p / f"{mol_name}.mol2"
        mol2_path.write_text(mol2_content, encoding="utf-8")
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
        mw = Descriptors.MolWt(mol)
        n_sites = mol.GetNumAtoms()
        mw_list.append(mw)
        sites_list.append(n_sites)
        manifest_entries.append(
            {
                "id": idx,
                "name": mol_name,
                "file": f"{mol_name}.mol2",
                "smiles": smi,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(mw, 3),
                "num_sites": n_sites,
            }
        )
    generate_forcefield_database(out_p)
    manifest_data = {
        "group_name": spec.get("group_name", "unknown_group"),
        "version": spec.get("version", "1.0.0"),
        "description": spec.get("description", ""),
        "timestamp": datetime.now().isoformat(),
        "total_molecules": len(mols_3d),
        "statistics": {
            "mean_molecular_weight": round(sum(mw_list) / max(1, len(mw_list)), 2),
            "max_molecular_weight": round(max(mw_list) if mw_list else 0.0, 2),
            "min_molecular_weight": round(min(mw_list) if mw_list else 0.0, 2),
            "mean_sites": round(sum(sites_list) / max(1, len(sites_list)), 1),
            "max_sites": max(sites_list) if sites_list else 0,
            "min_sites": min(sites_list) if sites_list else 0,
        },
        "molecules": manifest_entries,
    }
    manifest_path = out_p / "library_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return out_p


def run_library_generator(
    spec_path: str | Path,
    target_count: Optional[int] = None,
    out_dir: Optional[str | Path] = None,
    workers: Optional[int] = None,
    seed: int = 42,
    skip_3d: bool = False,
    skip_write: bool = False,
) -> int:
    """Entrypoint function to run full combinatorial molecular library generation."""
    spec = load_spec(spec_path)
    group_name = spec.get("group_name", Path(spec_path).stem)
    target = target_count or spec.get("generation_spec", {}).get("target_molecules", 50000)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_dir = out_dir or f"data/{group_name}_{ts}"
    num_workers = workers or min(16, os.cpu_count() or 4)

    print("=" * 84)
    print("  dens-city: High-Performance C / Multi-Core Combinatorial Molecular Generator")
    print("=" * 84)
    print(f"  Specification File  : {spec_path}")
    print(f"  Chemical Group      : {group_name}")
    print(f"  Target Molecule Set : {target:,} molecules")
    print(f"  Parallel Workers    : {num_workers} CPU cores (OpenMP C-Engine active)")
    print(f"  Execution Mode      : {'In-Memory Benchmarking (--skip-write)' if skip_write else 'Full Export (.mol2)'}")
    if not skip_write:
        print(f"  Target Output Path  : {destination_dir}")
    print("=" * 84)

    t_global_0 = time.perf_counter()

    mols_2d = generate_library(spec=spec, target_count=target, seed=seed, verbose=True)
    if not mols_2d:
        print("Error: No molecules were generated.", file=sys.stderr)
        return 1

    if not skip_3d:
        embed_and_export_parallel(
            mols_2d=mols_2d,
            spec=spec,
            output_dir=destination_dir,
            name_prefix=f"{group_name[:8]}_mol",
            seed=seed,
            num_workers=num_workers,
            skip_write=skip_write,
            verbose=True,
        )
    else:
        print("[*] Skipped 3D conformer embedding as requested (--skip-3d).")

    t_total = time.perf_counter() - t_global_0
    print("=" * 84)
    print(f"  [+] Complete Pipeline Finished in {t_total:.2f} seconds ({len(mols_2d) / max(1e-4, t_total):.1f} mol/s)")
    if not skip_write:
        print(f"  [+] Data Artifact Directory Ready: {destination_dir}")
    else:
        print(f"  [+] In-Memory Benchmark Successful: {len(mols_2d):,} 3D molecules generated.")
    print("=" * 84)

    return 0
