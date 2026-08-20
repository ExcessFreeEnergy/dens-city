#!/usr/bin/env python3
"""
Scrapes .mol2 molecular structure files and force field parameter databases (gaff.dat / forcefield_parameters.json)
to generate the complete cDFT Solute Input validation dataset for the dens-city 3D cDFT solver.

Generates the exact columns specified for cDFT Solute Input:
  Molecule ID | Site Name | Atom Type | X (Å) | Y (Å) | Z (Å) | Charge (q) | LJ σ (Å) | LJ ϵ (kcal/mol)

Output:
  data/input_validation.txt (and data/input_validation.csv)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"
DATA_DIR = REPO_ROOT / "data"
FF_JSON_PATH = TEST_DATA_DIR / "forcefield_parameters.json"
OUTPUT_TXT_PATH = DATA_DIR / "input_validation.txt"
OUTPUT_CSV_PATH = DATA_DIR / "input_validation.csv"


def load_forcefield_database(ff_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads atom type to LJ sigma / epsilon mapping from JSON forcefield database."""
    if not ff_path.exists():
        raise FileNotFoundError(f"Force field parameter database not found at {ff_path}")
    with open(ff_path, "r") as f:
        return json.load(f)


def parse_mol2_file(mol2_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses a TRIPOS .mol2 file to extract molecule ID and atom site records.

    Returns:
        (molecule_id, list_of_atoms)
    """
    lines = mol2_path.read_text().splitlines()
    mol_name = mol2_path.stem
    if mol_name == "water":
        mol_name = "water_spce"

    atoms = []
    in_molecule = False
    in_atom = False
    molecule_line_idx = -1

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("@<TRIPOS>MOLECULE"):
            in_molecule = True
            in_atom = False
            molecule_line_idx = idx + 1
            continue
        elif line.startswith("@<TRIPOS>ATOM"):
            in_molecule = False
            in_atom = True
            continue
        elif line.startswith("@<TRIPOS>"):
            in_atom = False
            in_molecule = False
            continue

        if in_molecule and idx == molecule_line_idx:
            # First line after @<TRIPOS>MOLECULE is the molecule name
            if line and not line.startswith("@") and mol2_path.stem not in ["water"]:
                # Use filename stem for unambiguous identification if descriptive
                pass

        if in_atom:
            parts = raw_line.split()
            if len(parts) >= 6:
                site_name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                atom_type = parts[5]
                charge = float(parts[8]) if len(parts) >= 9 else 0.0

                atoms.append({
                    "site_name": site_name,
                    "atom_type": atom_type,
                    "x": x,
                    "y": y,
                    "z": z,
                    "charge": charge,
                })

    return mol_name, atoms


def match_lj_parameters(
    atom_type: str,
    ff_db: Dict[str, Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Looks up LJ sigma (Å) and epsilon (kcal/mol) for a given atom type.
    Falls back to case-insensitive matching if direct lookup fails.
    """
    # 1. Exact match
    if atom_type in ff_db:
        entry = ff_db[atom_type]
        return float(entry["sigma_angstrom"]), float(entry["epsilon_kcal_mol"])

    # 2. Case-insensitive match
    for k, entry in ff_db.items():
        if k.lower() == atom_type.lower():
            return float(entry["sigma_angstrom"]), float(entry["epsilon_kcal_mol"])

    # 3. Stripped / generalized prefix match
    clean_type = atom_type.rstrip("0123456789")
    if clean_type in ff_db:
        entry = ff_db[clean_type]
        return float(entry["sigma_angstrom"]), float(entry["epsilon_kcal_mol"])

    print(f"Warning: Atom type '{atom_type}' not found in force field database. Using default (0.0, 0.0).", file=sys.stderr)
    return 0.0, 0.0


def generate_cdft_solute_input(
    mol2_dirs: List[Path],
    ff_db: Dict[str, Dict[str, Any]],
    output_txt: Path,
    output_csv: Path,
    include_all_freesolv: bool = False,
) -> int:
    """Scrapes mol2 files, resolves forcefield parameters, and writes formatted cDFT solute inputs."""
    records = []

    # Priority order for benchmark molecules
    benchmark_order = [
        "water", "water_spce", "nitrogen", "methane", "carbon_dioxide", "argon",
        "sodium_chloride", "calcium_chloride", "n_decane", "neopentane", "polyethylene",
        "methanol", "ammonia", "hydrogen_fluoride", "benzene", "5cb",
        "sodium_dodecyl_sulfate", "sulfur_hexafluoride", "acetone",
        "colloidal_hard_sphere", "hydrogen"
    ]

    discovered_files = []
    for m_dir in mol2_dirs:
        if m_dir.exists():
            discovered_files.extend(sorted(m_dir.glob("*.mol2")))

    # Sort files putting benchmark molecules first
    def sort_key(p: Path) -> Tuple[int, str]:
        stem = p.stem.lower()
        if stem in benchmark_order:
            return (0, f"{benchmark_order.index(stem):03d}")
        return (1, stem)

    unique_files = []
    seen_stems = set()
    for f in sorted(discovered_files, key=sort_key):
        if not include_all_freesolv and f.stem.startswith("mobley_") and f.parent.name == "mol2files_gaff":
            continue
        if f.stem not in seen_stems:
            seen_stems.add(f.stem)
            unique_files.append(f)

    for mol2_file in unique_files:
        mol_id, atoms = parse_mol2_file(mol2_file)
        for atom in atoms:
            sigma_A, eps_kcal = match_lj_parameters(atom["atom_type"], ff_db)
            records.append({
                "molecule_id": mol_id,
                "site_name": atom["site_name"],
                "atom_type": atom["atom_type"],
                "x": atom["x"],
                "y": atom["y"],
                "z": atom["z"],
                "charge": atom["charge"],
                "sigma_A": sigma_A,
                "epsilon_kcal": eps_kcal,
            })

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write formatted table in input_validation.txt
    header_title = "# =========================================================================================================\n"
    header_title += "# dens-city: 3D Classical Density Functional Theory (cDFT) Solute Input Validation Chart\n"
    header_title += "# Math Formulation:\n"
    header_title += "#   - Geometry (X, Y, Z): Centers of interaction sites relative to molecular frame (in Angstroms)\n"
    header_title += "#   - Electrostatics q: V_coulomb(r) = sum_i [ q_i / (4 * pi * eps_0 * |r - r_i|) ] (in elementary charges e)\n"
    header_title += "#   - LJ sigma: Collision diameter in Angstroms (steric repulsive hard core)\n"
    header_title += "#   - LJ epsilon: Well depth in kcal/mol (attractive dispersion stickiness)\n"
    header_title += "# =========================================================================================================\n"

    table_header = f"{'Molecule ID':<24} {'Site Name':<10} {'Atom Type':<10} {'X (Å)':>10} {'Y (Å)':>10} {'Z (Å)':>10} {'Charge (q)':>12} {'LJ σ (Å)':>10} {'LJ ϵ (kcal/mol)':>16}\n"
    separator = "-" * len(table_header.strip()) + "\n"

    txt_lines = [header_title, table_header, separator]
    for r in records:
        line = (
            f"{r['molecule_id']:<24} "
            f"{r['site_name']:<10} "
            f"{r['atom_type']:<10} "
            f"{r['x']:>10.4f} "
            f"{r['y']:>10.4f} "
            f"{r['z']:>10.4f} "
            f"{r['charge']:>12.4f} "
            f"{r['sigma_A']:>10.4f} "
            f"{r['epsilon_kcal']:>16.4f}\n"
        )
        txt_lines.append(line)

    output_txt.write_text("".join(txt_lines))
    print(f"Generated formatted solute input validation table: {output_txt} ({len(records)} sites)")

    # 2. Write CSV version for automated ingestion
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Molecule ID", "Site Name", "Atom Type", "X (Å)", "Y (Å)", "Z (Å)",
            "Charge (q)", "LJ σ (Å)", "LJ ϵ (kcal/mol)"
        ])
        for r in records:
            writer.writerow([
                r["molecule_id"], r["site_name"], r["atom_type"],
                f"{r['x']:.4f}", f"{r['y']:.4f}", f"{r['z']:.4f}",
                f"{r['charge']:.4f}", f"{r['sigma_A']:.4f}", f"{r['epsilon_kcal']:.4f}"
            ])
    print(f"Generated CSV solute input validation table: {output_csv}")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape .mol2 files and force field databases to produce complete cDFT solute inputs."
    )
    parser.add_argument(
        "--output-txt",
        "-o",
        type=str,
        default=str(OUTPUT_TXT_PATH),
        help="Path for output validation text table (default: data/input_validation.txt)",
    )
    parser.add_argument(
        "--output-csv",
        "-c",
        type=str,
        default=str(OUTPUT_CSV_PATH),
        help="Path for output validation CSV (default: data/input_validation.csv)",
    )
    parser.add_argument(
        "--all-freesolv",
        "-a",
        action="store_true",
        help="Include all 642 FreeSolv raw molecules in addition to benchmark test datasets",
    )

    args = parser.parse_args()

    ff_db = load_forcefield_database(FF_JSON_PATH)
    mol2_search_dirs = [TEST_DATA_DIR, DATA_DIR / "mol2files_gaff"]

    out_txt = Path(args.output_txt)
    out_csv = Path(args.output_csv)

    print("=== dens-city cDFT Solute Input Generator ===")
    print(f"Forcefield DB: {FF_JSON_PATH}")
    print(f"Output TXT: {out_txt}")
    print(f"Output CSV: {out_csv}")
    print("=============================================")

    total_sites = generate_cdft_solute_input(
        mol2_dirs=mol2_search_dirs,
        ff_db=ff_db,
        output_txt=out_txt,
        output_csv=out_csv,
        include_all_freesolv=args.all_freesolv,
    )

    print(f"\nSuccessfully generated solute input tables with {total_sites} interaction sites!")


if __name__ == "__main__":
    main()
