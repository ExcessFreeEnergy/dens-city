#!/usr/bin/env python3
"""
Initialize and extract FreeSolv dataset for dens-city.

Extracts mol2files_gaff.tar.gz from the FreeSolv submodule into the data/ directory,
along with associated database metadata (database.json, database.txt) for hydration free energy
calculations, cDFT simulations, and neural functional benchmarks.

Usage:
    python scripts/init_data.py
    python scripts/init_data.py --dest-dir data/
    python scripts/init_data.py --force
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).resolve().parent.parent


def check_and_init_submodule(repo_root: Path, freesolv_dir: Path) -> None:
    """Ensure the FreeSolv submodule is initialized and available."""
    archive_path = freesolv_dir / "mol2files_gaff.tar.gz"
    if not archive_path.exists():
        print(f"FreeSolv archive not found at {archive_path}. Attempting git submodule update...")
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.SubprocessError as e:
            print(f"Warning: git submodule update failed: {e}", file=sys.stderr)

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Could not find mol2files_gaff.tar.gz at {archive_path}. "
            "Please ensure the FreeSolv git submodule is properly initialized:\n"
            "  git submodule update --init --recursive"
        )


def extract_mol2_archive(
    archive_path: Path,
    dest_dir: Path,
    force: bool = False,
) -> int:
    """
    Extract mol2files_gaff.tar.gz into destination directory.

    Returns:
        Number of .mol2 files extracted.
    """
    target_mol2_dir = dest_dir / "mol2files_gaff"

    if target_mol2_dir.exists() and not force:
        existing_mol2 = list(target_mol2_dir.glob("*.mol2"))
        if existing_mol2:
            print(f"Destination '{target_mol2_dir}' already contains {len(existing_mol2)} .mol2 files.")
            print("Use --force to re-extract.")
            return len(existing_mol2)

    print(f"Extracting '{archive_path.name}' -> '{dest_dir}'...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with tarfile.open(archive_path, "r:gz") as tar:
        # Safe extraction filter for modern Python / security
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=dest_dir, filter="data")
        else:
            tar.extractall(path=dest_dir)

    extracted_files = list(target_mol2_dir.glob("*.mol2"))
    count = len(extracted_files)
    print(f"Successfully extracted {count} .mol2 files into '{target_mol2_dir}'.")
    return count


def copy_database_metadata(freesolv_dir: Path, dest_dir: Path) -> None:
    """Copy database metadata files (database.json, database.txt, etc.) to data directory."""
    metadata_files = [
        "database.json",
        "database.txt",
        "database.pickle",
        "iupac_to_cid.json",
        "smiles_to_cid.json",
    ]

    copied = 0
    for filename in metadata_files:
        src = freesolv_dir / filename
        dst = dest_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    if copied > 0:
        print(f"Copied {copied} FreeSolv metadata files to '{dest_dir}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract FreeSolv GAFF mol2 structures and metadata into dens-city data directory."
    )
    parser.add_argument(
        "--dest-dir",
        "-d",
        type=str,
        default=None,
        help="Destination directory for extracted data (default: <repo_root>/data)",
    )
    parser.add_argument(
        "--freesolv-dir",
        "-s",
        type=str,
        default=None,
        help="Path to FreeSolv submodule directory (default: <repo_root>/FreeSolv)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-extraction even if data already exists in destination",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip copying database JSON / metadata files",
    )

    args = parser.parse_args()
    repo_root = get_repo_root()

    dest_dir = Path(args.dest_dir) if args.dest_dir else (repo_root / "data")
    freesolv_dir = Path(args.freesolv_dir) if args.freesolv_dir else (repo_root / "FreeSolv")

    print("=== dens-city FreeSolv Data Initializer ===")
    print(f"Repository Root: {repo_root}")
    print(f"FreeSolv Source: {freesolv_dir}")
    print(f"Data Destination: {dest_dir}")
    print("===========================================")

    check_and_init_submodule(repo_root, freesolv_dir)

    archive_path = freesolv_dir / "mol2files_gaff.tar.gz"
    mol2_count = extract_mol2_archive(archive_path, dest_dir, force=args.force)

    if not args.no_metadata:
        copy_database_metadata(freesolv_dir, dest_dir)

    print(f"\nInitialization complete! Ready with {mol2_count} molecules in '{dest_dir}'.")


if __name__ == "__main__":
    main()
