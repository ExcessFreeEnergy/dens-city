#!/usr/bin/env python3
"""
Populates the data/test_data/ directory with standard Tripos .mol2 structure files
for benchmark materials in dens-city, copying directly from FreeSolv where available,
and generating canonical geometries for the remaining fluids.

Usage:
    python scripts/generate_test_data.py        # Populates 32 core benchmark materials
    python scripts/generate_test_data.py --all  # Populates ALL 642+ FreeSolv molecules + canonical fluids
"""

import argparse
import math
import shutil
import sys
import tarfile
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_forcefield_params import generate_json_and_csv

REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
TEST_DATA_DIR = DATA_DIR / "test_data"
FREESOLV_SUBMODULE_DIR = REPO_ROOT / "FreeSolv"
MOL2_GAFF_DIR = DATA_DIR / "mol2files_gaff"


def format_mol2(name: str, atoms: list, bonds: list, comment: str = "") -> str:
    """Formats atom and bond lists into standard TRIPOS .mol2 file format."""
    num_atoms = len(atoms)
    num_bonds = len(bonds)
    num_subst = 1

    lines = [
        "@<TRIPOS>MOLECULE",
        name,
        f"{num_atoms:>6}{num_bonds:>6}{num_subst:>6}     0     0",
        "SMALL",
        "GAFF_CHARGES",
        "",
        comment if comment else "",
        "@<TRIPOS>ATOM",
    ]

    for i, at in enumerate(atoms, start=1):
        # atom: (name, x, y, z, atom_type, charge)
        at_name, x, y, z, at_type, q = at
        lines.append(f"{i:>7} {at_name:<6} {x:>12.4f} {y:>10.4f} {z:>10.4f} {at_type:<8} 1 MOL {q:>12.6f}")

    lines.append("@<TRIPOS>BOND")
    for i, b in enumerate(bonds, start=1):
        # bond: (a1, a2, bond_type)
        a1, a2, b_type = b
        lines.append(f"{i:>6} {a1:>6} {a2:>6} {b_type:<4}")

    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append("     1 MOL         1 TEMP              0 ****  ****    0 ROOT")
    lines.append("")
    return "\n".join(lines)


def build_water() -> str:
    # SPC/E geometry: r(OH) = 1.0000 A, theta = 109.47 deg
    r_oh = 1.0000
    half_angle = math.radians(109.47 / 2.0)
    hx = r_oh * math.sin(half_angle)
    hz = r_oh * math.cos(half_angle)
    atoms = [
        ("O1", 0.0, 0.0, 0.0, "ow", -0.847600),
        ("H1", hx, 0.0, hz, "hw", 0.423800),
        ("H2", -hx, 0.0, hz, "hw", 0.423800),
    ]
    bonds = [
        (1, 2, "1"),
        (1, 3, "1"),
    ]
    return format_mol2("water", atoms, bonds, "SPC/E Water model (dens-city)")


def build_nitrogen() -> str:
    # TraPPE / diatomic N2: r(NN) = 1.1000 A
    atoms = [
        ("N1", 0.0, 0.0, -0.5500, "n1", 0.000000),
        ("N2", 0.0, 0.0, 0.5500, "n1", 0.000000),
    ]
    bonds = [
        (1, 2, "3"),
    ]
    return format_mol2("nitrogen", atoms, bonds, "TraPPE Diatomic Nitrogen (N2)")


def build_carbon_dioxide() -> str:
    # TraPPE CO2: linear, r(CO) = 1.1600 A, C=+0.70, O=-0.35
    atoms = [
        ("C1", 0.0, 0.0, 0.0000, "c1", 0.700000),
        ("O1", 0.0, 0.0, -1.1600, "o", -0.350000),
        ("O2", 0.0, 0.0, 1.1600, "o", -0.350000),
    ]
    bonds = [
        (1, 2, "2"),
        (1, 3, "2"),
    ]
    return format_mol2("carbon_dioxide", atoms, bonds, "TraPPE Carbon Dioxide (CO2)")


def build_argon() -> str:
    # Argon noble gas monoatomic: sigma = 3.405 A
    atoms = [
        ("Ar1", 0.0, 0.0, 0.0, "Ar", 0.000000),
    ]
    bonds = []
    return format_mol2("argon", atoms, bonds, "Monoatomic Argon (Ar)")


def build_sodium_chloride() -> str:
    # NaCl ion pair: r = 2.36 A
    atoms = [
        ("Na1", 0.0, 0.0, -1.1800, "Na", 1.000000),
        ("Cl1", 0.0, 0.0, 1.1800, "Cl", -1.000000),
    ]
    bonds = [
        (1, 2, "1"),
    ]
    return format_mol2("sodium_chloride", atoms, bonds, "Sodium Chloride (NaCl) 1:1 RPM electrolyte")


def build_calcium_chloride() -> str:
    # CaCl2 linear 2:1 electrolyte: r(CaCl) = 2.70 A
    atoms = [
        ("Ca1", 0.0, 0.0, 0.0000, "Ca", 2.000000),
        ("Cl1", 0.0, 0.0, -2.7000, "Cl", -1.000000),
        ("Cl2", 0.0, 0.0, 2.7000, "Cl", -1.000000),
    ]
    bonds = [
        (1, 2, "1"),
        (1, 3, "1"),
    ]
    return format_mol2("calcium_chloride", atoms, bonds, "Calcium Chloride (CaCl2) 2:1 electrolyte")


def build_hydrogen_fluoride() -> str:
    # HF associating dipole: r(HF) = 0.917 A, q = +/- 0.45e
    atoms = [
        ("F1", 0.0, 0.0, 0.0000, "f", -0.450000),
        ("H1", 0.0, 0.0, 0.9170, "ha", 0.450000),
    ]
    bonds = [
        (1, 2, "1"),
    ]
    return format_mol2("hydrogen_fluoride", atoms, bonds, "Hydrogen Fluoride (HF) 1D Associating fluid")


def build_sulfur_hexafluoride() -> str:
    # SF6 octahedral: r(SF) = 1.56 A
    r_sf = 1.5600
    atoms = [
        ("S1", 0.0, 0.0, 0.0, "s6", 1.500000),
        ("F1", r_sf, 0.0, 0.0, "f", -0.250000),
        ("F2", -r_sf, 0.0, 0.0, "f", -0.250000),
        ("F3", 0.0, r_sf, 0.0, "f", -0.250000),
        ("F4", 0.0, -r_sf, 0.0, "f", -0.250000),
        ("F5", 0.0, 0.0, r_sf, "f", -0.250000),
        ("F6", 0.0, 0.0, -r_sf, "f", -0.250000),
    ]
    bonds = [
        (1, 2, "1"),
        (1, 3, "1"),
        (1, 4, "1"),
        (1, 5, "1"),
        (1, 6, "1"),
        (1, 7, "1"),
    ]
    return format_mol2("sulfur_hexafluoride", atoms, bonds, "Sulfur Hexafluoride (SF6) octahedral fluid")


def build_hydrogen() -> str:
    # H2 diatomic: r(HH) = 0.7414 A
    atoms = [
        ("H1", 0.0, 0.0, -0.3707, "ha", 0.000000),
        ("H2", 0.0, 0.0, 0.3707, "ha", 0.000000),
    ]
    bonds = [
        (1, 2, "1"),
    ]
    return format_mol2("hydrogen", atoms, bonds, "Molecular Hydrogen (H2)")


def build_colloidal_hard_sphere() -> str:
    # Colloid hard sphere: giant single excluded volume particle
    atoms = [
        ("COL1", 0.0, 0.0, 0.0, "Col", 0.000000),
    ]
    bonds = []
    return format_mol2("colloidal_hard_sphere", atoms, bonds, "Large Colloidal Hard Sphere (D=15A)")


def build_polyethylene() -> str:
    # Extended polyethylene oligomer (eicosane C20H42 all-trans chain)
    n_c = 20
    atoms = []
    bonds = []
    c_idx = []

    # All-trans zigzag along x-axis: dx = 1.27 A, dy = +/- 0.44 A
    for i in range(n_c):
        x = i * 1.27
        y = 0.44 if (i % 2 == 0) else -0.44
        z = 0.0
        c_name = f"C{i + 1}"
        q_c = -0.18 if (i == 0 or i == n_c - 1) else -0.12
        atoms.append((c_name, x, y, z, "c3", q_c))
        c_idx.append(len(atoms))

        # Add hydrogens
        h_y = 0.88 if (i % 2 == 0) else -0.88
        h_z1 = 0.89
        h_z2 = -0.89
        q_h = 0.06

        atoms.append((f"H{i * 2 + 1}", x, h_y, h_z1, "hc", q_h))
        h1_idx = len(atoms)
        atoms.append((f"H{i * 2 + 2}", x, h_y, h_z2, "hc", q_h))
        h2_idx = len(atoms)

        bonds.append((c_idx[-1], h1_idx, "1"))
        bonds.append((c_idx[-1], h2_idx, "1"))

        # Terminal extra hydrogens
        if i == 0:
            atoms.append(("H0", x - 0.89, y, 0.0, "hc", q_h))
            bonds.append((c_idx[-1], len(atoms), "1"))
        elif i == n_c - 1:
            atoms.append((f"H{n_c * 2 + 1}", x + 0.89, y, 0.0, "hc", q_h))
            bonds.append((c_idx[-1], len(atoms), "1"))

        # Carbon-carbon backbone bond
        if i > 0:
            bonds.append((c_idx[i - 1], c_idx[i], "1"))

    return format_mol2("polyethylene", atoms, bonds, "Polyethylene oligomer (C20H42 eicosane chain)")


def build_5cb() -> str:
    # 5CB (4-Cyano-4'-pentylbiphenyl): C18H19N
    atoms = []
    bonds = []

    # Cyano group N#C- at origin / -z
    atoms.append(("N1", 0.0, 0.0, -7.0, "n1", -0.550000))
    atoms.append(("C1", 0.0, 0.0, -5.85, "c1", 0.450000))
    bonds.append((1, 2, "3"))

    # Ring 1: 4-substituted phenyl (z in [-4.7, -2.0])
    r1_coords = [
        ("C2", 0.0, 0.0, -4.45, "ca", 0.00),
        ("C3", 1.21, 0.0, -3.75, "ca", -0.13),
        ("C4", 1.21, 0.0, -2.35, "ca", -0.13),
        ("C5", 0.0, 0.0, -1.65, "ca", 0.05),
        ("C6", -1.21, 0.0, -2.35, "ca", -0.13),
        ("C7", -1.21, 0.0, -3.75, "ca", -0.13),
        ("H1", 2.14, 0.0, -4.28, "ha", 0.13),
        ("H2", 2.14, 0.0, -1.82, "ha", 0.13),
        ("H3", -2.14, 0.0, -1.82, "ha", 0.13),
        ("H4", -2.14, 0.0, -4.28, "ha", 0.13),
    ]
    r1_start = len(atoms) + 1
    for item in r1_coords:
        atoms.append(item)
    bonds.append((2, r1_start, "1"))  # C1 - C2
    bonds.append((r1_start, r1_start + 1, "ar"))
    bonds.append((r1_start + 1, r1_start + 2, "ar"))
    bonds.append((r1_start + 2, r1_start + 3, "ar"))
    bonds.append((r1_start + 3, r1_start + 4, "ar"))
    bonds.append((r1_start + 4, r1_start + 5, "ar"))
    bonds.append((r1_start + 5, r1_start, "ar"))
    # H bonds
    bonds.append((r1_start + 1, r1_start + 6, "1"))
    bonds.append((r1_start + 2, r1_start + 7, "1"))
    bonds.append((r1_start + 4, r1_start + 8, "1"))
    bonds.append((r1_start + 5, r1_start + 9, "1"))

    # Ring 2: biphenyl linked at C5 (z in [-0.15, 2.55], rotated ~30 deg)
    r2_start = len(atoms) + 1
    r2_coords = [
        ("C8", 0.0, 0.0, -0.15, "ca", 0.05),
        ("C9", 1.05, 0.60, 0.55, "ca", -0.13),
        ("C10", 1.05, 0.60, 1.95, "ca", -0.13),
        ("C11", 0.0, 0.0, 2.65, "ca", 0.08),
        ("C12", -1.05, -0.60, 1.95, "ca", -0.13),
        ("C13", -1.05, -0.60, 0.55, "ca", -0.13),
        ("H5", 1.85, 1.07, 0.02, "ha", 0.13),
        ("H6", 1.85, 1.07, 2.48, "ha", 0.13),
        ("H7", -1.85, -1.07, 2.48, "ha", 0.13),
        ("H8", -1.85, -1.07, 0.02, "ha", 0.13),
    ]
    for item in r2_coords:
        atoms.append(item)
    bonds.append((r1_start + 3, r2_start, "1"))  # C5 - C8 inter-ring bond
    bonds.append((r2_start, r2_start + 1, "ar"))
    bonds.append((r2_start + 1, r2_start + 2, "ar"))
    bonds.append((r2_start + 2, r2_start + 3, "ar"))
    bonds.append((r2_start + 3, r2_start + 4, "ar"))
    bonds.append((r2_start + 4, r2_start + 5, "ar"))
    bonds.append((r2_start + 5, r2_start, "ar"))
    # H bonds
    bonds.append((r2_start + 1, r2_start + 6, "1"))
    bonds.append((r2_start + 2, r2_start + 7, "1"))
    bonds.append((r2_start + 4, r2_start + 8, "1"))
    bonds.append((r2_start + 5, r2_start + 9, "1"))

    # Pentyl tail attached to C11 (5 carbons: C14-C18)
    tail_c = [
        ("C14", 0.0, 0.0, 4.15, "c3", -0.06),
        ("C15", 1.25, 0.0, 4.95, "c3", -0.12),
        ("C16", 1.25, 0.0, 6.45, "c3", -0.12),
        ("C17", 2.50, 0.0, 7.25, "c3", -0.12),
        ("C18", 2.50, 0.0, 8.75, "c3", -0.18),
    ]
    tail_start = len(atoms) + 1
    for c in tail_c:
        atoms.append(c)
    bonds.append((r2_start + 3, tail_start, "1"))  # C11 - C14
    for i in range(4):
        bonds.append((tail_start + i, tail_start + i + 1, "1"))

    # Pentyl hydrogens (11 H)
    h_tail = [
        ("H9", -0.55, 0.89, 4.40, "hc", 0.06),
        ("H10", -0.55, -0.89, 4.40, "hc", 0.06),
        ("H11", 1.80, 0.89, 4.70, "hc", 0.06),
        ("H12", 1.80, -0.89, 4.70, "hc", 0.06),
        ("H13", 0.70, 0.89, 6.70, "hc", 0.06),
        ("H14", 0.70, -0.89, 6.70, "hc", 0.06),
        ("H15", 3.05, 0.89, 7.00, "hc", 0.06),
        ("H16", 3.05, -0.89, 7.00, "hc", 0.06),
        ("H17", 1.95, 0.89, 9.05, "hc", 0.06),
        ("H18", 1.95, -0.89, 9.05, "hc", 0.06),
        ("H19", 3.45, 0.00, 9.25, "hc", 0.06),
    ]
    h_start = len(atoms) + 1
    for h in h_tail:
        atoms.append(h)
    bonds.append((tail_start, h_start, "1"))
    bonds.append((tail_start, h_start + 1, "1"))
    bonds.append((tail_start + 1, h_start + 2, "1"))
    bonds.append((tail_start + 1, h_start + 3, "1"))
    bonds.append((tail_start + 2, h_start + 4, "1"))
    bonds.append((tail_start + 2, h_start + 5, "1"))
    bonds.append((tail_start + 3, h_start + 6, "1"))
    bonds.append((tail_start + 3, h_start + 7, "1"))
    bonds.append((tail_start + 4, h_start + 8, "1"))
    bonds.append((tail_start + 4, h_start + 9, "1"))
    bonds.append((tail_start + 4, h_start + 10, "1"))

    return format_mol2("5cb", atoms, bonds, "4-Cyano-4'-pentylbiphenyl (5CB) nematic liquid crystal")


def build_sds() -> str:
    # Sodium Dodecyl Sulfate: C12H25-O-SO3(-) Na(+)
    atoms = []
    bonds = []

    # Sulfate headgroup + Na+
    atoms.append(("Na1", 0.0, 0.0, -3.20, "Na", 1.000000))
    atoms.append(("S1", 0.0, 0.0, -1.00, "s6", 1.300000))
    atoms.append(("O1", 1.40, 0.0, -1.30, "o", -0.650000))
    atoms.append(("O2", -0.70, 1.21, -1.30, "o", -0.650000))
    atoms.append(("O3", -0.70, -1.21, -1.30, "o", -0.650000))
    atoms.append(("O4", 0.0, 0.0, 0.60, "os", -0.450000))

    bonds.append((2, 3, "2"))
    bonds.append((2, 4, "2"))
    bonds.append((2, 5, "2"))
    bonds.append((2, 6, "1"))

    # Dodecyl tail: C1 to C12 along z/x zigzag
    c_start = len(atoms) + 1
    for i in range(12):
        z = 1.80 + i * 1.27
        x = 0.44 if (i % 2 == 0) else -0.44
        y = 0.0
        q_c = 0.10 if i == 0 else (-0.18 if i == 11 else -0.12)
        atoms.append((f"C{i + 1}", x, y, z, "c3", q_c))

    bonds.append((6, c_start, "1"))  # O4 - C1
    for i in range(11):
        bonds.append((c_start + i, c_start + i + 1, "1"))

    # Hydrogens
    for i in range(12):
        z = 1.80 + i * 1.27
        x = 0.88 if (i % 2 == 0) else -0.88
        q_h = 0.06
        atoms.append((f"H{i * 2 + 1}", x, 0.89, z, "hc", q_h))
        h1 = len(atoms)
        atoms.append((f"H{i * 2 + 2}", x, -0.89, z, "hc", q_h))
        h2 = len(atoms)
        bonds.append((c_start + i, h1, "1"))
        bonds.append((c_start + i, h2, "1"))
        if i == 11:
            atoms.append(("H25", x, 0.0, z + 0.89, "hc", q_h))
            bonds.append((c_start + i, len(atoms), "1"))

    return format_mol2("sds", atoms, bonds, "Sodium dodecyl sulfate (SDS) anionic surfactant")


# 32 Curated Benchmark Materials (20 Baseline + 12 High-Variance FreeSolv Additions)
BENCHMARK_MATERIALS = [
    # 1-20 Original Benchmark Suite
    ("Water", "generate", build_water, "water.mol2"),
    ("Nitrogen", "generate", build_nitrogen, "nitrogen.mol2"),
    ("Methane", "freesolv", "mobley_9055303.mol2", "methane.mol2"),
    ("Carbon dioxide", "generate", build_carbon_dioxide, "carbon_dioxide.mol2"),
    ("Argon", "generate", build_argon, "argon.mol2"),
    ("Sodium chloride", "generate", build_sodium_chloride, "sodium_chloride.mol2"),
    ("Calcium chloride", "generate", build_calcium_chloride, "calcium_chloride.mol2"),
    ("n-Decane", "freesolv", "mobley_2197088.mol2", "n_decane.mol2"),
    ("Neopentane", "freesolv", "mobley_1261349.mol2", "neopentane.mol2"),
    ("Polyethylene", "generate", build_polyethylene, "polyethylene.mol2"),
    ("Methanol", "freesolv", "mobley_1636752.mol2", "methanol.mol2"),
    ("Ammonia", "freesolv", "mobley_5631798.mol2", "ammonia.mol2"),
    ("Hydrogen fluoride", "generate", build_hydrogen_fluoride, "hydrogen_fluoride.mol2"),
    ("Benzene", "freesolv", "mobley_3053621.mol2", "benzene.mol2"),
    ("5CB (4-Cyano-4'-pentylbiphenyl)", "generate", build_5cb, "5cb.mol2"),
    ("Sodium dodecyl sulfate (SDS)", "generate", build_sds, "sodium_dodecyl_sulfate.mol2"),
    ("Sulfur hexafluoride", "generate", build_sulfur_hexafluoride, "sulfur_hexafluoride.mol2"),
    ("Acetone", "freesolv", "mobley_3867265.mol2", "acetone.mol2"),
    ("Large colloidal hard sphere", "generate", build_colloidal_hard_sphere, "colloidal_hard_sphere.mol2"),
    ("Hydrogen", "generate", build_hydrogen, "hydrogen.mol2"),
    # 21-32 High-Variance FreeSolv Additions (Reaching full 32 batch)
    ("Ethanol", "freesolv", "mobley_2310185.mol2", "ethanol.mol2"),
    ("Acetic acid", "freesolv", "mobley_3034976.mol2", "acetic_acid.mol2"),
    ("Ethyl acetate", "freesolv", "mobley_6973347.mol2", "ethyl_acetate.mol2"),
    ("Diethyl ether", "freesolv", "mobley_1144156.mol2", "diethyl_ether.mol2"),
    ("Pyridine", "freesolv", "mobley_296847.mol2", "pyridine.mol2"),
    ("Chlorobenzene", "freesolv", "mobley_7608462.mol2", "chlorobenzene.mol2"),
    ("Chloroform", "freesolv", "mobley_2996632.mol2", "chloroform.mol2"),
    ("Acetonitrile", "freesolv", "mobley_7532833.mol2", "acetonitrile.mol2"),
    ("Phenol", "freesolv", "mobley_20524.mol2", "phenol.mol2"),
    ("Aniline", "freesolv", "mobley_4883284.mol2", "aniline.mol2"),
    ("Cyclohexane", "freesolv", "mobley_2689721.mol2", "cyclohexane.mol2"),
    ("Ethanethiol", "freesolv", "mobley_1800170.mol2", "ethanethiol.mol2"),
]


def verify_and_extract_freesolv() -> Path:
    """
    Verifies that the FreeSolv submodule / GAFF database is available.
    Extracts mol2files_gaff.tar.gz and copies metadata files to data/ if not already present.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy metadata files if FreeSolv submodule exists
    if FREESOLV_SUBMODULE_DIR.exists():
        metadata_files = [
            "database.json",
            "database.pickle",
            "database.txt",
            "iupac_to_cid.json",
            "smiles_to_cid.json",
        ]
        for f in metadata_files:
            src = FREESOLV_SUBMODULE_DIR / f
            dst = DATA_DIR / f
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    # 2. Check if data/mol2files_gaff exists and has .mol2 files
    if MOL2_GAFF_DIR.exists() and any(MOL2_GAFF_DIR.glob("*.mol2")):
        return MOL2_GAFF_DIR

    # 3. Extract tarball from FreeSolv submodule or data/
    tar_paths = [
        FREESOLV_SUBMODULE_DIR / "mol2files_gaff.tar.gz",
        DATA_DIR / "mol2files_gaff.tar.gz",
    ]

    for tar_p in tar_paths:
        if tar_p.exists():
            print(f"Extracting FreeSolv GAFF archive from {tar_p} -> {MOL2_GAFF_DIR}...")
            MOL2_GAFF_DIR.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_p, "r:gz") as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(path=DATA_DIR, filter="data")
                else:
                    tar.extractall(path=DATA_DIR)
            return MOL2_GAFF_DIR

    if FREESOLV_SUBMODULE_DIR.exists() and (FREESOLV_SUBMODULE_DIR / "database.pickle").exists():
        return FREESOLV_SUBMODULE_DIR

    raise FileNotFoundError(
        "FreeSolv database not found!\n"
        "Please initialize the FreeSolv submodule by running:\n"
        "  git submodule update --init --recursive\n"
        f"or place mol2files_gaff.tar.gz into {FREESOLV_SUBMODULE_DIR}."
    )


def generate_all(populate_entire_freesolv: bool = False) -> None:
    """Populates data/test_data with benchmark materials and force field parameters."""
    print("================================================================================")
    print("  dens-city: Test Data & Benchmark Molecular Dataset Generator")
    print("================================================================================")

    # Step 1: Verify FreeSolv availability
    freesolv_src_dir = verify_and_extract_freesolv()
    print(f"  FreeSolv Source     : {freesolv_src_dir}")

    # Step 2: Ensure data/ and data/test_data/ exist
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Target Test Data Dir: {TEST_DATA_DIR}")
    mode_str = (
        "ENTIRE FreeSolv Database (642+ molecules)" if populate_entire_freesolv else "32 Core Benchmark Materials"
    )
    print(f"  Mode                : {mode_str}")
    print("--------------------------------------------------------------------------------")

    # Step 3: Populate canonical benchmark materials (32 materials)
    for idx, (name, src_type, src_val, out_filename) in enumerate(BENCHMARK_MATERIALS, 1):
        out_path = TEST_DATA_DIR / out_filename
        if src_type == "freesolv":
            src_path = freesolv_src_dir / src_val
            if not src_path.exists() and (freesolv_src_dir / "mol2files_gaff" / src_val).exists():
                src_path = freesolv_src_dir / "mol2files_gaff" / src_val

            if src_path.exists():
                shutil.copy2(src_path, out_path)
                print(f"[{idx:02d}/32] {name:<35} <- FreeSolv ({src_val}) -> {out_filename}")
            else:
                print(f"Warning: FreeSolv file {src_path} not found for {name}", file=sys.stderr)
        else:
            mol2_content = src_val()
            out_path.write_text(mol2_content)
            print(f"[{idx:02d}/32] {name:<35} <- Generated canonical model -> {out_filename}")

    # Step 4: If --all requested, copy EVERY remaining FreeSolv molecule into test_data/
    if populate_entire_freesolv:
        print("--------------------------------------------------------------------------------")
        print("  Populating entire FreeSolv database into data/test_data/...")
        all_freesolv_files = list(freesolv_src_dir.glob("*.mol2"))
        if not all_freesolv_files and (freesolv_src_dir / "mol2files_gaff").exists():
            all_freesolv_files = list((freesolv_src_dir / "mol2files_gaff").glob("*.mol2"))

        copied_extra = 0
        for f in all_freesolv_files:
            dst = TEST_DATA_DIR / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied_extra += 1
        print(f"  Copied {copied_extra} additional FreeSolv molecules into {TEST_DATA_DIR}.")

    # Step 5: Generate force field parameters and GAFF database in data/test_data/
    print("--------------------------------------------------------------------------------")
    print("  Generating Force Field Parameters & GAFF Database in data/test_data/...")
    generate_json_and_csv()

    # Step 6: Verification
    mol2_files = list(TEST_DATA_DIR.glob("*.mol2"))
    print("--------------------------------------------------------------------------------")
    print(f"  Verification: Found {len(mol2_files)} total .mol2 files in {TEST_DATA_DIR}")
    print("  Status: COMPLETE")
    print("================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate test_data directory with benchmark molecules and force field parameters."
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Extract and populate EVERY molecule from the FreeSolv database (642+ molecules) into test_data/",
    )
    args = parser.parse_args()
    generate_all(populate_entire_freesolv=args.all)


if __name__ == "__main__":
    main()
