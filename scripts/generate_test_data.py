#!/usr/bin/env python3
"""
Populates the data/test_data/ directory with standard Tripos .mol2 structure files
for benchmark materials in dens-city, copying directly from FreeSolv where available,
and generating canonical geometries for the remaining fluids. Also generates
forcefield_parameters.json, forcefield_parameters.csv, and gaff.dat.

Usage:
    python scripts/generate_test_data.py        # Populates 32 core benchmark materials
    python scripts/generate_test_data.py --all  # Populates ALL 642+ FreeSolv molecules + canonical fluids
"""

import argparse
import json
import math
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TEST_DATA_DIR = DATA_DIR / "test_data"
FREESOLV_SUBMODULE_DIR = REPO_ROOT / "FreeSolv"
MOL2_GAFF_DIR = DATA_DIR / "mol2files_gaff"

# Comprehensive atomtype lookup database
# Keys: atom type -> (R_min_half_A, eps_kcal_mol, mass, atomic_num, description)
ATOMTYPES_DB = {
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
    "ow": (1.7766, 0.1553, 16.00, 8, "SPC/E Water Oxygen (sigma=3.166 A, eps=78.2 K)"),
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
    "Ar": (1.9109, 0.2380, 39.95, 18, "Monoatomic Argon LJ (sigma=3.405 A, eps=119.8 K)"),
    "Na": (1.8680, 0.0028, 22.99, 11, "Sodium cation Na+"),
    "Cl": (2.4700, 0.1000, 35.45, 17, "Chloride anion Cl- (RPM / electrolyte)"),
    "Ca": (1.7130, 0.4590, 40.08, 20, "Calcium divalent cation Ca2+"),
    "Col": (8.4180, 0.0000, 1000.0, 0, "Large Colloidal Hard Sphere (sigma=15.0 A)"),
}


def compute_sigma(r_min_half_A: float) -> tuple:
    if r_min_half_A == 0.0:
        return 0.0, 0.0
    sigma_A = 2.0 * r_min_half_A / (2.0 ** (1.0 / 6.0))
    sigma_nm = sigma_A / 10.0
    return sigma_A, sigma_nm


def compute_epsilon(eps_kcal: float) -> tuple:
    eps_kj = eps_kcal * 4.184
    eps_K = (eps_kcal * 4184.0) / 8.314462618
    return eps_kj, eps_K


def generate_gaff_dat() -> str:
    lines = [
        "AMBER General Amber Force Field (GAFF) & Extended dens-city Parameters",
        "MOD4",
        "MASS",
    ]
    for atype, (r_min, eps_kcal, mass, at_num, desc) in sorted(ATOMTYPES_DB.items()):
        lines.append(f"{atype:<6} {mass:>7.3f}        0.878               {desc}")
    lines.append("")
    lines.append("NONBON")
    for atype, (r_min, eps_kcal, mass, at_num, desc) in sorted(ATOMTYPES_DB.items()):
        sig_A, sig_nm = compute_sigma(r_min)
        eps_kj, eps_K = compute_epsilon(eps_kcal)
        lines.append(
            f"  {atype:<6} {r_min:>10.4f} {eps_kcal:>10.4f}             {desc} (sigma={sig_A:.4f}A, eps={eps_K:.1f}K)"
        )
    lines.append("")
    return "\n".join(lines)


def generate_forcefield_files(dest_dir: Path) -> None:
    json_data = {}
    csv_lines = [
        "atom_type,atomic_number,mass_amu,r_min_half_A,sigma_A,sigma_nm,epsilon_kcal_mol,epsilon_kj_mol,epsilon_K,description"
    ]
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

    json_path = dest_dir / "forcefield_parameters.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    csv_path = dest_dir / "forcefield_parameters.csv"
    csv_path.write_text("\n".join(csv_lines))
    dat_path = dest_dir / "gaff.dat"
    dat_path.write_text(generate_gaff_dat())


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
        at_name, x, y, z, at_type, q = at
        lines.append(f"{i:>7} {at_name:<6} {x:>12.4f} {y:>10.4f} {z:>10.4f} {at_type:<8} 1 MOL {q:>12.6f}")

    lines.append("@<TRIPOS>BOND")
    for i, b in enumerate(bonds, start=1):
        a1, a2, b_type = b
        lines.append(f"{i:>6} {a1:>6} {a2:>6} {b_type:<4}")

    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append("     1 MOL         1 TEMP              0 ****  ****    0 ROOT")
    lines.append("")
    return "\n".join(lines)


def build_water() -> str:
    r_oh = 1.0000
    half_angle = math.radians(109.47 / 2.0)
    hx = r_oh * math.sin(half_angle)
    hz = r_oh * math.cos(half_angle)
    atoms = [
        ("O1", 0.0, 0.0, 0.0, "ow", -0.847600),
        ("H1", hx, 0.0, hz, "hw", 0.423800),
        ("H2", -hx, 0.0, hz, "hw", 0.423800),
    ]
    bonds = [(1, 2, "1"), (1, 3, "1")]
    return format_mol2("water", atoms, bonds, "SPC/E Water model (dens-city)")


def build_nitrogen() -> str:
    atoms = [
        ("N1", 0.0, 0.0, -0.5500, "n1", 0.000000),
        ("N2", 0.0, 0.0, 0.5500, "n1", 0.000000),
    ]
    bonds = [(1, 2, "3")]
    return format_mol2("nitrogen", atoms, bonds, "TraPPE Diatomic Nitrogen (N2)")


def build_carbon_dioxide() -> str:
    atoms = [
        ("C1", 0.0, 0.0, 0.0000, "c1", 0.700000),
        ("O1", 0.0, 0.0, -1.1600, "o", -0.350000),
        ("O2", 0.0, 0.0, 1.1600, "o", -0.350000),
    ]
    bonds = [(1, 2, "2"), (1, 3, "2")]
    return format_mol2("carbon_dioxide", atoms, bonds, "TraPPE Carbon Dioxide (CO2)")


def build_argon() -> str:
    atoms = [("Ar1", 0.0, 0.0, 0.0, "Ar", 0.000000)]
    bonds = []
    return format_mol2("argon", atoms, bonds, "Monoatomic Argon (Ar)")


def build_sodium_chloride() -> str:
    atoms = [
        ("Na1", 0.0, 0.0, -1.1800, "Na", 1.000000),
        ("Cl1", 0.0, 0.0, 1.1800, "Cl", -1.000000),
    ]
    bonds = [(1, 2, "1")]
    return format_mol2("sodium_chloride", atoms, bonds, "Sodium Chloride (NaCl) 1:1 RPM electrolyte")


def build_calcium_chloride() -> str:
    atoms = [
        ("Ca1", 0.0, 0.0, 0.0000, "Ca", 2.000000),
        ("Cl1", 0.0, 0.0, -2.7000, "Cl", -1.000000),
        ("Cl2", 0.0, 0.0, 2.7000, "Cl", -1.000000),
    ]
    bonds = [(1, 2, "1"), (1, 3, "1")]
    return format_mol2("calcium_chloride", atoms, bonds, "Calcium Chloride (CaCl2) 2:1 electrolyte")


def build_hydrogen_fluoride() -> str:
    atoms = [
        ("F1", 0.0, 0.0, 0.0000, "f", -0.450000),
        ("H1", 0.0, 0.0, 0.9170, "ha", 0.450000),
    ]
    bonds = [(1, 2, "1")]
    return format_mol2("hydrogen_fluoride", atoms, bonds, "Hydrogen Fluoride (HF) 1D Associating fluid")


def build_sulfur_hexafluoride() -> str:
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
    bonds = [(1, 2, "1"), (1, 3, "1"), (1, 4, "1"), (1, 5, "1"), (1, 6, "1"), (1, 7, "1")]
    return format_mol2("sulfur_hexafluoride", atoms, bonds, "Sulfur Hexafluoride (SF6) octahedral fluid")


def build_hydrogen() -> str:
    atoms = [
        ("H1", 0.0, 0.0, -0.3707, "ha", 0.000000),
        ("H2", 0.0, 0.0, 0.3707, "ha", 0.000000),
    ]
    bonds = [(1, 2, "1")]
    return format_mol2("hydrogen", atoms, bonds, "Molecular Hydrogen (H2)")


def build_colloidal_hard_sphere() -> str:
    atoms = [("COL1", 0.0, 0.0, 0.0, "Col", 0.000000)]
    bonds = []
    return format_mol2("colloidal_hard_sphere", atoms, bonds, "Large Colloidal Hard Sphere (D=15A)")


def build_polyethylene() -> str:
    n_c = 20
    atoms = []
    bonds = []
    c_idx = []
    for i in range(n_c):
        x = i * 1.27
        y = 0.44 if (i % 2 == 0) else -0.44
        z = 0.0
        c_name = f"C{i + 1}"
        q_c = -0.18 if (i == 0 or i == n_c - 1) else -0.12
        atoms.append((c_name, x, y, z, "c3", q_c))
        c_idx.append(len(atoms))

        h_y = 0.88 if (i % 2 == 0) else -0.88
        atoms.append((f"H{i * 2 + 1}", x, h_y, 0.89, "hc", 0.06))
        h1_idx = len(atoms)
        atoms.append((f"H{i * 2 + 2}", x, h_y, -0.89, "hc", 0.06))
        h2_idx = len(atoms)

        bonds.append((c_idx[-1], h1_idx, "1"))
        bonds.append((c_idx[-1], h2_idx, "1"))
        if i == 0:
            atoms.append(("H0", x - 0.89, y, 0.0, "hc", 0.06))
            bonds.append((c_idx[-1], len(atoms), "1"))
        elif i == n_c - 1:
            atoms.append((f"H{n_c * 2 + 1}", x + 0.89, y, 0.0, "hc", 0.06))
            bonds.append((c_idx[-1], len(atoms), "1"))
        if i > 0:
            bonds.append((c_idx[i - 1], c_idx[i], "1"))
    return format_mol2("polyethylene", atoms, bonds, "Polyethylene oligomer (C20H42 eicosane chain)")


def build_5cb() -> str:
    atoms = []
    bonds = []
    atoms.append(("N1", 0.0, 0.0, -7.0, "n1", -0.550000))
    atoms.append(("C1", 0.0, 0.0, -5.85, "c1", 0.450000))
    bonds.append((1, 2, "3"))

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
    bonds.append((2, r1_start, "1"))
    for i in range(5):
        bonds.append((r1_start + i, r1_start + i + 1, "ar"))
    bonds.append((r1_start + 5, r1_start, "ar"))
    bonds.append((r1_start + 1, r1_start + 6, "1"))
    bonds.append((r1_start + 2, r1_start + 7, "1"))
    bonds.append((r1_start + 4, r1_start + 8, "1"))
    bonds.append((r1_start + 5, r1_start + 9, "1"))

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
    bonds.append((r1_start + 3, r2_start, "1"))
    for i in range(5):
        bonds.append((r2_start + i, r2_start + i + 1, "ar"))
    bonds.append((r2_start + 5, r2_start, "ar"))
    bonds.append((r2_start + 1, r2_start + 6, "1"))
    bonds.append((r2_start + 2, r2_start + 7, "1"))
    bonds.append((r2_start + 4, r2_start + 8, "1"))
    bonds.append((r2_start + 5, r2_start + 9, "1"))

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
    bonds.append((r2_start + 3, tail_start, "1"))
    for i in range(4):
        bonds.append((tail_start + i, tail_start + i + 1, "1"))

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
    atoms = []
    bonds = []
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

    c_start = len(atoms) + 1
    for i in range(12):
        z = 1.80 + i * 1.27
        x = 0.44 if (i % 2 == 0) else -0.44
        y = 0.0
        q_c = 0.10 if i == 0 else (-0.18 if i == 11 else -0.12)
        atoms.append((f"C{i + 1}", x, y, z, "c3", q_c))

    bonds.append((6, c_start, "1"))
    for i in range(11):
        bonds.append((c_start + i, c_start + i + 1, "1"))

    for i in range(12):
        z = 1.80 + i * 1.27
        x = 0.88 if (i % 2 == 0) else -0.88
        atoms.append((f"H{i * 2 + 1}", x, 0.89, z, "hc", 0.06))
        h1 = len(atoms)
        atoms.append((f"H{i * 2 + 2}", x, -0.89, z, "hc", 0.06))
        h2 = len(atoms)
        bonds.append((c_start + i, h1, "1"))
        bonds.append((c_start + i, h2, "1"))
        if i == 11:
            atoms.append(("H25", x, 0.0, z + 0.89, "hc", 0.06))
            bonds.append((c_start + i, len(atoms), "1"))
    return format_mol2("sds", atoms, bonds, "Sodium dodecyl sulfate (SDS) anionic surfactant")


BENCHMARK_MATERIALS = [
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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

    if MOL2_GAFF_DIR.exists() and any(MOL2_GAFF_DIR.glob("*.mol2")):
        return MOL2_GAFF_DIR

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
    print("================================================================================")
    print("  dens-city: Test Data & Benchmark Molecular Dataset Generator")
    print("================================================================================")

    freesolv_src_dir = verify_and_extract_freesolv()
    print(f"  FreeSolv Source     : {freesolv_src_dir}")

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Target Test Data Dir: {TEST_DATA_DIR}")
    mode_str = (
        "ENTIRE FreeSolv Database (642+ molecules)" if populate_entire_freesolv else "32 Core Benchmark Materials"
    )
    print(f"  Mode                : {mode_str}")
    print("--------------------------------------------------------------------------------")

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

    print("--------------------------------------------------------------------------------")
    print("  Generating Force Field Parameters & GAFF Database in data/test_data/...")
    generate_forcefield_files(TEST_DATA_DIR)

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
