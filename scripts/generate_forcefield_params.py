#!/usr/bin/env python3
"""
Generates standard Amber gaff.dat and forcefield_parameters.json lookup files
for all GAFF and benchmark fluid atom types present in test_data/.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"

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
    """Computes sigma in Angstroms and nm from R_min/2."""
    if r_min_half_A == 0.0:
        return 0.0, 0.0
    # sigma = 2 * (R_min/2) / (2^(1/6))
    sigma_A = 2.0 * r_min_half_A / (2.0 ** (1.0 / 6.0))
    sigma_nm = sigma_A / 10.0
    return sigma_A, sigma_nm


def compute_epsilon(eps_kcal: float) -> tuple:
    """Computes epsilon in kJ/mol and Kelvin from kcal/mol."""
    eps_kj = eps_kcal * 4.184
    eps_K = (eps_kcal * 4184.0) / 8.314462618
    return eps_kj, eps_K


def generate_gaff_dat() -> str:
    """Generates standard Amber gaff.dat format file."""
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


def generate_json_and_csv() -> None:
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

    json_path = TEST_DATA_DIR / "forcefield_parameters.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"Generated: {json_path}")

    csv_path = TEST_DATA_DIR / "forcefield_parameters.csv"
    csv_path.write_text("\n".join(csv_lines))
    print(f"Generated: {csv_path}")

    dat_path = TEST_DATA_DIR / "gaff.dat"
    dat_path.write_text(generate_gaff_dat())
    print(f"Generated: {dat_path}")


if __name__ == "__main__":
    generate_json_and_csv()
