#!/usr/bin/env python3
"""
dens-city: Pure tinygrad Classical Density Functional Theory (cDFT) CLI runner.

Solves the variational grand free energy minimization problem for any material
(or list of materials) specified from the test_data/ directory.

Usage:
    # Run specific materials
    python scripts/run_cdft.py --materials argon water
    python scripts/run_cdft.py --materials [methane, benzene, 5cb]

    # Run all 20 benchmark fluids
    python scripts/run_cdft.py --materials all

    # With BEAM search compiler
    BEAM=2 python scripts/run_cdft.py --materials argon --steps 500
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import List

# Ensure src/ is on PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from tinygrad.helpers import colored, getenv
from dens_city.materials import MaterialLoader
from dens_city.cdft import TinyCDFT


def parse_materials_arg(mat_args: List[str]) -> List[str]:
    """Parses material names from CLI arguments, handling commas, brackets, and 'all'."""
    all_materials = MaterialLoader.list_available_materials()
    
    if not mat_args:
        return ["argon"]

    joined = " ".join(mat_args).strip()
    if joined.lower() in ["all", "[all]", "all_materials", "*"]:
        return all_materials

    # Clean brackets, commas, quotes
    cleaned = re.sub(r"[\[\]\'\",]", " ", joined)
    requested = [m.strip().lower() for m in cleaned.split() if m.strip()]

    if "all" in requested:
        return all_materials

    return requested if requested else ["argon"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve 3D/1D Classical Density Functional Theory (cDFT) in pure tinygrad."
    )
    parser.add_argument(
        "--materials",
        "-m",
        nargs="+",
        default=["argon"],
        help="Material names (e.g. argon, water, methane, 5cb, or 'all')",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=int(getenv("STEPS", 300)),
        help="Number of variational optimization gradient descent steps (default: 300)",
    )
    parser.add_argument(
        "--grid",
        "-g",
        type=int,
        default=int(getenv("GRID", 128)),
        help="Number of spatial grid bins (default: 128)",
    )
    parser.add_argument(
        "--slit-width",
        "-w",
        type=float,
        default=None,
        help="Confining slit width in Angstroms (default: dynamic 12.0*sigma)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(getenv("LR", 0.02)),
        help="Optimizer learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--temp",
        "-t",
        type=float,
        default=None,
        help="System temperature in Kelvin (default: 300.0 K)",
    )
    parser.add_argument(
        "--pressure",
        "-p",
        type=float,
        default=None,
        help="Target bulk reservoir pressure in bar for EOS density solving",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Target bulk chemical potential in k_B * T for EOS density solving",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable ASCII density profile visualization",
    )

    args = parser.parse_args()
    materials_list = parse_materials_arg(args.materials)

    print(colored("==========================================================================", "cyan"))
    print(colored("  dens-city: Variational cDFT Statistical Mechanics Solver (tinygrad)    ", "cyan"))
    print(colored("==========================================================================", "cyan"))
    print(f"Target Materials   : {materials_list}")
    print(f"Spatial Grid       : {args.grid} bins across {'dynamic scale-aware' if args.slit_width is None else f'{args.slit_width:.1f} Å'} slit")
    print(f"Optimization Steps : {args.steps} (lr = {args.lr})")
    print("--------------------------------------------------------------------------")

    results_summary = []
    t_start_total = time.time()

    for idx, mat_name in enumerate(materials_list, 1):
        try:
            print(f"\n[{idx:02d}/{len(materials_list):02d}] Initializing Fluid: {colored(mat_name, 'green')}...")
            mat = MaterialLoader.load_material(
                mat_name,
                temperature_k=args.temp,
                pressure_bar=args.pressure,
                chemical_potential_kbt=args.mu,
            )
            print(f"  Dimension Mode   : {mat.dimension_mode}")
            print(f"  Molecular Span   : {mat.molecular_span_a:.2f} Å (R_g = {mat.radius_of_gyration_a:.2f} Å, {mat.num_sites} sites)")
            print(f"  Effective LJ Core: σ = {mat.effective_sigma:.3f} Å, ε = {mat.effective_epsilon_k:.1f} K")
            print(f"  Bulk Reservoir   : ρ = {mat.bulk_density_a3:.5f} Å⁻³ ({mat.molarity_mol_l:.2f} M), μ_ex = {mat.bulk_mu:.3f} kBT")

            solver = TinyCDFT(
                material=mat,
                n_grid=args.grid,
                slit_width_a=args.slit_width,
                learning_rate=args.lr,
                temperature_k=args.temp,
            )

            t0 = time.time()
            res = solver.solve(steps=args.steps, verbose=True)
            elapsed = time.time() - t0

            if not args.no_plot:
                print("\n" + solver.ascii_plot() + "\n")

            print(f"  Converged in     : {elapsed:.3f} s ({args.steps / max(1e-4, elapsed):.1f} steps/s)")
            print(f"  Wall Pressure    : {res['wall_pressure_bar']:8.2f} bar")
            print(f"  Excess Adsorption: {res['excess_adsorption']:8.4f} molecules / Å²")
            print(f"  Peak Density     : {res['peak_density']:8.4f} Å⁻³ (Layering Peak)")

            results_summary.append({
                "material": mat.name,
                "mode": mat.dimension_mode,
                "sigma": mat.effective_sigma,
                "epsilon_k": mat.effective_epsilon_k,
                "bulk_density": mat.bulk_density_a3,
                "molarity": mat.molarity_mol_l,
                "loss": res["final_loss"],
                "wall_pressure_bar": res["wall_pressure_bar"],
                "excess_adsorption": res["excess_adsorption"],
                "peak_density": res["peak_density"],
                "elapsed": elapsed,
            })

        except Exception as e:
            print(colored(f"Error simulating {mat_name}: {e}", "red"), file=sys.stderr)

    total_time = time.time() - t_start_total
    print("\n" + colored("==========================================================================================================", "cyan"))
    print(colored(f"  cDFT Simulation Suite Finished in {total_time:.2f} s ({len(results_summary)} / {len(materials_list)} succeeded)", "cyan"))
    print(colored("==========================================================================================================", "cyan"))
    print(f"{'Material':<22} {'Mode':<13} {'σ (Å)':>6} {'ε (K)':>7} {'ρ_b (Å⁻³)':>10} {'M (mol/L)':>10} {'P_wall (bar)':>13} {'Time (s)':>9}")
    print("-" * 106)
    for r in results_summary:
        print(f"{r['material']:<22} {r['mode']:<13} {r['sigma']:>6.2f} {r['epsilon_k']:>7.1f} {r['bulk_density']:>10.5f} {r['molarity']:>10.2f} {r['wall_pressure_bar']:>13.2f} {r['elapsed']:>9.3f}")
    print("-" * 106)


if __name__ == "__main__":
    main()
