"""
dens-city: Unified Command-Line Interface.
Supports both batch/analytical cDFT optimization and high-performance 3D interactive Raylib visualization.

Usage:
    # 3D Interactive Raylib Visualizer
    uv run dens-city --interactive --materials argon
    uv run dens-city --interactive --materials water benzene 5cb

    # Standard Classical Density Functional Theory Solver
    uv run dens-city --materials argon water
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional

from tinygrad.helpers import colored, getenv

from dens_city.cdft import TinyCDFT
from dens_city.ui.viewer import run_interactive_viewer
from dens_city.utils.materials import MaterialLoader


def parse_materials_arg(mat_args: Optional[List[str]]) -> List[str]:
    """Parses material names from CLI arguments, handling commas, brackets, and 'all'."""
    all_materials = MaterialLoader.list_available_materials()

    if not mat_args:
        return ["argon"]

    joined = " ".join(mat_args).strip()
    if joined.lower() in ["all", "[all]", "all_materials", "*"]:
        return all_materials

    cleaned = re.sub(r"[\[\]\'\",]", " ", joined)
    requested = [m.strip().lower() for m in cleaned.split() if m.strip()]

    if "all" in requested:
        return all_materials

    return requested if requested else ["argon"]


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint for dens-city."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="dens-city",
        description="dens-city: Molecular Classical Density Functional Theory & 3D Interactive Platform.",
    )
    parser.add_argument(
        "--materials",
        "-m",
        nargs="+",
        default=["argon"],
        help="Material names (e.g. argon, water, methane, 5cb, or 'all')",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch the 3D Raylib molecular viewer for real-time visualization",
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

    args = parser.parse_args(argv)
    materials_list = parse_materials_arg(args.materials)

    if args.interactive:
        print(colored("==========================================================================", "cyan"))
        print(colored("  dens-city: 3D Interactive Raylib Molecular Visualizer                  ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"Target Materials   : {materials_list}")
        print("Controls           : Left Drag to Orbit | Scroll to Zoom | ← / → to Switch Materials")
        run_interactive_viewer(material_names=materials_list)
        return

    # Standard cDFT headless / ASCII runner
    print(colored("==========================================================================", "cyan"))
    print(colored("  dens-city: Variational cDFT Statistical Mechanics Solver (tinygrad)    ", "cyan"))
    print(colored("==========================================================================", "cyan"))
    print(f"Target Materials   : {materials_list}")
    print(
        f"Spatial Grid       : {args.grid} bins across {'dynamic scale-aware' if args.slit_width is None else f'{args.slit_width:.1f} Å'} slit"
    )
    print(
        f"Thermodynamics     : Temperature = {300.0 if args.temp is None else args.temp:.2f} K, "
        f"Pressure = {'default / EOS' if args.pressure is None else f'{args.pressure:.2f} bar'}"
    )
    print(f"Optimizer          : Adam (lr = {args.lr:.3f}, steps = {args.steps})")
    print(colored("==========================================================================", "cyan"))

    results_summary = []
    t_start_total = sys.modules["time"].time()

    for idx, mat_name in enumerate(materials_list, 1):
        print(f"\n[{idx:02d}/{len(materials_list):02d}] Solving cDFT for: {colored(mat_name, 'green')}")
        try:
            mat = MaterialLoader.load_material(
                mat_name,
                temperature_k=args.temp,
                pressure_bar=args.pressure,
                chemical_potential_kbt=args.mu,
            )
            print(f"  Molecular Model  : {len(mat.sites)}-site flexible/polar ({mat.dimension_mode})")
            print(f"  Effective σ      : {mat.effective_sigma:6.2f} Å")
            print(f"  Effective ε/k_B  : {mat.effective_epsilon_k:6.2f} K")
            print(f"  Derived Bulk ρ_b : {mat.bulk_density_a3:9.6f} Å⁻³ ({mat.molarity_mol_l:.2f} M)")
            print(f"  Derived Bulk μ   : {mat.bulk_mu:6.2f} k_B*T")

            solver = TinyCDFT(
                material=mat,
                n_grid=args.grid,
                slit_width_a=args.slit_width,
                learning_rate=args.lr,
                temperature_k=args.temp,
            )

            t0 = sys.modules["time"].time()
            res = solver.solve(steps=args.steps, verbose=True)
            elapsed = sys.modules["time"].time() - t0

            if not args.no_plot:
                print("\n" + solver.ascii_plot() + "\n")

            print(f"  Converged in     : {elapsed:.3f} s ({args.steps / max(1e-4, elapsed):.1f} steps/s)")
            print(f"  Wall Pressure    : {res['wall_pressure_bar']:8.2f} bar")
            print(f"  Excess Adsorption: {res['excess_adsorption']:8.4f} molecules / Å²")
            print(f"  Peak Density     : {res['peak_density']:8.4f} Å⁻³ (Layering Peak)")

            results_summary.append(
                {
                    "material": mat.name,
                    "mode": mat.dimension_mode,
                    "sigma": mat.effective_sigma,
                    "epsilon_k": mat.effective_epsilon_k,
                    "bulk_density": mat.bulk_density_a3,
                    "molarity": mat.molarity_mol_l,
                    "wall_pressure_bar": res["wall_pressure_bar"],
                    "elapsed": elapsed,
                }
            )
        except Exception as e:
            print(colored(f"Error simulating {mat_name}: {e}", "red"), file=sys.stderr)

    total_time = sys.modules["time"].time() - t_start_total
    print(colored("\n" + "=" * 106, "cyan"))
    print(
        colored(
            f"  cDFT Simulation Suite Finished in {total_time:.2f} s ({len(results_summary)} / {len(materials_list)} succeeded)",
            "cyan",
        )
    )
    print(colored("=" * 106, "cyan"))
    print(
        f"{'Material':<22} {'Mode':<14} {'σ (Å)':>7} {'ε (K)':>7} {'ρ_b (Å⁻³)':>10} {'M (mol/L)':>10} {'P_wall (bar)':>13} {'Time (s)':>9}"
    )
    print("-" * 106)
    for r in results_summary:
        print(
            f"{r['material']:<22} {r['mode']:<14} {r['sigma']:7.2f} {r['epsilon_k']:7.1f} "
            f"{r['bulk_density']:10.5f} {r['molarity']:10.2f} {r['wall_pressure_bar']:13.2f} {r['elapsed']:9.3f}"
        )
    print("-" * 106)


if __name__ == "__main__":
    main()
