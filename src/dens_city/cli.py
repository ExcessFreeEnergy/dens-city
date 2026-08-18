import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="dens-city",
        description="dens-city: High-Performance Molecular Density Functional Theory & Neural Operator Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Run unified PufferLib direct neural functional training")
    train_p.add_argument("--timesteps", type=int, default=50000, help="Total training steps")
    train_p.add_argument("--envs", type=int, default=16, help="Number of parallel C environments")
    train_p.add_argument("--save", type=str, default="dens_functional.pt", help="Path to save trained functional")

    # UI subcommand
    ui_p = subparsers.add_parser("ui", help="Launch interactive Raylib visualizer")
    ui_p.add_argument("--functional", type=str, default="dens_functional.pt", help="Path to neural functional")

    # 1. Water pipeline
    water_p = subparsers.add_parser("water", help="Execute water nanoconfinement and binodal pipeline")
    water_p.add_argument("--mode", choices=["confinement", "binodal", "all"], default="all")

    # 2. CO2 pipeline
    subparsers.add_parser("co2", help="Execute CO2 supercritical crossover pipeline")

    # 3. Electrolytes pipeline
    subparsers.add_parser("electrolytes", help="Execute RPM electrolyte double layer pipeline")

    # 4. CO2/Water binary mixture pipeline
    subparsers.add_parser("co2-water", help="Execute binary CO2/H2O solvation and competitive pore filling pipeline")

    # 5. Nitrogen flue gas separation pipeline
    subparsers.add_parser("nitrogen", help="Execute N2 linear diatomic flue gas separation pipeline")

    # 6. Methane shale gas recovery pipeline
    subparsers.add_parser("methane", help="Execute CH4 shale adsorption and gas recovery pipeline")

    # 7. Clay pore pipeline
    subparsers.add_parser("clay", help="Execute montmorillonite clay mineral swelling pressure pipeline")

    # 8. Liquid crystals pipeline
    subparsers.add_parser("liquid-crystals", help="Execute nematic liquid crystals and patchy particles pipeline")

    # 9. Argon pure Lennard-Jones baseline pipeline
    subparsers.add_parser("argon", help="Run pure Lennard-Jones Argon coexistence & FMT pipeline")
    subparsers.add_parser("interfaces", help="Run hydrophobic/hydrophilic wetting & capillary drying pipeline")

    # Benchmark / E2E subcommand
    bench_p = subparsers.add_parser("benchmark", help="Execute full end-to-end multi-material simulation & benchmark")
    bench_p.add_argument(
        "--materials",
        nargs="*",
        default=["all"],
        help="List of materials to simulate (e.g. water co2 liquid_crystals) or 'all'",
    )
    bench_p.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")

    # e2e alias
    e2e_p = subparsers.add_parser("e2e", help="Alias for benchmark")
    e2e_p.add_argument(
        "--materials",
        nargs="*",
        default=["all"],
        help="List of materials to simulate or 'all'",
    )
    e2e_p.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")

    args = parser.parse_args()

    if args.command == "train":
        from dens_city.envs.train import train_unified

        train_unified(total_timesteps=args.timesteps, num_envs=args.envs, save_path=args.save)
    elif args.command == "ui":
        from dens_city.ui.raylib_viewer import run_viewer

        run_viewer(functional_path=args.functional)
    elif args.command == "water":
        print("[dens-city] Executing Water Nanoconfinement & Binodal Pipeline...")
        from dens_city.pipelines.water.coexistence import compute_water_binodal
        from dens_city.pipelines.water.confinement import compute_confinement_isotherm

        def dummy_c1(rho, T):
            return -0.5 * (rho / 0.033)

        res_conf = compute_confinement_isotherm(dummy_c1, [8.0, 12.0, 16.0, 20.0])
        print(f"[dens-city] Water Disjoining Pressures across H: {res_conf['Pi_disjoining']}")
        res_bin = compute_water_binodal(dummy_c1, [300.0, 400.0, 500.0])
        print(f"[dens-city] Water Coexistence Densities rho_l: {res_bin['rho_l']}, rho_v: {res_bin['rho_v']}")
    elif args.command == "co2":
        print("[dens-city] Executing CO2 Supercritical Crossover Pipeline...")
        from dens_city.pipelines.co2.supercritical import compute_supercritical_crossovers

        def dummy_torch_c1(rho, T):
            return -0.4 * (rho / 0.02)

        res_co2 = compute_supercritical_crossovers(dummy_torch_c1, [320.0, 360.0, 400.0], [0.005, 0.010, 0.015, 0.020])
        print(f"[dens-city] CO2 Widom Line (max xi): {res_co2['widom_xi']}")
        print(f"[dens-city] CO2 Fisher-Widom Line: {res_co2['fisher_widom']}")
    elif args.command == "electrolytes":
        print("[dens-city] Executing RPM Electrolyte Double Layer Pipeline...")
        from dens_city.pipelines.electrolytes.double_layer import compute_differential_capacitance

        def dummy_c1(rho, T):
            return -0.3 * (rho / 0.005)

        v_arr, cap = compute_differential_capacitance(dummy_c1, [-1.0, -0.5, 0.0, 0.5, 1.0])
        print(f"[dens-city] Voltages: {v_arr} V | Capacitance: {cap}")
    elif args.command == "co2-water":
        print("[dens-city] Executing Binary CO2/H2O Mixture Pipeline...")
        from dens_city.pipelines.co2_water.mixture import (
            compute_competitive_pore_adsorption,
            compute_mutual_solubility,
        )

        res_sol = compute_mutual_solubility(T=310.0, P_atm=50.0)
        print(
            f"[dens-city] Mutual Solubility: x_CO2(aq) = {res_sol['x_CO2_liquid']:.4f}, y_H2O(gas) = {res_sol['y_H2O_vapor']:.4f}"
        )
        res_pore = compute_competitive_pore_adsorption(H=20.0, T=300.0, x_co2_feed=0.15)
        print(
            f"[dens-city] Competitive Slit Adsorption (H=20A): peak rho_water={res_pore['rho_water'].max():.3f}, center rho_co2={res_pore['rho_co2'].max():.3f}"
        )
    elif args.command == "nitrogen":
        print("[dens-city] Executing N2 Flue Gas Separation Pipeline...")
        from dens_city.pipelines.nitrogen.flue_gas import (
            compute_flue_gas_selectivity,
            compute_n2_orientational_isotherm,
        )

        res_sel = compute_flue_gas_selectivity(T=300.0, P_bar=1.0, y_co2=0.15, y_n2=0.85)
        print(
            f"[dens-city] CO2/N2 Selectivity: {res_sel['selectivity_CO2_N2']:.2f} (Adsorbed x_CO2={res_sel['x_CO2_adsorbed']:.3f})"
        )
        res_n2 = compute_n2_orientational_isotherm(None, H=20.0, T=298.15)
        print(f"[dens-city] N2 Near-Wall Nematic Order S_order min: {res_n2['S_order'].min():.3f} (Planar alignment)")
    elif args.command == "methane":
        print("[dens-city] Executing CH4 Shale Gas Pipeline...")
        from dens_city.pipelines.methane.shale import (
            compute_ch4_co2_gas_recovery_crossover,
            compute_methane_shale_isotherm,
        )

        res_shale = compute_methane_shale_isotherm([10.0, 20.0, 30.0], T=330.0)
        print(f"[dens-city] Methane Excess Adsorption across H (10-30A): {res_shale['excess_adsorption'][:, 2]}")
        res_egr = compute_ch4_co2_gas_recovery_crossover(T=330.0)
        print(f"[dens-city] Enhanced Gas Recovery Efficiency: {res_egr['recovery_efficiency']}")
    elif args.command == "clay":
        print("[dens-city] Executing Montmorillonite Clay Mineral Swelling Pipeline...")
        from dens_city.pipelines.clay_pore.mineral import compute_clay_swelling_pressure

        res_clay = compute_clay_swelling_pressure([9.5, 12.5, 15.5, 18.5, 25.0], T=298.15)
        print(f"[dens-city] Clay Swelling Pressures (MPa): {res_clay['Pi_swell_MPa']}")
    elif args.command == "liquid-crystals":
        print("[dens-city] Executing Nematic Liquid Crystals Pipeline...")
        from dens_city.pipelines.liquid_crystals.nematic import (
            compute_isotropic_nematic_binodal,
            compute_nematic_director_profile,
        )

        res_lc = compute_nematic_director_profile(None, H=30.0, anchoring_type="homeotropic")
        print(f"[dens-city] LC Director S_order: max={res_lc['S_order'].max():.3f}, min={res_lc['S_order'].min():.3f}")
        res_in = compute_isotropic_nematic_binodal()
        print(
            f"[dens-city] Isotropic-Nematic Coexistence rho_iso={res_in['rho_isotropic'][1]:.4f}, rho_nem={res_in['rho_nematic'][1]:.4f}"
        )
    elif args.command == "argon":
        print("[dens-city] Executing Argon Pure Lennard-Jones FMT Coexistence Pipeline...")
        from dens_city.pipelines.argon.coexistence import compute_argon_binodal

        bin_res = compute_argon_binodal([85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0])
        print(f"[dens-city] Argon Predicted T_c: {bin_res['T_c_K']:.1f} K (NIST: 150.86 K)")
        print(f"[dens-city] Argon Predicted rho_c: {bin_res['rho_c']:.4f} A^-3 (NIST: 0.00808 A^-3)")
        print(f"[dens-city] Argon Liquid Density (85K): {bin_res['rho_l'][0]:.4f} A^-3 (NIST 84K: 0.0214 A^-3)")
    elif args.command == "interfaces":
        print("[dens-city] Executing Hydrophobic/Hydrophilic Planar Wetting Pipeline...")
        from dens_city.pipelines.interfaces.wetting import (
            compute_capillary_drying_gap,
            compute_lum_chandler_weeks_crossover,
            compute_wetting_contact_angle,
        )

        res_wet = compute_wetting_contact_angle(gamma_sv=20.0, gamma_sl=60.0)
        print(f"[dens-city] Hydrophobic Contact Angle: {res_wet['theta_deg']:.1f} deg ({res_wet['wetting_regime']})")
        res_dry = compute_capillary_drying_gap(theta_deg=110.0)
        print(f"[dens-city] Critical Capillary Drying Gap: {res_dry['H_dry_nm']:.2f} nm")
        res_lcw = compute_lum_chandler_weeks_crossover()
        print(f"[dens-city] LCW Crossover Scale R_c: {res_lcw['R_c_nm']:.1f} nm")
    elif args.command in ["benchmark", "e2e"]:
        import subprocess
        import sys
        from pathlib import Path
        script_path = Path(__file__).parent.parent.parent / "scripts" / "run_e2e_benchmarks.py"
        cmd = [sys.executable, str(script_path), "--timesteps", str(args.timesteps), "--materials"] + args.materials
        subprocess.run(cmd, check=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
