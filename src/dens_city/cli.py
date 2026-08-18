import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="dens-city",
        description="dens-city: Unified Ab Initio Neural cDFT & PufferLib RL Platform for Programmable Fluids",
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

    # Water pipeline
    water_p = subparsers.add_parser("water", help="Execute water nanoconfinement and binodal pipeline")
    water_p.add_argument("--mode", choices=["confinement", "binodal", "all"], default="all")

    subparsers.add_parser("co2", help="Execute CO2 supercritical crossover pipeline")
    subparsers.add_parser("electrolytes", help="Execute RPM electrolyte double layer pipeline")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
