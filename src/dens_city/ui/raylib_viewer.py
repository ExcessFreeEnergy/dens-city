import argparse

import numpy as np

try:
    import pyray as rl
except ImportError:
    rl = None

from dens_city.envs.env import DensCityFluidEnv
from dens_city.ui.widgets import draw_button, draw_panel, draw_slider


def run_viewer(functional_path: str = "dens_functional.pt"):
    if rl is None:
        print("[dens-city-ui] Raylib not installed. Run 'pip install raylib' to use GUI.")
        return

    screen_w = 1280
    screen_h = 768
    rl.init_window(screen_w, screen_h, "dens-city: Ab Initio Neural cDFT & PufferLib RL Fluid Control")
    rl.set_target_fps(60)

    env = DensCityFluidEnv(num_envs=1)
    env.reset(0)

    target_filling = 0.5
    phi_0 = 0.0
    mode_m = 1.0
    v_bias = 0.0
    auto_rl = False
    paused = False

    while not rl.window_should_close():
        if not paused:
            if auto_rl:
                # Closed loop proportional control
                err = target_filling - env.current_filling
                phi_0 = float(np.clip(phi_0 + err * 0.1, -5.0, 5.0))
                env.phi_0 = phi_0
            else:
                env.phi_0 = phi_0
                env.mode_m = mode_m
                env.v_bias = v_bias

            env.target_filling = target_filling
            action = np.array([phi_0 / 5.0, (mode_m - 3.0) / 2.0, v_bias / 2.0], dtype=np.float32)
            env.step(action, env_idx=0)

        rl.begin_drawing()
        rl.clear_background((12, 16, 22, 255))

        # 1. Main Density Profile Graph Panel (Left)
        draw_panel(20, 20, 800, 480, "1D Equilibrium Fluid Density Profile rho(z) & Restructuring Potential phi_R(z)")
        graph_x, graph_y, graph_w, graph_h = 40, 60, 760, 420

        rho = env.rho
        phi_r = env.phi_R
        max_rho = max(0.06, float(np.max(rho)))

        # Draw grid lines
        for step in range(5):
            gy = graph_y + int(step * (graph_h / 4.0))
            rl.draw_line(graph_x, gy, graph_x + graph_w, gy, (28, 35, 45, 255))
            val = max_rho * (1.0 - step / 4.0)
            rl.draw_text(f"{val:.3f}", graph_x - 36, gy - 6, 11, (100, 115, 135, 255))

        # Draw rho(z) curve
        for i in range(len(rho) - 1):
            x1 = graph_x + int(i * (graph_w / len(rho)))
            y1 = graph_y + graph_h - int((rho[i] / max_rho) * graph_h)
            x2 = graph_x + int((i + 1) * (graph_w / len(rho)))
            y2 = graph_y + graph_h - int((rho[i + 1] / max_rho) * graph_h)
            rl.draw_line(x1, y1, x2, y2, (0, 200, 255, 255))
            rl.draw_line(x1, y1 + 1, x2, y2 + 1, (0, 150, 220, 255))

        # Draw phi_R(z) curve
        max_phi = max(1e-21, float(np.max(np.abs(phi_r))))
        for i in range(len(phi_r) - 1):
            x1 = graph_x + int(i * (graph_w / len(phi_r)))
            y1 = graph_y + int(graph_h / 2.0) - int((phi_r[i] / max_phi) * (graph_h * 0.4))
            x2 = graph_x + int((i + 1) * (graph_w / len(phi_r)))
            y2 = graph_y + int(graph_h / 2.0) - int((phi_r[i + 1] / max_phi) * (graph_h * 0.4))
            rl.draw_line(x1, y1, x2, y2, (255, 180, 50, 180))

        # Legend
        rl.draw_rectangle(graph_x + 20, graph_y + 20, 14, 4, (0, 200, 255, 255))
        rl.draw_text("Fluid Density rho(z)", graph_x + 40, graph_y + 14, 13, (220, 230, 240, 255))
        rl.draw_rectangle(graph_x + 180, graph_y + 20, 14, 4, (255, 180, 50, 255))
        rl.draw_text("Restructuring Potential phi_R(z)", graph_x + 200, graph_y + 14, 13, (220, 230, 240, 255))

        # 2. Control Panel (Right)
        draw_panel(840, 20, 420, 480, "Closed-Loop Control Parameters")
        target_filling = draw_slider(
            860, 60, 380, 40, "Target Pore Filling Fraction (theta*)", target_filling, 0.05, 0.95, "{:.2f}"
        )
        phi_0 = draw_slider(860, 120, 380, 40, "Harmonic Voltage Amplitude phi_0 (V)", phi_0, -5.0, 5.0, "{:.2f} V")
        mode_m = draw_slider(860, 180, 380, 40, "Harmonic Spatial Mode (m)", mode_m, 1.0, 5.0, "{:.0f}")
        v_bias = draw_slider(860, 240, 380, 40, "DC Gate Bias Offset V_bias (V)", v_bias, -2.0, 2.0, "{:.2f} V")

        if draw_button(860, 310, 180, 36, "Toggle Neural RL", active=auto_rl):
            auto_rl = not auto_rl
        if draw_button(1060, 310, 180, 36, "Pause / Resume", active=paused):
            paused = not paused

        # Live Metrics
        rl.draw_text(
            f"Current Pore Filling:   {env.current_filling:.3f} (Target: {target_filling:.2f})",
            860,
            370,
            14,
            (180, 220, 255, 255),
        )
        rl.draw_text(
            f"Euler-Lagrange Res:     {float(env._envs_ptr[0].el_residual):.6f}", 860, 400, 14, (180, 220, 255, 255)
        )
        rl.draw_text(
            f"Slit Width (L_z):        {float(env._envs_ptr[0].L_z):.1f} A", 860, 430, 14, (180, 220, 255, 255)
        )
        rl.draw_text(f"Temperature (T):         {float(env._envs_ptr[0].T):.1f} K", 860, 460, 14, (180, 220, 255, 255))

        # 3. 2D Slit-Pore Fluid Meniscus Visualization (Bottom)
        draw_panel(20, 520, 1240, 220, "2D Physical Nanofluidic Slit Pore Simulation (Dielectrocapillary Meniscus)")
        slit_x, slit_y, slit_w, slit_h = 40, 560, 1200, 160

        # Draw confining graphene plates
        rl.draw_rectangle(slit_x, slit_y, slit_w, 8, (60, 75, 95, 255))
        rl.draw_rectangle(slit_x, slit_y + slit_h - 8, slit_w, 8, (60, 75, 95, 255))
        rl.draw_text("Top Graphene Plate", slit_x + 10, slit_y - 16, 12, (140, 155, 175, 255))
        rl.draw_text("Bottom Graphene Plate", slit_x + 10, slit_y + slit_h + 4, 12, (140, 155, 175, 255))

        # Render fluid density field in 2D
        num_cols = len(rho)
        col_w = slit_w / num_cols
        for i in range(num_cols):
            alpha_f = min(255, max(10, int((rho[i] / 0.04) * 220)))
            fill_color = (0, 160, 240, alpha_f)
            rl.draw_rectangle(int(slit_x + i * col_w), slit_y + 8, max(1, int(col_w) + 1), slit_h - 16, fill_color)

        rl.end_drawing()

    rl.close_window()


def main():
    parser = argparse.ArgumentParser(description="dens-city Real-Time Raylib Scientific Dashboard")
    parser.add_argument(
        "--functional", type=str, default="dens_functional.pt", help="Path to trained neural functional"
    )
    args = parser.parse_args()
    run_viewer(functional_path=args.functional)


if __name__ == "__main__":
    main()
