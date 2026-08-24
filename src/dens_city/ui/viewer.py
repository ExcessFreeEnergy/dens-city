"""
High-performance 3D molecular viewer implemented in Raylib (pyray).
Renders atom sites with standard CPK color palette and covalent bond cylinders
with smooth 60+ FPS orbital mouse rotation and zoom.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pyray as pr

from dens_city.utils.materials import AtomSite, Material, MaterialLoader

# Standard CPK elemental color table: (R, G, B, A)
CPK_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "H": (245, 245, 245, 255),
    "C": (50, 50, 50, 255),
    "N": (48, 80, 248, 255),
    "O": (240, 30, 30, 255),
    "F": (144, 224, 80, 255),
    "CL": (31, 220, 31, 255),
    "BR": (166, 41, 41, 255),
    "I": (148, 0, 148, 255),
    "S": (255, 200, 50, 255),
    "P": (255, 128, 0, 255),
    "AR": (128, 209, 227, 255),
    "NA": (171, 92, 242, 255),
    "CA": (60, 180, 60, 255),
    "FE": (224, 102, 0, 255),
}

# Standard covalent visual radius in Angstroms
ELEMENT_RADII: Dict[str, float] = {
    "H": 0.35,
    "C": 0.55,
    "N": 0.52,
    "O": 0.50,
    "F": 0.48,
    "CL": 0.65,
    "BR": 0.72,
    "I": 0.80,
    "S": 0.68,
    "P": 0.68,
    "AR": 0.70,
    "NA": 0.75,
    "CA": 0.80,
}


def get_atom_element(atom_type: str) -> str:
    """Extracts elemental symbol from Tripos atom type (e.g., 'C.3' -> 'C', 'Cl' -> 'CL')."""
    raw = atom_type.split(".")[0].strip()
    return raw.upper()


def get_atom_color(atom_type: str) -> pr.Color:
    """Returns the CPK Color for a given Tripos atom type."""
    elem = get_atom_element(atom_type)
    r, g, b, a = CPK_COLORS.get(elem, (180, 180, 180, 255))
    return pr.Color(r, g, b, a)


def get_atom_radius(atom_type: str, sigma: float = 3.4) -> float:
    """Returns visual ball radius in Angstroms."""
    elem = get_atom_element(atom_type)
    if elem in ELEMENT_RADII:
        return ELEMENT_RADII[elem]
    return max(0.3, min(1.0, sigma * 0.25))


class MoleculeViewer:
    """
    Raylib 3D molecular renderer with orbital camera controls.
    """

    def __init__(
        self,
        materials: List[Material],
        width: int = 1280,
        height: int = 720,
        title: str = "dens-city 3D Molecular Viewer",
    ):
        if not materials:
            raise ValueError("Must provide at least one Material to MoleculeViewer.")
        self.materials = materials
        self.current_idx = 0
        self.width = width
        self.height = height
        self.title = title

        # Orbital Camera parameters
        self.azimuth = 0.6
        self.elevation = 0.4
        self.distance = 15.0
        self.target = pr.Vector3(0.0, 0.0, 0.0)
        self.default_distance = 15.0
        self.default_target = pr.Vector3(0.0, 0.0, 0.0)

        self._update_molecule_bounds()

    @property
    def current_material(self) -> Material:
        return self.materials[self.current_idx]

    def _update_molecule_bounds(self) -> None:
        """Calculates centroid and auto-frames camera distance for current material."""
        mat = self.current_material
        sites = mat.sites

        if not sites:
            self.target = pr.Vector3(0.0, 0.0, 0.0)
            self.distance = 10.0
            self.default_target = self.target
            self.default_distance = self.distance
            return

        cx = sum(s.x for s in sites) / len(sites)
        cy = sum(s.y for s in sites) / len(sites)
        cz = sum(s.z for s in sites) / len(sites)
        self.target = pr.Vector3(cx, cy, cz)
        self.default_target = pr.Vector3(cx, cy, cz)

        max_d = 0.0
        for s in sites:
            dx = s.x - cx
            dy = s.y - cy
            dz = s.z - cz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > max_d:
                max_d = d

        span = max(max_d, mat.effective_sigma * 0.5)
        self.distance = max(6.0, span * 2.8)
        self.default_distance = self.distance

    def next_material(self) -> None:
        """Switches to the next material in the list."""
        if len(self.materials) > 1:
            self.current_idx = (self.current_idx + 1) % len(self.materials)
            self._update_molecule_bounds()

    def prev_material(self) -> None:
        """Switches to the previous material in the list."""
        if len(self.materials) > 1:
            self.current_idx = (self.current_idx - 1 + len(self.materials)) % len(self.materials)
            self._update_molecule_bounds()

    def reset_view(self) -> None:
        """Resets camera orientation and zoom to defaults."""
        self.azimuth = 0.6
        self.elevation = 0.4
        self.distance = self.default_distance
        self.target = self.default_target

    def get_camera_position(self) -> pr.Vector3:
        """Computes 3D camera position from spherical coordinates."""
        cos_el = math.cos(self.elevation)
        sin_el = math.sin(self.elevation)
        cos_az = math.cos(self.azimuth)
        sin_az = math.sin(self.azimuth)

        x = self.target.x + self.distance * cos_el * sin_az
        y = self.target.y + self.distance * sin_el
        z = self.target.z + self.distance * cos_el * cos_az
        return pr.Vector3(x, y, z)

    def handle_input(self) -> None:
        """Handles mouse drag, scroll zoom, and keyboard hotkeys."""
        # 1. Left Mouse Button Hold & Drag -> Orbit
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
            delta = pr.get_mouse_delta()
            sensitivity = 0.006
            self.azimuth += delta.x * sensitivity
            self.elevation -= delta.y * sensitivity
            # Clamp elevation to prevent camera flipping at poles
            max_elevation = math.pi / 2.0 - 0.05
            self.elevation = max(-max_elevation, min(max_elevation, self.elevation))

        # 2. Right Mouse Button Hold & Drag -> Pan
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_RIGHT) or pr.is_mouse_button_down(pr.MOUSE_BUTTON_MIDDLE):
            delta = pr.get_mouse_delta()
            pan_speed = self.distance * 0.0015
            # Right vector perpendicular to camera forward in XZ
            sin_az = math.sin(self.azimuth)
            cos_az = math.cos(self.azimuth)
            self.target.x -= (delta.x * cos_az) * pan_speed
            self.target.z += (delta.x * sin_az) * pan_speed
            self.target.y += delta.y * pan_speed

        # 3. Mouse Wheel -> Zoom
        wheel = pr.get_mouse_wheel_move()
        if wheel != 0:
            zoom_factor = 1.0 - wheel * 0.1
            self.distance = max(1.5, min(250.0, self.distance * zoom_factor))

        # 4. Keyboard Navigation
        if pr.is_key_pressed(pr.KEY_RIGHT) or pr.is_key_pressed(pr.KEY_RIGHT_BRACKET):
            self.next_material()
        if pr.is_key_pressed(pr.KEY_LEFT) or pr.is_key_pressed(pr.KEY_LEFT_BRACKET):
            self.prev_material()
        if pr.is_key_pressed(pr.KEY_R) or pr.is_key_pressed(pr.KEY_SPACE):
            self.reset_view()

    def draw_molecule_3d(self) -> None:
        """Renders 3D atom spheres and bond cylinders for the active material."""
        mat = self.current_material
        sites: List[AtomSite] = mat.sites
        bonds = mat.bonds

        # 1. Draw Bonds (Cylinders)
        bond_color = pr.Color(160, 160, 160, 255)
        bond_radius = 0.10
        for b in bonds:
            i, j = b[0], b[1]
            if 0 <= i < len(sites) and 0 <= j < len(sites):
                s1 = sites[i]
                s2 = sites[j]
                p1 = pr.Vector3(s1.x, s1.y, s1.z)
                p2 = pr.Vector3(s2.x, s2.y, s2.z)
                pr.draw_cylinder_ex(p1, p2, bond_radius, bond_radius, 8, bond_color)

        # 2. Draw Atom Sites (Spheres)
        for s in sites:
            pos = pr.Vector3(s.x, s.y, s.z)
            radius = get_atom_radius(s.atom_type, s.sigma)
            color = get_atom_color(s.atom_type)
            pr.draw_sphere(pos, radius, color)

    def draw_hud(self) -> None:
        """Renders minimal HUD text information."""
        mat = self.current_material
        bg_dark = pr.Color(20, 24, 30, 220)
        pr.draw_rectangle(15, 15, 340, 130, bg_dark)
        pr.draw_rectangle_lines(15, 15, 340, 130, pr.Color(60, 70, 85, 255))

        title_text = f"Material: {mat.name} ({self.current_idx + 1}/{len(self.materials)})"
        pr.draw_text(title_text, 25, 25, 18, pr.RAYWHITE)

        mode_text = f"Mode: {mat.dimension_mode} | Sites: {len(mat.sites)} | Bonds: {len(mat.bonds)}"
        pr.draw_text(mode_text, 25, 50, 14, pr.LIGHTGRAY)

        sigma_text = f"Effective σ: {mat.effective_sigma:.2f} Å | ε/k_B: {mat.effective_epsilon_k:.1f} K"
        pr.draw_text(sigma_text, 25, 70, 14, pr.LIGHTGRAY)

        controls_text = "[Left Drag] Rotate | [Scroll] Zoom"
        pr.draw_text(controls_text, 25, 95, 12, pr.GRAY)
        nav_text = "[←/→] Switch Material | [R] Reset"
        pr.draw_text(nav_text, 25, 115, 12, pr.GRAY)

        pr.draw_fps(self.width - 90, 15)

    def run(self) -> None:
        """Main window loop."""
        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_RESIZABLE)
        pr.init_window(self.width, self.height, self.title)
        pr.set_target_fps(60)

        camera = pr.Camera3D(
            self.get_camera_position(),
            self.target,
            pr.Vector3(0.0, 1.0, 0.0),
            45.0,
            pr.CAMERA_PERSPECTIVE,
        )

        bg_color = pr.Color(16, 18, 22, 255)

        while not pr.window_should_close():
            self.handle_input()

            # Update camera vectors
            camera.position = self.get_camera_position()
            camera.target = self.target

            pr.begin_drawing()
            pr.clear_background(bg_color)

            pr.begin_mode_3d(camera)
            # Subtle reference floor grid
            pr.draw_grid(20, 1.0)
            self.draw_molecule_3d()
            pr.end_mode_3d()

            self.draw_hud()

            pr.end_drawing()

        pr.close_window()


def run_interactive_viewer(
    material_names: Optional[List[str]] = None,
    width: int = 1280,
    height: int = 720,
) -> None:
    """
    Helper function to load requested materials and launch the 3D Raylib viewer.
    """
    all_avail = MaterialLoader.list_available_materials()
    if not material_names or material_names == ["all"]:
        to_load = all_avail
    else:
        to_load = material_names

    materials: List[Material] = []
    for name in to_load:
        try:
            mat = MaterialLoader.load_material(name)
            materials.append(mat)
        except Exception as e:
            print(f"Warning: Failed to load material '{name}': {e}")

    if not materials:
        print("No valid materials loaded. Defaulting to 'argon'.")
        materials = [MaterialLoader.load_material("argon")]

    viewer = MoleculeViewer(materials=materials, width=width, height=height)
    viewer.run()
