"""
High-performance 3D molecular viewer implemented in Raylib (pyray).
Integrates variational cDFT and Boltzmann Generator workflows with non-blocking execution,
real-time 2D density profile overlay, Van der Waals (VDW) wireframe surface mesh,
bottom control deck with full Reset capability, and bottom-right telemetry output panel.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple, Union

import pyray as pr

from dens_city.ui.worker import CDFTBGWorker
from dens_city.utils.materials import AtomSite, Material, MaterialLoader

# Element color palette: (R, G, B, A)
# Nitrogen uses vibrant emerald green to match standard scientific / crystallographic references
ELEMENT_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "H": (240, 243, 246, 255),  # Crisp Silver / White
    "C": (55, 62, 72, 255),  # Charcoal / Slate
    "N": (34, 180, 115, 255),  # Emerald Green (matching reference)
    "O": (235, 45, 45, 255),  # Ruby Red
    "F": (120, 225, 140, 255),  # Light Aquamarine
    "CL": (45, 210, 65, 255),  # Leaf Green
    "BR": (165, 42, 42, 255),  # Crimson Brown
    "I": (148, 0, 211, 255),  # Purple
    "S": (245, 205, 35, 255),  # Golden Yellow
    "P": (250, 130, 20, 255),  # Flame Orange
    "AR": (100, 210, 240, 255),  # Cyan Ice
    "NA": (170, 90, 245, 255),  # Neon Violet
    "CA": (50, 160, 50, 255),  # Deep Green
    "FE": (224, 102, 0, 255),  # Metallic Rust
}

# Ball-and-stick visual radius in Angstroms (scaled to reveal covalent bonds clearly)
ELEMENT_RADII: Dict[str, float] = {
    "H": 0.18,
    "C": 0.30,
    "N": 0.28,
    "O": 0.26,
    "F": 0.25,
    "CL": 0.34,
    "BR": 0.38,
    "I": 0.42,
    "S": 0.35,
    "P": 0.35,
    "AR": 0.52,  # Substantial for monoatomic gases
    "NA": 0.42,
    "CA": 0.45,
}

# Crystallographic Van der Waals (VDW) radii in Angstroms
VDW_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "S": 1.80,
    "P": 1.80,
    "AR": 1.88,
    "NA": 2.27,
    "CA": 2.00,
    "FE": 2.05,
}

# Embedded GLSL Blinn-Phong lighting vertex shader
VS_PHONG = """#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec4 vertexColor;
uniform mat4 mvp;
uniform mat4 matModel;
out vec3 fragPosition;
out vec3 fragNormal;
out vec4 fragColor;
void main() {
    fragPosition = vec3(matModel * vec4(vertexPosition, 1.0));
    fragNormal = normalize(vec3(matModel * vec4(vertexNormal, 0.0)));
    fragColor = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

# Embedded GLSL Blinn-Phong lighting fragment shader with dual-directional light & specular shine
FS_PHONG = """#version 330
in vec3 fragPosition;
in vec3 fragNormal;
in vec4 fragColor;
uniform vec3 viewPos;
out vec4 finalColor;
void main() {
    vec3 normal = normalize(fragNormal);

    // Primary directional key light (upper right front)
    vec3 lightDir1 = normalize(vec3(0.6, 1.0, 0.8));
    float diff1 = max(dot(normal, lightDir1), 0.0);

    // Secondary soft fill light (lower left back)
    vec3 lightDir2 = normalize(vec3(-0.5, -0.3, -0.6));
    float diff2 = max(dot(normal, lightDir2), 0.0) * 0.3;

    // Ambient light
    vec3 ambient = vec3(0.28);

    // Blinn-Phong Specular Highlight
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 halfDir = normalize(lightDir1 + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0) * 0.45;

    vec3 lightIntensity = ambient + vec3(diff1 * 0.72) + vec3(diff2) + vec3(spec);
    vec3 rgb = fragColor.rgb * lightIntensity;
    finalColor = vec4(rgb, fragColor.a);
}
"""


def get_atom_element(atom_type: str, site_name: str = "") -> str:
    """
    Robustly extracts the chemical element symbol from Tripos/GAFF atom types and site names.
    Handles 'n1', 'N1', 'c3', 'C.3', 'ca', 'ha', 'hc', 'os', 's6', 'p5', 'cl', 'br', 'f', 'na', 'ca', 'ar', etc.
    """
    t = atom_type.strip()
    s = site_name.strip()

    # 1. Check 2-letter element symbols from site_name (e.g. Ca1 -> CA, Cl1 -> CL, Na1 -> NA, Ar1 -> AR)
    for sym in ["CL", "BR", "NA", "CA", "FE", "AR", "SI", "AL", "MG", "LI", "ZN", "HE", "NE", "KR", "XE"]:
        if s.upper().startswith(sym) and (len(s) == len(sym) or s[len(sym) :].isdigit() or s[len(sym)] in "_-."):
            return sym
        if t == sym.capitalize() or t == sym:
            return sym

    # 2. GAFF 2-character prefixes for standard organic elements:
    t_lower = t.lower()
    if t_lower in (
        "ca",
        "cp",
        "cq",
        "cc",
        "cd",
        "ce",
        "cf",
        "cg",
        "ch",
        "cx",
        "cy",
        "cz",
        "c.3",
        "c.2",
        "c.1",
        "c.ar",
    ):
        return "C"
    if t_lower in ("na", "nb", "nc", "nd", "ne", "nf", "nh", "no", "n.am", "n.pl3", "n.4", "n.ar"):
        return "N"
    if t_lower in ("ha", "hc", "hn", "ho", "hp", "hs", "hw", "hx", "h.spc", "h.tip3p"):
        return "H"
    if t_lower in ("oa", "ob", "oc", "od", "oe", "oh", "os", "ow", "o.3", "o.2", "o.co2"):
        return "O"
    if t_lower in ("sa", "sb", "sc", "sd", "se", "sh", "ss", "sp", "sq", "sx", "sy", "s.3", "s.2", "s.o", "s.o2"):
        return "S"
    if t_lower in ("pa", "pb", "pc", "pd", "pe", "pf", "p.3"):
        return "P"

    # 3. GAFF single-letter + number / punctuation (c1, c2, c3, n1, n2, h1, o, s6, p5, f, cl, br, etc.)
    if t_lower.startswith("cl"):
        return "CL"
    if t_lower.startswith("br"):
        return "BR"
    if t_lower.startswith("ar"):
        return "AR"

    first_char = t_lower[0] if t_lower else (s[0].lower() if s else "c")
    if first_char in ("c", "n", "o", "h", "f", "p", "s", "i", "b", "k", "v", "w", "u"):
        return first_char.upper()

    letters = re.sub(r"[^a-zA-Z]", "", t if t else s)
    return letters[:2].upper() if letters else "C"


def get_atom_color(atom_type: str, site_name: str = "") -> pr.Color:
    """Returns the CPK / publication Color for a given atom type or site name."""
    elem = get_atom_element(atom_type, site_name)
    r, g, b, a = ELEMENT_COLORS.get(elem, (170, 175, 185, 255))
    return pr.Color(r, g, b, a)


def get_atom_radius(atom_type: str, site_name: str = "", sigma: float = 3.4) -> float:
    """Returns visual ball-and-stick radius in Angstroms."""
    elem = get_atom_element(atom_type, site_name)
    if elem in ELEMENT_RADII:
        return ELEMENT_RADII[elem]
    return max(0.20, min(0.60, sigma * 0.12))


def get_vdw_radius(atom_type: str, site_name: str = "", sigma: float = 3.4) -> float:
    """Returns crystallographic Van der Waals radius in Angstroms."""
    elem = get_atom_element(atom_type, site_name)
    if elem in VDW_RADII:
        return VDW_RADII[elem]
    return max(1.20, sigma * 0.5)


class MoleculeViewer:
    """
    Raylib 3D molecular renderer with non-blocking cDFT and Boltzmann Generator workflows,
    real-time 2D density profile overlay, Van der Waals wireframe surface mesh,
    bottom control deck with reset capability, and bottom-right telemetry output panel.
    """

    def __init__(
        self,
        material: Union[Material, List[Material]],
        width: int = 1280,
        height: int = 720,
        title: str = "dens-city 3D Molecular Engine",
    ):
        if isinstance(material, list):
            if not material:
                raise ValueError("Must provide at least one Material to MoleculeViewer.")
            self.material = material[0]
        else:
            self.material = material

        self.width = width
        self.height = height
        self.title = title

        # Orbital Camera parameters
        self.azimuth = 0.75
        self.elevation = 0.35
        self.distance = 15.0
        self.target = pr.Vector3(0.0, 0.0, 0.0)
        self.default_distance = 15.0
        self.default_target = pr.Vector3(0.0, 0.0, 0.0)

        # Van der Waals Surface Mesh Display (Default OFF, toggle via [V] or [Mesh] button)
        self.show_vdw_surface: bool = False

        # Worker thread & ZeroMQ bridge
        self.worker = CDFTBGWorker(material=self.material, cdft_steps=100, bg_mcmc_steps=40)
        self.telemetry = self.worker.telemetry

        # 3D Coordinates (updated live during MCMC relaxation)
        self.rendered_coords: List[Tuple[float, float, float]] = [(s.x, s.y, s.z) for s in self.material.sites]

        self._update_molecule_bounds()

    def _update_molecule_bounds(self) -> None:
        """
        Calculates centroid, bounding radius, and auto-frames camera distance
        from perspective frustum limits (up to 128+ sites).
        """
        mat = self.material
        sites = mat.sites

        if not sites:
            self.target = pr.Vector3(0.0, 0.0, 0.0)
            self.distance = 8.0
            self.default_target = self.target
            self.default_distance = self.distance
            return

        cx = sum(c[0] for c in self.rendered_coords) / len(self.rendered_coords)
        cy = sum(c[1] for c in self.rendered_coords) / len(self.rendered_coords)
        cz = sum(c[2] for c in self.rendered_coords) / len(self.rendered_coords)
        self.target = pr.Vector3(cx, cy, cz)
        self.default_target = pr.Vector3(cx, cy, cz)

        max_d = 0.0
        for c in self.rendered_coords:
            dx = c[0] - cx
            dy = c[1] - cy
            dz = c[2] - cz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > max_d:
                max_d = d

        bounding_radius = max(max_d, mat.effective_sigma * 0.5, 1.0)

        # Perspective FOV frustum-based auto-framing:
        fov_half_rad = math.radians(22.5)
        fit_dist = (bounding_radius / math.sin(fov_half_rad)) * 1.15
        self.distance = max(6.0, fit_dist)
        self.default_distance = self.distance

    def reset_view(self) -> None:
        """Resets camera orientation and zoom to defaults."""
        self.azimuth = 0.75
        self.elevation = 0.35
        self.distance = self.default_distance
        self.target = self.default_target

    def reset_all(self) -> None:
        """
        Completely resets cDFT and Boltzmann calculations, clears graph and telemetry,
        resets coordinates to initial positions, and restores default camera view.
        """
        self.worker.reset()
        self.telemetry = self.worker.telemetry
        self.rendered_coords = [(s.x, s.y, s.z) for s in self.material.sites]
        self._update_molecule_bounds()
        self.reset_view()

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
        # 1. Left Mouse Button Hold & Drag -> Orbit (ignore if clicking in bottom UI deck)
        mouse_pos = pr.get_mouse_position()
        in_bottom_deck = mouse_pos.y >= (self.height - 80) and mouse_pos.x >= 20 and mouse_pos.x <= (self.width - 20)

        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT) and not in_bottom_deck:
            delta = pr.get_mouse_delta()
            sensitivity = 0.005
            self.azimuth += delta.x * sensitivity
            self.elevation -= delta.y * sensitivity
            max_elevation = math.pi / 2.0 - 0.05
            self.elevation = max(-max_elevation, min(max_elevation, self.elevation))

        # 2. Right / Middle Mouse Button -> Pan
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_RIGHT) or pr.is_mouse_button_down(pr.MOUSE_BUTTON_MIDDLE):
            delta = pr.get_mouse_delta()
            pan_speed = self.distance * 0.0015
            sin_az = math.sin(self.azimuth)
            cos_az = math.cos(self.azimuth)
            self.target.x -= (delta.x * cos_az) * pan_speed
            self.target.z += (delta.x * sin_az) * pan_speed
            self.target.y += delta.y * pan_speed

        # 3. Mouse Wheel -> Zoom
        wheel = pr.get_mouse_wheel_move()
        if wheel != 0:
            zoom_factor = 1.0 - wheel * 0.1
            self.distance = max(0.5, min(1000.0, self.distance * zoom_factor))

        # 4. Keyboard Hotkeys
        if pr.is_key_pressed(pr.KEY_R):
            self.reset_all()
        if pr.is_key_pressed(pr.KEY_V):
            self.show_vdw_surface = not self.show_vdw_surface
        if pr.is_key_pressed(pr.KEY_SPACE):
            # Play / Pause toggling
            if self.worker.is_running:
                self.worker.cancel()
            else:
                if self.telemetry.state in ("WAITING_CDFT", "RUNNING_CDFT"):
                    self.worker.solve_cdft()
                elif self.telemetry.state in ("CDFT_CONVERGED", "RUNNING_BG"):
                    self.worker.solve_bg()

    def _draw_cylinder_segment(
        self,
        p1: pr.Vector3,
        p2: pr.Vector3,
        radius: float,
        color1: pr.Color,
        color2: pr.Color,
    ) -> None:
        """Draws a cylinder split at the midpoint with dual elemental colors."""
        mid = pr.Vector3(
            0.5 * (p1.x + p2.x),
            0.5 * (p1.y + p2.y),
            0.5 * (p1.z + p2.z),
        )
        pr.draw_cylinder_ex(p1, mid, radius, radius, 10, color1)
        pr.draw_cylinder_ex(mid, p2, radius, radius, 10, color2)

    def draw_bond(
        self,
        s1: AtomSite,
        s2: AtomSite,
        p1_tuple: Tuple[float, float, float],
        p2_tuple: Tuple[float, float, float],
        bond_type: str,
        cam_pos: pr.Vector3,
    ) -> None:
        """
        Renders camera-aligned multi-bond cylinders (triple, double, single, aromatic).
        """
        p1 = pr.Vector3(p1_tuple[0], p1_tuple[1], p1_tuple[2])
        p2 = pr.Vector3(p2_tuple[0], p2_tuple[1], p2_tuple[2])

        c1 = get_atom_color(s1.atom_type, s1.site_name)
        c2 = get_atom_color(s2.atom_type, s2.site_name)

        vx = p2.x - p1.x
        vy = p2.y - p1.y
        vz = p2.z - p1.z
        length = math.sqrt(vx * vx + vy * vy + vz * vz)
        if length < 1e-4:
            return

        ux, uy, uz = vx / length, vy / length, vz / length

        mid_x = 0.5 * (p1.x + p2.x)
        mid_y = 0.5 * (p1.y + p2.y)
        mid_z = 0.5 * (p1.z + p2.z)
        cvx = cam_pos.x - mid_x
        cvy = cam_pos.y - mid_y
        cvz = cam_pos.z - mid_z
        cv_len = math.sqrt(cvx * cvx + cvy * cvy + cvz * cvz)

        if cv_len > 1e-4:
            cvx, cvy, cvz = cvx / cv_len, cvy / cv_len, cvz / cv_len
        else:
            cvx, cvy, cvz = 0.0, 1.0, 0.0

        nx = uy * cvz - uz * cvy
        ny = uz * cvx - ux * cvz
        nz = ux * cvy - uy * cvx
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)

        if n_len > 1e-3:
            nx, ny, nz = nx / n_len, ny / n_len, nz / n_len
        else:
            ax, ay, az = (0.0, 1.0, 0.0) if abs(uy) < 0.9 else (1.0, 0.0, 0.0)
            nx = uy * az - uz * ay
            ny = uz * ax - ux * az
            nz = ux * ay - uy * ax
            n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / n_len, ny / n_len, nz / n_len

        bt = str(bond_type).strip().lower()

        if bt in ("3", "tr", "triple"):
            offset = 0.095
            rad = 0.035
            self._draw_cylinder_segment(p1, p2, rad, c1, c2)
            p1_plus = pr.Vector3(p1.x + nx * offset, p1.y + ny * offset, p1.z + nz * offset)
            p2_plus = pr.Vector3(p2.x + nx * offset, p2.y + ny * offset, p2.z + nz * offset)
            self._draw_cylinder_segment(p1_plus, p2_plus, rad, c1, c2)
            p1_minus = pr.Vector3(p1.x - nx * offset, p1.y - ny * offset, p1.z - nz * offset)
            p2_minus = pr.Vector3(p2.x - nx * offset, p2.y - ny * offset, p2.z - nz * offset)
            self._draw_cylinder_segment(p1_minus, p2_minus, rad, c1, c2)

        elif bt in ("2", "db", "double", "ar", "aromatic"):
            offset = 0.065
            rad = 0.040
            p1_plus = pr.Vector3(p1.x + nx * offset, p1.y + ny * offset, p1.z + nz * offset)
            p2_plus = pr.Vector3(p2.x + nx * offset, p2.y + ny * offset, p2.z + nz * offset)
            self._draw_cylinder_segment(p1_plus, p2_plus, rad, c1, c2)
            p1_minus = pr.Vector3(p1.x - nx * offset, p1.y - ny * offset, p1.z - nz * offset)
            p2_minus = pr.Vector3(p2.x - nx * offset, p2.y - ny * offset, p2.z - nz * offset)
            self._draw_cylinder_segment(p1_minus, p2_minus, rad, c1, c2)

        else:
            rad = 0.060
            self._draw_cylinder_segment(p1, p2, rad, c1, c2)

    def draw_molecule_3d(self, cam_pos: pr.Vector3) -> None:
        """Renders 3D atom spheres and multi-bond cylinders for the active material."""
        mat = self.material
        sites: List[AtomSite] = mat.sites
        bonds = mat.bonds
        coords = self.rendered_coords

        # 1. Draw Bonds (Cylinders with camera-aligned multi-bond geometry)
        for b in bonds:
            i, j = b[0], b[1]
            b_type = b[2] if len(b) > 2 else "1"
            if 0 <= i < len(sites) and 0 <= j < len(sites) and i < len(coords) and j < len(coords):
                self.draw_bond(sites[i], sites[j], coords[i], coords[j], b_type, cam_pos)

        # 2. Draw Atom Sites (Spheres with ball-and-stick scaling)
        for idx, s in enumerate(sites):
            if idx < len(coords):
                c = coords[idx]
                pos = pr.Vector3(c[0], c[1], c[2])
                radius = get_atom_radius(s.atom_type, s.site_name, s.sigma)
                color = get_atom_color(s.atom_type, s.site_name)
                pr.draw_sphere(pos, radius, color)

    def draw_vdw_surface_3d(self) -> None:
        """
        Renders the Van der Waals surface wireframe mesh enclosing the molecule.
        Matches crystallographic reference visualizations (translucent envelope + crisp wireframe mesh).
        Dynamically updates as the Boltzmann Generator relaxes atomic coordinates in real-time.
        """
        if not self.show_vdw_surface:
            return

        mat = self.material
        sites = mat.sites
        coords = self.rendered_coords

        pr.begin_blend_mode(pr.BLEND_ALPHA)

        # Pass 1: Subtle translucent volume fill
        fill_color = pr.Color(220, 230, 245, 16)
        for idx, s in enumerate(sites):
            if idx < len(coords):
                c = coords[idx]
                pos = pr.Vector3(c[0], c[1], c[2])
                vdw_r = get_vdw_radius(s.atom_type, s.site_name, s.sigma)
                pr.draw_sphere(pos, vdw_r, fill_color)

        # Pass 2: Clean wireframe mesh (matching reference screenshot)
        wire_color = pr.Color(160, 175, 195, 80)
        for idx, s in enumerate(sites):
            if idx < len(coords):
                c = coords[idx]
                pos = pr.Vector3(c[0], c[1], c[2])
                vdw_r = get_vdw_radius(s.atom_type, s.site_name, s.sigma)
                pr.draw_sphere_wires(pos, vdw_r, 16, 16, wire_color)

        pr.end_blend_mode()

    def draw_hud(self) -> None:
        """Renders top-left HUD with State badge and camera controls."""
        mat = self.material
        bg_dark = pr.Color(16, 20, 26, 225)
        pr.draw_rectangle(15, 15, 370, 140, bg_dark)
        pr.draw_rectangle_lines(15, 15, 370, 140, pr.Color(50, 60, 75, 255))

        title_text = f"Material: {mat.name}"
        pr.draw_text(title_text, 25, 25, 18, pr.RAYWHITE)

        mode_text = f"Mode: {mat.dimension_mode} | Sites: {len(mat.sites)} | Bonds: {len(mat.bonds)}"
        pr.draw_text(mode_text, 25, 50, 14, pr.LIGHTGRAY)

        sigma_text = f"Effective σ: {mat.effective_sigma:.2f} Å | ε/k_B: {mat.effective_epsilon_k:.1f} K"
        pr.draw_text(sigma_text, 25, 70, 14, pr.LIGHTGRAY)

        # Pipeline Status Badge
        st = self.telemetry.state
        state_color = pr.Color(245, 205, 35, 255)  # WAITING
        if st == "RUNNING_CDFT":
            state_color = pr.Color(80, 200, 255, 255)
        elif st == "CDFT_CONVERGED":
            state_color = pr.Color(60, 220, 100, 255)
        elif st == "RUNNING_BG":
            state_color = pr.Color(200, 100, 255, 255)
        elif st == "COMPLETE":
            state_color = pr.Color(40, 240, 80, 255)

        pr.draw_text("State: ", 25, 92, 15, pr.LIGHTGRAY)
        pr.draw_text(st, 75, 92, 15, state_color)

        controls_text = "[Left Drag] Orbit | [Scroll] Zoom | [R] Reset | [V] Mesh | [Space] Play"
        pr.draw_text(controls_text, 25, 120, 11, pr.GRAY)

        pr.draw_fps(self.width - 90, 15)

    def draw_cdft_density_overlay(self) -> None:
        """Renders real-time 2D line graph of spatial density profile rho(z) in bottom-left."""
        if not self.telemetry.rho_z or len(self.telemetry.rho_z) < 2:
            return

        box_x, box_y, box_w, box_h = 20, self.height - 245, 320, 155
        bg_dark = pr.Color(14, 18, 24, 230)
        pr.draw_rectangle(box_x, box_y, box_w, box_h, bg_dark)
        pr.draw_rectangle_lines(box_x, box_y, box_w, box_h, pr.Color(45, 55, 70, 255))

        title = "cDFT Density Profile ρ(z)"
        pr.draw_text(title, box_x + 10, box_y + 8, 13, pr.RAYWHITE)

        # Plot area
        plot_x = box_x + 35
        plot_y = box_y + 30
        plot_w = box_w - 50
        plot_h = box_h - 55

        # Draw axes
        pr.draw_line(plot_x, plot_y, plot_x, plot_y + plot_h, pr.DARKGRAY)
        pr.draw_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, pr.DARKGRAY)

        rho_arr = self.telemetry.rho_z
        max_rho = max(max(rho_arr), self.material.bulk_density_a3 * 1.5, 1e-4)

        # Draw horizontal bulk density dashed line
        bulk_rho = self.material.bulk_density_a3
        bulk_y = int(plot_y + plot_h - (bulk_rho / max_rho) * plot_h)
        if plot_y <= bulk_y <= (plot_y + plot_h):
            pr.draw_line(plot_x, bulk_y, plot_x + plot_w, bulk_y, pr.Color(80, 100, 120, 200))
            pr.draw_text("ρ_bulk", plot_x + plot_w - 40, bulk_y - 12, 10, pr.Color(120, 140, 160, 255))

        # Plot points & curve
        n_pts = len(rho_arr)
        prev_px, prev_py = -1, -1
        line_color = pr.Color(50, 220, 160, 255)

        for idx, rho_val in enumerate(rho_arr):
            px = int(plot_x + (idx / (n_pts - 1)) * plot_w)
            py = int(plot_y + plot_h - (min(max_rho, max(0.0, rho_val)) / max_rho) * plot_h)

            if prev_px != -1:
                pr.draw_line(prev_px, prev_py, px, py, line_color)
            prev_px, prev_py = px, py

        # Peak annotation
        peak_text = f"Peak: {max(rho_arr):.4f} Å⁻³"
        pr.draw_text(peak_text, box_x + 10, box_y + box_h - 18, 11, pr.LIGHTGRAY)
        p_text = f"P_wall: {self.telemetry.wall_pressure_bar:.1f} bar"
        pr.draw_text(p_text, box_x + box_w - 120, box_y + box_h - 18, 11, pr.Color(100, 200, 255, 255))

    def draw_telemetry_panel(self) -> None:
        """Renders monospace telemetry panel positioned neatly in the bottom-right."""
        t = self.telemetry
        panel_w = 345
        panel_h = 265
        panel_x = self.width - panel_w - 20
        panel_y = self.height - panel_h - 85  # Nearer bottom right, sitting above the bottom control deck

        bg_dark = pr.Color(14, 18, 24, 235)
        pr.draw_rectangle(panel_x, panel_y, panel_w, panel_h, bg_dark)
        pr.draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, pr.Color(50, 65, 80, 255))

        pr.draw_text("TELEMETRY OUTPUT", panel_x + 15, panel_y + 12, 14, pr.RAYWHITE)
        pr.draw_line(panel_x + 15, panel_y + 32, panel_x + panel_w - 15, panel_y + 32, pr.Color(40, 50, 65, 255))

        y = panel_y + 40
        line_spacing = 22

        # 1. Thermodynamics
        pr.draw_text("Thermodynamics:", panel_x + 15, y, 12, pr.Color(110, 180, 255, 255))
        y += line_spacing - 4
        free_en_text = f"  Excess Free Energy (Ω) : {t.excess_free_energy:8.2f} k_BT"
        pr.draw_text(free_en_text, panel_x + 15, y, 12, pr.LIGHTGRAY)
        y += line_spacing - 4
        p_wall_text = f"  Wall Pressure (P_wall) : {t.wall_pressure_bar:8.2f} bar"
        pr.draw_text(p_wall_text, panel_x + 15, y, 12, pr.LIGHTGRAY)

        y += line_spacing
        # 2. Structural Health
        pr.draw_text("Structural Health:", panel_x + 15, y, 12, pr.Color(110, 180, 255, 255))
        y += line_spacing - 4
        clash_text = f"  Steric Clashes         : {t.steric_clashes}"
        clash_col = pr.Color(50, 220, 120, 255) if t.steric_clashes == 0 else pr.Color(240, 160, 40, 255)
        pr.draw_text(clash_text, panel_x + 15, y, 12, clash_col)
        y += line_spacing - 4
        acc_text = f"  Torsional Acceptance   : {t.torsional_acceptance_pct:5.1f} %"
        pr.draw_text(acc_text, panel_x + 15, y, 12, pr.LIGHTGRAY)

        y += line_spacing
        # 3. Geometry
        pr.draw_text("Geometry:", panel_x + 15, y, 12, pr.Color(110, 180, 255, 255))
        y += line_spacing - 4
        rg_text = f"  Radius of Gyration (Rg): {t.radius_of_gyration:6.2f} Å"
        pr.draw_text(rg_text, panel_x + 15, y, 12, pr.LIGHTGRAY)
        y += line_spacing - 4
        ree_text = f"  End-to-End Dist (Ree)  : {t.end_to_end_dist:6.2f} Å"
        pr.draw_text(ree_text, panel_x + 15, y, 12, pr.LIGHTGRAY)

        y += line_spacing + 2
        pr.draw_line(panel_x + 15, y - 4, panel_x + panel_w - 15, y - 4, pr.Color(40, 50, 65, 255))
        # 4. Material Viability
        viab_col = (
            pr.Color(50, 220, 100, 255)
            if t.is_wetting and t.coating_viability != "PENDING"
            else pr.Color(230, 60, 60, 255)
            if t.coating_viability != "PENDING"
            else pr.GRAY
        )
        viab_text = f"Coating Viability: {t.coating_viability}"
        pr.draw_text(viab_text, panel_x + 15, y, 13, viab_col)

    def draw_control_deck(self) -> None:
        """Renders minimalist bottom execution control deck with Reset button."""
        deck_x = 20
        deck_y = self.height - 75
        deck_w = self.width - 40
        deck_h = 58

        bg_deck = pr.Color(14, 18, 24, 230)
        pr.draw_rectangle(deck_x, deck_y, deck_w, deck_h, bg_deck)
        pr.draw_rectangle_lines(deck_x, deck_y, deck_w, deck_h, pr.Color(45, 55, 70, 255))

        mouse_pos = pr.get_mouse_position()
        mouse_clicked = pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT)

        curr_x = deck_x + 15
        btn_y = deck_y + 12
        btn_h = 34

        is_running_cdft = self.worker.is_running and self.telemetry.state == "RUNNING_CDFT"
        is_running_bg = self.worker.is_running and self.telemetry.state == "RUNNING_BG"
        cdft_converged = self.telemetry.state in ("CDFT_CONVERGED", "RUNNING_BG", "COMPLETE")

        # 0. [Reset] Button
        btn_reset = pr.Rectangle(curr_x, btn_y, 75, btn_h)
        hover0 = pr.check_collision_point_rec(mouse_pos, btn_reset)
        col0 = pr.Color(55, 50, 60, 255) if not hover0 else pr.Color(80, 65, 85, 255)
        pr.draw_rectangle_rec(btn_reset, col0)
        pr.draw_rectangle_lines_ex(btn_reset, 1, pr.Color(85, 75, 95, 255))
        pr.draw_text("Reset", curr_x + 18, btn_y + 10, 13, pr.RAYWHITE)

        if hover0 and mouse_clicked:
            self.reset_all()

        curr_x += 85

        # 1. [Step cDFT] Button
        btn_step_cdft = pr.Rectangle(curr_x, btn_y, 90, btn_h)
        hover1 = pr.check_collision_point_rec(mouse_pos, btn_step_cdft)
        col1 = pr.Color(40, 50, 65, 255) if not hover1 else pr.Color(55, 70, 90, 255)
        pr.draw_rectangle_rec(btn_step_cdft, col1)
        pr.draw_rectangle_lines_ex(btn_step_cdft, 1, pr.Color(70, 85, 105, 255))
        pr.draw_text("Step cDFT", curr_x + 12, btn_y + 10, 13, pr.RAYWHITE)

        if hover1 and mouse_clicked and not self.worker.is_running:
            self.worker.step_cdft(n_steps=5)

        curr_x += 100

        # 2. [Solve cDFT] / [Cancel] Button
        btn_solve_cdft = pr.Rectangle(curr_x, btn_y, 100, btn_h)
        hover2 = pr.check_collision_point_rec(mouse_pos, btn_solve_cdft)
        if is_running_cdft:
            col2 = pr.Color(180, 45, 45, 255) if not hover2 else pr.Color(210, 60, 60, 255)
            label2 = "Cancel"
        else:
            col2 = pr.Color(30, 80, 130, 255) if not hover2 else pr.Color(40, 105, 170, 255)
            label2 = "Solve cDFT"

        pr.draw_rectangle_rec(btn_solve_cdft, col2)
        pr.draw_rectangle_lines_ex(btn_solve_cdft, 1, pr.Color(80, 130, 180, 255))
        pr.draw_text(label2, curr_x + (25 if is_running_cdft else 13), btn_y + 10, 13, pr.RAYWHITE)

        if hover2 and mouse_clicked:
            if is_running_cdft:
                self.worker.cancel()
            elif not self.worker.is_running:
                self.worker.solve_cdft()

        curr_x += 112

        # 3. Horizontal cDFT Progress Bar
        bar_w = 175
        bar_h = 16
        bar_y = btn_y + 9
        pr.draw_rectangle(curr_x, bar_y, bar_w, bar_h, pr.Color(25, 30, 40, 255))
        pr.draw_rectangle_lines(curr_x, bar_y, bar_w, bar_h, pr.Color(50, 60, 75, 255))

        prog = max(0.0, min(1.0, self.telemetry.cdft_progress))
        fill_w = int(bar_w * prog)
        fill_col = pr.Color(40, 180, 120, 255) if prog >= 1.0 else pr.Color(50, 150, 240, 255)
        pr.draw_rectangle(curr_x, bar_y, fill_w, bar_h, fill_col)

        pct_text = f"{int(prog * 100)}%"
        pr.draw_text(pct_text, curr_x + bar_w + 8, btn_y + 10, 12, pr.LIGHTGRAY)

        curr_x += bar_w + 50

        # 4. [Step MCMC] Button (Grayed out until cDFT 100%)
        btn_step_mcmc = pr.Rectangle(curr_x, btn_y, 95, btn_h)
        hover3 = pr.check_collision_point_rec(mouse_pos, btn_step_mcmc)
        if cdft_converged:
            col3 = pr.Color(50, 45, 75, 255) if not hover3 else pr.Color(70, 60, 105, 255)
            text_col3 = pr.RAYWHITE
        else:
            col3 = pr.Color(25, 28, 35, 255)
            text_col3 = pr.DARKGRAY

        pr.draw_rectangle_rec(btn_step_mcmc, col3)
        pr.draw_rectangle_lines_ex(btn_step_mcmc, 1, pr.Color(50, 60, 75, 255))
        pr.draw_text("Step MCMC", curr_x + 10, btn_y + 10, 13, text_col3)

        if hover3 and mouse_clicked and cdft_converged and not self.worker.is_running:
            self.worker.step_mcmc(n_steps=5)

        curr_x += 105

        # 5. [Fully Solve BG] / [Cancel] Button (Grayed out until cDFT 100%)
        btn_solve_bg = pr.Rectangle(curr_x, btn_y, 125, btn_h)
        hover4 = pr.check_collision_point_rec(mouse_pos, btn_solve_bg)
        if is_running_bg:
            col4 = pr.Color(180, 45, 45, 255) if not hover4 else pr.Color(210, 60, 60, 255)
            label4 = "Cancel"
            text_col4 = pr.RAYWHITE
        elif cdft_converged:
            col4 = pr.Color(85, 45, 135, 255) if not hover4 else pr.Color(115, 60, 180, 255)
            label4 = "Fully Solve BG"
            text_col4 = pr.RAYWHITE
        else:
            col4 = pr.Color(25, 28, 35, 255)
            label4 = "Fully Solve BG"
            text_col4 = pr.DARKGRAY

        pr.draw_rectangle_rec(btn_solve_bg, col4)
        pr.draw_rectangle_lines_ex(btn_solve_bg, 1, pr.Color(60, 65, 80, 255))
        pr.draw_text(label4, curr_x + (40 if is_running_bg else 16), btn_y + 10, 13, text_col4)

        if hover4 and mouse_clicked and cdft_converged:
            if is_running_bg:
                self.worker.cancel()
            elif not self.worker.is_running:
                self.worker.solve_bg()

        curr_x += 135

        # 6. [VDW Mesh] Toggle Button
        btn_vdw = pr.Rectangle(curr_x, btn_y, 100, btn_h)
        hover5 = pr.check_collision_point_rec(mouse_pos, btn_vdw)
        if self.show_vdw_surface:
            col5 = pr.Color(40, 90, 85, 255) if not hover5 else pr.Color(55, 120, 110, 255)
            border_col5 = pr.Color(60, 180, 150, 255)
        else:
            col5 = pr.Color(35, 40, 50, 255) if not hover5 else pr.Color(50, 60, 75, 255)
            border_col5 = pr.Color(60, 70, 85, 255)

        pr.draw_rectangle_rec(btn_vdw, col5)
        pr.draw_rectangle_lines_ex(btn_vdw, 1, border_col5)
        vdw_label = "Mesh: ON" if self.show_vdw_surface else "Mesh: OFF"
        pr.draw_text(vdw_label, curr_x + 14, btn_y + 10, 13, pr.RAYWHITE)

        if hover5 and mouse_clicked:
            self.show_vdw_surface = not self.show_vdw_surface

    def run(self) -> None:
        """Main window loop with non-blocking ZeroMQ polling and 60 FPS rendering."""
        pr.set_config_flags(pr.FLAG_MSAA_4X_HINT | pr.FLAG_WINDOW_RESIZABLE)
        pr.init_window(self.width, self.height, self.title)
        pr.set_target_fps(60)

        # Load and compile Blinn-Phong lighting shader
        shader_enabled = False
        shader = None
        view_pos_loc = -1
        try:
            shader = pr.load_shader_from_memory(VS_PHONG, FS_PHONG)
            view_pos_loc = pr.get_shader_location(shader, "viewPos")
            shader_enabled = True
        except Exception:
            shader_enabled = False

        camera = pr.Camera3D(
            self.get_camera_position(),
            self.target,
            pr.Vector3(0.0, 1.0, 0.0),
            45.0,
            pr.CAMERA_PERSPECTIVE,
        )

        bg_color = pr.Color(16, 19, 25, 255)

        while not pr.window_should_close():
            # 1. Non-blockingly poll ZeroMQ SUB socket & queue for background worker updates
            self.telemetry = self.worker.poll_telemetry()
            if self.telemetry.current_coords:
                self.rendered_coords = self.telemetry.current_coords

            self.handle_input()

            cam_pos = self.get_camera_position()
            camera.position = cam_pos
            camera.target = self.target

            # Update shader camera position
            if shader_enabled and shader is not None and view_pos_loc != -1:
                cam_arr = pr.ffi.new("float[3]", [cam_pos.x, cam_pos.y, cam_pos.z])
                pr.set_shader_value(shader, view_pos_loc, cam_arr, pr.SHADER_UNIFORM_VEC3)

            pr.begin_drawing()
            pr.clear_background(bg_color)

            # 3D Viewport
            pr.begin_mode_3d(camera)

            # Shaded Opaque Geometry (Bonds & Atoms)
            if shader_enabled and shader is not None:
                pr.begin_shader_mode(shader)
                self.draw_molecule_3d(cam_pos)
                pr.end_shader_mode()
            else:
                self.draw_molecule_3d(cam_pos)

            # Van der Waals (VDW) Wireframe Surface Mesh
            self.draw_vdw_surface_3d()

            pr.end_mode_3d()

            # 2D Overlays & HUD Elements
            self.draw_hud()
            self.draw_cdft_density_overlay()
            self.draw_telemetry_panel()
            self.draw_control_deck()

            pr.end_drawing()

        # Clean shutdown
        self.worker.close()
        if shader_enabled and shader is not None:
            pr.unload_shader(shader)
        pr.close_window()


def run_interactive_viewer(
    material_names: Optional[List[str]] = None,
    width: int = 1280,
    height: int = 720,
) -> None:
    """
    Helper function to load the requested material and launch the 3D Raylib viewer.
    """
    if not material_names or material_names == ["all"]:
        target_name = "argon"
    else:
        target_name = material_names[0]

    try:
        mat = MaterialLoader.load_material(target_name)
    except Exception as e:
        print(f"Warning: Failed to load material '{target_name}': {e}. Defaulting to 'argon'.")
        mat = MaterialLoader.load_material("argon")

    viewer = MoleculeViewer(material=mat, width=width, height=height)
    viewer.run()
