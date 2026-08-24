"""
High-performance 3D molecular viewer implemented in Raylib (pyray).
Renders atom sites with realistic ball-and-stick proportions, camera-aligned multi-bond cylinder geometry
(triple, double, single, aromatic), dynamic ground plane positioning, Phong specular highlights,
and alpha-blended probability density clouds for arbitrary molecules up to 128+ sites.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyray as pr

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


@dataclass
class DensityCloud:
    """
    Volumetric 3D spatial probability density cloud for cDFT density or Boltzmann Generator priors.
    """

    points: List[Tuple[float, float, float]] = field(default_factory=list)
    densities: List[float] = field(default_factory=list)
    color_rgb: Tuple[int, int, int] = (60, 160, 255)
    max_density: float = 1.0
    alpha_scale: float = 0.5


class MoleculeViewer:
    """
    Raylib 3D molecular renderer with camera-aligned multi-bond cylinder geometry,
    scale-invariant auto-framing, dynamic ground grid, specular lighting,
    and alpha-blended probability density clouds.
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
        self.azimuth = 0.75
        self.elevation = 0.35
        self.distance = 15.0
        self.target = pr.Vector3(0.0, 0.0, 0.0)
        self.default_distance = 15.0
        self.default_target = pr.Vector3(0.0, 0.0, 0.0)

        # Dynamic Grid parameters
        self.grid_y = -3.0
        self.grid_slices = 30
        self.grid_spacing = 1.0
        self.show_grid = True

        # Volumetric Probability Cloud
        self.density_cloud: Optional[DensityCloud] = None
        self.show_cloud: bool = True

        self._update_molecule_bounds()

    @property
    def current_material(self) -> Material:
        return self.materials[self.current_idx]

    def set_probability_cloud(self, cloud: Optional[DensityCloud]) -> None:
        """Attaches a volumetric probability density cloud to the 3D scene."""
        self.density_cloud = cloud

    def _update_molecule_bounds(self) -> None:
        """
        Calculates centroid, bounding radius, dynamic ground grid altitude,
        and auto-frames camera distance from perspective frustum limits (up to 128+ sites).
        """
        mat = self.current_material
        sites = mat.sites

        if not sites:
            self.target = pr.Vector3(0.0, 0.0, 0.0)
            self.distance = 8.0
            self.default_target = self.target
            self.default_distance = self.distance
            self.grid_y = -2.0
            self.grid_slices = 20
            self.grid_spacing = 1.0
            return

        cx = sum(s.x for s in sites) / len(sites)
        cy = sum(s.y for s in sites) / len(sites)
        cz = sum(s.z for s in sites) / len(sites)
        self.target = pr.Vector3(cx, cy, cz)
        self.default_target = pr.Vector3(cx, cy, cz)

        min_y = min(s.y for s in sites)

        max_d = 0.0
        for s in sites:
            dx = s.x - cx
            dy = s.y - cy
            dz = s.z - cz
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > max_d:
                max_d = d

        bounding_radius = max(max_d, mat.effective_sigma * 0.5, 1.0)

        # 1. Perspective FOV frustum-based auto-framing:
        # Distance guarantees the entire bounding sphere is contained in 45-degree vertical FOV
        fov_half_rad = math.radians(22.5)
        fit_dist = (bounding_radius / math.sin(fov_half_rad)) * 1.15
        self.distance = max(6.0, fit_dist)
        self.default_distance = self.distance

        # 2. Dynamic Ground Grid: Positioned strictly beneath the lowest atom in the molecule
        grid_margin = max(1.2, 0.15 * bounding_radius)
        self.grid_y = min_y - grid_margin
        self.grid_slices = max(24, min(120, int(math.ceil(bounding_radius * 2.2))))
        self.grid_spacing = max(1.0, round(bounding_radius / 15.0))

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
        self.azimuth = 0.75
        self.elevation = 0.35
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

        # 4. Keyboard Navigation
        if pr.is_key_pressed(pr.KEY_RIGHT) or pr.is_key_pressed(pr.KEY_RIGHT_BRACKET):
            self.next_material()
        if pr.is_key_pressed(pr.KEY_LEFT) or pr.is_key_pressed(pr.KEY_LEFT_BRACKET):
            self.prev_material()
        if pr.is_key_pressed(pr.KEY_R) or pr.is_key_pressed(pr.KEY_SPACE):
            self.reset_view()
        if pr.is_key_pressed(pr.KEY_G):
            self.show_grid = not self.show_grid
        if pr.is_key_pressed(pr.KEY_C):
            self.show_cloud = not self.show_cloud

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
        bond_type: str,
        cam_pos: pr.Vector3,
    ) -> None:
        """
        Renders accurate multi-bond cylinder geometry (triple, double, single, aromatic).
        Multi-bond offsets are oriented perpendicular to the camera line-of-sight to ensure
        all parallel bond cylinders remain visible from any rotation angle without self-occlusion.
        """
        p1 = pr.Vector3(s1.x, s1.y, s1.z)
        p2 = pr.Vector3(s2.x, s2.y, s2.z)

        c1 = get_atom_color(s1.atom_type, s1.site_name)
        c2 = get_atom_color(s2.atom_type, s2.site_name)

        vx = p2.x - p1.x
        vy = p2.y - p1.y
        vz = p2.z - p1.z
        length = math.sqrt(vx * vx + vy * vy + vz * vz)
        if length < 1e-4:
            return

        ux, uy, uz = vx / length, vy / length, vz / length

        # Compute line-of-sight view vector from bond midpoint to camera
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

        # Cross product of bond axis u and view ray cv gives camera-facing perpendicular normal
        nx = uy * cvz - uz * cvy
        ny = uz * cvx - ux * cvz
        nz = ux * cvy - uy * cvx
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)

        if n_len > 1e-3:
            nx, ny, nz = nx / n_len, ny / n_len, nz / n_len
        else:
            # Fallback if looking directly down the bond axis
            ax, ay, az = (0.0, 1.0, 0.0) if abs(uy) < 0.9 else (1.0, 0.0, 0.0)
            nx = uy * az - uz * ay
            ny = uz * ax - ux * az
            nz = ux * ay - uy * ax
            n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / n_len, ny / n_len, nz / n_len

        bt = str(bond_type).strip().lower()

        if bt in ("3", "tr", "triple"):
            # Triple Bond: 3 parallel cylinders
            offset = 0.095
            rad = 0.035
            # Center cylinder
            self._draw_cylinder_segment(p1, p2, rad, c1, c2)

            # Offset cylinder +
            p1_plus = pr.Vector3(p1.x + nx * offset, p1.y + ny * offset, p1.z + nz * offset)
            p2_plus = pr.Vector3(p2.x + nx * offset, p2.y + ny * offset, p2.z + nz * offset)
            self._draw_cylinder_segment(p1_plus, p2_plus, rad, c1, c2)

            # Offset cylinder -
            p1_minus = pr.Vector3(p1.x - nx * offset, p1.y - ny * offset, p1.z - nz * offset)
            p2_minus = pr.Vector3(p2.x - nx * offset, p2.y - ny * offset, p2.z - nz * offset)
            self._draw_cylinder_segment(p1_minus, p2_minus, rad, c1, c2)

        elif bt in ("2", "db", "double", "ar", "aromatic"):
            # Double / Aromatic Bond: 2 parallel cylinders
            offset = 0.065
            rad = 0.040
            p1_plus = pr.Vector3(p1.x + nx * offset, p1.y + ny * offset, p1.z + nz * offset)
            p2_plus = pr.Vector3(p2.x + nx * offset, p2.y + ny * offset, p2.z + nz * offset)
            self._draw_cylinder_segment(p1_plus, p2_plus, rad, c1, c2)

            p1_minus = pr.Vector3(p1.x - nx * offset, p1.y - ny * offset, p1.z - nz * offset)
            p2_minus = pr.Vector3(p2.x - nx * offset, p2.y - ny * offset, p2.z - nz * offset)
            self._draw_cylinder_segment(p1_minus, p2_minus, rad, c1, c2)

        else:
            # Single Bond: 1 central cylinder
            rad = 0.060
            self._draw_cylinder_segment(p1, p2, rad, c1, c2)

    def draw_molecule_3d(self, cam_pos: pr.Vector3) -> None:
        """Renders 3D atom spheres and multi-bond cylinders for the active material."""
        mat = self.current_material
        sites: List[AtomSite] = mat.sites
        bonds = mat.bonds

        # 1. Draw Bonds (Cylinders with camera-aligned multi-bond geometry)
        for b in bonds:
            i, j = b[0], b[1]
            b_type = b[2] if len(b) > 2 else "1"
            if 0 <= i < len(sites) and 0 <= j < len(sites):
                self.draw_bond(sites[i], sites[j], b_type, cam_pos)

        # 2. Draw Atom Sites (Spheres with ball-and-stick scaling)
        for s in sites:
            pos = pr.Vector3(s.x, s.y, s.z)
            radius = get_atom_radius(s.atom_type, s.site_name, s.sigma)
            color = get_atom_color(s.atom_type, s.site_name)
            pr.draw_sphere(pos, radius, color)

    def draw_probability_cloud_3d(self) -> None:
        """Renders volumetric alpha-blended probability density cloud."""
        if not self.density_cloud or not self.show_cloud:
            return

        pr.begin_blend_mode(pr.BLEND_ALPHA)
        r, g, b = self.density_cloud.color_rgb
        max_d = max(1e-6, self.density_cloud.max_density)

        for pt, rho in zip(self.density_cloud.points, self.density_cloud.densities):
            norm_rho = min(1.0, rho / max_d)
            alpha = int(norm_rho * self.density_cloud.alpha_scale * 255)
            if alpha > 5:
                pos = pr.Vector3(pt[0], pt[1], pt[2])
                pr.draw_sphere(pos, 0.08, pr.Color(r, g, b, alpha))

        pr.end_blend_mode()

    def draw_hud(self) -> None:
        """Renders clean HUD information."""
        mat = self.current_material
        bg_dark = pr.Color(16, 20, 26, 225)
        pr.draw_rectangle(15, 15, 360, 135, bg_dark)
        pr.draw_rectangle_lines(15, 15, 360, 135, pr.Color(50, 60, 75, 255))

        title_text = f"Material: {mat.name} ({self.current_idx + 1}/{len(self.materials)})"
        pr.draw_text(title_text, 25, 25, 18, pr.RAYWHITE)

        mode_text = f"Mode: {mat.dimension_mode} | Sites: {len(mat.sites)} | Bonds: {len(mat.bonds)}"
        pr.draw_text(mode_text, 25, 50, 14, pr.LIGHTGRAY)

        sigma_text = f"Effective σ: {mat.effective_sigma:.2f} Å | ε/k_B: {mat.effective_epsilon_k:.1f} K"
        pr.draw_text(sigma_text, 25, 70, 14, pr.LIGHTGRAY)

        controls_text = "[Left Drag] Rotate | [Scroll] Zoom | [Right Drag] Pan"
        pr.draw_text(controls_text, 25, 95, 12, pr.GRAY)
        nav_text = "[←/→] Switch Material | [R] Reset | [G] Grid | [C] Cloud"
        pr.draw_text(nav_text, 25, 115, 12, pr.GRAY)

        pr.draw_fps(self.width - 90, 15)

    def run(self) -> None:
        """Main window loop."""
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

        bg_color = pr.Color(18, 21, 28, 255)

        while not pr.window_should_close():
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

            pr.begin_mode_3d(camera)

            # 1. Dynamic Floor Reference Grid (Cleanly positioned beneath the molecule)
            if self.show_grid:
                pr.rl_push_matrix()
                pr.rl_translatef(self.target.x, self.grid_y, self.target.z)
                pr.draw_grid(self.grid_slices, self.grid_spacing)
                pr.rl_pop_matrix()

            # 2. Shaded Opaque Geometry (Bonds & Atoms)
            if shader_enabled and shader is not None:
                pr.begin_shader_mode(shader)
                self.draw_molecule_3d(cam_pos)
                pr.end_shader_mode()
            else:
                self.draw_molecule_3d(cam_pos)

            # 3. Translucent Volumetric Probability Cloud
            self.draw_probability_cloud_3d()

            pr.end_mode_3d()

            self.draw_hud()

            pr.end_drawing()

        if shader_enabled and shader is not None:
            pr.unload_shader(shader)
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
