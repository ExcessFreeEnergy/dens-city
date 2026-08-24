"""
Unit and functional tests for the Raylib 3D Molecular Viewer in dens_city.ui.
Verifies CPK color mappings, orbital camera math, auto-framing, multi-material navigation,
and CLI argument parsing.
"""

import math

from dens_city.ui.cli import parse_materials_arg
from dens_city.ui.viewer import (
    CPK_COLORS,
    MoleculeViewer,
    get_atom_color,
    get_atom_element,
    get_atom_radius,
)
from dens_city.utils.materials import MaterialLoader


def test_atom_element_parsing():
    """Validates that Tripos atom types are correctly parsed into element symbols."""
    assert get_atom_element("C.3") == "C"
    assert get_atom_element("C.ar") == "C"
    assert get_atom_element("H") == "H"
    assert get_atom_element("O.3") == "O"
    assert get_atom_element("N.am") == "N"
    assert get_atom_element("Cl") == "CL"
    assert get_atom_element("Ar") == "AR"


def test_cpk_color_and_radius_mapping():
    """Validates standard CPK colors and visual radii for common elements."""
    for elem in ["H", "C", "N", "O", "CL", "S", "AR"]:
        assert elem in CPK_COLORS
        color = get_atom_color(elem)
        assert color.a == 255
        radius = get_atom_radius(elem)
        assert radius > 0.1

    # Fallback for unknown element
    fallback_color = get_atom_color("Xx")
    assert fallback_color.a == 255
    fallback_radius = get_atom_radius("Xx", sigma=4.0)
    assert 0.3 <= fallback_radius <= 1.0


def test_molecule_viewer_bounds_and_auto_framing():
    """
    Validates that MoleculeViewer accurately calculates centroid and distance
    for monoatomic, small molecule, and polymer/liquid crystal materials.
    """
    argon = MaterialLoader.load_material("argon")
    water = MaterialLoader.load_material("water")
    benzene = MaterialLoader.load_material("benzene")
    five_cb = MaterialLoader.load_material("5cb")

    viewer = MoleculeViewer(materials=[argon, water, benzene, five_cb])
    assert len(viewer.materials) == 4
    assert viewer.current_material.name == "argon"

    # Argon (1 atom at 0,0,0)
    assert viewer.target.x == 0.0
    assert viewer.target.y == 0.0
    assert viewer.target.z == 0.0
    assert viewer.distance >= 6.0

    # Switch to Water (3 atoms)
    viewer.next_material()
    assert viewer.current_material.name == "water"
    assert viewer.distance > 0.0

    # Switch to Benzene (12 atoms)
    viewer.next_material()
    assert viewer.current_material.name == "benzene"

    # Switch to 5CB (38 atoms)
    viewer.next_material()
    assert viewer.current_material.name == "5cb"
    assert viewer.distance > 10.0  # Larger molecule requires larger framing distance

    # Cycle back to Argon
    viewer.next_material()
    assert viewer.current_material.name == "argon"

    # Test previous material navigation
    viewer.prev_material()
    assert viewer.current_material.name == "5cb"


def test_camera_spherical_coordinate_math():
    """Validates camera 3D position calculation from spherical azimuth/elevation."""
    water = MaterialLoader.load_material("water")
    viewer = MoleculeViewer(materials=[water])

    viewer.distance = 10.0
    viewer.elevation = 0.0
    viewer.azimuth = 0.0

    pos = viewer.get_camera_position()
    # At elevation 0, azimuth 0: camera is on Z-axis behind target (x=target.x, y=target.y, z=target.z + 10)
    assert math.isclose(pos.x, viewer.target.x, abs_tol=1e-5)
    assert math.isclose(pos.y, viewer.target.y, abs_tol=1e-5)
    assert math.isclose(pos.z, viewer.target.z + 10.0, abs_tol=1e-5)

    # Elevation 90 deg (pi/2): camera is directly above target on Y-axis
    viewer.elevation = math.pi / 2.0
    pos_top = viewer.get_camera_position()
    assert math.isclose(pos_top.x, viewer.target.x, abs_tol=1e-5)
    assert math.isclose(pos_top.y, viewer.target.y + 10.0, abs_tol=1e-5)
    assert math.isclose(pos_top.z, viewer.target.z, abs_tol=1e-5)


def test_cli_materials_arg_parsing():
    """Validates CLI argument parsing for material lists, commas, brackets, and 'all'."""
    assert parse_materials_arg(None) == ["argon"]
    assert parse_materials_arg([]) == ["argon"]
    assert parse_materials_arg(["water"]) == ["water"]
    assert parse_materials_arg(["argon", "water", "benzene"]) == ["argon", "water", "benzene"]
    assert parse_materials_arg(["[argon, water]"]) == ["argon", "water"]

    all_mats = parse_materials_arg(["all"])
    assert len(all_mats) >= 20
    assert "argon" in all_mats
    assert "water" in all_mats
    assert "5cb" in all_mats
