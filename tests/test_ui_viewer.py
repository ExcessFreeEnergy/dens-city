"""
Unit and functional tests for the Raylib 3D Molecular Viewer in dens_city.ui.
Verifies CPK/publication color mappings, GAFF/Tripos element parsing, orbital camera math,
multi-bond geometry, auto-framing, multi-material navigation, and probability clouds.
"""

import math

from dens_city.ui.cli import parse_materials_arg
from dens_city.ui.viewer import (
    ELEMENT_COLORS,
    DensityCloud,
    MoleculeViewer,
    get_atom_color,
    get_atom_element,
    get_atom_radius,
)
from dens_city.utils.materials import MaterialLoader


def test_atom_element_parsing():
    """Validates that Tripos and GAFF atom types are correctly parsed into element symbols."""
    assert get_atom_element("C.3") == "C"
    assert get_atom_element("C.ar") == "C"
    assert get_atom_element("c3") == "C"
    assert get_atom_element("ca") == "C"
    assert get_atom_element("c1") == "C"
    assert get_atom_element("H") == "H"
    assert get_atom_element("ha") == "H"
    assert get_atom_element("hc") == "H"
    assert get_atom_element("O.3") == "O"
    assert get_atom_element("o") == "O"
    assert get_atom_element("os") == "O"
    assert get_atom_element("N.am") == "N"
    assert get_atom_element("n1") == "N"
    assert get_atom_element("N1") == "N"
    assert get_atom_element("n2") == "N"
    assert get_atom_element("s6") == "S"
    assert get_atom_element("p5") == "P"
    assert get_atom_element("Cl") == "CL"
    assert get_atom_element("cl") == "CL"
    assert get_atom_element("Ar") == "AR"
    assert get_atom_element("ar") == "AR"
    assert get_atom_element("Na") == "NA"


def test_element_color_and_radius_mapping():
    """Validates standard colors and visual radii for common elements."""
    for elem in ["H", "C", "N", "O", "CL", "S", "AR"]:
        assert elem in ELEMENT_COLORS
        color = get_atom_color(elem)
        assert color.a == 255
        radius = get_atom_radius(elem)
        assert 0.15 <= radius <= 0.60

    # Nitrogen specifically maps to rich emerald green (matching reference)
    n_color = get_atom_color("n1", "N1")
    assert n_color.r == 34 and n_color.g == 180 and n_color.b == 115

    # Fallback for unknown element
    fallback_color = get_atom_color("Xx")
    assert fallback_color.a == 255
    fallback_radius = get_atom_radius("Xx", sigma=4.0)
    assert 0.20 <= fallback_radius <= 0.60


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
    assert viewer.distance >= 4.5

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
    assert viewer.distance > 8.0  # Larger molecule requires larger framing distance

    # Cycle back to Argon
    viewer.next_material()
    assert viewer.current_material.name == "argon"

    # Test previous material navigation
    viewer.prev_material()
    assert viewer.current_material.name == "5cb"


def test_nitrogen_and_co2_multi_bond_detection():
    """Validates that multi-bonds (triple in N2, double in CO2) are loaded from .mol2."""
    nitrogen = MaterialLoader.load_material("nitrogen")
    assert len(nitrogen.sites) == 2
    assert len(nitrogen.bonds) == 1
    # Nitrogen bond is triple bond "3"
    assert str(nitrogen.bonds[0][2]).strip() == "3"

    co2 = MaterialLoader.load_material("carbon_dioxide")
    assert len(co2.sites) == 3
    assert len(co2.bonds) == 2
    # Both bonds in CO2 are double bonds "2"
    assert str(co2.bonds[0][2]).strip() == "2"
    assert str(co2.bonds[1][2]).strip() == "2"

    five_cb = MaterialLoader.load_material("5cb")
    # 5CB has nitrile triple bond (bond 1 between site 1 and 2)
    has_triple = any(str(b[2]).strip() == "3" for b in five_cb.bonds)
    assert has_triple, "5CB must contain a triple bond in its nitrile group"


def test_probability_density_cloud_structure():
    """Validates DensityCloud instantiation and viewer attachment."""
    water = MaterialLoader.load_material("water")
    viewer = MoleculeViewer(materials=[water])

    cloud = DensityCloud(
        points=[(0.0, 0.0, 1.0), (0.0, 0.5, 1.2)],
        densities=[0.02, 0.035],
        color_rgb=(80, 180, 255),
        max_density=0.035,
        alpha_scale=0.6,
    )
    viewer.set_probability_cloud(cloud)
    assert viewer.density_cloud is not None
    assert len(viewer.density_cloud.points) == 2
    assert viewer.density_cloud.max_density == 0.035


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
