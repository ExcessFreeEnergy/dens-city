"""
Unit and functional tests for the Raylib 3D Molecular Viewer in dens_city.ui.
Verifies CPK/publication color mappings, GAFF/Tripos element parsing, orbital camera math,
multi-bond geometry, scale-invariant auto-framing (up to 128+ sites), Van der Waals surface radii,
full reset functionality, and CLI argument parsing.
"""

import math

from dens_city.ui.cli import parse_materials_arg
from dens_city.ui.viewer import (
    ELEMENT_COLORS,
    VDW_RADII,
    MoleculeViewer,
    get_atom_color,
    get_atom_element,
    get_atom_radius,
    get_vdw_radius,
)
from dens_city.utils.materials import AtomSite, Material, MaterialLoader


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


def test_vdw_radius_mapping():
    """Validates crystallographic Van der Waals radii for molecular surface reconstruction."""
    for elem in ["H", "C", "N", "O", "CL", "S", "AR"]:
        assert elem in VDW_RADII
        vdw_r = get_vdw_radius(elem)
        assert 1.0 <= vdw_r <= 2.5

    assert math.isclose(get_vdw_radius("c3"), 1.70, abs_tol=1e-3)
    assert math.isclose(get_vdw_radius("ha"), 1.20, abs_tol=1e-3)
    assert math.isclose(get_vdw_radius("n1"), 1.55, abs_tol=1e-3)
    assert math.isclose(get_vdw_radius("os"), 1.52, abs_tol=1e-3)

    # Fallback
    fallback_vdw = get_vdw_radius("Xx", sigma=3.8)
    assert math.isclose(fallback_vdw, 1.9, abs_tol=1e-3)


def test_molecule_viewer_bounds_and_auto_framing():
    """
    Validates that MoleculeViewer accurately calculates centroid and distance
    for monoatomic, small molecule, and complex fluids.
    """
    argon = MaterialLoader.load_material("argon")
    viewer_ar = MoleculeViewer(material=argon)
    assert viewer_ar.material.name == "argon"
    assert viewer_ar.target.x == 0.0
    assert viewer_ar.target.y == 0.0
    assert viewer_ar.target.z == 0.0
    assert viewer_ar.distance >= 4.5
    assert viewer_ar.show_vdw_surface is True
    viewer_ar.worker.close()

    water = MaterialLoader.load_material("water")
    viewer_w = MoleculeViewer(material=water)
    assert viewer_w.material.name == "water"
    assert viewer_w.distance > 0.0
    viewer_w.worker.close()

    five_cb = MaterialLoader.load_material("5cb")
    viewer_5cb = MoleculeViewer(material=five_cb)
    assert viewer_5cb.material.name == "5cb"
    assert viewer_5cb.distance > 8.0
    viewer_5cb.worker.close()


def test_molecule_viewer_reset_all():
    """Validates that reset_all() restores camera, calculations, telemetry, and coordinates."""
    benzene = MaterialLoader.load_material("benzene")
    viewer = MoleculeViewer(material=benzene)
    try:
        # Step solver and perturb camera
        viewer.worker.step_cdft()
        viewer.distance = 99.0
        viewer.azimuth = 2.5
        viewer.elevation = 0.8
        viewer.rendered_coords[0] = (10.0, 10.0, 10.0)

        # Reset all
        viewer.reset_all()
        assert viewer.worker.telemetry.state == "WAITING_CDFT"
        assert viewer.worker.telemetry.cdft_step == 0
        assert viewer.distance == viewer.default_distance
        assert math.isclose(viewer.azimuth, 0.75, abs_tol=1e-3)
        assert math.isclose(viewer.elevation, 0.35, abs_tol=1e-3)
        assert viewer.rendered_coords[0] == (benzene.sites[0].x, benzene.sites[0].y, benzene.sites[0].z)
    finally:
        viewer.worker.close()


def test_sodium_dodecyl_sulfate_bounds():
    """
    Validates that Sodium Dodecyl Sulfate (43 sites, elongated surfactant)
    is properly centered and auto-framed without frustum clipping.
    """
    sds = MaterialLoader.load_material("sodium_dodecyl_sulfate")
    assert len(sds.sites) == 43
    assert len(sds.bonds) == 41

    viewer = MoleculeViewer(material=sds)
    try:
        assert viewer.distance >= 25.0, f"Camera distance {viewer.distance} must scale for ~20 Å SDS span"

        # Sulfate head group has 3 S=O double bonds
        so_double_bonds = [b for b in sds.bonds if str(b[2]).strip() == "2"]
        assert len(so_double_bonds) == 3, f"Expected 3 S=O double bonds, got {len(so_double_bonds)}"
    finally:
        viewer.worker.close()


def test_large_128_site_molecule_scaling():
    """
    Validates scale-invariant auto-framing for large 128-site polymers/macromolecules.
    """
    sites = []
    for i in range(128):
        # 128-site helical/linear polymer chain spanning ~100 Å in Z
        sites.append(
            AtomSite(
                site_name=f"C{i + 1}",
                atom_type="c3",
                x=math.cos(i * 0.3) * 2.0,
                y=math.sin(i * 0.3) * 2.0,
                z=i * 0.8,
                charge=0.0,
                sigma=3.4,
                epsilon_kcal=0.1,
                epsilon_k=50.0,
                mass=12.011,
            )
        )

    bonds = [(i, i + 1, "1") for i in range(127)]
    mat_128 = Material(
        name="poly_128",
        identifier="poly_128",
        dimension_mode="3D_MOLECULAR",
        sites=sites,
        bonds=bonds,
        effective_sigma=3.4,
        effective_epsilon_k=50.0,
        temperature_k=300.0,
    )

    viewer = MoleculeViewer(material=mat_128)
    try:
        assert len(viewer.material.sites) == 128
        # Target must be at center of chain (z ~ 50.8 Å)
        assert math.isclose(viewer.target.z, 50.8, abs_tol=1.0)
        # Camera distance must scale to contain the full ~102 Å bounding sphere
        assert viewer.distance > 100.0, f"Expected camera distance > 100 Å, got {viewer.distance}"
    finally:
        viewer.worker.close()


def test_nitrogen_and_co2_multi_bond_detection():
    """Validates that multi-bonds (triple in N2, double in CO2) are loaded from .mol2."""
    nitrogen = MaterialLoader.load_material("nitrogen")
    assert len(nitrogen.sites) == 2
    assert len(nitrogen.bonds) == 1
    assert str(nitrogen.bonds[0][2]).strip() == "3"

    co2 = MaterialLoader.load_material("carbon_dioxide")
    assert len(co2.sites) == 3
    assert len(co2.bonds) == 2
    assert str(co2.bonds[0][2]).strip() == "2"
    assert str(co2.bonds[1][2]).strip() == "2"

    five_cb = MaterialLoader.load_material("5cb")
    has_triple = any(str(b[2]).strip() == "3" for b in five_cb.bonds)
    assert has_triple, "5CB must contain a triple bond in its nitrile group"


def test_camera_spherical_coordinate_math():
    """Validates camera 3D position calculation from spherical azimuth/elevation."""
    water = MaterialLoader.load_material("water")
    viewer = MoleculeViewer(material=water)
    try:
        viewer.distance = 10.0
        viewer.elevation = 0.0
        viewer.azimuth = 0.0

        pos = viewer.get_camera_position()
        assert math.isclose(pos.x, viewer.target.x, abs_tol=1e-5)
        assert math.isclose(pos.y, viewer.target.y, abs_tol=1e-5)
        assert math.isclose(pos.z, viewer.target.z + 10.0, abs_tol=1e-5)

        viewer.elevation = math.pi / 2.0
        pos_top = viewer.get_camera_position()
        assert math.isclose(pos_top.x, viewer.target.x, abs_tol=1e-5)
        assert math.isclose(pos_top.y, viewer.target.y + 10.0, abs_tol=1e-5)
        assert math.isclose(pos_top.z, viewer.target.z, abs_tol=1e-5)
    finally:
        viewer.worker.close()


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
    assert "sodium_dodecyl_sulfate" in all_mats
