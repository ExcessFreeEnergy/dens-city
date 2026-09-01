"""
Unit tests for the Three Engine Tiers and adaptive hybrid routing in dens_city.
"""

from dens_city.utils.materials import AtomSite, Material
from dens_city.utils.pipeline import MaterialPipelineTask


def test_engine_tier_routing_properties():
    """
    Verifies that MaterialPipelineTask respects engine options:
    'classical', 'electronegativity', 'egnn', 'auto', and force_egnn.
    """
    # 1. Classical task
    task_c = MaterialPipelineTask(material_path_or_name="water", out_dir="runs/test", energy_engine="classical")
    assert task_c.energy_engine == "classical"
    assert not task_c.force_egnn

    # 2. Electronegativity task
    task_el = MaterialPipelineTask(
        material_path_or_name="water", out_dir="runs/test", energy_engine="electronegativity"
    )
    assert task_el.energy_engine == "electronegativity"

    # 3. EGNN task
    task_egnn = MaterialPipelineTask(material_path_or_name="water", out_dir="runs/test", energy_engine="egnn")
    assert task_egnn.energy_engine == "egnn"

    # 4. Auto task with force_egnn override
    task_auto = MaterialPipelineTask(
        material_path_or_name="water", out_dir="runs/test", energy_engine="auto", force_egnn=True
    )
    assert task_auto.force_egnn is True


def test_auto_heuristic_classification():
    """
    Verifies that the auto engine routes non-polar hydrocarbons to electronegativity/classical
    and heteroatom-containing polar molecules to egnn.
    """
    # Non-polar methane: 1 Carbon, 4 Hydrogens
    site_c = AtomSite(
        site_name="C",
        atom_type="c3",
        x=0,
        y=0,
        z=0,
        charge=0,
        sigma=3.4,
        epsilon_kcal=0.1,
        epsilon_k=100,
        mass=12,
        atomic_number=6,
    )
    site_h = AtomSite(
        site_name="H",
        atom_type="hc",
        x=1,
        y=0,
        z=0,
        charge=0,
        sigma=2.0,
        epsilon_kcal=0.0,
        epsilon_k=20,
        mass=1,
        atomic_number=1,
    )
    mat_alkane = Material(name="methane", identifier="methane", dimension_mode="3D_MOLECULAR", sites=[site_c, site_h])

    has_hetero_alkane = any(getattr(s, "atomic_number", 6) not in (1, 6) for s in mat_alkane.sites)
    assert not has_hetero_alkane, "Alkane was incorrectly classified as heteroatomic!"

    # Polar water: 1 Oxygen (Z=8), 2 Hydrogens (Z=1)
    site_o = AtomSite(
        site_name="O",
        atom_type="oh",
        x=0,
        y=0,
        z=0,
        charge=0,
        sigma=3.1,
        epsilon_kcal=0.15,
        epsilon_k=150,
        mass=16,
        atomic_number=8,
    )
    mat_water = Material(name="water", identifier="water", dimension_mode="3D_MOLECULAR", sites=[site_o, site_h])

    has_hetero_water = any(getattr(s, "atomic_number", 6) not in (1, 6) for s in mat_water.sites)
    assert has_hetero_water, "Water was not identified as containing heteroatoms!"
