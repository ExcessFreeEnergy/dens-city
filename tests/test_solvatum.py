"""
Tests for Solv@TUM (Solvatum) multi-solvent benchmark integration,
physical solvent registry, universal BenchmarkDataset abstraction layer,
and multi-solvent verification reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dens_city.utils.benchmark_dataset import (
    RT_LN10_KCAL_MOL,
    FreeSolvDataset,
    SolvatumDataset,
    get_benchmark_dataset,
)
from dens_city.utils.solvents import (
    estimate_dielectric_from_polarizability_and_dipole,
    get_solvent_properties,
    list_registered_solvents,
    normalize_solvent_name,
)
from dens_city.utils.verification import verify_and_generate_solvatum_report


def test_solvent_registry_coverage():
    """Verifies that all standard Solvatum solvents resolve to physical dielectric constants."""
    registered = list_registered_solvents()
    assert len(registered) >= 140

    # Test key solvent classes
    water = get_solvent_properties("WATER")
    assert water.dielectric_constant == pytest.approx(78.4, rel=1e-2)
    assert water.solvent_class == "polar_protic"

    m_meoh = get_solvent_properties("methanol")
    assert m_meoh.dielectric_constant == pytest.approx(32.7, rel=1e-2)
    assert m_meoh.solvent_class == "polar_protic"

    chlf = get_solvent_properties("CHLOROFORM")
    assert chlf.dielectric_constant == pytest.approx(4.81, rel=1e-2)
    assert chlf.solvent_class == "chlorinated"

    hexad = get_solvent_properties("HEXADECANE")
    assert hexad.dielectric_constant == pytest.approx(2.05, rel=1e-2)
    assert hexad.solvent_class == "alkane_nonpolar"

    dmf = get_solvent_properties("DIMETHYLFORMAMIDE")
    assert dmf.dielectric_constant == pytest.approx(36.7, rel=1e-2)
    assert dmf.solvent_class == "polar_aprotic"

    dmso = get_solvent_properties("DMSO")
    assert dmso.dielectric_constant == pytest.approx(46.7, rel=1e-2)


def test_solvent_name_normalization():
    """Verifies unicode and racemic prefix cleaning."""
    assert normalize_solvent_name("(\\xb1)-1,2-PROPANEDIOL") == "1,2-PROPANEDIOL"
    assert normalize_solvent_name("(±)-2-BUTANOL") == "2-BUTANOL"
    assert normalize_solvent_name(" 1-OCTANOL  ") == "1-OCTANOL"

    # Should match despite racemic prefixes
    p1 = get_solvent_properties("(\\xb1)-2-BUTANOL")
    assert p1.name == "2-BUTANOL"
    assert p1.dielectric_constant == pytest.approx(16.6, rel=1e-1)


def test_onsager_kirkwood_dynamic_fallback():
    """Verifies dynamic Onsager/Kirkwood-Fröhlich polarization equation fallback."""
    # Test for water-like parameters: mu = 1.85 D, alpha = 1.45 A^3, M = 18.015, rho = 0.997
    eps_est = estimate_dielectric_from_polarizability_and_dipole(
        dipole_debye=1.85,
        polarizability_angstrom3=1.45,
        molecular_weight=18.015,
        density_g_cm3=0.997,
        temp_k=298.15,
    )
    assert eps_est > 20.0  # Captures strong dipole orientational polarization

    # Test for nonpolar alkane: mu = 0.0 D, alpha = 10.0 A^3, M = 100.0, rho = 0.70
    eps_nonpolar = estimate_dielectric_from_polarizability_and_dipole(
        dipole_debye=0.0,
        polarizability_angstrom3=10.0,
        molecular_weight=100.0,
        density_g_cm3=0.70,
        temp_k=298.15,
    )
    assert 1.8 <= eps_nonpolar <= 2.5


def test_solvatum_dataset_loading_and_entries():
    """Verifies full loading of Solvatum SDF, entry conversion, and thermodynamic math."""
    sdf_path = Path("Solvatum/solvatum/data/solvatum.sdf")
    if not sdf_path.exists():
        pytest.skip("Solvatum submodule SDF file not found.")

    ds = SolvatumDataset(db_path=sdf_path)
    entries = ds.load_entries()
    assert len(entries) == 5952

    solutes = ds.get_unique_solutes()
    assert len(solutes) == 658

    solvents = ds.get_unique_solvents()
    assert len(solvents) == 146

    # Verify thermodynamic conversion: Delta G_solv = -1.364233 * log_k
    sample_entry = entries[0]
    raw_log_k = sample_entry.properties["log_k"]
    expected_dg = -RT_LN10_KCAL_MOL * raw_log_k
    assert sample_entry.expt_dG_solv == pytest.approx(expected_dg, rel=1e-5)
    assert sample_entry.solvent_dielectric > 1.0


def test_solvatum_mol2_generation_and_material_instantiation():
    """Verifies dynamic .mol2 string generation from RDKit Mol and loading into Material."""
    sdf_path = Path("Solvatum/solvatum/data/solvatum.sdf")
    if not sdf_path.exists():
        pytest.skip("Solvatum submodule SDF file not found.")

    ds = SolvatumDataset(db_path=sdf_path)
    # Test on a known organic molecule in Solvatum (e.g., solute ID '081' which is Methanol, or '001')
    mat = ds.get_material("001")
    if mat is None:
        # Fallback to the first available solute
        first_id = list(ds.get_unique_solutes().keys())[0]
        mat = ds.get_material(first_id)

    assert mat is not None
    assert mat.num_sites > 0
    assert mat.effective_sigma > 0.0
    assert len(mat.sites) == mat.num_sites
    assert hasattr(mat, "num_rotatable_bonds")


def test_benchmark_dataset_factory():
    """Verifies get_benchmark_dataset factory dispatch."""
    ds_free = get_benchmark_dataset("freesolv")
    assert isinstance(ds_free, FreeSolvDataset)
    assert ds_free.name == "freesolv"

    ds_solv = get_benchmark_dataset("solvatum")
    assert isinstance(ds_solv, SolvatumDataset)
    assert ds_solv.name == "solvatum"

    with pytest.raises(ValueError):
        get_benchmark_dataset("non_existent_dataset")


def test_solvatum_report_generation(tmp_path: Path):
    """Verifies that verify_and_generate_solvatum_report produces a valid markdown report."""
    sdf_path = Path("Solvatum/solvatum/data/solvatum.sdf")
    if not sdf_path.exists():
        pytest.skip("Solvatum submodule SDF file not found.")

    # Create mock pipeline summary with some Solvatum molecules (e.g. Helium, Argon, Methane, Methanol)
    mock_summary = tmp_path / "pipeline_summary.jsonl"
    mock_records = [
        {
            "material_name": "solvatum_000_helium",
            "status": "SUCCESS",
            "solvation_free_energy_kcal_mol": 2.15,
            "wall_pressure_bar": 12.5,
            "num_sites": 1,
            "cdft_runtime_seconds": 0.01,
            "bg_runtime_seconds": 0.02,
            "runtime_seconds": 0.03,
            "solvent_name": "HEXADECANE",
        },
        {
            "material_name": "solvatum_001_neon",
            "status": "SUCCESS",
            "solvation_free_energy_kcal_mol": 1.90,
            "wall_pressure_bar": 10.2,
            "num_sites": 1,
            "cdft_runtime_seconds": 0.01,
            "bg_runtime_seconds": 0.02,
            "runtime_seconds": 0.03,
            "solvent_name": "CHLOROFORM",
        },
    ]
    with open(mock_summary, "w", encoding="utf-8") as f:
        for r in mock_records:
            f.write(json.dumps(r) + "\n")

    report_file = tmp_path / "test_solvatum_report.md"
    stats = verify_and_generate_solvatum_report(
        results_dir=tmp_path,
        db_path=sdf_path,
        report_out=report_file,
    )

    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Solv@TUM (Solvatum) Multi-Solvent Validation Report" in content
    assert "Performance Breakdown Across Solvent Chemical Classes" in content
    assert stats["successful_runs"] == 2


def test_cli_dataset_options():
    """Verifies that CLI parser accepts --dataset, --all-solvatum, --verify-solvatum, --verify-dataset."""
    from dens_city.ui.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["--dataset", "solvatum", "--all-solvatum", "--verify-solvatum"])
    assert args.dataset == "solvatum"
    assert args.all_solvatum is True
    assert args.verify_solvatum is True

    args2 = parser.parse_args(["--verify-dataset", "freesolv"])
    assert args2.verify_dataset == "freesolv"


def test_dynamic_bmcsl_cavity_free_energy():
    """
    Verifies that compute_bmcsl_cavity_free_energy accurately models Scaled Particle Theory
    and BMCSL hard-sphere cavity creation across solvents of differing sizes.
    """
    from dens_city.utils.materials import compute_bmcsl_cavity_free_energy
    from dens_city.utils.solvents import get_solvent_properties

    # Carbon tetrachloride (CCl4) effective diameter ~5.6 Å
    sig_ccl4 = 5.6

    # In water (small kinetic diameter ~2.8 Å, high density ~0.0333 Å^-3)
    dg_cav_water = compute_bmcsl_cavity_free_energy(
        sigma_solute=sig_ccl4,
        solvent_sigma=2.8,
        solvent_rho=0.0333,
        temp_k=298.15,
    )
    assert 10.0 < dg_cav_water < 20.0, f"Expected water cavitation ~15 kcal/mol, got {dg_cav_water}"

    # In hexadecane (large kinetic diameter ~9.75 Å, low number density ~0.00206 Å^-3)
    hex_props = get_solvent_properties("HEXADECANE")
    rho_hex = (hex_props.density_g_cm3 * 6.02214076e23) / (hex_props.molecular_weight * 1e24)
    dg_cav_hex = compute_bmcsl_cavity_free_energy(
        sigma_solute=sig_ccl4,
        solvent_sigma=hex_props.kinetic_diameter_a,
        solvent_rho=rho_hex,
        temp_k=298.15,
    )
    # Using hexadecane's dynamic diameter reduces cavitation work by ~8-10 kcal/mol relative to water,
    # eliminating the +6.29 kcal/mol outlier error observed when water's radius was forced.
    assert 4.0 < dg_cav_hex < 9.0, f"Expected hexadecane cavitation ~6.5 kcal/mol, got {dg_cav_hex}"
    assert dg_cav_water - dg_cav_hex > 4.0, "Hexadecane cavitation must be significantly lower than water"


def test_adaptive_gated_cooperative_pooling():
    """
    Verifies that EGNN cooperative pooling is adaptively gated by the solvent's
    hydrogen-bond capacity (alpha + beta), preventing unphysical cooperative bridging in alkanes.
    """
    import numpy as np
    from tinygrad import Tensor

    from dens_city.boltzmann.egnn import EGNNForceField

    egnn = EGNNForceField(num_layers=2, hidden_dim=64, n_particles=32)

    # 1 molecule, 4 atoms (e.g. ethylene glycol fragment)
    x = Tensor(np.random.normal(0, 1, (1, 32, 3)).astype(np.float32))
    z = Tensor(np.array([[6, 8, 8, 1] + [0] * 28], dtype=np.float32))
    mask = Tensor(np.array([[1.0, 1.0, 1.0, 1.0] + [0.0] * 28], dtype=np.float32)).reshape(1, 32, 1)

    # In hexane (alpha=0, beta=0 -> hbond_capacity = 0.0): gate must be 0.0
    _, _, _, coop_hexane, _ = egnn.compute_solvation_readouts(
        x=x,
        atomic_numbers=z,
        atom_mask=mask,
        solvent_hbond_capacity=0.0,
        return_global=True,
    )
    assert abs(float(coop_hexane.numpy()[0])) < 1e-6, "Cooperative correction must be strictly zero in hexane"

    # In water (alpha=1.17, beta=0.47 -> hbond_capacity = 1.64): gate retains magnitude
    # Set global_mlp final bias to non-zero to test gating
    egnn.global_mlp[2].bias.assign(Tensor([5.0])).realize()
    _, _, _, coop_water, _ = egnn.compute_solvation_readouts(
        x=x,
        atomic_numbers=z,
        atom_mask=mask,
        solvent_hbond_capacity=1.64,
        return_global=True,
    )
    coop_val = float(coop_water.numpy()[0])
    assert abs(coop_val) > 0.5, f"Cooperative correction in water must be non-zero, got {coop_val}"


def test_solvent_descriptors_and_krr_inference():
    """
    Verifies that solvent descriptors are physical and predict_krr_residual supports
    solvent-aware queries.
    """
    import numpy as np

    from dens_city.boltzmann.train_charges import predict_krr_residual
    from dens_city.utils.solvents import get_solvent_descriptors_vector, get_solvent_properties

    # Check 7D descriptor vectors for representative solvents
    for s_name in ["water", "HEXANE", "METHANOL", "BENZENE", "HEXADECANE"]:
        props = get_solvent_properties(s_name)
        assert props.molar_volume_cm3_mol > 0.0
        assert 2.5 < props.kinetic_diameter_a < 15.0
        vec = get_solvent_descriptors_vector(s_name)
        assert vec.shape == (7,)
        assert np.all(np.isfinite(vec))

    # Test predict_krr_residual with solvent string and vector
    z_dummy = np.zeros(384, dtype=np.float32)
    d_dummy = np.zeros(6, dtype=np.float32)
    res_water = predict_krr_residual(z_dummy, d_dummy, s_solv="water")
    res_hex = predict_krr_residual(z_dummy, d_dummy, s_solv="HEXANE")
    assert isinstance(res_water, float)
    assert isinstance(res_hex, float)
