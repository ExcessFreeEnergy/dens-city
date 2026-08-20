"""
End-to-end integration tests for multi-material ingestion and cDFT solving across test_data/.
"""

import pytest
from dens_city.materials import MaterialLoader
from dens_city.cdft import TinyCDFT


def test_list_all_available_materials():
    """Verifies that all test_data molecular datasets are discovered."""
    mats = MaterialLoader.list_available_materials()
    assert len(mats) >= 20, f"Expected at least 20 materials in test_data, found {len(mats)}: {mats}"
    assert "argon" in mats
    assert "water" in mats
    assert "methane" in mats
    assert "5cb" in mats


@pytest.mark.parametrize(
    "mat_name, expected_mode",
    [
        ("argon", "1D_SPHERICAL"),
        ("methane", "1D_SPHERICAL"),
        ("nitrogen", "1D_ANGULAR"),
        ("carbon_dioxide", "1D_ANGULAR"),
        ("5cb", "1D_ANGULAR"),
        ("water", "3D_MOLECULAR"),
        ("benzene", "3D_MOLECULAR"),
    ],
)
def test_material_classification_and_solve(mat_name: str, expected_mode: str):
    """Tests loading, dimension classification, and cDFT optimization for distinct materials."""
    mat = MaterialLoader.load_material(mat_name)
    assert mat.dimension_mode == expected_mode
    assert mat.effective_sigma > 0.0
    assert mat.effective_epsilon_k > 0.0

    solver = TinyCDFT(mat, n_grid=64, slit_width_a=30.0, learning_rate=0.03)
    res = solver.solve(steps=40, verbose=False)

    assert res["wall_pressure_bar"] > 0.0
    assert len(res["rho"]) == 64
    assert res["peak_density"] > 0.0
