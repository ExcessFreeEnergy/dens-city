"""
End-to-end integration tests for multi-material ingestion and cDFT solving across test_data/.
Verifies pure parser parameter derivation and gradient descent on all fluids without hardcoding.
"""

import pytest
from dens_city.materials import MaterialLoader
from dens_city.cdft import TinyCDFT


def test_list_all_available_materials():
    """Verifies that all test_data molecular datasets are discovered."""
    mats = MaterialLoader.list_available_materials()
    assert len(mats) >= 20, f"Expected at least 20 materials in test_data, found {len(mats)}: {mats}"


def test_all_materials_have_distinct_derived_parameters():
    """Verifies that each material has distinct parameters calculated directly from its geometry and force field."""
    mats = MaterialLoader.list_available_materials()
    loaded = [MaterialLoader.load_material(name) for name in mats]

    sigmas = [m.effective_sigma for m in loaded]
    epsilons = [m.effective_epsilon_k for m in loaded]
    densities = [m.bulk_density_a3 for m in loaded]

    # Must have high diversity across distinct fluids (not all identical fallback constants)
    assert len(set(round(s, 2) for s in sigmas)) >= 10, "Effective sigmas must reflect distinct molecular sizes"
    assert len(set(round(e, 1) for e in epsilons)) >= 10, "Effective epsilons must reflect distinct molecular energies"
    assert len(set(round(d, 5) for d in densities)) >= 10, "Bulk densities must reflect distinct molecular packings"


@pytest.mark.parametrize(
    "mat_name",
    [
        "argon",
        "methane",
        "nitrogen",
        "carbon_dioxide",
        "water",
        "benzene",
        "5cb",
    ],
)
def test_pure_gradient_descent_solve(mat_name: str):
    """Tests loading and variational gradient descent optimization for distinct materials."""
    mat = MaterialLoader.load_material(mat_name)
    assert mat.effective_sigma > 0.0
    assert mat.effective_epsilon_k >= 0.0
    assert mat.bulk_density_a3 > 0.0

    solver = TinyCDFT(mat, n_grid=64, slit_width_a=35.0, learning_rate=0.02)
    res = solver.solve(steps=40, verbose=False)

    assert res["wall_pressure_bar"] > 0.0
    assert len(res["rho"]) == 64
    assert res["peak_density"] > 0.0
