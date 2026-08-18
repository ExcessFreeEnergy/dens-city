import numpy as np

from dens_city.pipelines.co2.supercritical import compute_supercritical_crossovers
from dens_city.pipelines.electrolytes.double_layer import solve_electric_double_layer
from dens_city.pipelines.water.confinement import compute_confinement_isotherm


def test_water_confinement_pipeline():
    def c1_fn(rho, T):
        return -0.6 * (rho / 0.033)

    res = compute_confinement_isotherm(c1_fn, H_values=[10.0, 15.0, 20.0], T=300.0, grid_size=128)

    assert len(res["H"]) == 3
    assert len(res["P_eff"]) == 3
    assert len(res["Pi_disjoining"]) == 3
    assert not np.isnan(res["Pi_disjoining"]).any()


def test_co2_supercritical_pipeline():
    def torch_c1(rho, T):
        return -0.4 * (rho / 0.02)

    res = compute_supercritical_crossovers(
        torch_c1,
        temperatures=[320.0, 360.0],
        densities=[0.005, 0.010, 0.015, 0.020],
    )

    assert "widom_xi" in res
    assert "fisher_widom" in res
    assert len(res["widom_xi"]) == 2


def test_electrolyte_double_layer():
    def c1_fn(rho, T):
        return -0.3 * (rho / 0.005)

    res = solve_electric_double_layer(c1_fn, voltage=1.0, T=300.0, grid_size=128)

    assert len(res["rho_pos"]) == 128
    assert len(res["rho_neg"]) == 128
    assert res["total_charge"] != 0.0
