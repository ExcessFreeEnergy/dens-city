import numpy as np

from dens_city.pipelines.clay_pore.mineral import (
    compute_clay_swelling_pressure,
    make_montmorillonite_slit_potential,
)
from dens_city.pipelines.co2.supercritical import compute_supercritical_crossovers
from dens_city.pipelines.co2_water.mixture import (
    compute_competitive_pore_adsorption,
    compute_mutual_solubility,
    compute_solvation_free_energy,
)
from dens_city.pipelines.electrolytes.double_layer import solve_electric_double_layer
from dens_city.pipelines.liquid_crystals.nematic import (
    compute_isotropic_nematic_binodal,
    compute_nematic_director_profile,
)
from dens_city.pipelines.methane.shale import (
    compute_ch4_co2_gas_recovery_crossover,
    compute_methane_shale_isotherm,
)
from dens_city.pipelines.nitrogen.flue_gas import (
    compute_flue_gas_selectivity,
    compute_n2_orientational_isotherm,
)
from dens_city.pipelines.water.confinement import compute_confinement_isotherm


# 1. Water Pipeline Test
def test_water_confinement_pipeline():
    def c1_fn(rho, T):
        return -0.6 * (rho / 0.033)

    res = compute_confinement_isotherm(c1_fn, H_values=[10.0, 15.0, 20.0], T=300.0, grid_size=128)

    assert len(res["H"]) == 3
    assert len(res["P_eff"]) == 3
    assert len(res["Pi_disjoining"]) == 3
    assert not np.isnan(res["Pi_disjoining"]).any()


# 2. CO2 Pipeline Test
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


# 3. Electrolyte Double Layer Pipeline Test
def test_electrolyte_double_layer():
    def c1_fn(rho, T):
        return -0.3 * (rho / 0.005)

    res = solve_electric_double_layer(c1_fn, voltage=1.0, T=300.0, grid_size=128)

    assert len(res["rho_pos"]) == 128
    assert len(res["rho_neg"]) == 128
    assert res["total_charge"] != 0.0


# 4. CO2 / Water Binary Mixture Pipeline Test
def test_co2_water_mixture_pipeline():
    def dummy_water_c1(rho, T):
        return -0.5 * (rho / 0.033)

    delta_mu = compute_solvation_free_energy(dummy_water_c1, T=300.0)
    assert not np.isnan(delta_mu)

    sol_res = compute_mutual_solubility(T=310.0, P_atm=50.0)
    assert 0.0 < sol_res["x_CO2_liquid"] < 0.1
    assert 0.0 < sol_res["y_H2O_vapor"] < 0.1

    pore_res = compute_competitive_pore_adsorption(H=20.0, T=300.0, x_co2_feed=0.15)
    assert len(pore_res["rho_water"]) == 128
    assert len(pore_res["rho_co2"]) == 128
    assert pore_res["rho_water"].max() > 0.0
    assert pore_res["rho_co2"].max() > 0.0


# 5. Nitrogen Flue Gas Pipeline Test
def test_nitrogen_flue_gas_pipeline():
    sel_res = compute_flue_gas_selectivity(T=300.0, P_bar=1.0, y_co2=0.15, y_n2=0.85)
    assert sel_res["selectivity_CO2_N2"] > 1.0  # CO2 preferentially adsorbs over N2
    assert sel_res["x_CO2_adsorbed"] > sel_res["y_CO2_feed"]

    n2_isotherm = compute_n2_orientational_isotherm(None, H=20.0, T=298.15)
    assert len(n2_isotherm["S_order"]) == 64
    assert n2_isotherm["S_order"].min() < 0.0  # Planar quadrupolar alignment at walls


# 6. Methane Shale Gas Pipeline Test
def test_methane_shale_pipeline():
    shale_res = compute_methane_shale_isotherm([10.0, 20.0, 30.0], T=330.0)
    assert shale_res["excess_adsorption"].shape == (3, 5)
    assert np.all(shale_res["excess_adsorption"] >= 0.0)

    egr_res = compute_ch4_co2_gas_recovery_crossover(T=330.0)
    assert len(egr_res["recovery_efficiency"]) == 4
    assert np.all(egr_res["recovery_efficiency"] > 0.5)


# 7. Clay Pore Swelling Pipeline Test
def test_clay_pore_pipeline():
    z_coords, v_ext, dv_ext = make_montmorillonite_slit_potential(H=15.0, L_z=40.0)
    assert len(z_coords) == 128
    assert len(v_ext) == 128

    swell_res = compute_clay_swelling_pressure([9.5, 12.5, 15.5, 18.5, 25.0], T=298.15)
    assert len(swell_res["Pi_swell_MPa"]) == 5
    assert not np.isnan(swell_res["Pi_swell_MPa"]).any()


# 8. Liquid Crystals Pipeline Test
def test_liquid_crystals_pipeline():
    lc_homeotropic = compute_nematic_director_profile(None, H=30.0, anchoring_type="homeotropic")
    assert len(lc_homeotropic["S_order"]) == 64
    assert lc_homeotropic["S_order"].max() > 0.6  # Strong homeotropic alignment

    lc_planar = compute_nematic_director_profile(None, H=30.0, anchoring_type="planar")
    assert lc_planar["S_order"].min() < 0.0  # Planar alignment

    in_binodal = compute_isotropic_nematic_binodal()
    assert len(in_binodal["rho_isotropic"]) == 5
    assert np.all(in_binodal["rho_nematic"] > in_binodal["rho_isotropic"])
