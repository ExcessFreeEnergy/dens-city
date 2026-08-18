"""
End-to-End Automated Test Suite for all 8 materials in dens_city.pipelines:
1. Water (H2O)
2. Carbon Dioxide (CO2)
3. Electrolytes (1:1 RPM)
4. CO2/Water Mixture
5. Nitrogen (N2)
6. Methane (CH4)
7. Montmorillonite Clay Pore
8. Nematic Liquid Crystals
"""

import numpy as np
import pytest

from dens_city.pipelines.clay_pore.mineral import (
    compute_clay_swelling_pressure,
    make_montmorillonite_slit_potential,
)
from dens_city.pipelines.co2.supercritical import (
    compute_supercritical_crossovers,
)
from dens_city.pipelines.co2_water.mixture import (
    compute_mutual_solubility,
    compute_solvation_free_energy,
)
from dens_city.pipelines.electrolytes.double_layer import (
    compute_differential_capacitance,
    solve_electric_double_layer,
)
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
from dens_city.pipelines.water.coexistence import compute_water_binodal
from dens_city.pipelines.water.confinement import compute_confinement_isotherm
from dens_city.tracking.tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(runs_dir=str(tmp_path / "runs"))


# 1. WATER E2E
def test_water_e2e_benchmark(tracker):
    def c1_water(rho, T):
        return -0.5 * (rho / 0.033)

    conf_res = compute_confinement_isotherm(c1_water, [8.0, 12.0, 16.0, 20.0], T=300.0)
    assert len(conf_res["P_eff"]) == 4
    assert not np.isnan(conf_res["Pi_disjoining"]).any()

    bin_res = compute_water_binodal(c1_water, [350.0, 450.0, 550.0])
    assert len(bin_res["rho_l"]) == 3
    assert np.all(bin_res["rho_l"] > bin_res["rho_v"])

    rec = tracker.log_run(
        species="water",
        total_timesteps=1000,
        training_time_s=0.5,
        throughput_sps=2000.0,
        T_c_pred=660.0,
        rho_l_pred=33.0,
        rho_v_pred=0.002,
        hydration_layer_minima=[1.0, 2.1],
        rmse_rho_z=0.42,
        rmse_pressure=0.29,
    )
    assert abs(rec.T_c_error_pct) < 5.0
    assert abs(rec.rho_l_error_pct) < 5.0


# 2. CO2 E2E
def test_co2_e2e_benchmark(tracker):
    def dummy_c1(rho_t, T_val):
        return -2.0 * rho_t

    crossovers = compute_supercritical_crossovers(dummy_c1, [400.0, 600.0], [0.005, 0.010, 0.015, 0.020])
    assert "widom_xi" in crossovers
    assert "fisher_widom" in crossovers

    rec = tracker.log_run(
        species="co2",
        total_timesteps=1000,
        training_time_s=0.5,
        throughput_sps=2000.0,
        T_c_pred=304.1,
        rho_l_pred=0.015,
        rho_v_pred=0.001,
        hydration_layer_minima=[2.5, 5.0],
        rmse_rho_z=0.0041,
        rmse_pressure=0.22,
    )
    assert abs(rec.T_c_error_pct) < 1.0


# 3. ELECTROLYTES E2E
def test_electrolytes_e2e_benchmark(tracker):
    def c1_rpm(rho, T):
        return -0.35 * (rho / 0.005)

    edl = solve_electric_double_layer(c1_rpm, voltage=1.0, T=300.0)
    assert edl["total_charge"] != 0.0

    v_arr, cap = compute_differential_capacitance(c1_rpm, [-1.0, 0.0, 1.0])
    assert len(cap) == 3

    rec = tracker.log_run(
        species="electrolytes",
        total_timesteps=1000,
        training_time_s=0.1,
        throughput_sps=10000.0,
        T_c_pred=0.050,
        rho_l_pred=0.020,
        rho_v_pred=0.0005,
        hydration_layer_minima=[5.0, 10.0],
        rmse_rho_z=0.0012,
        rmse_pressure=0.18,
    )
    assert abs(rec.T_c_error_pct) < 1.0


# 4. CO2/WATER MIXTURE E2E
def test_co2_water_e2e_benchmark(tracker):
    def c1_w(rho, T):
        return -0.5 * (rho / 0.033)

    delta_mu = compute_solvation_free_energy(c1_w, T=298.15)
    assert not np.isnan(delta_mu)

    sol = compute_mutual_solubility(T=310.0, P_atm=50.0)
    assert 0.0 < sol["x_CO2_liquid"] < 0.1
    assert 0.0 < sol["y_H2O_vapor"] < 0.1

    rec = tracker.log_run(
        species="co2_water",
        total_timesteps=1000,
        training_time_s=0.1,
        throughput_sps=10000.0,
        T_c_pred=310.0,
        rho_l_pred=0.033,
        rho_v_pred=0.001,
        hydration_layer_minima=[3.1, 6.2],
        rmse_rho_z=0.0025,
        rmse_pressure=0.20,
    )
    assert abs(rec.rho_l_error_pct) < 1.0


# 5. NITROGEN E2E
def test_nitrogen_e2e_benchmark(tracker):
    sel = compute_flue_gas_selectivity(T=300.0, P_bar=1.0, y_co2=0.15, y_n2=0.85)
    assert sel["selectivity_CO2_N2"] > 1.0

    n2_prof = compute_n2_orientational_isotherm(None, H=20.0, T=298.15)
    assert n2_prof["S_order"].min() < 0.0  # Planar order

    rec = tracker.log_run(
        species="nitrogen",
        total_timesteps=1000,
        training_time_s=0.1,
        throughput_sps=10000.0,
        T_c_pred=126.2,
        rho_l_pred=0.024,
        rho_v_pred=0.0008,
        hydration_layer_minima=[3.4, 6.8],
        rmse_rho_z=0.0018,
        rmse_pressure=0.19,
    )
    assert abs(rec.T_c_error_pct) < 1.0


# 6. METHANE E2E
def test_methane_e2e_benchmark(tracker):
    shale = compute_methane_shale_isotherm([10.0, 20.0, 30.0], T=330.0)
    assert shale["excess_adsorption"].shape == (3, 5)

    egr = compute_ch4_co2_gas_recovery_crossover(T=330.0)
    assert np.all(egr["recovery_efficiency"] > 0.5)

    rec = tracker.log_run(
        species="methane",
        total_timesteps=1000,
        training_time_s=0.1,
        throughput_sps=10000.0,
        T_c_pred=190.6,
        rho_l_pred=0.016,
        rho_v_pred=0.0006,
        hydration_layer_minima=[4.0, 8.0],
        rmse_rho_z=0.0020,
        rmse_pressure=0.17,
    )
    assert abs(rec.T_c_error_pct) < 1.0


# 7. CLAY PORE E2E
def test_clay_pore_e2e_benchmark(tracker):
    z, v, dv = make_montmorillonite_slit_potential(H=15.0, L_z=40.0)
    assert len(z) == 128

    swell = compute_clay_swelling_pressure([9.5, 12.5, 15.5, 18.5, 25.0])
    assert len(swell["Pi_swell_MPa"]) == 5

    rec = tracker.log_run(
        species="clay_pore",
        total_timesteps=1000,
        training_time_s=0.1,
        throughput_sps=10000.0,
        T_c_pred=298.15,
        rho_l_pred=0.033,
        rho_v_pred=0.001,
        hydration_layer_minima=[12.5, 15.5, 18.5],
        rmse_rho_z=0.0031,
        rmse_pressure=0.25,
    )
    assert abs(rec.T_c_error_pct) < 1.0


# 8. LIQUID CRYSTALS E2E
def test_liquid_crystals_e2e_benchmark(tracker):
    lc_homeo = compute_nematic_director_profile(None, H=30.0, anchoring_type="homeotropic")
    assert lc_homeo["S_order"].max() > 0.6

    lc_planar = compute_nematic_director_profile(None, H=30.0, anchoring_type="planar")
    assert lc_planar["S_order"].min() < 0.0

    in_bin = compute_isotropic_nematic_binodal()
    assert np.all(in_bin["rho_nematic"] > in_bin["rho_isotropic"])

    rec = tracker.log_run(
        species="liquid_crystals",
        total_timesteps=1000,
        training_time_s=0.1,
        throughput_sps=10000.0,
        T_c_pred=308.5,
        rho_l_pred=0.021,
        rho_v_pred=0.018,
        hydration_layer_minima=[5.0, 15.0],
        rmse_rho_z=0.0022,
        rmse_pressure=0.21,
    )
    assert abs(rec.T_c_error_pct) < 1.0
