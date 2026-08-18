r"""
Automated Unit Tests for 10 Extreme Statistical Mechanics Edge Cases & Trapdoors.

Validates that dens-city solves:
1. Helium-4 Nuclear Quantum Effects (NQE) via Feynman-Hibbs
2. RTIL [BMIM][PF6] Camel-Shaped Capacitance & Steric Overscreening
3. Polyethylene (N=100) Wertheim TPT1 Connectivity & Entropic Depletion
4. Liquid Gallium Electron Gas Friedel Oscillations & Surface Tension
5. Water-Ethanol Non-Ideal VLE & Minimum-Boiling Azeotrope
6. SDS Surfactant Self-Assembly & Critical Micelle Concentration (CMC)
7. Hydrogen Fluoride 1D Chain / Hexamer (HF)_6 Vapor Compressibility
8. Binary Colloids Asakura-Oosawa Pure Entropic Depletion Demixing
9. Kob-Andersen 80/20 Supercooled Glass & Radial Distribution Splitting
10. SF6 Octahedral Steric Fluorine Shielding & Triple Point
"""

import numpy as np

from dens_city.pipelines.associating_1d.hf import run_hf_vapor_association_simulation
from dens_city.pipelines.azeotropes.water_ethanol import compute_water_ethanol_vle
from dens_city.pipelines.colloids.depletion import run_colloidal_depletion_simulation
from dens_city.pipelines.fluorinated.sf6 import compute_sf6_phase_boundaries
from dens_city.pipelines.glasses.kob_andersen import compute_kob_andersen_glass_structure
from dens_city.pipelines.ionic_liquids.rtil import compute_rtil_camel_capacitance, compute_rtil_charge_layering
from dens_city.pipelines.liquid_metals.gallium import compute_liquid_metal_friedel_profile
from dens_city.pipelines.polymers.polyethylene import run_polyethylene_confinement_simulation
from dens_city.pipelines.quantum.helium import run_helium_quantum_simulation
from dens_city.pipelines.surfactants.sds import compute_sds_micellization
from dens_city.solver.quantum import compute_feynman_hibbs_potential


def test_helium4_quantum_nqe_effective_potential():
    # 1. Feynman-Hibbs potential creates softer core than classical LJ
    r_arr = np.linspace(2.0, 6.0, 100)
    u_fh = compute_feynman_hibbs_potential(r_arr, T=5.2)
    assert np.min(u_fh) < 0.0  # Has attractive well

    # 2. Critical temperature matches NIST 5.195 K within 1%
    sim = run_helium_quantum_simulation()
    t_c = sim["T_c_K"]
    error_pct = abs(t_c - 5.1953) / 5.1953 * 100.0
    assert error_pct < 1.0, f"Helium-4 Tc error {error_pct:.2f}% exceeds 1.0%"
    assert sim["rho_l_2_5k_A3"] > sim["rho_c_A3"]


def test_rtil_camel_shaped_capacitance():
    cap_res = compute_rtil_camel_capacitance()
    # Must exhibit camel-shaped bimodal profile (peak > 1.4 * PZC)
    assert cap_res["is_camel_shaped"]
    assert cap_res["C_peak_uF_cm2"] > cap_res["C_pzc_uF_cm2"]

    # Alternating charge layering
    z = np.linspace(0.0, 30.0, 300)
    layer_res = compute_rtil_charge_layering(z)
    assert 0.7 < layer_res["layering_period_nm"] < 1.1
    assert len(layer_res["charge_density_profile"]) == 300


def test_polyethylene_tpt1_chain_and_depletion():
    sim = run_polyethylene_confinement_simulation(m_chain=100)
    # Radius of gyration for N=100 must be ~ 1.85 nm
    assert 1.6 < sim["R_g_nm"] < 2.2
    assert 2.2 < sim["depletion_thickness_nm"] < 3.0
    # Near-wall density must be depleted (rho(0) < 0.1 * rho_bulk)
    assert sim["rho_profile"][0] < 0.005


def test_liquid_gallium_friedel_oscillations_and_surface_tension():
    z = np.linspace(0.0, 25.0, 500)
    sim = compute_liquid_metal_friedel_profile(z)
    # Friedel oscillation wavelength ~ 2.56 A
    assert 2.3 < sim["lambda_F_A"] < 2.8
    # Surface tension must match NIST / experimental ~ 718 mN/m within 2%
    err_gamma = abs(sim["surface_tension_mN_m"] - 718.0) / 718.0 * 100.0
    assert err_gamma < 2.0, f"Gallium surface tension error {err_gamma:.2f}% exceeds 2.0%"


def test_water_ethanol_nonideal_azeotrope():
    vle = compute_water_ethanol_vle()
    # Azeotropic composition 89.3 mol% = 95.63 wt%
    assert np.isclose(vle["x_azeotrope_mol"], 0.893, atol=1e-3)
    assert np.isclose(vle["T_azeotrope_K"], 351.30, atol=0.5)

    # At azeotropic point, relative volatility \alpha_12 = 1.0 (vapor equals liquid composition)
    x_idx = np.argmin(np.abs(vle["x_ethanol"] - 0.893))
    assert np.isclose(vle["y_ethanol"][x_idx], vle["x_ethanol"][x_idx], atol=0.01)


def test_sds_surfactant_micellization_and_cmc():
    sds_res = compute_sds_micellization()
    # Critical Micelle Concentration CMC = 8.2 mM
    assert np.isclose(sds_res["CMC_mM"], 8.20, atol=1e-2)
    assert np.isclose(sds_res["aggregation_number_N"], 62.0, atol=1.0)
    assert 1.7 < sds_res["core_radius_nm"] < 2.0
    # Post-CMC surface tension plateaus
    assert np.isclose(sds_res["surface_tension_mN_m"][-1], 38.5, atol=0.5)


def test_hf_1d_associating_vapor_compressibility():
    hf_res = run_hf_vapor_association_simulation()
    # Gas compressibility factor Z < 0.5 due to cyclic hexamer formation
    assert hf_res["Z_at_1atm"] < 0.50
    assert np.isclose(hf_res["T_boiling_K"], 292.68, atol=0.5)
    assert np.isclose(hf_res["T_c_K"], 461.0, atol=1.0)


def test_colloidal_asakura_oosawa_depletion_demixing():
    sim = run_colloidal_depletion_simulation(R_colloid_nm=50.0, r_depletant_nm=5.0, eta_depletant=0.20)
    # Contact depletion well depth W_AO(0) = -1.5 * eta_d * (R/r + 2/3) = -3.20 k_B T
    assert np.isclose(sim["W_contact_kBT"], -3.20, atol=0.1)
    assert sim["is_phase_separated"] is True


def test_kob_andersen_glassy_dynamics_and_peak_splitting():
    glass_res = compute_kob_andersen_glass_structure(T=0.45)
    # Mode coupling temperature
    assert np.isclose(glass_res["T_MCT"], 0.435, atol=1e-3)
    assert glass_res["is_glassy_basin"] is True
    # First peak at r = 1.08 \sigma
    assert glass_res["first_peak_r"] == 1.08
    # Second peak split at r = 1.75 and r = 2.02 \sigma
    assert glass_res["split_peak_1_r"] == 1.75
    assert glass_res["split_peak_2_r"] == 2.02


def test_sf6_octahedral_fluorine_steric_shielding():
    sf6_res = compute_sf6_phase_boundaries()
    # Triple point T_t = 222.35 K (NIST exact)
    assert np.isclose(sf6_res["T_triple_K"], 222.35, atol=0.5)
    # Critical point T_c = 318.72 K (NIST exact)
    assert np.isclose(sf6_res["T_c_K"], 318.72, atol=0.5)
    assert len(sf6_res["rho_l"]) == len(sf6_res["rho_v"])
    assert np.all(sf6_res["rho_l"] > sf6_res["rho_v"])
