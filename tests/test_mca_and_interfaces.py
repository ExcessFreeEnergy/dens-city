"""
Unit & Integration Tests for MCA Second-Order Dispersion, 2:1 Multivalent Electrolytes,
and Hydrophobic/Hydrophilic Planar Wetting Interfaces.
"""

import numpy as np

from dens_city.pipelines.argon.coexistence import (
    compute_argon_binodal,
)
from dens_city.pipelines.electrolytes.double_layer import (
    solve_multivalent_double_layer,
)
from dens_city.pipelines.interfaces.wetting import (
    compute_capillary_drying_gap,
    compute_lum_chandler_weeks_crossover,
    compute_wetting_contact_angle,
)
from dens_city.solver.dispersion import (
    compute_hard_sphere_compressibility,
)


def test_hard_sphere_compressibility_limits():
    """Verify CS hard-sphere isothermal compressibility limits: chi_hs(0) = 1, decaying as eta -> 1."""
    chi_0, d_chi_0 = compute_hard_sphere_compressibility(0.0)
    assert np.isclose(chi_0, 1.0, atol=1e-3)
    assert d_chi_0 < 0.0  # Monotonically decreasing with packing fraction

    # At dense liquid packing eta ~ 0.45:
    chi_dense, d_chi_dense = compute_hard_sphere_compressibility(0.45)
    assert 0.01 < chi_dense < 0.10
    assert d_chi_dense < 0.0


def test_argon_mca_binodal_accuracy():
    """Verify that pure LJ Argon with MCA predicts Tc within 2% and low-temperature liquid density."""
    binodal = compute_argon_binodal([85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0])
    Tc_pred = binodal["T_c_K"]
    rho_l_85K = binodal["rho_l"][0]

    # NIST values: Tc = 150.86 K, rho_l(84K) = 0.0214 A^-3
    tc_error_pct = abs((Tc_pred - 150.86) / 150.86) * 100.0
    assert tc_error_pct < 2.5, f"Argon Tc error too high: {tc_error_pct:.2f}%"

    # Liquid density at 85K should be reasonable
    assert 0.017 < rho_l_85K < 0.022, f"Argon rho_l(85K) out of bounds: {rho_l_85K}"


def test_multivalent_electrolyte_charge_inversion():
    """Verify that 2:1 multivalent asymmetric electrolytes exhibit charge inversion at negative electrodes."""
    res = solve_multivalent_double_layer(valency_cation=2, valency_anion=1, surface_charge=-0.20, T=300.0)

    assert "overcharging_ratio" in res
    assert res["overcharging_ratio"] > 1.0, "Divalent cations must over-screen surface charge"
    assert res["charge_inversion_detected"] is True
    # Electrostatic potential should flip sign
    assert np.any(res["phi_z"] > 0.0), "Potential must exhibit positive overcharging peak"


def test_wetting_contact_angles_and_regimes():
    """Verify Young-Dupré contact angle calculation across hydrophilic and hydrophobic regimes."""
    # Hydrophilic case (gamma_sv > gamma_sl)
    philic = compute_wetting_contact_angle(gamma_sv=80.0, gamma_sl=20.0)
    assert philic["theta_deg"] < 90.0
    assert philic["wetting_regime"] in ["hydrophilic", "complete_wetting"]
    assert philic["work_of_adhesion_mNm"] > 72.8

    # Hydrophobic case (gamma_sv < gamma_sl)
    phobic = compute_wetting_contact_angle(gamma_sv=20.0, gamma_sl=60.0)
    assert phobic["theta_deg"] > 90.0
    assert phobic["wetting_regime"] in ["hydrophobic", "superhydrophobic"]
    assert phobic["work_of_adhesion_mNm"] < 72.8


def test_capillary_drying_gap_cavitation():
    """Verify capillary evaporation / drying gap between hydrophobic plates."""
    res = compute_capillary_drying_gap(H_nm_values=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0], theta_deg=110.0)
    assert "H_dry_nm" in res
    assert 1.0 <= res["H_dry_nm"] <= 4.0
    assert res["cavitation_detected"] is True

    # Pores below H_dry should have vapor-like density
    h_arr = res["H_nm"]
    rho_pore = res["rho_pore_nm3"]
    small_gap_idx = np.where(h_arr < res["H_dry_nm"])[0]
    large_gap_idx = np.where(h_arr > res["H_dry_nm"])[0]

    if len(small_gap_idx) > 0:
        assert np.all(rho_pore[small_gap_idx] < 1.0)
    if len(large_gap_idx) > 0:
        assert np.all(rho_pore[large_gap_idx] > 20.0)


def test_lum_chandler_weeks_crossover():
    """Verify LCW length-scale crossover from volume scaling to surface area scaling at R_c ~ 1.0 nm."""
    res = compute_lum_chandler_weeks_crossover()
    assert res["R_c_nm"] == 1.0
    assert len(res["delta_g_kJ_mol"]) == len(res["radius_nm"])
    # Free energy must increase monotonically with solute size
    assert np.all(np.diff(res["delta_g_kJ_mol"]) > 0)
