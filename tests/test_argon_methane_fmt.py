"""
Automated verification tests for Fundamental Measure Theory (FMT) and
First-Principles Pure Lennard-Jones Argon & TraPPE Methane pipelines without empirical patches.
"""

import numpy as np
import pytest

from dens_city.pipelines.argon.coexistence import (
    ARGON_EPSILON_K,
    ARGON_SIGMA,
    compute_argon_binodal,
    compute_argon_isotherms,
    compute_argon_pressure,
)
from dens_city.pipelines.methane.shale import (
    METHANE_EPSILON_K,
    METHANE_SIGMA,
    compute_methane_binodal,
    compute_methane_pressure,
)
from dens_city.solver.dispersion import (
    LennardJonesFMTDispersion1D,
    compute_barker_henderson_diameter,
    compute_planar_attractive_kernel,
)
from dens_city.solver.fmt import FundamentalMeasureTheory1D


def test_fmt_weighted_densities():
    fmt = FundamentalMeasureTheory1D(diameter=3.405)
    z = np.linspace(-20.0, 20.0, 256)
    rho = np.full_like(z, 0.02)  # Homogeneous fluid

    n0, n1, n2, n3, nv1, nv2 = fmt.compute_weighted_densities(z, rho)
    R = 3.405 / 2.0

    # In bulk homogeneous 3D fluid:
    # n3 = rho * (4/3) * pi * R^3 = eta
    # n2 = rho * 4 * pi * R^2
    # nv1 = nv2 = 0 (symmetric)
    expected_eta = 0.02 * (np.pi / 6.0) * (3.405**3)
    # Center of domain should match bulk value
    mid = len(z) // 2
    assert np.isclose(n3[mid], expected_eta, rtol=1e-2)
    assert np.abs(nv1[mid]) < 1e-4
    assert np.abs(nv2[mid]) < 1e-4

    c1_hs = fmt.compute_c1_hs(z, rho)
    assert not np.isnan(c1_hs).any()


def test_barker_henderson_diameter():
    # As T increases, effective hard sphere diameter d(T) decreases due to soft core penetration
    d_80 = compute_barker_henderson_diameter(ARGON_SIGMA, ARGON_EPSILON_K, T=80.0)
    d_150 = compute_barker_henderson_diameter(ARGON_SIGMA, ARGON_EPSILON_K, T=150.0)
    d_300 = compute_barker_henderson_diameter(ARGON_SIGMA, ARGON_EPSILON_K, T=300.0)

    assert d_80 > d_150 > d_300
    assert 3.0 < d_150 < 3.6


def test_argon_first_principles_binodal():
    # Pure Lennard-Jones Argon without empirical patches
    binodal = compute_argon_binodal([85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0])
    T_c = binodal["T_c_K"]

    # Must match NIST critical temperature 150.86 K within 2%
    error_pct = abs(T_c - 150.86) / 150.86 * 100.0
    assert error_pct < 2.0, f"Argon Tc error {error_pct:.2f}% exceeds 2.0% (predicted: {T_c:.1f} K)"

    # Liquid density at 85K must be positive and physically realistic
    assert 0.015 < binodal["rho_l"][0] < 0.025
    assert len(binodal["rho_v"]) == len(binodal["rho_l"])
    assert np.all(binodal["rho_l"] > binodal["rho_v"])


def test_methane_first_principles_binodal():
    # TraPPE United-Atom Methane without empirical patches
    binodal = compute_methane_binodal([110.0, 125.0, 140.0, 155.0, 170.0, 180.0])
    T_c = binodal["T_c_K"]

    # Must match NIST critical temperature 190.56 K within 3.5%
    error_pct = abs(T_c - 190.56) / 190.56 * 100.0
    assert error_pct < 3.5, f"Methane Tc error {error_pct:.2f}% exceeds 3.5% (predicted: {T_c:.1f} K)"

    assert 0.010 < binodal["rho_l"][0] < 0.020
    assert np.all(binodal["rho_l"] > binodal["rho_v"])


def test_argon_isotherms_monotonicity():
    # Subcritical isotherm (100K) has van der Waals loop; supercritical isotherm (220K) is strictly monotonic
    iso_res = compute_argon_isotherms([100.0, 220.0])
    p_100 = iso_res["isotherms"]["T_100K"]
    p_220 = iso_res["isotherms"]["T_220K"]

    # 220K should be strictly monotonic with density
    dp_220 = np.diff(p_220)
    assert np.all(dp_220 > 0)
