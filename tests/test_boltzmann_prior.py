"""
Unit and physical validation tests for CDFTBaseDistribution in dens_city.boltzmann.prior.
Verifies inverse-CDF 1D sampling overlap with cDFT equilibrium profiles (R^2 > 0.99),
transverse spatial uniformity, and exact base probability log-likelihoods.
"""

import math

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.prior import CDFTBaseDistribution
from dens_city.cdft import TinyCDFT
from dens_city.utils.materials import MaterialLoader


def test_cdft_prior_histogram_r2_overlap():
    """
    Validates that 1,000,000 particles sampled from the CDFTBaseDistribution
    produce a 1D Z-density histogram overlapping the cDFT rho(z) profile with R^2 > 0.99.
    """
    argon = MaterialLoader.load_material("argon")
    n_grid = 128
    slit_width_a = 20.0
    cdft = TinyCDFT(argon, n_grid=n_grid, slit_width_a=slit_width_a)
    result = cdft.solve(steps=50, verbose=False)

    rho_np = result["rho"]
    dz = cdft.dz_val

    prior = CDFTBaseDistribution(
        rho_z=rho_np,
        l_z=slit_width_a,
        box_size_xy=(30.0, 30.0),
        n_particles=1,
    )

    # Sample 1,000,000 particles
    n_samples = 1_000_000
    samples = prior.sample(n_samples=n_samples)  # (1000000, 1, 3)
    z_coords = samples[..., 2].numpy().flatten()

    # Histogram matching cDFT grid
    z_edges = np.linspace(0.0, slit_width_a, n_grid + 1)
    counts, _ = np.histogram(z_coords, bins=z_edges)

    # Normalized sampled density
    rho_sampled = (counts / (n_samples * dz)) * prior.total_mass

    # Compute R^2 against cDFT target density
    ss_res = np.sum((rho_np - rho_sampled) ** 2)
    ss_tot = np.sum((rho_np - np.mean(rho_np)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    assert r2 > 0.99, f"Histogram R^2 {r2:.6f} <= 0.99"


def test_cdft_prior_xy_uniformity():
    """
    Validates that sampled X and Y positions are uniformly distributed over [0, Lx] x [0, Ly].
    """
    rho_dummy = np.ones(64, dtype=np.float64)
    lx, ly, lz = 25.0, 35.0, 15.0
    prior = CDFTBaseDistribution(rho_z=rho_dummy, l_z=lz, box_size_xy=(lx, ly), n_particles=1)

    samples = prior.sample(n_samples=50_000).numpy()
    x = samples[..., 0].flatten()
    y = samples[..., 1].flatten()

    # Mean must be ~ Lx / 2 and Ly / 2
    assert math.isclose(np.mean(x), 0.5 * lx, rel_tol=0.05)
    assert math.isclose(np.mean(y), 0.5 * ly, rel_tol=0.05)

    # Standard deviation of uniform(0, L) is L / sqrt(12)
    expected_std_x = lx / math.sqrt(12.0)
    expected_std_y = ly / math.sqrt(12.0)
    assert math.isclose(np.std(x), expected_std_x, rel_tol=0.05)
    assert math.isclose(np.std(y), expected_std_y, rel_tol=0.05)


def test_cdft_prior_log_prob_consistency():
    """
    Validates that log_prob evaluates the exact theoretical formula:
    log p_0 = sum_{i=1}^N [ -ln(Lx * Ly) + ln(rho(z_i)) - ln(int rho dz) ]
    """
    n_grid = 100
    lz = 20.0
    lx, ly = 30.0, 30.0
    area = lx * ly
    z_centers = np.linspace(0.1, lz - 0.1, n_grid)
    # Synthetic density profile with spatial structure
    rho_syn = np.sin(z_centers * (math.pi / lz)) + 0.2
    dz = lz / n_grid
    total_mass = np.sum(rho_syn * dz)

    prior = CDFTBaseDistribution(
        rho_z=rho_syn,
        l_z=lz,
        box_size_xy=(lx, ly),
        n_particles=2,
    )

    # Particle 1 at z=5.0, Particle 2 at z=15.0
    test_pos = Tensor([[[10.0, 10.0, 5.0], [20.0, 20.0, 15.0]]])  # (1, 2, 3)
    lp = prior.log_prob(test_pos)

    # Analytical values
    rho1 = np.interp(5.0, z_centers, rho_syn)
    rho2 = np.interp(15.0, z_centers, rho_syn)
    exact_lp = (
        -math.log(area) + math.log(rho1) - math.log(total_mass) - math.log(area) + math.log(rho2) - math.log(total_mass)
    )

    assert math.isclose(lp.item(), exact_lp, rel_tol=1e-3), f"Log prob {lp.item()} != exact {exact_lp}"


def test_cdft_prior_batched_sampling():
    """
    Validates batched sampling and log-probability evaluation for multi-particle configurations (B, N, 3).
    """
    rho_dummy = np.ones(50, dtype=np.float64)
    prior = CDFTBaseDistribution(rho_z=rho_dummy, l_z=10.0, box_size_xy=(20.0, 20.0), n_particles=4)

    B = 8
    samples = prior.sample(n_samples=B)
    assert samples.shape == (B, 4, 3)

    lp = prior.log_prob(samples)
    assert lp.shape == (B,)
    assert np.all(np.isfinite(lp.numpy()))
