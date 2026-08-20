"""
Unit and physical validation tests for BoltzmannGenerator in dens_city.boltzmann.generator.
Verifies Reverse KL Divergence training, convergence to analytical Boltzmann distributions
in harmonic potential wells, and cDFT prior-informed flow training.
"""

import math
import numpy as np
import pytest
from tinygrad import Tensor
from dens_city.boltzmann.bijectors import RealNVPFlow
from dens_city.boltzmann.prior import CDFTBaseDistribution
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.materials import MaterialLoader


def test_harmonic_well_analytical_gaussian_convergence():
    """
    Validates that training a RealNVP BoltzmannGenerator on a 1D/2D harmonic potential well
    U(x) = 0.5 * k * ||x||^2 converges to the analytical Boltzmann Gaussian distribution N(0, k_B T / k).
    """
    k_spring = 2.0
    temp_k = 1.0  # k_B T in Kelvin
    target_var = temp_k / k_spring  # 0.5

    def harmonic_energy(x: Tensor) -> Tensor:
        return 0.5 * k_spring * (x * x).sum(axis=-1)

    flow = RealNVPFlow(dim=2, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=harmonic_energy,
        temperature_k=temp_k,
        learning_rate=0.03,
    )

    # Train for 80 steps
    losses = generator.train(steps=80, batch_size=256)

    # Loss must decrease
    assert losses[-1] < losses[0], f"Final loss {losses[-1]} must be lower than initial loss {losses[0]}"

    # Sample 50,000 points from trained flow
    samples = generator.sample(n_samples=50_000)
    samples_np = samples.numpy()

    mean_val = np.mean(samples_np, axis=0)
    var_val = np.var(samples_np, axis=0)

    # Validate mean ~ 0 and variance ~ target_var
    assert np.allclose(mean_val, 0.0, atol=0.10), f"Mean {mean_val} != 0.0"
    assert np.allclose(var_val, target_var, rtol=0.18), f"Variance {var_val} != {target_var}"


def test_boltzmann_generator_with_cdft_prior():
    """
    Validates end-to-end BoltzmannGenerator training informed by the cDFT spatial prior and MicroscopicEnergy.
    """
    # Load Argon
    argon = MaterialLoader.load_material("argon")
    box_size = (30.0, 30.0, 20.0)

    # Synthetic cDFT profile
    n_grid = 64
    rho_syn = np.sin(np.linspace(0.2, math.pi - 0.2, n_grid)) + 0.1

    prior = CDFTBaseDistribution(
        rho_z=rho_syn,
        l_z=20.0,
        box_size_xy=(30.0, 30.0),
        n_particles=2,
    )

    energy_fn = MicroscopicEnergy(
        material=argon,
        box_size=box_size,
        r_cut=10.0,
    )

    dim = 2 * 3  # 2 particles in 3D = 6 degrees of freedom
    flow = RealNVPFlow(dim=dim, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        prior=prior,
        temperature_k=300.0,
        learning_rate=0.01,
    )

    # Run 10 training steps
    losses = generator.train(steps=10, batch_size=16)
    assert len(losses) == 10
    assert np.all(np.isfinite(losses)), "Losses must be finite numbers"

    # Sample configurations
    confs = generator.sample(n_samples=5)
    assert confs.shape == (5, 2, 3)

    # Evaluate generated log probability
    lp = generator.log_prob(confs)
    assert lp.shape == (5,)
    assert np.all(np.isfinite(lp.numpy()))
