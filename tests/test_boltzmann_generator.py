"""
Unit and physical validation tests for BoltzmannGenerator in dens_city.boltzmann.generator.
Verifies Reverse KL Divergence training, convergence to analytical Boltzmann distributions
in harmonic potential wells, and cDFT prior-informed flow training.
"""

import math

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.bijectors import Base2CartesianFlow, CompositeFlow, RealNVPFlow
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.boltzmann.prior import CDFTBaseDistribution
from dens_city.utils.materials import MaterialLoader


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

    # Train for 100 steps
    losses = generator.train(steps=100, batch_size=256)

    # Moving window loss must decrease
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (
        f"Final 10-step average loss {np.mean(losses[-10:])} must be lower than initial 10-step average {np.mean(losses[:10])}"
    )

    # Sample 50,000 points from trained flow
    samples = generator.sample(n_samples=50_000)
    samples_np = samples.numpy()

    mean_val = np.mean(samples_np, axis=0)
    var_val = np.var(samples_np, axis=0)

    # Validate mean ~ 0 and variance ~ target_var
    assert np.allclose(mean_val, 0.0, atol=0.15), f"Mean {mean_val} != 0.0"
    assert np.allclose(var_val, target_var, rtol=0.25), f"Variance {var_val} != {target_var}"


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
        sigmas=[argon.effective_sigma, argon.effective_sigma],
        epsilons=[argon.effective_epsilon_k, argon.effective_epsilon_k],
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


def test_boltzmann_generator_with_base2_cartesian_flow():
    """
    Validates end-to-end BoltzmannGenerator with 4-channel Base2CartesianFlow.
    Ensures 100% dyadic factorable base-2 execution and automatic slicing of dummy sites.
    """
    water = MaterialLoader.load_material("water")
    energy_fn = MicroscopicEnergy(
        material=water,
        box_size=(30.0, 30.0, 30.0),
        r_cut=10.0,
        pad_to_power_of_2=True,
    )
    assert energy_fn.n_particles == 4
    assert energy_fn.n_real_particles == 3

    flow = Base2CartesianFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    assert flow.dim == 16  # 4 atoms * 4 channels = 16 (2^4)

    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=300.0,
        learning_rate=0.01,
    )

    losses = generator.train(steps=20, batch_size=16)
    assert len(losses) == 20
    assert np.all(np.isfinite(losses)), "Losses must be finite"

    # Sample configurations: must be automatically sliced to real 3 atoms
    samples = generator.sample(n_samples=8)
    assert samples.shape == (8, 3, 3), f"Samples shape {samples.shape} != (8, 3, 3)"
    assert np.all(np.isfinite(samples.numpy())), "Sampled coordinates must be finite"

    lp = generator.log_prob(samples)
    assert lp.shape == (8,)
    assert np.all(np.isfinite(lp.numpy()))


def test_boltzmann_generator_with_composite_flow():
    """
    Validates end-to-end BoltzmannGenerator training using CompositeFlow (RealNVP + ZMatrix)
    directly mapping noise (B, 3N-6) -> Internal Coordinates -> Cartesian (B, N, 3)
    and evaluating exact microscopic potential energy.
    """
    water = MaterialLoader.load_material("water")  # 3 atoms (O, H, H)
    n_atoms = len(water.sites)
    assert n_atoms == 3

    flow = CompositeFlow(n_atoms=n_atoms, n_layers=4, hidden_dim=32)
    assert flow.dim == 3  # 3N - 6 = 3

    energy_fn = MicroscopicEnergy(
        material=water,
        box_size=(30.0, 30.0, 30.0),
        r_cut=10.0,
        pad_to_power_of_2=False,
    )

    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=300.0,
        learning_rate=0.01,
    )

    # Train for 15 steps
    losses = generator.train(steps=15, batch_size=16)
    assert len(losses) == 15
    assert np.all(np.isfinite(losses)), "Losses must be finite numbers"

    # Sample 3D Cartesian configurations
    samples = generator.sample(n_samples=6)
    assert samples.shape == (6, 3, 3), f"Samples shape {samples.shape} != (6, 3, 3)"
    assert np.all(np.isfinite(samples.numpy())), "Sampled coordinates must be finite"

    # Evaluate exact log-probability on 3D configurations
    lp = generator.log_prob(samples)
    assert lp.shape == (6,), f"Log prob shape {lp.shape} != (6,)"
    assert np.all(np.isfinite(lp.numpy())), "Log probability must be finite"


def test_boltzmann_generator_with_power_of_2_padded_composite_flow():
    """
    Validates end-to-end BoltzmannGenerator with 3-site water padded to power-of-2 (N_pad = 4).
    Verifies that generator.sample automatically slices dummy sites back to (B, 3, 3).
    """
    water = MaterialLoader.load_material("water")
    energy_fn = MicroscopicEnergy(
        material=water,
        box_size=(30.0, 30.0, 30.0),
        r_cut=10.0,
        pad_to_power_of_2=True,
        e_high=1e4,
    )
    assert energy_fn.n_particles == 4
    assert energy_fn.n_real_particles == 3

    flow = CompositeFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=300.0,
        learning_rate=0.005,
    )

    losses = generator.train(steps=15, batch_size=16)
    assert len(losses) == 15
    assert np.all(np.isfinite(losses)), "Losses must be finite"

    # Sample configurations: must be automatically sliced to real 3 atoms
    samples = generator.sample(n_samples=6)
    assert samples.shape == (6, 3, 3), f"Samples shape {samples.shape} != (6, 3, 3)"
    assert np.all(np.isfinite(samples.numpy())), "Sampled coordinates must be finite"
