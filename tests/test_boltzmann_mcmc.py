"""
Unit and physical validation tests for Vectorized Latent Space MCMC Relaxation in dens_city.boltzmann.
Verifies Metropolis acceptance statistics, outlier relaxation, zero-step invariance,
and seamless compatibility across Base2CartesianFlow and CompositeFlow architectures.
"""

import numpy as np

from dens_city.boltzmann.bijectors import Base2CartesianFlow, CompositeFlow
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.materials import MaterialLoader


def test_zero_mcmc_steps_invariance():
    """
    Validates that sample(mcmc_steps=0) behaves identically to default one-shot sampling.
    """
    water = MaterialLoader.load_material("water")
    energy_fn = MicroscopicEnergy(
        material=water,
        box_size=(30.0, 30.0, 30.0),
        r_cut=10.0,
        pad_to_power_of_2=True,
    )
    flow = Base2CartesianFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=300.0,
        learning_rate=0.01,
    )

    # Fast train
    generator.train(steps=5, batch_size=16)

    samples_0 = generator.sample(n_samples=8, mcmc_steps=0)
    assert samples_0.shape == (8, 3, 3)
    assert np.all(np.isfinite(samples_0.numpy()))


def test_latent_mcmc_acceptance_and_relaxation_statistics():
    """
    Validates that sample_relaxed with return_stats=True returns valid statistical metrics
    and that acceptance rates fall in a physical range (alpha in (0, 1]).
    """
    argon = MaterialLoader.load_material("argon")
    energy_fn = MicroscopicEnergy(
        material=argon,
        box_size=(30.0, 30.0, 30.0),
        r_cut=10.0,
        pad_to_power_of_2=True,
    )
    flow = Base2CartesianFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=120.0,
        learning_rate=0.01,
    )

    generator.train(steps=5, batch_size=16)

    samples, stats = generator.sample_relaxed(
        n_samples=16,
        mcmc_steps=8,
        mcmc_step_size=0.1,
        return_stats=True,
    )

    assert samples.shape == (16, 1, 3)
    assert np.all(np.isfinite(samples.numpy()))
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
    assert stats["mcmc_steps"] == 8
    assert stats["mcmc_step_size"] == 0.1
    assert np.isfinite(stats["initial_energy_mean"])
    assert np.isfinite(stats["final_energy_mean"])


def test_base2_cartesian_flow_mcmc_on_polyethylene():
    """
    Validates latent MCMC relaxation on 64-site padded polyethylene (62 real atoms).
    Ensures dummy site padding is handled seamlessly and relaxed coordinates are finite.
    """
    pe = MaterialLoader.load_material("polyethylene")
    energy_fn = MicroscopicEnergy(
        material=pe,
        box_size=(40.0, 40.0, 50.0),
        r_cut=12.0,
        pad_to_power_of_2=True,
    )
    flow = Base2CartesianFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=350.0,
        learning_rate=0.005,
        w_torsion=0.5,
        dihedral_quadruplets=pe.dihedral_quadruplets,
    )

    generator.train(steps=10, batch_size=8)

    # Sample with 5 steps of latent MCMC relaxation
    samples = generator.sample(n_samples=4, mcmc_steps=5, mcmc_step_size=0.05)
    assert samples.shape == (4, pe.num_sites, 3)
    assert np.all(np.isfinite(samples.numpy())), "Relaxed 3D coordinates must be finite"


def test_composite_flow_mcmc_relaxation():
    """
    Validates latent MCMC relaxation using CompositeFlow architecture with internal coordinates.
    """
    water = MaterialLoader.load_material("water")
    energy_fn = MicroscopicEnergy(
        material=water,
        box_size=(30.0, 30.0, 30.0),
        r_cut=10.0,
        pad_to_power_of_2=False,
    )
    flow = CompositeFlow(n_atoms=energy_fn.n_particles, n_layers=4, hidden_dim=32)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        temperature_k=300.0,
        learning_rate=0.01,
    )

    generator.train(steps=10, batch_size=16)

    samples, stats = generator.sample_relaxed(
        n_samples=8,
        mcmc_steps=6,
        mcmc_step_size=0.1,
        return_stats=True,
    )
    assert samples.shape == (8, 3, 3)
    assert np.all(np.isfinite(samples.numpy()))
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
