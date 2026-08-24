"""
Unit tests for BatchedTinyCDFT.
Verifies batched grand potential minimization, grouped planar convolutions,
density profiles, and wall contact pressures across multiple materials in a single JIT pass.
"""

import numpy as np

from dens_city.cdft.cdft import BatchedTinyCDFT
from dens_city.utils.materials import MaterialLoader, MolecularBatch


def test_batched_tiny_cdft_execution():
    argon = MaterialLoader.load_material("argon")
    water = MaterialLoader.load_material("water")
    benzene = MaterialLoader.load_material("benzene")

    batch = MolecularBatch.create_batch([argon, water, benzene], batch_size=32, target_n_particles=128)
    batched_cdft = BatchedTinyCDFT(batch=batch, n_grid=128, learning_rate=0.02)

    # Solve 30 steps
    losses = batched_cdft.solve(steps=30, verbose=False)
    assert len(losses) == 30
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]  # Grand potential minimized

    # Profiles and physical observables
    profiles = batched_cdft.get_density_profiles()
    assert len(profiles) == 32
    assert profiles[0].shape == (128,)
    assert np.all(profiles[0] > 0)  # Positive density
    assert np.all(profiles[1] > 0)
    assert np.all(profiles[2] > 0)

    pressures = batched_cdft.get_wall_contact_pressures()
    assert len(pressures) == 32
    assert np.isfinite(pressures[0])
    assert np.isfinite(pressures[1])
    assert np.isfinite(pressures[2])
    # Dummy slots must have 0 pressure
    assert np.all(np.array(pressures[3:]) == 0.0)

    gammas = batched_cdft.get_excess_adsorptions()
    assert len(gammas) == 32
    assert np.isfinite(gammas[0])
    assert np.isfinite(gammas[1])
    assert np.isfinite(gammas[2])
    assert np.all(np.array(gammas[3:]) == 0.0)
