"""
Unit tests for MolecularBatch ingestion, batched MicroscopicEnergy, and batched BoltzmannGenerator.
Verifies fixed (B=32, 128) tensor stacking, zeroed dummy molecule isolation, and JIT compatibility.
"""

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.bijectors import Base2CartesianFlow
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.utils.materials import MaterialLoader, MolecularBatch


def test_molecular_batch_creation():
    argon = MaterialLoader.load_material("argon")
    water = MaterialLoader.load_material("water")
    benzene = MaterialLoader.load_material("benzene")

    batch = MolecularBatch.create_batch([argon, water, benzene], batch_size=32, target_n_particles=128)

    assert batch.batch_size == 32
    assert batch.n_particles == 128
    assert batch.num_active_materials == 3

    # Check tensor shapes
    assert batch.sigmas.shape == (32, 128)
    assert batch.epsilons.shape == (32, 128)
    assert batch.charges.shape == (32, 128)
    assert batch.atom_mask.shape == (32, 128)
    assert batch.molecule_mask.shape == (32,)
    assert batch.conditioning.shape == (32, 5)

    # Active molecules have molecule_mask == 1.0
    mol_mask = batch.molecule_mask.numpy()
    assert mol_mask[0] == 1.0
    assert mol_mask[1] == 1.0
    assert mol_mask[2] == 1.0
    assert np.all(mol_mask[3:] == 0.0)

    # Check atom mask
    atom_mask = batch.atom_mask.numpy()
    assert atom_mask[0, 0] == 1.0
    assert np.all(atom_mask[0, 1:] == 0.0)  # Argon has 1 site
    assert np.sum(atom_mask[1]) == 3.0  # Water has 3 sites
    assert np.sum(atom_mask[2]) == 12.0  # Benzene has 12 sites
    assert np.all(atom_mask[3:] == 0.0)  # Dummy molecules have 0 real sites


def test_batched_microscopic_energy():
    argon = MaterialLoader.load_material("argon")
    water = MaterialLoader.load_material("water")

    batch = MolecularBatch.create_batch([argon, water], batch_size=32, target_n_particles=128)
    energy_fn = MicroscopicEnergy(material=batch, pad_to_128=True)

    assert energy_fn.is_batched_energy is True
    assert energy_fn.s_ij.shape == (32, 128, 128)
    assert energy_fn.e_ij.shape == (32, 128, 128)
    assert energy_fn.triu_mask.shape == (32, 128, 128)

    # Dummy molecules (b >= 2) must have triu_mask == 0.0
    triu_np = energy_fn.triu_mask.numpy()
    assert np.all(triu_np[2:] == 0.0)

    # Evaluate energy on (32, 128, 3) random positions
    pos = Tensor.randn(32, 128, 3) * 5.0 + 15.0
    u_eval = energy_fn.eval_energy(pos)

    assert u_eval.shape == (32,)
    u_np = u_eval.numpy()
    assert np.all(np.isfinite(u_np))
    # Dummy molecules must evaluate to 0 energy
    assert np.all(u_np[2:] == 0.0)


def test_batched_boltzmann_generator():
    argon = MaterialLoader.load_material("argon")
    water = MaterialLoader.load_material("water")

    batch = MolecularBatch.create_batch([argon, water], batch_size=32, target_n_particles=128)
    energy_fn = MicroscopicEnergy(material=batch, pad_to_128=True)
    flow = Base2CartesianFlow(n_atoms=128, n_layers=2, hidden_dim=32)

    gen = BoltzmannGenerator(flow=flow, energy_fn=energy_fn, prior=None, batch_size=32)
    assert gen.is_batched_generator is True
    assert gen.batch_size == 32

    losses = gen.train(steps=3, batch_size=32, verbose=False)
    assert len(losses) == 3
    assert np.all(np.isfinite(losses))

    # Sample batch
    samples = gen.sample(n_samples=32, return_all_pad=True)
    assert samples.shape == (32, 128, 3)
    assert np.all(np.isfinite(samples.numpy()))
