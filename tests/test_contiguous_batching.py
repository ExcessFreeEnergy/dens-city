"""
Unit tests for in-memory material ingestion and contiguous array batch construction.
"""

import numpy as np
from tinygrad import Tensor

from dens_city.utils.materials import MaterialLoader, MolecularBatch


def test_material_loader_from_mol2_string():
    argon_mol2 = """@<TRIPOS>MOLECULE
argon
     1     0     1     0     0
SMALL
GAFF_CHARGES

CDFT Argon
@<TRIPOS>ATOM
      1 Ar         0.0000     0.0000     0.0000 ar            1 MOL     0.000000
@<TRIPOS>SUBSTRUCTURE
     1 MOL         1 TEMP              0 ****  ****    0 ROOT
"""
    mat = MaterialLoader.from_mol2_string(argon_mol2, identifier="argon_test")
    assert mat.name == "argon_test"
    assert mat.num_sites == 1
    assert mat.sites[0].sigma > 0.0
    assert mat.bulk_density_a3 > 0.0
    assert mat.bulk_mu != 0.0


def test_material_loader_from_raw_arrays_bypass():
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float32)
    sigmas = np.array([3.4, 3.4], dtype=np.float32)
    epsilons_k = np.array([120.0, 120.0], dtype=np.float32)
    charges = np.array([0.0, 0.0], dtype=np.float32)
    atomic_numbers = np.array([6, 6], dtype=np.int32)

    mat = MaterialLoader.from_raw_arrays(
        coords=coords,
        sigmas=sigmas,
        epsilons_k=epsilons_k,
        charges=charges,
        atomic_numbers=atomic_numbers,
        identifier="bypass_dimer",
    )
    assert mat.num_sites == 2
    assert mat.dimension_mode == "1D_ANGULAR"
    assert mat.effective_sigma > 0.0
    assert mat.bulk_density_a3 > 0.0


def test_molecular_batch_from_contiguous_arrays():
    B = 16
    N = 128
    sigmas = np.full((B, N), 3.4, dtype=np.float32)
    epsilons = np.full((B, N), 120.0, dtype=np.float32)
    charges = np.zeros((B, N), dtype=np.float32)
    atomic_numbers = np.full((B, N), 6, dtype=np.int32)
    atom_mask = np.ones((B, N), dtype=np.float32)
    molecule_mask = np.ones(B, dtype=np.float32)
    temp = np.full(B, 300.0, dtype=np.float32)
    rho = np.full(B, 0.02, dtype=np.float32)
    mu = np.full(B, -8.0, dtype=np.float32)
    slit = np.full(B, 40.0, dtype=np.float32)
    cond = np.zeros((B, 5), dtype=np.float32)

    batch = MolecularBatch.from_contiguous_arrays(
        sigmas=sigmas,
        epsilons=epsilons,
        charges=charges,
        atomic_numbers=atomic_numbers,
        atom_mask=atom_mask,
        molecule_mask=molecule_mask,
        temperature_k=temp,
        bulk_density_a3=rho,
        bulk_mu=mu,
        slit_width_a=slit,
        conditioning=cond,
    )

    assert batch.batch_size == B
    assert batch.n_particles == N
    assert batch.sigmas.shape == (B, N)
    assert batch.epsilons.shape == (B, N)
    assert batch.charges.shape == (B, N)
    assert batch.molecule_mask.shape == (B,)
    assert isinstance(batch.sigmas, Tensor)
