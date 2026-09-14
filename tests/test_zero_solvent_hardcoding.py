"""
Automated regression tests verifying zero solvent hardcoding, strict environmental
parameter requirements, dynamic temperature adjustment, and first-principles scaling.
"""

from pathlib import Path

import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.boltzmann.energy import EGNNMicroscopicEnergy
from dens_city.cdft.generalized_born import GeneralizedBornSolvation
from dens_city.server.models import CDFTThermoRequest, RunFullPipelineRequest
from dens_city.utils.materials import (
    MaterialLoader,
    compute_bmcsl_cavity_free_energy,
)
from dens_city.utils.pipeline import (
    MaterialPipelineTask,
    PipelineStatus,
    process_material_task,
)
from dens_city.utils.solvents import (
    get_solvent_descriptors_vector,
    get_solvent_dielectric,
)


def test_generalized_born_requires_dielectric():
    """
    Verifies that GeneralizedBornSolvation does not assume an aqueous dielectric (78.4)
    and strictly raises ValueError on energy evaluation if dielectric is omitted.
    """
    gb_unspecified = GeneralizedBornSolvation()
    assert gb_unspecified.dielectric_constant is None

    # Born radii computation is purely steric/geometric and does not require dielectric
    x = Tensor.randn(1, 4, 3)
    z = Tensor([[6, 1, 1, 1]], dtype=dtypes.float32)
    mask = Tensor.ones(1, 4, 1)
    q = Tensor([[-0.3, 0.1, 0.1, 0.1]], dtype=dtypes.float32)

    radii = gb_unspecified.compute_born_radii(x, z, mask)
    assert radii.shape == (1, 4, 1)

    # Free energy evaluation without dielectric MUST fail with ValueError
    with pytest.raises(ValueError, match="Solvent dielectric constant must be explicitly provided"):
        gb_unspecified.compute_solvation_free_energy(x, q, z, mask)

    # Calling with explicit dielectric constant succeeds
    dg_water = gb_unspecified.compute_solvation_free_energy(x, q, z, mask, dielectric_constant=78.4)
    assert np.isfinite(float(dg_water.numpy()[0]))

    dg_vacuum = gb_unspecified.compute_solvation_free_energy(x, q, z, mask, dielectric_constant=1.0)
    assert float(dg_vacuum.numpy()[0]) == 0.0


def test_compute_ensembled_solvation_readouts_requires_environment():
    """
    Verifies that compute_ensembled_solvation_readouts requires explicit dielectric_constant
    and temperature_k, rather than defaulting to ambient water.
    """
    ff = EGNNForceField(num_layers=1, hidden_dim=16, n_particles=16, load_default_weights=False)
    coords = Tensor.randn(1, 2, 16, 3)
    z = Tensor.full((1, 16), 6.0, dtype=dtypes.float32)
    mask = Tensor.ones(1, 16, 1)

    # Missing dielectric_constant raises ValueError
    with pytest.raises(ValueError, match="dielectric_constant must be explicitly provided"):
        ff.compute_ensembled_solvation_readouts(
            x_ensemble=coords,
            atomic_numbers=z,
            atom_mask=mask,
            temperature_k=298.15,
        )

    # Missing temperature_k raises ValueError
    with pytest.raises(ValueError, match="temperature_k must be explicitly provided"):
        ff.compute_ensembled_solvation_readouts(
            x_ensemble=coords,
            atomic_numbers=z,
            atom_mask=mask,
            dielectric_constant=78.4,
        )


def test_solvent_dielectric_dynamic_resolution():
    """
    Verifies that get_solvent_dielectric resolves dynamically, supports temperature scaling,
    and raises KeyError for unknown fluids when no default is provided.
    """
    # Water at 298.15 K
    eps_water = get_solvent_dielectric("water")
    assert 77.0 < eps_water < 80.0

    # Temperature adjustment: dielectric decreases at elevated temperatures
    eps_water_hot = get_solvent_dielectric("water", temp_k=373.15)
    assert eps_water_hot < eps_water - 10.0

    # Non-aqueous solvent
    eps_hexane = get_solvent_dielectric("hexane")
    assert 1.8 < eps_hexane < 2.0

    # Unknown fluid without default must raise KeyError
    with pytest.raises(KeyError):
        get_solvent_dielectric("non_existent_unregistered_fluid_12345")

    # Unknown fluid with explicit default returns default
    eps_default = get_solvent_dielectric("non_existent_unregistered_fluid_12345", default=4.5)
    assert eps_default == 4.5


def test_bmcsl_cavity_free_energy_requires_parameters():
    """
    Verifies that compute_bmcsl_cavity_free_energy and Material.compute_solvation_in_solvent
    enforce explicit physical parameters or structured solvent objects.
    """
    # Invalid non-positive parameter must raise ValueError
    with pytest.raises(ValueError):
        compute_bmcsl_cavity_free_energy(sigma_solute=3.5, solvent_sigma=-1.0, solvent_rho=0.03, temp_k=300.0)

    loader = MaterialLoader()
    methane = loader.load_material("methane")

    # Calling compute_solvation_in_solvent without arguments must raise ValueError
    with pytest.raises(ValueError, match="Explicit solvent parameters"):
        methane.compute_solvation_in_solvent()

    # Calling with explicit solvent object computes successfully
    dg_ethanol = methane.compute_solvation_in_solvent(solvent="ethanol", temp_k=300.0)
    assert np.isfinite(dg_ethanol)


def test_egnn_wall_potential_uses_fluid_density():
    """
    Verifies that EGNNMicroscopicEnergy uses explicit wall_fluid_density or material bulk density
    rather than hardcoded water density 0.0333.
    """
    energy_fn_1 = EGNNMicroscopicEnergy(wall_fluid_density=0.0100)
    assert abs(energy_fn_1.wall_fluid_density - 0.0100) < 1e-6

    energy_fn_2 = EGNNMicroscopicEnergy(wall_fluid_density=0.0500)
    assert abs(energy_fn_2.wall_fluid_density - 0.0500) < 1e-6

    # Prefactor must scale linearly with fluid density
    ratio = energy_fn_2.wall_prefactor / energy_fn_1.wall_prefactor
    np.testing.assert_allclose(ratio, 5.0, rtol=1e-5)


def test_pipeline_executes_vacuum_without_hydration(tmp_path: Path):
    """
    Verifies that MaterialPipelineTask defaults to vacuum and executes with zero solvation penalty.
    """
    task = MaterialPipelineTask(
        material_path_or_name="water",
        out_dir=str(tmp_path),
        temperature_k=300.0,
        pressure_bar=1.0,
        grid=64,
        cdft_steps=10,
        skip_bg=True,
    )
    assert task.solvent_name == "vacuum"
    assert task.dielectric_constant is None

    res = process_material_task(task)
    assert res is not None
    assert res.status in (PipelineStatus.SUCCESS.value, PipelineStatus.SUCCESS_CDFT_ONLY.value)
    assert res.solvent_name == "vacuum"
    assert res.solvent_dielectric == 1.0
    assert res.solvation_free_energy_kcal_mol == 0.0
    assert res.born_solvation_kcal_mol is None or res.born_solvation_kcal_mol == 0.0


def test_solvent_descriptors_continuous_scaling():
    """
    Verifies that get_solvent_descriptors_vector produces continuous, bounded, finite vectors
    without water-anchored normalization factors.
    """
    for s_name in ["water", "hexane", "ethanol", "benzene", "vacuum"]:
        vec = get_solvent_descriptors_vector(s_name)
        assert vec.shape == (7,)
        assert np.all(np.isfinite(vec))
        # Surface tension scaling gamma / (gamma + 25) must be in [0, 1)
        assert 0.0 <= vec[2] < 1.0


def test_server_defaults_to_vacuum():
    """
    Verifies that server request schemas default to vacuum rather than water.
    """
    req_pipe = RunFullPipelineRequest(target_spec={"name": "test"})
    assert req_pipe.solvent_id == "vacuum"

    req_thermo = CDFTThermoRequest(smiles_list=["C"])
    assert req_thermo.solvent_id == "vacuum"
