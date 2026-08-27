"""
Unit test suite for Batched L-BFGS Quasi-Newton Geometry Optimizer.
"""

from __future__ import annotations

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.lbfgs import BatchedLBFGS, LBFGSResult
from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.sampler import SwarmCandidateSampler
from dens_city.swarm.spec_loader import SwarmSpecLoader


def test_batched_lbfgs_quadratic_bowl():
    """
    Validates that BatchedLBFGS rapidly minimizes an anisotropic quadratic bowl
    U(x) = 0.5 * sum_i (w_i * x_i^2) across batch items.
    """
    B, N, D = 4, 8, 3
    dim = N * D

    # Diagonal Hessian weights with condition number ~ 100
    weights_np = np.linspace(1.0, 100.0, dim, dtype=np.float32)[np.newaxis, :]  # (1, 3N)
    weights_tensor = Tensor(weights_np)

    def quadratic_energy_fn(x: Tensor) -> Tensor:
        # x is (B, N, 3)
        x_flat = x.reshape(B, -1)
        # 0.5 * sum(w_i * x_i^2)
        return 0.5 * (weights_tensor * (x_flat**2)).sum(axis=-1)

    # Initial random points away from minimum
    np.random.seed(42)
    x_init = np.random.randn(B, N, D).astype(np.float32)

    lbfgs = BatchedLBFGS(m=5, max_iter=50, grad_tol=1e-3, lr=1.0, verbose=False)
    res: LBFGSResult = lbfgs.minimize(quadratic_energy_fn, x_init)

    # Verify energy decreased dramatically towards 0.0
    assert np.all(res.final_energies < res.initial_energies * 1e-4)
    assert np.all(res.final_energies < 1e-3)
    assert np.all(res.converged)


def test_batched_lbfgs_molecular_clash_untangling():
    """
    Verifies that un-relaxed molecular conformations sampled from C-FFI
    have their potential energy monotonically decreased into negative/stable minima.
    """
    spec_path = "tests/data/conjugated_oled_semiconductors.yaml"
    spec_data = SwarmSpecLoader.load_yaml(spec_path)
    _ = SwarmSpecLoader.derive_target_spec(spec_data)
    policy = MolecularSwarmPolicy(obs_size=88, hidden_size=256)
    sampler = SwarmCandidateSampler(policy=policy, spec_path=spec_path, num_envs=4)

    batch = sampler.sample_candidates(total_candidates=4, max_rollout_steps=5000)
    mol_batch = batch.slice_molecular_batch(0, 4, batch_size=4)
    energy_fn = MicroscopicEnergy(material=mol_batch, pad_to_128=True)

    coords_chunk = np.zeros((4, batch.n_particles, 3), dtype=np.float32)
    coords_chunk[:4] = batch.coords[:4]

    lbfgs = BatchedLBFGS(m=6, max_iter=40, grad_tol=1e-3, lr=1.0, verbose=False)
    res = lbfgs.minimize(energy_fn, coords_chunk, atom_mask=mol_batch.atom_mask)

    # Verify all 4 molecules relaxed to lower energy and remain within physical bounds
    for i in range(4):
        assert res.final_energies[i] <= res.initial_energies[i]
        # Eliminates nuclear collapse (must be physically bounded > -500 kcal/mol, i.e. > -250,000 K)
        assert res.final_energies[i] > -250000.0, f"Energy {res.final_energies[i]} K suffered Coulomb collapse!"
        # Energy history should show monotonic or stable decrease
        assert res.energy_history[-1][i] <= res.energy_history[0][i]


def test_batched_lbfgs_convergence_masking():
    """
    Validates that converged molecules stop receiving updates while unconverged molecules
    continue stepping.
    """
    B, N, D = 2, 4, 3

    # Molecule 0 starts at minimum (x=0), Molecule 1 starts slightly perturbed (x=0.5)
    x_init = np.zeros((B, N, D), dtype=np.float32)
    x_init[1] = 0.5

    def mock_energy_fn(x: Tensor) -> Tensor:
        x_flat = x.reshape(B, -1)
        return (x_flat**2).sum(axis=-1)

    lbfgs = BatchedLBFGS(m=4, max_iter=25, grad_tol=1e-3, lr=0.5, verbose=False)
    res = lbfgs.minimize(mock_energy_fn, x_init)

    # Molecule 0 should have taken 0 effective displacement
    mol0_disp = np.linalg.norm(res.x_relaxed[0] - x_init[0])
    assert mol0_disp < 1e-5

    # Molecule 1 should have moved from 0.5 to near 0.0
    mol1_final_dist = np.linalg.norm(res.x_relaxed[1])
    assert mol1_final_dist < 1e-2
    assert res.converged[0]
    assert res.converged[1]


def test_batched_lbfgs_autodiff_parity():
    """
    Verifies that reverse-mode autograd Cartesian forces match numerical finite-difference gradients.
    """
    B, N, D = 1, 2, 3
    x_np = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)

    def simple_potential(x: Tensor) -> Tensor:
        diff = x[:, 0, :] - x[:, 1, :]
        r_sq = (diff * diff).sum(axis=-1)
        return (r_sq - 4.0) ** 2

    lbfgs = BatchedLBFGS()
    atom_mask_3d = np.ones((B, N, D), dtype=np.float32)
    _, g_analytic = lbfgs._eval_energy_and_grad(simple_potential, x_np.reshape(B, -1), atom_mask_3d, (B, N, D))

    # Numerical finite difference
    eps = 1e-4
    g_num = np.zeros_like(g_analytic)
    for i in range(N * D):
        x_plus = x_np.reshape(B, -1).copy()
        x_plus[0, i] += eps
        u_plus = simple_potential(Tensor(x_plus.reshape(B, N, D))).numpy()[0]

        x_minus = x_np.reshape(B, -1).copy()
        x_minus[0, i] -= eps
        u_minus = simple_potential(Tensor(x_minus.reshape(B, N, D))).numpy()[0]

        g_num[0, i] = (u_plus - u_minus) / (2.0 * eps)

    # Parity comparison: relative error < 1e-2 in float32
    np.testing.assert_allclose(g_analytic, g_num, rtol=1e-2, atol=1e-2)
