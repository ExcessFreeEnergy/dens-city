"""
Unit and performance test suite for TinyJIT dynamic graph compilation in dens-city.
Validates numerical parity, dummy atom force masking, convergence, and GPU execution acceleration.
"""

from __future__ import annotations

import time

import numpy as np
from tinygrad import Tensor

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.boltzmann.energy import MicroscopicEnergy
from dens_city.boltzmann.lbfgs import BatchedLBFGS


def test_egnn_jit_numerical_parity():
    """
    Verifies that the TinyJit-compiled EGNN evaluator produces numerically identical
    potential energies and conservative forces to eager autograd execution.
    """
    B, N = 4, 16
    np.random.seed(42)
    x_np = np.random.randn(B, N, 3).astype(np.float32)
    z_np = np.random.randint(1, 10, size=(B, N)).astype(np.float32)
    mask_np = np.ones((B, N, 1), dtype=np.float32)
    mask_np[:, 12:] = 0.0  # Mask dummy atoms
    z_np[:, 12:] = 0.0
    mol_mask_np = np.ones(B, dtype=np.float32)

    egnn = EGNNForceField(num_layers=3, hidden_dim=64, max_atomic_number=128, n_particles=N)

    # 1. Eager execution
    x_eager = Tensor(x_np)
    u_eager, f_eager = egnn.compute_energy_and_forces(
        x=x_eager,
        atomic_numbers=Tensor(z_np),
        atom_mask=Tensor(mask_np),
        molecule_mask=Tensor(mol_mask_np),
    )
    u_eager_np = u_eager.numpy()
    f_eager_np = f_eager.numpy()

    # 2. JIT execution (with warmup)
    jit_eval = egnn.get_jit_evaluator()
    for _ in range(3):
        xt = Tensor(x_np)
        xt.requires_grad = True
        u_tmp, f_tmp = jit_eval(xt, Tensor(z_np), Tensor(mask_np), Tensor(mol_mask_np))
        u_tmp.numpy()
        f_tmp.numpy()

    xt = Tensor(x_np)
    xt.requires_grad = True
    u_jit, f_jit = jit_eval(xt, Tensor(z_np), Tensor(mask_np), Tensor(mol_mask_np))
    u_jit_np = u_jit.numpy()
    f_jit_np = f_jit.numpy()

    # Assert exact numerical parity
    np.testing.assert_allclose(u_jit_np, u_eager_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(f_jit_np, f_eager_np, rtol=1e-4, atol=1e-4)

    # Verify dummy atoms have zero force
    assert np.allclose(f_jit_np[:, 12:], 0.0)


def test_egnn_jit_speedup():
    """
    Verifies that TinyJit compilation provides significant throughput speedup over eager evaluation.
    """
    B, N = 8, 32
    np.random.seed(100)
    x_np = np.random.randn(B, N, 3).astype(np.float32)
    z_np = np.full((B, N), 6.0, dtype=np.float32)
    mask_np = np.ones((B, N, 1), dtype=np.float32)
    mol_mask_np = np.ones(B, dtype=np.float32)

    egnn = EGNNForceField(num_layers=4, hidden_dim=64, max_atomic_number=128, n_particles=N)
    jit_eval = egnn.get_jit_evaluator()

    # Warmup JIT
    for _ in range(3):
        xt = Tensor(x_np)
        xt.requires_grad = True
        u_w, f_w = jit_eval(xt, Tensor(z_np), Tensor(mask_np), Tensor(mol_mask_np))
        u_w.numpy()
        f_w.numpy()

    n_iters = 6

    # Eager benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        xt = Tensor(x_np)
        u_e, f_e = egnn.compute_energy_and_forces(
            x=xt,
            atomic_numbers=Tensor(z_np),
            atom_mask=Tensor(mask_np),
            molecule_mask=Tensor(mol_mask_np),
        )
        u_e.numpy()
        f_e.numpy()
    t_eager = (time.perf_counter() - t0) / n_iters

    # JIT benchmark
    t0 = time.perf_counter()
    for _ in range(n_iters):
        xt = Tensor(x_np)
        xt.requires_grad = True
        u_j, f_j = jit_eval(xt, Tensor(z_np), Tensor(mask_np), Tensor(mol_mask_np))
        u_j.numpy()
        f_j.numpy()
    t_jit = (time.perf_counter() - t0) / n_iters

    # JIT must be strictly faster
    assert t_jit < t_eager, f"JIT ({t_jit * 1000:.1f}ms) was not faster than eager ({t_eager * 1000:.1f}ms)"


def test_lbfgs_jit_parity_and_convergence():
    """
    Verifies that BatchedLBFGS with JIT compilation enabled converges to identical
    energy minima as non-JIT execution.
    """
    B, N = 4, 16
    np.random.seed(42)
    sigmas = [3.4] * N
    epsilons = [120.0] * N
    charges = [0.0] * N

    energy_fn = MicroscopicEnergy(
        sigmas=sigmas,
        epsilons=epsilons,
        charges=charges,
        pad_to_128=False,
        target_n_particles=N,
    )

    x_init = np.random.randn(B, N, 3).astype(np.float32) * 2.0

    # 1. Non-JIT L-BFGS
    lbfgs_eager = BatchedLBFGS(max_iter=30, lr=0.5, grad_tol=1e-3, use_jit=False, verbose=False)
    res_eager = lbfgs_eager.minimize(energy_fn, x_init.copy())

    # 2. JIT L-BFGS
    lbfgs_jit = BatchedLBFGS(max_iter=30, lr=0.5, grad_tol=1e-3, use_jit=True, verbose=False)
    res_jit = lbfgs_jit.minimize(energy_fn, x_init.copy())

    # Both must decrease energy monotonically
    assert np.all(res_eager.final_energies <= res_eager.initial_energies)
    assert np.all(res_jit.final_energies <= res_jit.initial_energies)

    # Parity: relaxed energies and coordinates should match closely
    np.testing.assert_allclose(res_jit.final_energies, res_eager.final_energies, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(res_jit.x_relaxed, res_eager.x_relaxed, rtol=1e-2, atol=1e-2)


def test_egnn_jit_partial_batch_padding():
    """
    Verifies that passing a partial batch padded with zeros to static JIT buffers
    correctly masks out dummy interactions and produces zero forces on dummy molecules.
    """
    B_static, N = 8, 16
    count_valid = 5  # 5 valid molecules, 3 padded ghost molecules

    np.random.seed(77)
    x_chunk = np.zeros((B_static, N, 3), dtype=np.float32)
    z_chunk = np.zeros((B_static, N), dtype=np.float32)
    mask_chunk = np.zeros((B_static, N, 1), dtype=np.float32)
    mol_mask_chunk = np.zeros(B_static, dtype=np.float32)

    x_chunk[:count_valid] = np.random.randn(count_valid, N, 3).astype(np.float32)
    z_chunk[:count_valid] = np.random.randint(1, 10, size=(count_valid, N)).astype(np.float32)
    mask_chunk[:count_valid] = 1.0
    mol_mask_chunk[:count_valid] = 1.0

    egnn = EGNNForceField(num_layers=3, hidden_dim=32, max_atomic_number=128, n_particles=N)
    jit_eval = egnn.get_jit_evaluator()

    x_buf = Tensor.zeros(B_static, N, 3).realize()
    z_buf = Tensor.zeros(B_static, N).realize()
    mask_buf = Tensor.zeros(B_static, N, 1).realize()
    mol_mask_buf = Tensor.zeros(B_static).realize()

    # In-place assign
    x_buf.grad = None
    x_buf.assign(Tensor(x_chunk)).realize()
    x_buf.requires_grad = True
    z_buf.assign(Tensor(z_chunk)).realize()
    mask_buf.assign(Tensor(mask_chunk)).realize()
    mol_mask_buf.assign(Tensor(mol_mask_chunk)).realize()

    u_t, f_t = jit_eval(x_buf, z_buf, mask_buf, mol_mask_buf)
    u_np = u_t.numpy()
    f_np = f_t.numpy()

    # Valid molecules must have non-zero energies and valid forces
    assert not np.isnan(u_np[:count_valid]).any()
    assert not np.isnan(f_np[:count_valid]).any()

    # Padded ghost molecules (5:8) must have strictly zero energy and zero forces
    assert np.allclose(u_np[count_valid:], 0.0)
    assert np.allclose(f_np[count_valid:], 0.0)


def test_lbfgs_jit_speedup():
    """
    Verifies that static-buffer JIT L-BFGS is significantly faster than eager execution.
    """
    B, N = 8, 32
    np.random.seed(55)
    energy_fn = MicroscopicEnergy(
        sigmas=[3.4] * N,
        epsilons=[120.0] * N,
        charges=[0.0] * N,
        pad_to_128=False,
        target_n_particles=N,
    )
    x_init = np.random.randn(B, N, 3).astype(np.float32)

    lbfgs_eager = BatchedLBFGS(max_iter=15, lr=0.5, use_jit=False, verbose=False)
    lbfgs_jit = BatchedLBFGS(max_iter=15, lr=0.5, use_jit=True, verbose=False)

    # Warmup JIT
    _ = lbfgs_jit.minimize(energy_fn, x_init.copy())

    t0 = time.perf_counter()
    _ = lbfgs_eager.minimize(energy_fn, x_init.copy())
    t_eager = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = lbfgs_jit.minimize(energy_fn, x_init.copy())
    t_jit = time.perf_counter() - t0

    assert t_jit < t_eager, f"JIT ({t_jit * 1000:.1f}ms) not faster than eager ({t_eager * 1000:.1f}ms)"
