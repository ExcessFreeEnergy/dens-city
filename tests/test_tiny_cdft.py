"""
Unit and physical validation tests for pure tinygrad Classical Density Functional Theory (cDFT).
"""

import math
import numpy as np
import pytest
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import getenv

from dens_city.cdft import TinyCDFT
from dens_city.kernels import KernelBuilder
from dens_city.materials import MaterialLoader


def test_anti_aliased_fmt_kernel_integrals():
    """Validates that cell-integrated FMT kernels match exact geometric sphere integrals."""
    sigma = 3.405
    R = sigma / 2.0
    dz = 0.05
    kernels = KernelBuilder.build_fmt_planar_kernels(sigma, dz)

    # 1D integral of weight functions: \sum w_i * dz
    w3_int = float(kernels["w3"].numpy().sum() * dz)
    w2_int = float(kernels["w2"].numpy().sum() * dz)
    w1_int = float(kernels["w1"].numpy().sum() * dz)
    w0_int = float(kernels["w0"].numpy().sum() * dz)

    # Exact 3D sphere geometric measures
    exact_v = (4.0 / 3.0) * math.pi * (R**3)  # Volume
    exact_s = 4.0 * math.pi * (R**2)         # Surface Area
    exact_r = R                               # Mean Radius
    exact_chi = 1.0                           # Euler Characteristic

    assert math.isclose(w3_int, exact_v, rel_tol=1e-3), f"w3 integral {w3_int} != {exact_v}"
    assert math.isclose(w2_int, exact_s, rel_tol=1e-3), f"w2 integral {w2_int} != {exact_s}"
    assert math.isclose(w1_int, exact_r, rel_tol=1e-3), f"w1 integral {w1_int} != {exact_r}"
    assert math.isclose(w0_int, exact_chi, rel_tol=1e-3), f"w0 integral {w0_int} != {exact_chi}"


def test_positivity_guarantee_and_optimization():
    """Validates that density rho(z) > 0 strictly throughout optimization via exponential field."""
    argon = MaterialLoader.load_material("argon")
    solver = TinyCDFT(argon, n_grid=64, slit_width_a=30.0, learning_rate=0.03)

    initial_rho = solver.get_density_profile().copy()
    assert np.all(initial_rho > 0.0), "Initial density must be strictly positive"
    assert np.all(np.isfinite(initial_rho)), "Initial density must be finite"

    # Run optimization
    res = solver.solve(steps=60, verbose=False)
    final_rho = res["rho"]

    assert np.all(final_rho > 0.0), "Optimized density must be strictly positive (no Adam trap)"
    assert np.all(np.isfinite(final_rho)), "Optimized density must not have NaNs"
    assert not np.allclose(final_rho, initial_rho), "Density must adapt to wall potential"
    assert res["wall_pressure_bar"] > 0.0, "Wall pressure must be positive"


def test_beam_jit_compilation():
    """Validates that TinyJit compiles the complete forward-backward-optimizer graph."""
    argon = MaterialLoader.load_material("argon")
    solver = TinyCDFT(argon, n_grid=64, slit_width_a=25.0, learning_rate=0.02)

    # First step triggers JIT trace
    loss1 = solver.train_step().item()
    # Subsequent steps run compiled kernel
    loss2 = solver.train_step().item()
    loss3 = solver.train_step().item()

    assert np.isfinite(loss1) and np.isfinite(loss2) and np.isfinite(loss3)
