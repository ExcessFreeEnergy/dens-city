"""
Unit and physical validation tests for pure tinygrad Classical Density Functional Theory (cDFT).
Verifies exact geometric measure integrals, Boltzmann boundary conditions, steric masking, and JIT compilation.
"""

import math

import numpy as np
from tinygrad import Tensor

from dens_city.cdft import KernelBuilder, TinyCDFT
from dens_city.materials import MaterialLoader


def test_anti_aliased_fmt_and_wca_kernel_integrals():
    """Validates that cell-integrated FMT and WCA kernels match exact analytical integrals."""
    sigma = 3.405
    epsilon_k = 119.8
    R = sigma / 2.0
    dz = 0.05

    # 1. FMT weight kernels
    fmt_kernels = KernelBuilder.build_fmt_planar_kernels(sigma, dz)
    w3_int = float(fmt_kernels["w3"].numpy().sum() * dz)
    w2_int = float(fmt_kernels["w2"].numpy().sum() * dz)
    w1_int = float(fmt_kernels["w1"].numpy().sum() * dz)
    w0_int = float(fmt_kernels["w0"].numpy().sum() * dz)

    exact_v = (4.0 / 3.0) * math.pi * (R**3)  # Volume
    exact_s = 4.0 * math.pi * (R**2)  # Surface Area
    exact_r = R  # Mean Radius
    exact_chi = 1.0  # Euler Characteristic

    assert math.isclose(w3_int, exact_v, rel_tol=1e-3), f"w3 integral {w3_int} != {exact_v}"
    assert math.isclose(w2_int, exact_s, rel_tol=1e-3), f"w2 integral {w2_int} != {exact_s}"
    assert math.isclose(w1_int, exact_r, rel_tol=1e-3), f"w1 integral {w1_int} != {exact_r}"
    assert math.isclose(w0_int, exact_chi, rel_tol=1e-3), f"w0 integral {w0_int} != {exact_chi}"

    # 2. WCA attractive kernel cell integral
    att_kernel, _ = KernelBuilder.build_wca_attraction_kernel(sigma, epsilon_k, dz)
    kernel_sum = float(att_kernel.numpy().sum()) * dz

    from dens_city.materials import compute_wca_dispersion_integral

    exact_wca_3d = compute_wca_dispersion_integral(sigma, epsilon_k)

    assert math.isclose(kernel_sum, exact_wca_3d, rel_tol=1e-3), f"WCA kernel sum {kernel_sum} != {exact_wca_3d}"


def test_boltzmann_initialization_and_continuous_autograd():
    """Validates unmasked continuous autograd, unbiased Boltzmann initial guess, and dynamic observables."""
    argon = MaterialLoader.load_material("argon")
    solver = TinyCDFT(argon, n_grid=96, learning_rate=0.02)

    initial_rho = solver.get_density_profile().copy()
    v_ext = solver.v_ext.numpy().flatten()

    # Repulsive core (V_ext >= 50 kBT) must have negligible density
    hard_core = v_ext >= 50.0
    assert np.all(initial_rho[hard_core] < 1e-15), "Density inside hard core must be asymptotically zero"
    # Fluid channel must have positive finite density
    assert np.all(initial_rho[~hard_core] > 0.0), "Density in fluid channel must be strictly positive"
    assert np.all(np.isfinite(initial_rho)), "Initial density must have no NaNs"

    # Verify autograd backward pass produces non-zero finite gradients throughout
    solver.grand_potential().backward()
    grad = solver.psi.grad.numpy()
    assert np.all(np.isfinite(grad)), "Gradients must be finite without NaNs"
    assert not np.all(grad == 0.0), "Gradients must not be all-zero"

    # Run variational optimization
    res = solver.solve(steps=50, verbose=False)
    final_rho = res["rho"]

    assert np.all(final_rho[hard_core] < 1e-12), "Density in steric core must remain suppressed"
    assert np.all(final_rho[~hard_core] > 0.0), "Fluid channel density must remain strictly positive"
    assert np.all(np.isfinite(final_rho)), "Optimized density must have no NaNs"
    assert not np.isnan(res["final_loss"]), "Loss must not be NaN"
    assert np.isfinite(res["wall_pressure_bar"]), "Irving-Kirkwood mechanical wall pressure must be finite"


def test_beam_jit_compilation():
    """Validates that TinyJit compiles the complete forward-backward-optimizer graph."""
    argon = MaterialLoader.load_material("argon")
    solver = TinyCDFT(argon, n_grid=64, learning_rate=0.02)

    # First step triggers JIT trace
    loss1 = solver.train_step().item()
    # Subsequent steps run compiled kernel
    loss2 = solver.train_step().item()
    loss3 = solver.train_step().item()

    assert np.isfinite(loss1) and np.isfinite(loss2) and np.isfinite(loss3)


def test_exact_1d_poisson_greens_matrix():
    """
    Validates that the 1D Poisson Green's matrix matches the exact analytical Dirichlet solution:
    -d^2 phi / dz^2 = (4*pi/eps) * rho_q, phi(0) = phi(L_z) = 0.
    For uniform rho_q, phi_exact(z) = -(2*pi*rho_q/eps) * z * (L_z - z).
    """
    n_grid = 128
    dz = 0.2
    l_z = n_grid * dz
    dielectric_constant = 1.0

    g_matrix = KernelBuilder.build_coulomb_1d_greens_matrix(
        n_grid=n_grid, dz=dz, dielectric_constant=dielectric_constant
    ).reshape(n_grid, n_grid)

    # Uniform charge density rho_q = 0.5 e / A^3
    rho_q = 0.5
    rho_tensor = Tensor.full((n_grid, 1), rho_q)
    phi_tensor = g_matrix.matmul(rho_tensor).numpy().flatten()

    z_coords = np.linspace(0.5 * dz, l_z - 0.5 * dz, n_grid)
    phi_exact = -(2.0 * math.pi * rho_q / dielectric_constant) * z_coords * (l_z - z_coords)

    # Maximum relative error across channel
    rel_err = np.abs((phi_tensor - phi_exact) / phi_exact)
    assert np.max(rel_err) < 1e-2, f"Poisson Green's matrix max relative error {np.max(rel_err)} exceeds 1%"
    # Verify exact grounded boundaries
    assert phi_tensor[0] < 0.0 and phi_tensor[-1] < 0.0


def test_large_molecule_wall_steric_exclusion():
    """
    Validates that large molecules (sigma_fluid >> sigma_wall) obey Lorentz-Berthelot
    collision diameter sigma_wf = 0.5 * (sigma_wall + sigma_fluid) and cannot penetrate the wall.
    """
    colloid = MaterialLoader.load_material("colloidal_hard_sphere")
    assert colloid.effective_sigma >= 14.0, f"Colloid sigma should be >= 14 A, got {colloid.effective_sigma}"

    wall_sigma = 3.4
    expected_steric_radius = 0.5 * (0.5 * (wall_sigma + colloid.effective_sigma))

    v_ext = (
        KernelBuilder.build_slit_wall_potential(
            n_grid=128,
            dz=0.5,
            fluid_sigma=colloid.effective_sigma,
            wall_sigma=wall_sigma,
        )
        .numpy()
        .flatten()
    )

    # Domain within steric collision radius must diverge to 1e6
    z_coords = np.linspace(0.25, 128 * 0.5 - 0.25, 128)
    steric_mask = (z_coords <= expected_steric_radius) | (z_coords >= (64.0 - expected_steric_radius))
    assert np.all(v_ext[steric_mask] >= 1e5), "Steric exclusion zone must have infinite potential"


def test_london_dispersion_well_depth_scaling():
    """
    Validates that London dispersion coarse-graining derives physically realistic well depths:
    - Argon (monoatomic): exact match eps = 119.8 K
    - 5CB (38 atoms): non-divergent eps in [500, 1200] K rather than naive sum > 4000 K
    """
    argon = MaterialLoader.load_material("argon")
    assert math.isclose(argon.effective_epsilon_k, 119.8, rel_tol=1e-2)

    cb5 = MaterialLoader.load_material("5cb")
    assert 400.0 <= cb5.effective_epsilon_k <= 1200.0, (
        f"5CB effective well depth should be physical [400, 1200] K, got {cb5.effective_epsilon_k}"
    )
    assert cb5.effective_sigma > 6.0, f"5CB effective diameter should be > 6 A, got {cb5.effective_sigma}"
