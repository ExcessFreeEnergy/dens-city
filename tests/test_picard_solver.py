import numpy as np

from dens_city.solver.picard_solver import CdftPicardSolver
from dens_city.solver.thermo_integration import compute_bulk_pressure


def test_picard_solver_convergence():
    def c1_fn(rho, T):
        return -0.8 * (rho / 0.033)

    solver = CdftPicardSolver(c1_fn, grid_size=128, tol=1e-4)

    z_coords = np.linspace(0, 20.0, 128)
    v_ext = np.zeros(128)

    rho, converged, it, res = solver.solve(z_coords, v_ext, T=300.0, mu=-3000.0 * 1.380649e-23, rho_bulk=0.033)

    assert converged
    assert len(rho) == 128
    assert (rho > 0.0).all()


def test_thermodynamic_integration():
    def c1_fn(rho, T):
        return -0.5 * (rho / 0.033)

    p_bulk = compute_bulk_pressure(c1_fn, rho_bulk=0.033, T=300.0, L_z=20.0, grid_size=128)
    assert p_bulk > 0.0
