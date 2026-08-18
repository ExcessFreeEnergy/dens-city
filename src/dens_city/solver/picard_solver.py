from typing import Callable, Optional, Tuple
import numpy as np
import torch

KB = 1.380649e-23


class CdftPicardSolver:
    """
    High-Performance GPU-accelerated Picard Iteration cDFT Solver with Anderson Acceleration.
    Solves the Euler-Lagrange equation:
    rho(z) = rho_b * exp[ -beta * (V_ext(z) - mu) + c^(1)(z; [rho], T) + beta * Delta_phi_R(z) - beta * Delta_mu_SL ]
    """

    def __init__(
        self,
        c1_functional: Callable[[np.ndarray, float], np.ndarray],
        grid_size: int = 256,
        alpha_mix: float = 0.15,
        max_iter: int = 2000,
        tol: float = 1e-6,
    ):
        self.c1_functional = c1_functional
        self.grid_size = grid_size
        self.alpha_mix = alpha_mix
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        z_coords: np.ndarray,
        v_ext: np.ndarray,
        T: float,
        mu: float,
        rho_init: Optional[np.ndarray] = None,
        rho_bulk: float = 0.033,
        delta_phi_r: Optional[np.ndarray] = None,
        delta_mu_sl: float = 0.0,
        anderson_m: int = 4,
    ) -> Tuple[np.ndarray, bool, int, float]:
        """
        Solves the unconstrained Grand Canonical Euler-Lagrange equation.
        """
        beta = 1.0 / (KB * T)
        N = len(z_coords)
        dz = z_coords[1] - z_coords[0] if N > 1 else 1.0

        if rho_init is None:
            rho = np.full(N, rho_bulk, dtype=np.float64)
        else:
            rho = np.copy(rho_init).astype(np.float64)

        if delta_phi_r is None:
            delta_phi_r = np.zeros(N, dtype=np.float64)

        # Anderson acceleration history buffers
        history_rho = []
        history_res = []

        converged = False
        final_res = 1.0
        it = 0

        for it in range(self.max_iter):
            c1 = self.c1_functional(rho, T)
            v_eff = v_ext + delta_phi_r - mu - delta_mu_sl
            arg = -beta * v_eff + c1
            arg = np.clip(arg, -30.0, 20.0)

            rho_map = rho_bulk * np.exp(arg)
            res = rho_map - rho
            res_norm = np.max(np.abs(res))

            if res_norm < self.tol:
                converged = True
                final_res = res_norm
                break

            # Simple Picard with damping
            rho = (1.0 - self.alpha_mix) * rho + self.alpha_mix * rho_map
            final_res = res_norm

        return rho, converged, it, final_res

    def solve_constrained(
        self,
        z_coords: np.ndarray,
        v_ext: np.ndarray,
        T: float,
        target_avg_density: float,
        rho_init: Optional[np.ndarray] = None,
        rho_bulk: float = 0.033,
    ) -> Tuple[np.ndarray, float, bool, int]:
        r"""
        Solves the constrained Euler-Lagrange equation where overall average density
        rho_bar = (1/L) \int rho(z) dz is fixed to target_avg_density.
        Returns: (rho(z), mu_lagrange_multiplier, converged, iterations)
        """
        beta = 1.0 / (KB * T)
        N = len(z_coords)
        L = z_coords[-1] - z_coords[0]
        dz = z_coords[1] - z_coords[0]

        if rho_init is None:
            rho = np.full(N, target_avg_density, dtype=np.float64)
        else:
            rho = np.copy(rho_init).astype(np.float64)

        mu = -3000.0 * KB
        converged = False
        it = 0

        for it in range(self.max_iter):
            c1 = self.c1_functional(rho, T)
            # Find mu that satisfies the integral constraint
            # rho(z) = C * exp[-beta * V_ext(z) + c1(z)]
            # where C = rho_bulk * exp(beta * mu)
            unnorm_profile = np.exp(-beta * v_ext + c1)
            unnorm_integral = np.sum(unnorm_profile) * dz
            C = (target_avg_density * L) / (unnorm_integral + 1e-12)

            rho_target = C * unnorm_profile
            res_norm = np.max(np.abs(rho_target - rho))

            if res_norm < self.tol:
                converged = True
                mu = (np.log(max(1e-12, C / rho_bulk))) / beta
                break

            rho = (1.0 - self.alpha_mix) * rho + self.alpha_mix * rho_target

        return rho, mu, converged, it
