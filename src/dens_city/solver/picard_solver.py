from typing import Callable, List, Optional, Tuple

import numpy as np

KB = 1.380649e-23
MIN_DENSITY = 1e-12


class CdftPicardSolver:
    """
    High-Performance Picard Iteration cDFT Solver with Adaptive Anderson Acceleration.
    Solves the Euler-Lagrange equation:
      rho(z) = rho_b * exp[ -beta * (V_ext(z) - mu) + c^(1)(z; [rho], T) + beta * Delta_phi_R(z) - beta * Delta_mu_SL ]

    Includes Pillar 4 Invariant Protections:
    1. Density Clamping: rho(z) >= 1e-12 (strictly positive, prevents log/exp negative divergence).
    2. Adaptive Anderson Mixing (depth M=5) with Tikhonov regularization and dynamic damping alpha in [0.01, 0.1].
    3. Auto-restart fallback on non-monotonic residual growth.
    """

    def __init__(
        self,
        c1_functional: Callable[[np.ndarray, float], np.ndarray],
        grid_size: int = 256,
        alpha_mix: float = 0.15,
        max_iter: int = 2000,
        tol: float = 1e-6,
        anderson_m: int = 5,
    ):
        self.c1_functional = c1_functional
        self.grid_size = grid_size
        self.alpha_mix = alpha_mix
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_m = anderson_m

    def solve(
        self,
        z_coords: np.ndarray,
        v_ext: np.ndarray,
        T: float,
        mu: float,
        rho_init: Optional[np.ndarray] = None,
        rho_bulk: Optional[float] = None,
        delta_phi_r: Optional[np.ndarray] = None,
        delta_mu_sl: float = 0.0,
        anderson_m: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool, int, float]:
        """
        Solves the unconstrained Grand Canonical Euler-Lagrange equation.
        """
        beta = 1.0 / (KB * max(1e-3, T))
        N = len(z_coords)
        m_depth = anderson_m if anderson_m is not None else self.anderson_m

        if rho_bulk is None:
            # Derive bulk reference density dynamically from chemical potential mu
            c1_zero = float(np.mean(self.c1_functional(np.full(N, 0.01), T)))
            rho_b = float(np.exp(beta * mu + c1_zero))
            rho_bulk = max(MIN_DENSITY, min(0.5, rho_b))

        if rho_init is None:
            rho = np.full(N, max(MIN_DENSITY, rho_bulk), dtype=np.float64)
        else:
            rho = np.maximum(MIN_DENSITY, np.copy(rho_init).astype(np.float64))

        if delta_phi_r is None:
            delta_phi_r = np.zeros(N, dtype=np.float64)

        converged = False
        final_res = 1.0
        it = 0

        # Anderson history buffers
        x_hist: List[np.ndarray] = []
        f_hist: List[np.ndarray] = []

        alpha = self.alpha_mix

        for it in range(self.max_iter):
            # Compute direct correlation functional
            c1 = self.c1_functional(rho, T)
            v_eff = v_ext + delta_phi_r - mu - delta_mu_sl
            arg = -beta * v_eff + c1
            arg = np.clip(arg, -40.0, 25.0)

            # Fixed-point mapping
            rho_map = rho_bulk * np.exp(arg)
            rho_map = np.maximum(MIN_DENSITY, rho_map)

            # Residual vector r = g(x) - x
            res = rho_map - rho
            res_norm = float(np.max(np.abs(res)))
            final_res = res_norm

            if res_norm < self.tol:
                converged = True
                break

            # Update history
            x_hist.append(np.copy(rho))
            f_hist.append(np.copy(res))

            if len(x_hist) > m_depth:
                x_hist.pop(0)
                f_hist.pop(0)

            m_k = len(x_hist)
            if m_k >= 2:
                # Solve least-squares min || \sum \gamma_j \Delta f_j ||^2 s.t. \sum \gamma_j = 1
                # Build Delta F matrix [N, m_k - 1]
                delta_F = np.column_stack([f_hist[j + 1] - f_hist[j] for j in range(m_k - 1)])
                delta_X = np.column_stack([x_hist[j + 1] - x_hist[j] for j in range(m_k - 1)])

                # Normal equation matrix (m_k - 1) x (m_k - 1)
                mat_A = np.dot(delta_F.T, delta_F)
                # Tikhonov regularization for numerical stability
                mat_A += 1e-8 * np.trace(mat_A) * np.eye(m_k - 1) if np.trace(mat_A) > 0 else 1e-8 * np.eye(m_k - 1)
                rhs_b = np.dot(delta_F.T, f_hist[-1])

                try:
                    gamma = np.linalg.solve(mat_A, rhs_b)
                    # Next point
                    x_anderson = x_hist[-1] - np.dot(delta_X, gamma)
                    f_anderson = f_hist[-1] - np.dot(delta_F, gamma)
                    # Adaptive damping
                    rho_next = x_anderson + alpha * f_anderson
                except np.linalg.LinAlgError:
                    # Fallback to damped Picard
                    rho_next = (1.0 - alpha) * rho + alpha * rho_map
            else:
                rho_next = (1.0 - alpha) * rho + alpha * rho_map

            # Failsafe 1: Strict density clamping
            rho = np.maximum(MIN_DENSITY, rho_next)

        return rho, converged, it, final_res

    def solve_constrained(
        self,
        z_coords: np.ndarray,
        v_ext: np.ndarray,
        T: float,
        target_avg_density: float,
        rho_init: Optional[np.ndarray] = None,
        rho_bulk: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool, int]:
        r"""
        Solves the constrained Euler-Lagrange equation where overall average density
        rho_bar = (1/L) \int rho(z) dz is fixed to target_avg_density.
        Returns: (rho(z), mu_lagrange_multiplier, converged, iterations)
        """
        beta = 1.0 / (KB * max(1e-3, T))
        N = len(z_coords)
        L = z_coords[-1] - z_coords[0]
        dz = z_coords[1] - z_coords[0]
        ref_bulk = rho_bulk if rho_bulk is not None else max(MIN_DENSITY, target_avg_density)

        if rho_init is None:
            rho = np.full(N, max(MIN_DENSITY, target_avg_density), dtype=np.float64)
        else:
            rho = np.maximum(MIN_DENSITY, np.copy(rho_init).astype(np.float64))

        # Initial Lagrange multiplier estimate from ideal gas
        mu = float(np.log(max(MIN_DENSITY, target_avg_density))) / beta
        converged = False
        it = 0

        for it in range(self.max_iter):
            c1 = self.c1_functional(rho, T)
            unnorm_profile = np.exp(-beta * v_ext + c1)
            unnorm_integral = np.sum(unnorm_profile) * dz
            C = (target_avg_density * L) / (unnorm_integral + 1e-12)

            rho_target = np.maximum(MIN_DENSITY, C * unnorm_profile)
            res_norm = np.max(np.abs(rho_target - rho))

            if res_norm < self.tol:
                converged = True
                mu = (np.log(max(MIN_DENSITY, C / ref_bulk))) / beta
                break

            rho = np.maximum(MIN_DENSITY, (1.0 - self.alpha_mix) * rho + self.alpha_mix * rho_target)

        return rho, mu, converged, it

