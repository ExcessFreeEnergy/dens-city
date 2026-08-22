"""
cDFT Spatial Prior (Base Distribution) for Normalizing Flows.
Bridges 1D mean-field Classical Density Functional Theory equilibrium density profiles
with many-body particle configuration sampling in 3D slit pores.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
from tinygrad import Tensor, dtypes


class CDFTBaseDistribution:
    """
    Base probability distribution derived from a 1D cDFT equilibrium density profile rho_cDFT(z).
    Samples particles uniformly in the transverse slit plane (X, Y) and according to the
    piecewise-linear Cumulative Distribution Function (CDF) of rho_cDFT(z) in the confined Z dimension.
    """

    def __init__(
        self,
        rho_z: Union[np.ndarray, Tensor],
        l_z: float,
        box_size_xy: Tuple[float, float] = (30.0, 30.0),
        n_particles: int = 1,
        n_grid: Optional[int] = None,
    ):
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        self.n_particles = n_particles
        self.l_z = float(l_z)
        self.lx, self.ly = float(box_size_xy[0]), float(box_size_xy[1])
        self.area = self.lx * self.ly

        if isinstance(rho_z, Tensor):
            rho_np = rho_z.numpy().astype(np.float64)
        else:
            rho_np = np.asarray(rho_z, dtype=np.float64)

        if len(rho_np) < 2:
            raise ValueError(f"rho_z profile must contain at least 2 grid points, got {len(rho_np)}")

        self.rho_np = np.maximum(rho_np, 1e-12)
        self.n_grid = len(self.rho_np)
        self.dz = self.l_z / float(self.n_grid)

        # Precompute 1D cumulative distribution function (CDF) for inverse transform sampling
        prob_mass = self.rho_np * self.dz
        self.total_mass = float(np.sum(prob_mass))
        if self.total_mass <= 1e-12:
            raise ValueError("cDFT density profile has zero or negative integral mass.")

        cdf = np.cumsum(prob_mass) / self.total_mass
        self.cdf = np.concatenate([[0.0], cdf])

        # Precompute tinygrad tensors for vectorized differentiable log_prob and pure tensor sampling
        self.rho_tensor = Tensor(self.rho_np.astype(np.float32), dtype=dtypes.float32)
        self.cdf_tensor = Tensor(self.cdf.astype(np.float32), dtype=dtypes.float32)

    def sample(self, n_samples: int = 1, as_4channel: bool = False) -> Tensor:
        """
        Draws N-particle configurations from the cDFT base distribution using pure on-device Tensor operations.
        Returns shape (n_samples, N, 4) if as_4channel=True else (n_samples, N, 3).
        """
        # Uniform sampling in transverse X and Y
        x = Tensor.rand(n_samples, self.n_particles) * self.lx
        y = Tensor.rand(n_samples, self.n_particles) * self.ly

        # Pure tinygrad vectorized inverse-CDF piecewise linear interpolation
        u = Tensor.rand(n_samples, self.n_particles)
        k = ((u.unsqueeze(-1) >= self.cdf_tensor).sum(axis=-1) - 1).clip(0, self.n_grid - 1)
        u0 = self.cdf_tensor[k]
        u1 = self.cdf_tensor[(k + 1).clip(0, self.n_grid)]
        du = (u1 - u0).maximum(1e-12)
        alpha = (u - u0) / du
        z = (k.cast(dtypes.float32) + alpha) * self.dz

        if as_4channel:
            w = Tensor.zeros(n_samples, self.n_particles, dtype=dtypes.float32)
            pts = Tensor.stack(x, y, z, w, dim=-1)
        else:
            pts = Tensor.stack(x, y, z, dim=-1)
        return pts if n_samples > 1 else pts.squeeze(0)

    def log_prob(self, pos: Tensor) -> Tensor:
        r"""
        Computes exact base distribution log probability:
        \log p_0(\vec{r}_1 \dots \vec{r}_N) = \sum_{i=1}^N \left[ -\ln(L_x L_y) + \ln(\rho_{\rm cDFT}(z_i)) - \ln\left(\int \rho_{\rm cDFT}(z) dz\right) \right]
        """
        if len(pos.shape) == 2:
            if pos.shape[-1] in (3, 4):
                pos_b = pos.unsqueeze(0)
            else:
                ch = 4 if pos.shape[-1] == self.n_particles * 4 else 3
                pos_b = pos.reshape(-1, self.n_particles, ch)
        elif len(pos.shape) == 3:
            pos_b = pos
        else:
            pos_b = pos.reshape(1, self.n_particles, -1)

        is_batched = len(pos.shape) >= 2
        z = pos_b[..., 2]  # (B, N)

        # 1D linear grid interpolation for rho(z)
        u = (z - 0.5 * self.dz) / self.dz
        u_clamped = u.clip(0.0, self.n_grid - 1 - 1e-5)
        k0 = u_clamped.floor().cast(dtypes.int32)
        k1 = (k0 + 1).clip(0, self.n_grid - 1)
        alpha = u_clamped - k0.cast(dtypes.float32)

        val0 = self.rho_tensor[k0]
        val1 = self.rho_tensor[k1]

        inside = (z >= 0.0) & (z <= self.l_z)
        rho_interp = (1.0 - alpha) * val0 + alpha * val1
        rho_safe = inside.where(rho_interp, 1e-12).maximum(1e-12)

        log_pz = rho_safe.log() - math.log(self.total_mass)
        log_pxy = -math.log(self.area)
        log_p1 = log_pxy + log_pz  # (B, N)

        log_p = log_p1.sum(axis=-1)  # (B,)
        return log_p if is_batched else log_p.squeeze(0)
