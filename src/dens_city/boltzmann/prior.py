"""
cDFT Spatial Prior (Base Distribution) for Normalizing Flows.
Bridges 1D mean-field Classical Density Functional Theory equilibrium density profiles
with many-body particle configuration sampling in 3D slit pores.
"""

import math
from typing import Tuple, Union
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
    ):
        """
        Initializes the cDFT prior distribution.

        Parameters
        ----------
        rho_z : Union[np.ndarray, Tensor]
            1D equilibrium density profile on a uniform grid in Z.
        l_z : float
            Confined slit width (in Å).
        box_size_xy : Tuple[float, float]
            Transverse periodic box dimensions (Lx, Ly) in Å.
        n_particles : int
            Number of particles N per configuration.
        """
        if isinstance(rho_z, Tensor):
            self.rho_np = np.asarray(rho_z.numpy(), dtype=np.float64).flatten()
        else:
            self.rho_np = np.asarray(rho_z, dtype=np.float64).flatten()

        self.n_grid = len(self.rho_np)
        self.l_z = float(l_z)
        self.lx, self.ly = float(box_size_xy[0]), float(box_size_xy[1])
        self.area = self.lx * self.ly
        self.dz = self.l_z / self.n_grid
        self.n_particles = int(n_particles)

        # Coordinate grid centers and cell boundaries
        self.z_edges = np.linspace(0.0, self.l_z, self.n_grid + 1)
        self.z_centers = np.linspace(0.5 * self.dz, self.l_z - 0.5 * self.dz, self.n_grid)

        # Build Discrete Cumulative Distribution Function (CDF) at cell boundaries
        cdf = np.zeros(self.n_grid + 1, dtype=np.float64)
        cdf[1:] = np.cumsum(self.rho_np * self.dz)
        self.total_mass = float(cdf[-1])
        if self.total_mass <= 0.0:
            raise ValueError("cDFT density profile mass integral must be strictly positive.")
        self.cdf = cdf / self.total_mass

        # Precompute tinygrad tensors for vectorized differentiable log_prob evaluation
        self.rho_tensor = Tensor(self.rho_np.astype(np.float32), dtype=dtypes.float32)

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Draws N-particle configurations from the cDFT base distribution.

        Parameters
        ----------
        n_samples : int
            Number of configuration samples B.

        Returns
        -------
        Tensor
            Sampled coordinates of shape (n_samples, n_particles, 3) if n_samples > 1 else (n_particles, 3).
        """
        total_pts = n_samples * self.n_particles
        # Uniform sampling in transverse X and Y
        x = np.random.uniform(0.0, self.lx, size=total_pts)
        y = np.random.uniform(0.0, self.ly, size=total_pts)

        # Inverse-CDF transform sampling in confined Z
        u = np.random.uniform(0.0, 1.0, size=total_pts)
        z = np.interp(u, self.cdf, self.z_edges)

        pts = np.stack([x, y, z], axis=-1).reshape(n_samples, self.n_particles, 3).astype(np.float32)
        out_tensor = Tensor(pts, dtype=dtypes.float32)
        return out_tensor if n_samples > 1 else out_tensor.squeeze(0)

    def log_prob(self, pos: Tensor) -> Tensor:
        r"""
        Computes exact base distribution log probability:
        \log p_0(\vec{r}_1 \dots \vec{r}_N) = \sum_{i=1}^N \left[ -\ln(L_x L_y) + \ln(\rho_{\rm cDFT}(z_i)) - \ln\left(\int \rho_{\rm cDFT}(z) dz\right) \right]

        Parameters
        ----------
        pos : Tensor
            Particle coordinates of shape (N, 3) or (B, N, 3).

        Returns
        -------
        Tensor
            Log-probabilities (scalar or (B,)).
        """
        is_batched = len(pos.shape) == 3
        pos_b = pos if is_batched else pos.unsqueeze(0)  # (B, N, 3)

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
