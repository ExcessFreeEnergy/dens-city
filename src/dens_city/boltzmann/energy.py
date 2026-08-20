"""
Microscopic Hamiltonian and Ground-Truth Energy Environment for Many-Body Configurations.
Evaluates O(N^2) pairwise Lennard-Jones and Coulomb interactions with Minimum Image Convention in X/Y
and exact Steele 9-3 / hard-core steric wall potentials in confined Z dimension.
"""

import math
from typing import Optional, Tuple, Union, List
import numpy as np
from tinygrad import Tensor, dtypes
from dens_city.materials import Material


class MicroscopicEnergy:
    """
    Evaluates the exact 3D microscopic Hamiltonian U(x) for discrete particle configurations
    in a slit pore confined between impenetrable Steele 9-3 planar walls in Z with periodic boundaries in X and Y.
    """

    def __init__(
        self,
        material: Optional[Material] = None,
        sigmas: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        epsilons: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        charges: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        box_size: Tuple[float, float, float] = (30.0, 30.0, 40.0),
        r_cut: Optional[float] = None,
        dielectric_constant: float = 1.0,
        wall_sigma: float = 3.4,
        wall_epsilon_k: float = 50.0,
        wall_type: str = "stele93",
    ):
        """
        Initializes the microscopic energy evaluator.

        Parameters
        ----------
        material : Optional[Material]
            Material instance from dens_city.materials.
        sigmas : Optional[Tensor-like]
            Particle Lennard-Jones diameters (in Å).
        epsilons : Optional[Tensor-like]
            Particle Lennard-Jones well depths (in Kelvin).
        charges : Optional[Tensor-like]
            Particle partial charges (in elementary charge units e).
        box_size : Tuple[float, float, float]
            Periodic box dimensions (Lx, Ly, Lz) in Å.
        r_cut : Optional[float]
            Spherical interaction cutoff (in Å). Enforces r_cut <= min(Lx, Ly) / 2.
        dielectric_constant : float
            Relative dielectric permittivity epsilon_r.
        wall_sigma : float
            Substrate wall atom diameter (in Å).
        wall_epsilon_k : float
            Substrate wall interaction energy (in Kelvin).
        wall_type : str
            Wall potential model ('stele93' or 'hard').
        """
        # Extract particle parameters
        if material is not None:
            if material.sites:
                s_list = [s.sigma for s in material.sites]
                e_list = [s.epsilon_k for s in material.sites]
                q_list = [s.charge for s in material.sites]
            else:
                s_list = [material.effective_sigma]
                e_list = [material.effective_epsilon_k]
                q_list = [material.total_charge]
        elif sigmas is not None and epsilons is not None:
            s_list = sigmas.tolist() if isinstance(sigmas, (np.ndarray, Tensor)) else list(sigmas)
            e_list = epsilons.tolist() if isinstance(epsilons, (np.ndarray, Tensor)) else list(epsilons)
            q_list = (
                charges.tolist()
                if isinstance(charges, (np.ndarray, Tensor))
                else list(charges)
                if charges is not None
                else [0.0] * len(s_list)
            )
        else:
            raise ValueError("Must provide either a Material instance or explicit (sigmas, epsilons).")

        self.n_particles = len(s_list)
        self.sigmas = Tensor(s_list, dtype=dtypes.float32)
        self.epsilons = Tensor(e_list, dtype=dtypes.float32)
        self.charges = Tensor(q_list, dtype=dtypes.float32)

        # Box dimensions
        self.lx, self.ly, self.lz = float(box_size[0]), float(box_size[1]), float(box_size[2])
        max_valid_rcut = 0.5 * min(self.lx, self.ly)
        if r_cut is not None:
            self.r_cut = min(float(r_cut), max_valid_rcut)
        else:
            self.r_cut = max_valid_rcut

        # Coulomb prefactor C_coul = e^2 / (4 * pi * eps_0 * eps_r * k_B) in Kelvin * Å
        self.dielectric_constant = max(1e-6, float(dielectric_constant))
        self.c_coul = 167101.0 / self.dielectric_constant

        # Precompute Lorentz-Berthelot pairwise combining matrices (N, N)
        self.s_ij = 0.5 * (self.sigmas.unsqueeze(1) + self.sigmas.unsqueeze(0))
        self.e_ij = (self.epsilons.unsqueeze(1) * self.epsilons.unsqueeze(0)).sqrt()
        self.q_ij = self.charges.unsqueeze(1) * self.charges.unsqueeze(0)

        # Precompute cutoff potential shifts at r = r_cut
        sr_cut = self.s_ij / self.r_cut
        sr6_cut = sr_cut**6
        self.u_lj_cut = 4.0 * self.e_ij * (sr6_cut * sr6_cut - sr6_cut)
        self.u_coul_cut = (self.q_ij * self.c_coul) / self.r_cut

        # Upper triangular mask (excludes diagonal and double counting)
        self.triu_mask = Tensor(np.triu(np.ones((self.n_particles, self.n_particles), dtype=np.float32), k=1))
        self.eye = Tensor(np.eye(self.n_particles, dtype=np.float32))

        # Wall potential parameters in Z
        self.wall_sigma = float(wall_sigma)
        self.wall_epsilon_k = float(wall_epsilon_k)
        self.wall_type = wall_type
        self.v_wall_inf = 1e6

        # Lorentz-Berthelot collision diameter with wall
        self.sigma_wf = 0.5 * (self.wall_sigma + self.sigmas)  # (N,)
        self.steric_radius = 0.5 * self.sigma_wf  # (N,)
        self.wall_prefactor = (2.0 * math.pi * self.wall_epsilon_k * (self.sigma_wf**3)) / 3.0  # (N,)

    def compute_pair_energy(self, pos: Tensor, shift: bool = True) -> Tensor:
        """
        Computes pairwise Lennard-Jones + Coulomb energy with Minimum Image Convention in X and Y.

        Parameters
        ----------
        pos : Tensor
            Coordinates tensor of shape (N, 3) or (B, N, 3).
        shift : bool
            If True, shifts potentials to 0 at r_cut to guarantee continuous energy and finite gradients.
            If False, evaluates bare unshifted potentials for exact single-point verifications.

        Returns
        -------
        Tensor
            Total pairwise energy (scalar for unbatched, (B,) for batched) in Kelvin.
        """
        is_batched = len(pos.shape) == 3
        pos_b = pos if is_batched else pos.unsqueeze(0)  # (B, N, 3)

        # Vectorized pairwise displacement matrix (B, N, N, 3)
        diff = pos_b.unsqueeze(2) - pos_b.unsqueeze(1)

        # Minimum Image Convention in periodic X and Y dimensions
        dx = diff[..., 0] - self.lx * (diff[..., 0] / self.lx + 0.5).floor()
        dy = diff[..., 1] - self.ly * (diff[..., 1] / self.ly + 0.5).floor()
        dz = diff[..., 2]

        r_sq = dx * dx + dy * dy + dz * dz
        # Regularize diagonal self-interaction distance to prevent 0/0 NaNs
        r = (r_sq + self.eye.unsqueeze(0)).sqrt()  # (B, N, N)

        # 1. Lennard-Jones 12-6 pairwise term
        s_ij = self.s_ij.unsqueeze(0)  # (1, N, N)
        e_ij = self.e_ij.unsqueeze(0)  # (1, N, N)
        sr6 = (s_ij / r) ** 6
        u_lj_bare = 4.0 * e_ij * (sr6 * sr6 - sr6)
        u_lj = (u_lj_bare - self.u_lj_cut.unsqueeze(0)) if shift else u_lj_bare

        # 2. Coulomb pairwise electrostatic term
        q_ij = self.q_ij.unsqueeze(0)  # (1, N, N)
        u_coul_bare = (q_ij * self.c_coul) / r
        u_coul = (u_coul_bare - self.u_coul_cut.unsqueeze(0)) if shift else u_coul_bare

        # Spherical cutoff mask & upper triangular exclusion
        within_cutoff = (r <= self.r_cut).float()
        mask = self.triu_mask.unsqueeze(0) * within_cutoff

        u_pair_mat = (u_lj + u_coul) * mask
        u_pair = u_pair_mat.sum(axis=(-1, -2))  # (B,)

        return u_pair if is_batched else u_pair.squeeze(0)

    def compute_wall_energy(self, pos: Tensor) -> Tensor:
        """
        Computes external confining slit wall potential energy in Z for all particles.

        Parameters
        ----------
        pos : Tensor
            Coordinates tensor of shape (N, 3) or (B, N, 3).

        Returns
        -------
        Tensor
            Total external wall potential energy in Kelvin.
        """
        is_batched = len(pos.shape) == 3
        pos_b = pos if is_batched else pos.unsqueeze(0)  # (B, N, 3)

        z = pos_b[..., 2]  # (B, N)
        z_l = z
        z_r = self.lz - z

        sigma_wf = self.sigma_wf.unsqueeze(0)  # (1, N)
        steric_r = self.steric_radius.unsqueeze(0)  # (1, N)
        pref = self.wall_prefactor.unsqueeze(0)  # (1, N)

        # Distance ratios safely bounded from below to avoid negative/zero division
        s_l = sigma_wf / z_l.maximum(1e-6)
        s_r = sigma_wf / z_r.maximum(1e-6)

        if self.wall_type == "stele93":
            v_l = pref * ((2.0 / 15.0) * (s_l**9) - (s_l**3))
            v_r = pref * ((2.0 / 15.0) * (s_r**9) - (s_r**3))
            v_stele = (v_l + v_r).minimum(self.v_wall_inf)
            is_steric = (z_l <= steric_r) | (z_r <= steric_r)
            v_wall = is_steric.where(self.v_wall_inf, v_stele)
        else:
            # Pure hard wall
            is_steric = (z_l < sigma_wf) | (z_r < sigma_wf)
            v_wall = is_steric.where(self.v_wall_inf, 0.0)

        u_wall = v_wall.sum(axis=-1)  # (B,)
        return u_wall if is_batched else u_wall.squeeze(0)

    def __call__(self, pos: Tensor, shift: bool = True) -> Tensor:
        """
        Computes total microscopic Hamiltonian U(pos) = U_pair(pos) + U_ext(pos).

        Parameters
        ----------
        pos : Tensor
            Coordinates tensor of shape (N, 3) or (B, N, 3).
        shift : bool
            Whether to apply potential shifts at r_cut.

        Returns
        -------
        Tensor
            Total system potential energy in Kelvin.
        """
        return self.compute_pair_energy(pos, shift=shift) + self.compute_wall_energy(pos)
