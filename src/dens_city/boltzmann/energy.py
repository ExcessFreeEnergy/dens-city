"""
Microscopic Hamiltonian and Ground-Truth Energy Environment for Many-Body Configurations.
Evaluates O(N^2) pairwise Lennard-Jones and Coulomb interactions with Minimum Image Convention in X/Y
and exact Steele 9-3 / hard-core steric wall potentials in confined Z dimension.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes

from dens_city.materials import Material


def regularize_energy(
    energy: Tensor,
    e_high: Union[float, Tensor] = 1e4,
    e_max: Union[float, Tensor] = 1e20,
) -> Tensor:
    r"""
    Applies Frank Noé's continuous logarithmic energy regularization for energies exceeding e_high:
    $$E_{\rm reg} = \begin{cases}
        E & E < E_{\rm high} \\
        E_{\rm high} + \log(E - E_{\rm high} + 1) & E_{\rm high} \le E < E_{\rm max} \\
        E_{\rm high} + \log(E_{\rm max} - E_{\rm high} + 1) & E \ge E_{\rm max}
    \end{cases}$$
    Provides continuous, non-zero gradients \nabla E / (E - E_high + 1) to push clashing atoms
    apart during early training iterations without numerical gradient explosions.

    Hard-bounds excess to 0.0 before evaluating .log() to prevent NaNs in both forward
    and autograd backward compiler passes.
    """
    eh = Tensor(float(e_high), dtype=dtypes.float32) if not isinstance(e_high, Tensor) else e_high
    em = Tensor(float(e_max), dtype=dtypes.float32) if not isinstance(e_max, Tensor) else e_max
    excess = (energy - eh).maximum(0.0)
    excess_clamped = excess.minimum(em - eh)
    e_reg = eh + (excess_clamped + 1.0).log()
    is_high = energy >= eh
    return is_high.where(e_reg, energy)


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
        wall_sigma: float = 3.405,
        wall_epsilon_k: float = 119.8,
        wall_type: str = "stele93",
        dielectric_constant: float = 1.0,
        pad_to_power_of_2: bool = True,
        e_high: Optional[float] = 1e4,
        e_max: float = 1e20,
    ):
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

        self.material = material
        self.n_real_particles = len(s_list)
        # Pad number of sites to the nearest power of 2 (1, 2, 4, 8, 16, 32, 64, ...) if requested
        if pad_to_power_of_2:
            self.n_particles = 1 << (self.n_real_particles - 1).bit_length() if self.n_real_particles > 1 else 1
        else:
            self.n_particles = self.n_real_particles

        n_pad = self.n_particles - self.n_real_particles
        if n_pad > 0:
            s_list = s_list + [1.0] * n_pad
            e_list = e_list + [0.0] * n_pad
            q_list = q_list + [0.0] * n_pad

        self.sigmas = Tensor(s_list, dtype=dtypes.float32).realize()
        self.epsilons = Tensor(e_list, dtype=dtypes.float32).realize()
        self.charges = Tensor(q_list, dtype=dtypes.float32).realize()

        # Real-atom exclusion and reduction mask
        idx = Tensor.arange(self.n_particles)
        self.is_real_atom = (idx < self.n_real_particles).float().realize()  # (N,)
        atom_mask_2d = self.is_real_atom.unsqueeze(1) * self.is_real_atom.unsqueeze(0)  # (N, N)

        # Box dimensions as realized device buffers
        self.lx = Tensor([float(box_size[0])], dtype=dtypes.float32).realize()
        self.ly = Tensor([float(box_size[1])], dtype=dtypes.float32).realize()
        self.lz = Tensor([float(box_size[2])], dtype=dtypes.float32).realize()
        self.box_xy = Tensor([float(box_size[0]), float(box_size[1])], dtype=dtypes.float32).realize()
        max_valid_rcut = 0.5 * min(float(box_size[0]), float(box_size[1]))
        r_cut_val = min(float(r_cut), max_valid_rcut) if r_cut is not None else max_valid_rcut
        self.r_cut = Tensor([r_cut_val], dtype=dtypes.float32).realize()

        # Coulomb prefactor C_coul = e^2 / (4 * pi * eps_0 * eps_r * k_B) in Kelvin * Å
        self.dielectric_constant = max(1e-6, float(dielectric_constant))
        c_coul_val = 167101.0 / self.dielectric_constant
        self.c_coul = Tensor([c_coul_val], dtype=dtypes.float32).realize()
        self.has_charges = bool((self.charges.abs() > 1e-6).any().item())

        # Precompute Lorentz-Berthelot pairwise combining matrices (1, N, N) as realized device buffers
        s_ij_2d = 0.5 * (self.sigmas.unsqueeze(1) + self.sigmas.unsqueeze(0))
        e_ij_2d = (self.epsilons.unsqueeze(1) * self.epsilons.unsqueeze(0)).sqrt()
        q_ij_2d = self.charges.unsqueeze(1) * self.charges.unsqueeze(0)
        self.s_ij = s_ij_2d.unsqueeze(0).realize()
        self.e_ij = e_ij_2d.unsqueeze(0).realize()
        self.q_ij = q_ij_2d.unsqueeze(0).realize()

        # Precompute cutoff potential shifts and force gradients at r = r_cut for Shifted-Force (SF)
        sr_cut = self.s_ij / self.r_cut
        sr2_cut = sr_cut * sr_cut
        sr6_cut = sr2_cut * sr2_cut * sr2_cut
        sr12_cut = sr6_cut * sr6_cut
        self.u_lj_cut = (4.0 * self.e_ij * (sr12_cut - sr6_cut)).realize()
        self.du_lj_cut = (-(24.0 * self.e_ij / self.r_cut) * (2.0 * sr12_cut - sr6_cut)).realize()
        self.u_coul_cut = ((self.q_ij * self.c_coul) / self.r_cut).realize()
        self.du_coul_cut = (-(self.q_ij * self.c_coul) / (self.r_cut * self.r_cut)).realize()

        # Upper triangular mask (excludes diagonal, double counting, and dummy atoms)
        self.triu_mask = ((idx.unsqueeze(1) < idx.unsqueeze(0)).float() * atom_mask_2d).unsqueeze(0).realize()
        self.eye = Tensor.eye(self.n_particles).unsqueeze(0).realize()

        # Wall potential parameters in Z as realized device buffers
        self.wall_sigma = float(wall_sigma)
        self.wall_epsilon_k = float(wall_epsilon_k)
        self.wall_type = wall_type
        self.v_wall_inf = Tensor([1e6], dtype=dtypes.float32).realize()

        # Lorentz-Berthelot collision diameter with wall (masked for dummy atoms)
        self.sigma_wf = (0.5 * (self.wall_sigma + self.sigmas)).realize()  # (N,)
        self.steric_radius = ((0.5 * self.sigma_wf) * self.is_real_atom).realize()  # (N,)
        self.wall_prefactor = (
            (((2.0 * math.pi * self.wall_epsilon_k) / 3.0) * (self.sigma_wf**3)) * self.is_real_atom
        ).realize()  # (N,)

        # Energy regularization parameters as realized device buffers
        self.e_high_val = float(e_high) if e_high is not None else None
        self.e_max_val = float(e_max)
        if self.e_high_val is not None:
            self.e_high = Tensor(self.e_high_val, dtype=dtypes.float32).realize()
            self.e_max = Tensor(self.e_max_val, dtype=dtypes.float32).realize()
        else:
            self.e_high = None
            self.e_max = None

        # JIT-compiled energy evaluation
        self.eval_energy = TinyJit(self.__call__)

    def compute_pair_energy(self, pos: Tensor, shift: bool = True) -> Tensor:
        """
        Computes pairwise Lennard-Jones + Coulomb energy with Minimum Image Convention in X and Y,
        applying exact Shifted-Force (SF) boundary continuity.
        Supports both (B, N, 3) and (B, N, 4) coordinates.
        """
        is_batched = len(pos.shape) >= 2 and (
            len(pos.shape) == 3 or (len(pos.shape) == 2 and pos.shape[-1] not in (3, 4))
        )
        if len(pos.shape) == 2:
            if pos.shape[-1] in (3, 4):
                pos_b = pos.unsqueeze(0)
            else:
                # Flat (B, N*3) or (B, N*4)
                ch = 4 if pos.shape[-1] == self.n_particles * 4 else 3
                pos_b = pos.reshape(-1, self.n_particles, ch)
        elif len(pos.shape) == 3:
            pos_b = pos
        else:
            pos_b = pos.reshape(1, self.n_particles, -1)

        # Slice 3D Cartesian coordinates (x, y, z)
        pos_3d = pos_b[..., :3]

        # Vectorized pairwise displacement matrix (B, N, N, 3)
        diff = pos_3d.unsqueeze(2) - pos_3d.unsqueeze(1)

        # Minimum Image Convention in periodic X and Y dimensions
        dx = diff[..., 0] - self.lx * (diff[..., 0] / self.lx + 0.5).floor()
        dy = diff[..., 1] - self.ly * (diff[..., 1] / self.ly + 0.5).floor()
        dz = diff[..., 2]

        r_sq = dx * dx + dy * dy + dz * dz
        # Regularize diagonal self-interaction and overlapping particles to eliminate 0/0 NaN singularities
        r = (r_sq + self.eye + 1e-4).sqrt()  # (B, N, N)
        dr = r - self.r_cut

        # 1. Lennard-Jones 12-6 pairwise Shifted-Force term via native ALU products
        sr = self.s_ij / r
        sr2 = sr * sr
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        u_lj_bare = 4.0 * self.e_ij * (sr12 - sr6)
        u_lj_sf = u_lj_bare - self.u_lj_cut - self.du_lj_cut * dr
        u_lj = u_lj_sf if shift else u_lj_bare

        # 2. Coulomb pairwise electrostatic Shifted-Force term (unified ALU arithmetic, zero for non-polar)
        u_coul_bare = (self.q_ij * self.c_coul) / r
        u_coul_sf = u_coul_bare - self.u_coul_cut - self.du_coul_cut * dr
        u_coul = u_coul_sf if shift else u_coul_bare
        u_pair_ij = u_lj + u_coul

        # Spherical cutoff mask & upper triangular exclusion
        within_cutoff = (r <= self.r_cut).float()
        mask = self.triu_mask * within_cutoff

        u_pair_mat = u_pair_ij * mask
        # Collapse contiguous (N, N) matrix into 1D (N*N) before summing to collapse reduction axes
        u_pair = u_pair_mat.flatten(1).sum(axis=-1)  # (B,)

        return u_pair if is_batched else u_pair.squeeze(0)

    def compute_wall_energy(self, pos: Tensor) -> Tensor:
        """
        Computes external confining slit wall potential energy in Z for all particles.
        """
        is_batched = len(pos.shape) >= 2 and (
            len(pos.shape) == 3 or (len(pos.shape) == 2 and pos.shape[-1] not in (3, 4))
        )
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

        z = pos_b[..., 2]  # (B, N)
        z_l = z
        z_r = self.lz - z

        # Distance ratios safely bounded by steric radius to eliminate float32 (s_l**9) overflow
        s_l = self.sigma_wf / z_l.maximum(self.steric_radius)
        s_r = self.sigma_wf / z_r.maximum(self.steric_radius)

        if self.wall_type == "stele93":
            s3_l = s_l * s_l * s_l
            s9_l = s3_l * s3_l * s3_l
            v_l = self.wall_prefactor * ((2.0 / 15.0) * s9_l - s3_l)

            s3_r = s_r * s_r * s_r
            s9_r = s3_r * s3_r * s3_r
            v_r = self.wall_prefactor * ((2.0 / 15.0) * s9_r - s3_r)

            v_stele = (v_l + v_r).minimum(self.v_wall_inf)
            is_steric = (z_l <= self.steric_radius) | (z_r <= self.steric_radius)
            v_wall = is_steric.where(self.v_wall_inf, v_stele)
        else:
            # Pure hard wall
            is_steric = (z_l < self.sigma_wf) | (z_r < self.sigma_wf)
            v_wall = is_steric.where(self.v_wall_inf, 0.0)

        u_wall = (v_wall * self.is_real_atom.unsqueeze(0)).sum(axis=-1)  # (B,)
        if self.n_real_particles < self.n_particles:
            # Subtle anchor to prevent dummy unconstrained coordinate divergence
            dummy_mask = (1.0 - self.is_real_atom).unsqueeze(0)
            u_dummy = (1e-4 * (pos_b[..., :3] * pos_b[..., :3]).sum(axis=-1) * dummy_mask).sum(axis=-1)
            u_wall = u_wall + u_dummy
        return u_wall if is_batched else u_wall.squeeze(0)

    def regularize_energy(
        self,
        energy: Tensor,
        e_high: Optional[Union[float, Tensor]] = None,
        e_max: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        """
        Regularizes high energy configurations using Noé's soft logarithmic ceiling:
        E_reg = E_high + log(E - E_high + 1) for E >= E_high.
        """
        eh = e_high if e_high is not None else self.e_high
        em = e_max if e_max is not None else self.e_max
        if eh is None:
            return energy
        return regularize_energy(energy, e_high=eh, e_max=em if em is not None else 1e20)

    def __call__(self, pos: Tensor, shift: bool = True, regularize: bool = True) -> Tensor:
        """
        Computes total microscopic Hamiltonian U(pos) = U_pair(pos) + U_ext(pos).
        Optionally applies Noé energy regularization for E >= E_high.
        """
        u = self.compute_pair_energy(pos, shift=shift) + self.compute_wall_energy(pos)
        if regularize and self.e_high is not None:
            return self.regularize_energy(u)
        return u
