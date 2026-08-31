"""
Microscopic Hamiltonian and Ground-Truth Energy Environment for Many-Body Configurations.
Evaluates O(N^2) pairwise Lennard-Jones and Coulomb interactions with Minimum Image Convention in X/Y
and exact Steele 9-3 / hard-core steric wall potentials in confined Z dimension.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes

if TYPE_CHECKING:
    from dens_city.utils.materials import Material


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
        pad_to_128: bool = True,
        pad_to_power_of_2: bool = False,
        target_n_particles: int = 128,
        e_high: Optional[float] = 1e4,
        e_max: float = 1e20,
    ):
        if material is not None:
            if hasattr(material, "atom_mask") and hasattr(material, "molecule_mask"):
                # material is a MolecularBatch
                self.is_batched_energy = True
                self.batch = material
                self.material = None
                self.batch_size = material.batch_size
                self.n_particles = material.n_particles
                self.n_real_particles = self.n_particles
                self.sigmas = material.sigmas.realize()
                self.epsilons = material.epsilons.realize()
                self.charges = material.charges.realize()
                self.is_real_atom = material.atom_mask.realize()  # (B, N)
                self.molecule_mask = material.molecule_mask.realize()  # (B,)
            elif material.sites:
                self.is_batched_energy = False
                self.material = material
                s_list = [s.sigma for s in material.sites]
                e_list = [s.epsilon_k for s in material.sites]
                q_list = [s.charge for s in material.sites]
            else:
                self.is_batched_energy = False
                self.material = material
                s_list = [material.effective_sigma]
                e_list = [material.effective_epsilon_k]
                q_list = [material.total_charge]
        elif sigmas is not None and epsilons is not None:
            self.is_batched_energy = False
            self.material = None
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
            raise ValueError("Must provide either a Material/MolecularBatch instance or explicit (sigmas, epsilons).")

        if not getattr(self, "is_batched_energy", False):
            self.n_real_particles = len(s_list)

            # Padding logic: default is uniform 128-site bucketing
            if pad_to_power_of_2:
                self.n_particles = 1 << (self.n_real_particles - 1).bit_length() if self.n_real_particles > 1 else 1
            elif not pad_to_128:
                self.n_particles = self.n_real_particles
            else:
                self.n_particles = max(self.n_real_particles, target_n_particles)

            n_pad = self.n_particles - self.n_real_particles
            if n_pad > 0:
                # Physical parameters for dummy atoms are strictly zeroed out
                s_list = s_list + [0.0] * n_pad
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
        lz_val = (
            float(material.slit_width_a.numpy().max())
            if material is not None and hasattr(material, "slit_width_a") and material.slit_width_a is not None
            else float(box_size[2])
        )
        self.lx = Tensor([float(box_size[0])], dtype=dtypes.float32).realize()
        self.ly = Tensor([float(box_size[1])], dtype=dtypes.float32).realize()
        self.lz = Tensor([lz_val], dtype=dtypes.float32).realize()
        self.box_xy = Tensor([float(box_size[0]), float(box_size[1])], dtype=dtypes.float32).realize()
        max_valid_rcut = 0.5 * min(float(box_size[0]), float(box_size[1]))
        r_cut_val = min(float(r_cut), max_valid_rcut) if r_cut is not None else max_valid_rcut
        self.r_cut = Tensor([r_cut_val], dtype=dtypes.float32).realize()

        # Coulomb prefactor C_coul = e^2 / (4 * pi * eps_0 * eps_r * k_B) in Kelvin * Å
        self.dielectric_constant = max(1e-6, float(dielectric_constant))
        c_coul_val = 167101.0 / self.dielectric_constant
        self.c_coul = Tensor([c_coul_val], dtype=dtypes.float32).realize()
        self.has_charges = bool((self.charges.abs() > 1e-6).any().item())

        idx = Tensor.arange(self.n_particles)
        if getattr(self, "is_batched_energy", False):
            self.s_ij = 0.5 * (self.sigmas.unsqueeze(2) + self.sigmas.unsqueeze(1))
            self.e_ij = (self.epsilons.unsqueeze(2) * self.epsilons.unsqueeze(1)).sqrt()
            self.q_ij = self.charges.unsqueeze(2) * self.charges.unsqueeze(1)
            atom_mask_3d = self.is_real_atom.unsqueeze(2) * self.is_real_atom.unsqueeze(1)  # (B, N, N)
            triu_base = (idx.unsqueeze(1) < idx.unsqueeze(0)).float().unsqueeze(0)  # (1, N, N)
            if hasattr(material, "exclusions") and material.exclusions is not None:
                excl_3d = material.exclusions.realize()
                non_excl = (1.0 - excl_3d).maximum(0.0)
                self.triu_mask = triu_base * atom_mask_3d * self.molecule_mask.reshape(-1, 1, 1) * non_excl
            else:
                self.triu_mask = triu_base * atom_mask_3d * self.molecule_mask.reshape(-1, 1, 1)
        else:
            s_ij_2d = 0.5 * (self.sigmas.unsqueeze(1) + self.sigmas.unsqueeze(0))
            e_ij_2d = (self.epsilons.unsqueeze(1) * self.epsilons.unsqueeze(0)).sqrt()
            q_ij_2d = self.charges.unsqueeze(1) * self.charges.unsqueeze(0)
            self.s_ij = s_ij_2d.unsqueeze(0)
            self.e_ij = e_ij_2d.unsqueeze(0)
            self.q_ij = q_ij_2d.unsqueeze(0)
            excl_2d = getattr(material, "exclusions", None) if material is not None else None
            if excl_2d is not None:
                non_excl = (1.0 - excl_2d).maximum(0.0)
                self.triu_mask = (((idx.unsqueeze(1) < idx.unsqueeze(0)).float() * atom_mask_2d) * non_excl).unsqueeze(
                    0
                )
            else:
                self.triu_mask = ((idx.unsqueeze(1) < idx.unsqueeze(0)).float() * atom_mask_2d).unsqueeze(0)

        # Precompute cutoff potential shifts and force gradients at r = r_cut for Shifted-Force (SF)
        sr_cut = self.s_ij / self.r_cut
        sr2_cut = sr_cut * sr_cut
        sr6_cut = sr2_cut * sr2_cut * sr2_cut
        sr12_cut = sr6_cut * sr6_cut
        self.u_lj_cut = 4.0 * self.e_ij * (sr12_cut - sr6_cut)
        self.du_lj_cut = -(24.0 * self.e_ij / self.r_cut) * (2.0 * sr12_cut - sr6_cut)
        self.u_coul_cut = (self.q_ij * self.c_coul) / self.r_cut
        self.du_coul_cut = -(self.q_ij * self.c_coul) / (self.r_cut * self.r_cut)

        Tensor.realize(
            self.s_ij,
            self.e_ij,
            self.q_ij,
            self.triu_mask,
            self.u_lj_cut,
            self.du_lj_cut,
            self.u_coul_cut,
            self.du_coul_cut,
        )

        self.eye = Tensor.eye(self.n_particles).unsqueeze(0).realize()

        # Wall potential parameters in Z as realized device buffers
        self.wall_sigma = float(wall_sigma)
        self.wall_epsilon_k = float(wall_epsilon_k)
        self.wall_type = wall_type
        self.v_wall_inf = Tensor([1e6], dtype=dtypes.float32).realize()

        # Lorentz-Berthelot collision diameter with wall (masked for dummy atoms)
        self.sigma_wf = (0.5 * (self.wall_sigma + self.sigmas)).realize()
        self.steric_radius = ((0.5 * self.sigma_wf) * self.is_real_atom).realize()
        self.wall_prefactor = (
            (((2.0 * math.pi * self.wall_epsilon_k) / 3.0) * (self.sigma_wf**3)) * self.is_real_atom
        ).realize()

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
            pos_b = pos.reshape(1, -1, 3)

        if pos_b.shape[1] < self.n_particles:
            pad_len = self.n_particles - pos_b.shape[1]
            pos_b = pos_b.pad(((0, 0), (0, pad_len), (0, 0)))

        # Slice 3D Cartesian coordinates (x, y, z)
        pos_3d = pos_b[..., :3]

        # Vectorized pairwise displacement matrix (B, N, N, 3)
        diff = pos_3d.unsqueeze(2) - pos_3d.unsqueeze(1)

        # Minimum Image Convention in periodic X and Y dimensions
        dx = diff[..., 0] - self.lx * (diff[..., 0] / self.lx + 0.5).floor()
        dy = diff[..., 1] - self.ly * (diff[..., 1] / self.ly + 0.5).floor()
        dz = diff[..., 2]

        r_sq = dx * dx + dy * dy + dz * dz
        # Regularize diagonal self-interaction and clamp minimum distance to 0.5 Angstrom (r_sq >= 0.25)
        # to eliminate 0/0 NaN singularities and unphysical nuclear collapse
        r = (r_sq + self.eye + 1e-4).maximum(0.25).sqrt()  # (B, N, N)
        dr = r - self.r_cut

        # 1. Lennard-Jones 12-6 pairwise Shifted-Force term via native ALU products
        sr = (self.s_ij / r).minimum(10.0)
        sr2 = sr * sr
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        u_lj_bare = 4.0 * self.e_ij * (sr12 - sr6)
        u_lj_sf = u_lj_bare - self.u_lj_cut - self.du_lj_cut * dr
        u_lj = u_lj_sf if shift else u_lj_bare

        # 2. Coulomb pairwise electrostatic Shifted-Force term with Reaction-Field Soft-Core Damping
        # (r_soft = 0.60 A, r_soft^2 = 0.36; prevents 1/r -> infty singularity when sigma = 0)
        r_coul = (r_sq + self.eye + 0.36).sqrt()
        dr_coul = r_coul - self.r_cut
        u_coul_bare = (self.q_ij * self.c_coul) / r_coul
        u_coul_sf = u_coul_bare - self.u_coul_cut - self.du_coul_cut * dr_coul
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
            pos_b = pos.reshape(1, -1, 3)

        if pos_b.shape[1] < self.n_particles:
            pad_len = self.n_particles - pos_b.shape[1]
            pos_b = pos_b.pad(((0, 0), (0, pad_len), (0, 0)))

        z = pos_b[..., 2]  # (B, N)
        z_l = z
        z_r = self.lz - z

        # Distance ratios safely bounded by steric radius (floored at 0.5 for autodiff safety on dummy atoms)
        steric_rad_safe = self.steric_radius.maximum(0.5)
        s_l = self.sigma_wf / z_l.maximum(steric_rad_safe)
        s_r = self.sigma_wf / z_r.maximum(steric_rad_safe)

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

        if getattr(self, "is_batched_energy", False):
            u_wall = (v_wall * self.is_real_atom).sum(axis=-1) * self.molecule_mask  # (B,)
            dummy_mask = (1.0 - self.is_real_atom) * self.molecule_mask.unsqueeze(1)
            u_dummy = (1e-4 * (pos_b[..., :3] * pos_b[..., :3]).sum(axis=-1) * dummy_mask).sum(axis=-1)
            u_wall = u_wall + u_dummy
        else:
            u_wall = (v_wall * self.is_real_atom.unsqueeze(0)).sum(axis=-1)  # (B,)
            if self.n_real_particles < self.n_particles:
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


class EGNNMicroscopicEnergy:
    """
    Quantum-accurate Machine Learned Force Field energy environment.
    Evaluates internal molecular potential U_egnn(x) via a 7-layer EGNN
    coupled with analytical 1D Steele 9-3 / WCA slit wall potentials in the Z dimension.
    """

    def __init__(
        self,
        material: Optional[Union[Material, Any]] = None,
        egnn_ff: Optional[Any] = None,
        atomic_numbers: Optional[Union[List[int], np.ndarray, Tensor]] = None,
        box_size: Tuple[float, float, float] = (30.0, 30.0, 40.0),
        wall_sigma: float = 3.405,
        wall_epsilon_k: float = 119.8,
        wall_type: str = "stele93",
        e_high: Optional[float] = 1e4,
        e_max: float = 1e20,
    ):
        from dens_city.boltzmann.egnn import EGNNForceField

        self.box_size = Tensor(list(box_size), dtype=dtypes.float32)
        self.wall_type = wall_type
        self.e_high = e_high
        self.e_max = e_max

        # Setup EGNN model
        self.egnn_ff = egnn_ff if egnn_ff is not None else EGNNForceField()

        if material is not None and hasattr(material, "atomic_numbers") and hasattr(material, "atom_mask"):
            # MolecularBatch
            self.is_batched_energy = True
            self.batch = material
            self.batch_size = material.batch_size
            self.n_particles = material.n_particles
            self.atomic_numbers = (
                material.atomic_numbers.realize()
                if material.atomic_numbers is not None
                else (material.atom_mask * 6.0).realize()
            )
            self.is_real_atom = material.atom_mask.realize()
            self.molecule_mask = material.molecule_mask.realize()
            self.base_charges = getattr(material, "base_charges", None)
        elif material is not None and getattr(material, "sites", None):
            self.is_batched_energy = False
            self.material = material
            z_list = [getattr(s, "atomic_number", 6) for s in material.sites]
            n_real = len(z_list)
            self.n_particles = 128
            self.n_real_particles = n_real
            z_padded = np.zeros(128, dtype=np.float32)
            z_padded[:n_real] = z_list
            mask_padded = np.zeros(128, dtype=np.float32)
            mask_padded[:n_real] = 1.0
            self.atomic_numbers = Tensor(z_padded, dtype=dtypes.float32).realize()
            self.is_real_atom = Tensor(mask_padded, dtype=dtypes.float32).realize()
            self.molecule_mask = Tensor([1.0], dtype=dtypes.float32).realize()

            bq = material.base_charges or material.compute_topological_base_charges()
            bq_padded = np.zeros(128, dtype=np.float32)
            bq_padded[: min(n_real, 128)] = bq[: min(n_real, 128)]
            self.base_charges = Tensor(bq_padded, dtype=dtypes.float32).realize()
        elif atomic_numbers is not None:
            self.is_batched_energy = False
            z_arr = np.array(atomic_numbers, dtype=np.float32)
            n_real = len(z_arr)
            self.n_particles = 128
            self.n_real_particles = n_real
            z_padded = np.zeros(128, dtype=np.float32)
            z_padded[: min(n_real, 128)] = z_arr[: min(n_real, 128)]
            mask_padded = np.zeros(128, dtype=np.float32)
            mask_padded[: min(n_real, 128)] = 1.0
            self.atomic_numbers = Tensor(z_padded, dtype=dtypes.float32).realize()
            self.is_real_atom = Tensor(mask_padded, dtype=dtypes.float32).realize()
            self.molecule_mask = Tensor([1.0], dtype=dtypes.float32).realize()
            self.base_charges = None
        else:
            self.is_batched_energy = False
            self.n_particles = 128
            self.n_real_particles = 128
            self.atomic_numbers = Tensor.full((128,), 6.0, dtype=dtypes.float32).realize()
            self.is_real_atom = Tensor.ones(128, dtype=dtypes.float32).realize()
            self.molecule_mask = Tensor([1.0], dtype=dtypes.float32).realize()
            self.base_charges = None

        # Wall potential parameters
        self.sigma_wf = wall_sigma
        self.eps_wf_k = wall_epsilon_k
        self.wall_prefactor = 2.0 * math.pi * (self.sigma_wf**2) * 0.0333 * self.eps_wf_k
        self.steric_radius = 0.2 * self.sigma_wf
        self.v_wall_inf = 1e6

    def compute_pair_energy(self, pos: Tensor) -> Tensor:
        """Evaluates E(n)-invariant EGNN potential energy U_egnn(x)."""
        is_batched = len(pos.shape) == 3
        pos_b = pos if is_batched else pos.unsqueeze(0)
        B, N, _ = pos_b.shape

        if getattr(self, "is_batched_energy", False):
            z_in = self.atomic_numbers
            a_mask = self.is_real_atom
            m_mask = self.molecule_mask
        else:
            z_in = self.atomic_numbers.reshape(1, N).expand(B, N)
            a_mask = self.is_real_atom.reshape(1, N).expand(B, N)
            m_mask = Tensor.ones(B, dtype=dtypes.float32)

        u_egnn = self.egnn_ff.compute_energy(
            x=pos_b,
            atomic_numbers=z_in,
            atom_mask=a_mask,
            molecule_mask=m_mask,
        )
        return u_egnn if is_batched else u_egnn.squeeze(0)

    def compute_wall_energy(self, pos: Tensor) -> Tensor:
        """Evaluates 1D Steele 9-3 / steric wall potential in Z."""
        is_batched = len(pos.shape) == 3
        pos_b = pos if is_batched else pos.unsqueeze(0)
        z = pos_b[..., 2]
        Lz = self.box_size[2]
        z_l = z
        z_r = Lz - z

        if self.wall_type == "stele93":
            z_l_safe = z_l.maximum(self.steric_radius)
            z_r_safe = z_r.maximum(self.steric_radius)
            s_l = self.sigma_wf / z_l_safe
            s3_l = s_l * s_l * s_l
            s9_l = s3_l * s3_l * s3_l
            v_l = self.wall_prefactor * ((2.0 / 15.0) * s9_l - s3_l)

            s_r = self.sigma_wf / z_r_safe
            s3_r = s_r * s_r * s_r
            s9_r = s3_r * s3_r * s3_r
            v_r = self.wall_prefactor * ((2.0 / 15.0) * s9_r - s3_r)

            v_stele = (v_l + v_r).minimum(self.v_wall_inf)
            is_steric = (z_l <= self.steric_radius) | (z_r <= self.steric_radius)
            v_wall = is_steric.where(self.v_wall_inf, v_stele)
        else:
            is_steric = (z_l < self.sigma_wf) | (z_r < self.sigma_wf)
            v_wall = is_steric.where(self.v_wall_inf, 0.0)

        if getattr(self, "is_batched_energy", False):
            u_wall = (v_wall * self.is_real_atom).sum(axis=-1) * self.molecule_mask
        else:
            u_wall = (v_wall * self.is_real_atom.unsqueeze(0)).sum(axis=-1)

        return u_wall if is_batched else u_wall.squeeze(0)

    def compute_quantum_charges(
        self,
        pos: Tensor,
        total_charge: Optional[Union[Tensor, float]] = None,
        base_charges: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluates dynamic quantum partial charges q(x) via the EGNN secondary readout head."""
        is_batched = len(pos.shape) == 3
        pos_b = pos if is_batched else pos.unsqueeze(0)
        B, N, _ = pos_b.shape

        if getattr(self, "is_batched_energy", False):
            z_in = self.atomic_numbers
            a_mask = self.is_real_atom
            m_mask = self.molecule_mask
            bq_in = base_charges if base_charges is not None else getattr(self, "base_charges", None)
        else:
            z_in = self.atomic_numbers.reshape(1, N).expand(B, N)
            a_mask = self.is_real_atom.reshape(1, N).expand(B, N)
            m_mask = Tensor.ones(B, dtype=dtypes.float32)
            bq_in = (
                base_charges
                if base_charges is not None
                else (
                    self.base_charges.reshape(1, N).expand(B, N)
                    if getattr(self, "base_charges", None) is not None
                    else None
                )
            )

        q = self.egnn_ff.compute_charges(
            x=pos_b,
            atomic_numbers=z_in,
            atom_mask=a_mask,
            molecule_mask=m_mask,
            total_charge=total_charge,
            base_charges=bq_in,
        )
        return q if is_batched else q.squeeze(0)

    def __call__(self, pos: Tensor, shift: bool = True, regularize: bool = True) -> Tensor:
        """Computes total potential energy U(x) = U_egnn(x) + U_wall(z)."""
        u = self.compute_pair_energy(pos) + self.compute_wall_energy(pos)
        if regularize and self.e_high is not None:
            return regularize_energy(u, e_high=self.e_high, e_max=self.e_max)
        return u
