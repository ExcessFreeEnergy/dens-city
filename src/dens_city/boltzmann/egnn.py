"""
Equivariant Graph Neural Network (EGNN) Machine Learned Force Field (MLFF).
Predicts total potential energy U(x) with Density Functional Theory (DFT) level accuracy
and derives exact conservative forces F = -∇_x U with strict E(n) rotational and translational invariance.

Adheres strictly to tinygrad best practices and power-of-2 vector alignments (N=128, F=128, L=7).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes, nn


class EGNNLayer:
    """
    Single E(n)-invariant message passing layer.
    Computes edge interaction messages from node embeddings, relative squared distances,
    and edge masks with smooth radial cutoff attenuation and degree normalization,
    then updates node embeddings with residual connections.
    Uses decomposed linear projections to avoid materializing huge (B, N, N, 2F+2) tensors in memory.
    """

    def __init__(self, hidden_dim: int = 128, edge_in_dim: Optional[int] = None, r_cut: float = 5.0):
        self.hidden_dim = hidden_dim
        self.r_cut = r_cut

        # Decomposed first linear projection of edge features
        self.edge_hi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_hj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_d = nn.Linear(1, hidden_dim, bias=False)
        self.edge_a = nn.Linear(1, hidden_dim, bias=True)

        # Second linear layer of edge MLP
        self.edge_l2 = nn.Linear(hidden_dim, hidden_dim)

        # Node Update MLP: phi_h: R^(256) -> R^(128) -> R^(128) with Swish (SiLU)
        self.node_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, hidden_dim),
        ]

    def __call__(
        self,
        h: Tensor,
        d_sq: Tensor,
        edge_mask: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass for a single message-passing step.
        h: (B, N, F) node features
        d_sq: (B, N, N, 1) pairwise squared distances ||x_i - x_j||^2
        edge_mask: (B, N, N, 1) edge validity mask
        atom_mask: (B, N, 1) atom validity mask
        """
        B, N, F = h.shape

        # 1. Project node features directly on (B, N, F) before spatial broadcast
        h_i_proj = self.edge_hi(h).reshape(B, N, 1, F)
        h_j_proj = self.edge_hj(h).reshape(B, 1, N, F)
        d_proj = self.edge_d(d_sq)
        a_proj = self.edge_a(edge_mask)

        # 2. Sum linear projections and apply SiLU
        e_hidden = (h_i_proj + h_j_proj + d_proj + a_proj).silu()

        # 3. Smooth cosine radial distance cutoff envelope: f_cut(r_ij) = 0.5 * (cos(pi * r / r_cut) + 1)
        r_ij = (d_sq + 1e-8).sqrt()
        cutoff_mask = (r_ij < self.r_cut).cast(dtypes.float32)
        f_cut = cutoff_mask * 0.5 * ((r_ij * (math.pi / self.r_cut)).cos() + 1.0)
        effective_edge_mask = edge_mask * f_cut

        # 4. Message generation via second linear layer & smooth cutoff masking
        m_ij = self.edge_l2(e_hidden).silu() * effective_edge_mask

        # 5. Message aggregation normalized by active neighbor degree to guarantee extensive O(N) energy scaling
        deg_i = (edge_mask * cutoff_mask).sum(axis=2).maximum(1.0)  # (B, N, 1)
        m_i = m_ij.sum(axis=2) / deg_i  # (B, N, F)

        # 6. Node update with residual connection
        node_inputs = Tensor.cat(h, m_i, dim=-1)
        h_delta = node_inputs.sequential(self.node_mlp)
        h_next = (h + h_delta) * atom_mask

        return h_next


class EGNNForceField:
    """
    7-layer Equivariant Graph Neural Network (EGNN) Potential Energy Surface & Force Field.
    Ingests atomic numbers Z and Cartesian coordinates x, predicting scalar potential energy U(x).
    Derives analytical conservative forces via reverse-mode autograd: F = -∇_x U(x).
    """

    DEFAULT_CHECKPOINT = Path("data/checkpoints/egnn_charges_trained.npz")

    def __init__(
        self,
        num_layers: int = 7,
        hidden_dim: int = 128,
        max_atomic_number: int = 128,
        n_particles: int = 128,
        r_cut: float = 5.0,
        weights_path: Optional[str | Path] = None,
        load_default_weights: bool = True,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_atomic_number = max_atomic_number
        self.n_particles = n_particles
        self.r_cut = r_cut

        # Linear embedding layer for atomic numbers Z -> h^0 in R^(128)
        self.embedding = nn.Linear(max_atomic_number, hidden_dim)

        # 7 sequential message-passing layers with smooth radial cutoff and degree normalization
        self.layers = [EGNNLayer(hidden_dim=hidden_dim, r_cut=r_cut) for _ in range(num_layers)]

        # Readout MLP: node-wise energy contribution eps_i
        self.readout_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(hidden_dim, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, 1),
        ]

        # Secondary Readout MLP: node-wise dynamic quantum partial charge q_i
        # Ingests geometric node embedding h_i (128) + 4-channel continuous solvent descriptors
        # (alpha_i, beta_i = alpha/rho, base_q, chi) -> 132 dimensions
        self.charge_in_dim = hidden_dim + 4
        self.charge_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(self.charge_in_dim, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, 1),
        ]
        self.max_delta_q = 0.25
        # Zero-initialize output layer so initial neural perturbations start cleanly at 0.0 around physical baseline
        self.charge_mlp[2].weight = Tensor.zeros(1, hidden_dim, dtype=dtypes.float32)
        self.charge_mlp[2].bias = Tensor.zeros(1, dtype=dtypes.float32)

        # Tertiary Readout MLP: node-wise volumetric cavitation & dispersion free energy correction delta_g_i^vdw
        # Ingests geometric node embedding h_i (128) + 4-channel continuous solvent descriptors (132 dimensions total)
        # Predicts continuous atomic nonpolar cavitation & dispersion energy modulation (kcal/mol)
        self.vdw_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(self.charge_in_dim, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, 1),
        ]
        self.max_delta_vdw = 3.5  # Max allowed atomic nonpolar perturbation |Δg_i^vdw| <= 3.5 kcal/mol
        # Zero-initialize output layer so initial nonpolar perturbations start cleanly at 0.0 around physical baseline
        self.vdw_mlp[2].weight = Tensor.zeros(1, hidden_dim, dtype=dtypes.float32)
        self.vdw_mlp[2].bias = Tensor.zeros(1, dtype=dtypes.float32)

        # Quaternary Readout MLP: molecular graph-level cooperative free energy correction delta_g_coop
        # Ingests multi-scale pooled node embeddings (mean, max, std across active atoms: 3 * 128 = 384 dimensions)
        # Resolves cooperative multi-center polarization and collective hydrogen-bonding networks in polyols/sugars
        self.global_in_dim = hidden_dim * 3
        self.global_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(self.global_in_dim, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, 1),
        ]
        self.max_delta_global = 25.0  # Max allowed molecular cooperative perturbation |ΔG_coop| <= 25.0 kcal/mol
        # Zero-initialize output layer so initial cooperative perturbations start cleanly at 0.0
        self.global_mlp[2].weight = Tensor.zeros(1, hidden_dim, dtype=dtypes.float32)
        self.global_mlp[2].bias = Tensor.zeros(1, dtype=dtypes.float32)

        # Load weights if specified or present at default location
        if weights_path is not None:
            self.load_weights(weights_path)
        elif load_default_weights and self.DEFAULT_CHECKPOINT.exists():
            self.load_weights(self.DEFAULT_CHECKPOINT)

    def load_weights(self, filepath: str | Path) -> None:
        """Loads model state from a .npz checkpoint if layer shapes match the instance architecture."""
        p = Path(filepath)
        if not p.exists():
            return
        np_data = np.load(str(p))
        state_dict = {k: Tensor(np_data[k]) for k in np_data.files}
        model_dict = nn.state.get_state_dict(self)
        matched = {
            k: state_dict[k] for k, v in model_dict.items() if k in state_dict and state_dict[k].shape == v.shape
        }
        if matched:
            nn.state.load_state_dict(self, matched, strict=False, verbose=False)

    def save_weights(self, filepath: str | Path) -> None:
        """Saves model state to a .npz checkpoint."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        state_dict = nn.state.get_state_dict(self)
        np_dict = {k: v.numpy() for k, v in state_dict.items()}
        np.savez(str(p), **np_dict)

    def _prepare_inputs(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Validates and broadcasts input dimensions to static (B, N, ...) shapes."""
        # Ensure x is (B, N, 3)
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 3)
        B, N, _ = x.shape

        # Ensure atomic_numbers is (B, N)
        if len(atomic_numbers.shape) == 1:
            atomic_numbers = atomic_numbers.reshape(B, N)
        elif len(atomic_numbers.shape) == 3:
            atomic_numbers = atomic_numbers.reshape(B, N)

        # Default masks if not supplied
        if atom_mask is None:
            atom_mask = (atomic_numbers > 0).cast(dtypes.float32).reshape(B, N, 1)
        elif len(atom_mask.shape) == 2:
            atom_mask = atom_mask.reshape(B, N, 1)

        if molecule_mask is None:
            molecule_mask = Tensor.ones(B, dtype=dtypes.float32)
        elif len(molecule_mask.shape) == 2:
            molecule_mask = molecule_mask.reshape(B)

        # Edge mask a_ij = mask_i * mask_j * (1 - delta_ij)
        mask_i = atom_mask.reshape(B, N, 1, 1)
        mask_j = atom_mask.reshape(B, 1, N, 1)
        diag_zero = (1.0 - Tensor.eye(N, dtype=dtypes.float32)).reshape(1, N, N, 1)
        edge_mask = (mask_i * mask_j * diag_zero).realize()

        return x, atomic_numbers, atom_mask, molecule_mask, edge_mask

    def compute_energy(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes total potential energy U(x) for batch across all molecules.
        Returns: Tensor of shape (B,) in internal energy units (kcal/mol or reduced units).
        """
        x, atomic_numbers, atom_mask, molecule_mask, edge_mask = self._prepare_inputs(
            x, atomic_numbers, atom_mask, molecule_mask
        )
        B, N, _ = x.shape

        # 1. One-hot encode atomic numbers Z -> (B, N, max_atomic_number) -> h^0 (B, N, 128)
        z_clamped = atomic_numbers.cast(dtypes.int32)
        z_one_hot = Tensor.one_hot(z_clamped, num_classes=self.max_atomic_number)
        h = self.embedding(z_one_hot) * atom_mask

        # 2. Evaluate Pairwise Relative Squared Distances d_ij^2 = ||x_i - x_j||^2
        x_i = x.reshape(B, N, 1, 3)
        x_j = x.reshape(B, 1, N, 3)
        diff = x_i - x_j
        d_sq = (diff * diff).sum(axis=-1, keepdim=True)  # (B, N, N, 1)

        # 3. Pass through L=7 message-passing layers
        for layer in self.layers:
            h = layer(h, d_sq, edge_mask, atom_mask)

        # 4. Readout: Map node embeddings to atomic energies eps_i -> sum over atoms
        eps_i = self.readout_mlp[0](h)
        eps_i = self.readout_mlp[1](eps_i)
        eps_i = self.readout_mlp[2](eps_i) * atom_mask  # (B, N, 1)

        u_total = eps_i.sum(axis=(1, 2)) * molecule_mask
        return u_total

    def compute_solvation_readouts(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
        total_charge: Optional[Tensor | float] = None,
        base_charges: Optional[Tensor] = None,
        solvent_features: Optional[Tensor] = None,
        solvent_hbond_capacity: Optional[Union[float, Tensor]] = None,
        detach_trunk: bool = False,
        return_global: bool = False,
    ) -> Tuple[Tensor, ...]:
        """
        Predicts:
        1. Formally conserved quantum partial charges q_i(x) for electrostatic Born solvation.
        2. Volumetric cavitation & dispersion corrections delta_g_i^vdw(x) for nonpolar solvation.
        3. Molecular graph-level cooperative free energy delta_g_coop(x) for hydrogen-bonding networks.
        Returns:
            q_pred: Tensor of shape (B, N)
            delta_vdw_mol: Tensor of shape (B,) - total molecular nonpolar & cooperative correction (kcal/mol)
            delta_vdw_atomic: Tensor of shape (B, N, 1) - atomic nonpolar contributions
            (Optional if return_global=True) delta_g_coop: Tensor of shape (B,)
        """
        x, atomic_numbers, atom_mask, molecule_mask, edge_mask = self._prepare_inputs(
            x, atomic_numbers, atom_mask, molecule_mask
        )
        B, N, _ = x.shape

        # 1. One-hot encode atomic numbers Z -> (B, N, max_atomic_number) -> h^0 (B, N, 128)
        z_clamped = atomic_numbers.cast(dtypes.int32)
        z_one_hot = Tensor.one_hot(z_clamped, num_classes=self.max_atomic_number)
        h = self.embedding(z_one_hot) * atom_mask

        # 2. Evaluate Pairwise Relative Squared Distances d_ij^2 = ||x_i - x_j||^2
        x_i = x.reshape(B, N, 1, 3)
        x_j = x.reshape(B, 1, N, 3)
        diff = x_i - x_j
        d_sq = (diff * diff).sum(axis=-1, keepdim=True)  # (B, N, N, 1)

        # 3. Pass through L=7 message-passing layers
        for layer in self.layers:
            h = layer(h, d_sq, edge_mask, atom_mask)

        # Halt gradient traversal into the 7-layer message-passing trunk per tinyspec.tex \op{Detach}
        if detach_trunk:
            h = h.detach()

        # Augment with 4-channel continuous 3D solvent descriptors (alpha, beta=alpha/rho, bq, chi)
        if solvent_features is not None:
            sf = solvent_features
        else:
            from dens_city.cdft.generalized_born import GeneralizedBornSolvation

            gb = GeneralizedBornSolvation()
            sf = gb.compute_solvent_descriptors(x, atomic_numbers, atom_mask, base_charges=base_charges)

        node_inputs = Tensor.cat(h, sf, dim=-1)

        # Head 1: Neural Charge Readout: delta_q_i(x) with physical tanh squashing
        delta_q_raw = self.charge_mlp[0](node_inputs)
        delta_q_raw = self.charge_mlp[1](delta_q_raw)
        delta_q_raw = self.charge_mlp[2](delta_q_raw)
        delta_q = self.max_delta_q * (delta_q_raw / self.max_delta_q).tanh() * atom_mask  # (B, N, 1)

        # Residual Superposition: q_raw = q_base + delta_q
        if base_charges is not None:
            bq = base_charges.reshape(B, N, 1)
            q_raw = (bq + delta_q) * atom_mask
        else:
            q_raw = delta_q

        # Exact Formal Charge Mean-Shift Conservation Projection
        num_real = atom_mask.sum(axis=1, keepdim=True).maximum(1.0)  # (B, 1, 1)
        q_sum = q_raw.sum(axis=1, keepdim=True)  # (B, 1, 1)
        if total_charge is not None:
            if isinstance(total_charge, (int, float)):
                q_target = Tensor.full((B, 1, 1), float(total_charge), dtype=dtypes.float32)
            else:
                q_target = total_charge.reshape(B, 1, 1).cast(dtypes.float32)
        else:
            q_target = Tensor.zeros(B, 1, 1, dtype=dtypes.float32)

        q_shift = (q_sum - q_target) / num_real
        q_final = ((q_raw - q_shift) * atom_mask).reshape(B, N)
        q_masked = q_final * molecule_mask.reshape(B, 1)

        # Head 2: Volumetric Cavitation & Dispersion Readout: delta_g_i^vdw with physical tanh squashing
        delta_vdw_raw = self.vdw_mlp[0](node_inputs)
        delta_vdw_raw = self.vdw_mlp[1](delta_vdw_raw)
        delta_vdw_raw = self.vdw_mlp[2](delta_vdw_raw)
        delta_vdw_atomic = self.max_delta_vdw * (delta_vdw_raw / self.max_delta_vdw).tanh() * atom_mask  # (B, N, 1)
        delta_vdw_mol_atomic = (delta_vdw_atomic * molecule_mask.reshape(B, 1, 1)).sum(axis=(1, 2))  # (B,)

        # Head 3: Multi-Scale Molecular Graph Pooling & Cooperative Readout delta_g_coop
        num_real_nodes = atom_mask.sum(axis=1).maximum(1.0)  # (B, 1)
        mean_pool = (h * atom_mask).sum(axis=1) / num_real_nodes  # (B, F)
        h_masked = h * atom_mask - (1.0 - atom_mask) * 1e4
        max_pool = h_masked.max(axis=1)  # (B, F)
        h_diff = (h - mean_pool.reshape(B, 1, self.hidden_dim)) * atom_mask
        var_pool = (h_diff * h_diff).sum(axis=1) / num_real_nodes
        std_pool = (var_pool + 1e-6).sqrt()  # (B, F)

        graph_features = Tensor.cat(mean_pool, max_pool, std_pool, dim=-1)  # (B, 384)
        delta_coop_raw = self.global_mlp[0](graph_features)
        delta_coop_raw = self.global_mlp[1](delta_coop_raw)
        delta_coop_raw = self.global_mlp[2](delta_coop_raw).reshape(B)
        delta_g_coop = (
            self.max_delta_global * (delta_coop_raw / self.max_delta_global).tanh() * molecule_mask.reshape(B)
        )
        if solvent_hbond_capacity is not None:
            if isinstance(solvent_hbond_capacity, (int, float)):
                gate_s = math.tanh(max(0.0, float(solvent_hbond_capacity)) / 1.5)
                delta_g_coop = delta_g_coop * gate_s
            else:
                gate_s = (solvent_hbond_capacity.maximum(0.0) / 1.5).tanh().reshape(B)
                delta_g_coop = delta_g_coop * gate_s

        # Total molecular nonpolar + cooperative free energy modulation
        delta_vdw_mol = delta_vdw_mol_atomic + delta_g_coop
        if not Tensor.training:
            q_masked = q_masked.realize()
            delta_vdw_mol = delta_vdw_mol.realize()
            delta_vdw_atomic = delta_vdw_atomic.realize()
            delta_g_coop = delta_g_coop.realize()
            graph_features = graph_features.realize()

        if return_global:
            return q_masked, delta_vdw_mol, delta_vdw_atomic, delta_g_coop, graph_features
        return q_masked, delta_vdw_mol, delta_vdw_atomic

    def compute_ensembled_solvation_readouts(
        self,
        x_ensemble: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
        total_charge: Optional[Union[float, Tensor]] = None,
        base_charges: Optional[Tensor] = None,
        solvent_features: Optional[Tensor] = None,
        solvent_hbond_capacity: Optional[Union[float, Tensor]] = None,
        dielectric_constant: Optional[float] = None,
        gb_solver=None,
        detach_trunk: bool = False,
        internal_energies: Optional[Tensor] = None,
        temperature_k: Optional[float] = None,
        return_global: bool = False,
    ) -> Tuple[Tensor, ...]:
        r"""
        Computes Boltzmann conformational ensemble averages across s configurations per molecule:
        Ingests x_ensemble of shape (B, s, N, 3) or (s, N, 3).
        Evaluates dynamic charges, Born electrostatic free energy, and nonpolar/cooperative
        cavitation across the ensemble in a single vector-parallel pass without Python loops.
        Weights configurations using normalized Boltzmann factors w_k \propto exp(-\Delta E_k / k_B T).
        Returns:
            q_mean: Tensor of shape (B, N) - ensemble-averaged quantum partial charges
            total_solv_mean: Tensor of shape (B,) - ensemble-averaged (delta_vdw + delta_g_born) free energy
            gb_mean: Tensor of shape (B,) - ensemble-averaged Born electrostatic free energy
            (optional) h_mol_mean: Tensor of shape (B, 384) if return_global is True
            (optional) coop_mean: Tensor of shape (B,) if return_global is True
        """
        if dielectric_constant is None or dielectric_constant <= 0.0:
            raise ValueError(
                f"dielectric_constant must be explicitly provided as a positive float, got {dielectric_constant}"
            )
        if temperature_k is None or temperature_k <= 0.0:
            raise ValueError(
                f"temperature_k must be explicitly provided as a positive float in Kelvin, got {temperature_k}"
            )

        if len(x_ensemble.shape) == 3:
            # (s, N, 3) -> treat as single molecule with B=1, s=s
            x_ensemble = x_ensemble.reshape(1, x_ensemble.shape[0], x_ensemble.shape[1], 3)
        B, s, N, _ = x_ensemble.shape

        # Flatten (B, s, N, 3) -> (B * s, N, 3)
        x_flat = x_ensemble.reshape(B * s, N, 3)

        # Broadcast atomic numbers: (B, N) -> (B * s, N)
        if len(atomic_numbers.shape) == 1:
            atomic_numbers = atomic_numbers.reshape(1, N)
        z_flat = atomic_numbers.reshape(B, 1, N).expand(B, s, N).reshape(B * s, N)

        # Broadcast atom mask: (B, N, 1) -> (B * s, N, 1)
        if atom_mask is None:
            atom_mask = (atomic_numbers > 0).cast(dtypes.float32).reshape(B, N, 1)
        elif len(atom_mask.shape) == 2:
            atom_mask = atom_mask.reshape(B, N, 1)
        m_flat = atom_mask.reshape(B, 1, N, 1).expand(B, s, N, 1).reshape(B * s, N, 1)

        # Broadcast molecule mask: (B,) -> (B * s,)
        if molecule_mask is None:
            molecule_mask = Tensor.ones(B, dtype=dtypes.float32)
        elif len(molecule_mask.shape) == 2:
            molecule_mask = molecule_mask.reshape(B)
        mol_flat = molecule_mask.reshape(B, 1).expand(B, s).reshape(B * s)

        # Broadcast base charges: (B, N) -> (B * s, N)
        if base_charges is not None:
            if len(base_charges.shape) == 1:
                base_charges = base_charges.reshape(1, N)
            bq_flat = base_charges.reshape(B, 1, N).expand(B, s, N).reshape(B * s, N)
        else:
            bq_flat = None

        # Broadcast total charge: (B, 1, 1) -> (B * s, 1, 1)
        if total_charge is not None:
            if isinstance(total_charge, (int, float)):
                tq_flat = Tensor.full((B * s, 1, 1), float(total_charge), dtype=dtypes.float32)
            else:
                tq_flat = total_charge.reshape(B, 1, 1, 1).expand(B, s, 1, 1).reshape(B * s, 1, 1).cast(dtypes.float32)
        else:
            tq_flat = None

        # Solvent features on flat ensemble
        if solvent_features is not None:
            if len(solvent_features.shape) == 2:
                solvent_features = solvent_features.reshape(1, N, solvent_features.shape[-1])
            if len(solvent_features.shape) == 4:
                sf_flat = solvent_features.reshape(B * s, N, solvent_features.shape[-1])
            else:
                sf_flat = solvent_features.reshape(B, 1, N, -1).expand(B, s, N, -1).reshape(B * s, N, -1)
        else:
            sf_flat = None

        if solvent_hbond_capacity is not None:
            if isinstance(solvent_hbond_capacity, (int, float)):
                cap_flat = float(solvent_hbond_capacity)
            elif len(solvent_hbond_capacity.shape) == 0:
                cap_flat = solvent_hbond_capacity
            elif len(solvent_hbond_capacity.shape) == 1:
                cap_flat = solvent_hbond_capacity.reshape(B, 1).expand(B, s).reshape(B * s)
            else:
                cap_flat = solvent_hbond_capacity.reshape(B * s)
        else:
            cap_flat = None

        # Execute single fused forward pass over all B * s conformers
        q_pred, delta_vdw_mol, delta_vdw_atomic, delta_g_coop, graph_features = self.compute_solvation_readouts(
            x=x_flat,
            atomic_numbers=z_flat,
            atom_mask=m_flat,
            molecule_mask=mol_flat,
            total_charge=tq_flat,
            base_charges=bq_flat,
            solvent_features=sf_flat,
            solvent_hbond_capacity=cap_flat,
            detach_trunk=detach_trunk,
            return_global=True,
        )

        if gb_solver is None:
            from dens_city.cdft.generalized_born import GeneralizedBornSolvation

            gb_solver = GeneralizedBornSolvation(dielectric_constant=dielectric_constant)

        gb_tensor = gb_solver.compute_solvation_free_energy(
            x=x_flat,
            charges=q_pred,
            atomic_numbers=z_flat,
            atom_mask=m_flat,
            dielectric_constant=dielectric_constant,
        )

        # Normalized Boltzmann weights: w_k \propto exp(-\Delta E_k / k_B T)
        kb = 0.001987204  # kcal / (mol * K)
        kb_t = kb * max(10.0, float(temperature_k))
        if internal_energies is not None:
            if len(internal_energies.shape) == 1:
                e_int = internal_energies.reshape(B, s)
            else:
                e_int = internal_energies.reshape(B, s)

            # In polar solution (dielectric > 2.5), weight conformers by solution-phase potential of mean force:
            # G_eff(k) = E_int(k) + (Delta G_born(k) - Delta G_born(0))
            if dielectric_constant > 2.5:
                gb_2d = gb_tensor.reshape(B, s)
                gb_diff = gb_2d - gb_2d[:, :1]
                e_eff = e_int + gb_diff
            else:
                e_eff = e_int

            e_min = e_eff.min(axis=1, keepdim=True)
            delta_e = (e_eff - e_min).maximum(0.0).minimum(50.0)
            w_unnorm = (-delta_e / kb_t).exp()
            w = w_unnorm / w_unnorm.sum(axis=1, keepdim=True).maximum(1e-8)
        else:
            w = Tensor.full((B, s), 1.0 / float(s), dtype=dtypes.float32)

        total_solv_flat = delta_vdw_mol + gb_tensor  # (B * s,)
        total_solv_mean = (total_solv_flat.reshape(B, s) * w).sum(axis=1)  # (B,)
        gb_mean = (gb_tensor.reshape(B, s) * w).sum(axis=1)  # (B,)
        q_mean = (q_pred.reshape(B, s, N) * w.reshape(B, s, 1)).sum(axis=1)  # (B, N)

        if return_global:
            h_mol_mean = (graph_features.reshape(B, s, -1) * w.reshape(B, s, 1)).sum(axis=1)
            coop_mean = (delta_g_coop.reshape(B, s) * w).sum(axis=1)
            if not Tensor.training:
                h_mol_mean = h_mol_mean.realize()
                coop_mean = coop_mean.realize()
            return q_mean, total_solv_mean, gb_mean, h_mol_mean, coop_mean

        return q_mean, total_solv_mean, gb_mean

    def compute_charges(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
        total_charge: Optional[Tensor | float] = None,
        base_charges: Optional[Tensor] = None,
        solvent_features: Optional[Tensor] = None,
        detach_trunk: bool = False,
    ) -> Tensor:
        """
        Backwards-compatible interface predicting conformation-dependent quantum partial charges.
        Delegates to compute_solvation_readouts.
        """
        q_masked, _, _ = self.compute_solvation_readouts(
            x=x,
            atomic_numbers=atomic_numbers,
            atom_mask=atom_mask,
            molecule_mask=molecule_mask,
            total_charge=total_charge,
            base_charges=base_charges,
            solvent_features=solvent_features,
            detach_trunk=detach_trunk,
        )
        return q_masked

    def compute_forces(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Evaluates exact conservative forces F = -∇_x U(x) via reverse-mode autograd.
        Returns: Tensor of shape (B, N, 3).
        """
        _, forces = self.compute_energy_and_forces(
            x=x,
            atomic_numbers=atomic_numbers,
            atom_mask=atom_mask,
            molecule_mask=molecule_mask,
        )
        return forces

    def compute_energy_and_forces(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluates total potential energy U(x) and conservative forces F = -∇_x U(x)
        via reverse-mode autograd in a single unified graph traversal.
        Returns: Tuple of (energy (B,), forces (B, N, 3)).
        """
        x_in = x.detach()
        x_in.requires_grad = True

        u_total = self.compute_energy(
            x=x_in,
            atomic_numbers=atomic_numbers,
            atom_mask=atom_mask,
            molecule_mask=molecule_mask,
        )

        loss = u_total.sum()
        loss.backward()

        _, _, a_mask, _, _ = self._prepare_inputs(x_in, atomic_numbers, atom_mask, molecule_mask)
        grad = x_in.grad if x_in.grad is not None else Tensor.zeros_like(x_in)
        forces = (-grad * a_mask).realize()
        return u_total.realize(), forces

    def compute_energy_forces_and_charges(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
        total_charge: Optional[Tensor | float] = None,
        base_charges: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluates total potential energy U(x), conservative forces F = -∇_x U(x),
        and dynamic quantum partial charges q(x) in a single unified graph traversal.
        Returns: Tuple of (energy (B,), forces (B, N, 3), charges (B, N)).
        """
        x_in = x.detach()
        x_in.requires_grad = True

        x_prep, atomic_numbers, atom_mask, molecule_mask, edge_mask = self._prepare_inputs(
            x_in, atomic_numbers, atom_mask, molecule_mask
        )
        B, N, _ = x_prep.shape

        z_clamped = atomic_numbers.cast(dtypes.int32)
        z_one_hot = Tensor.one_hot(z_clamped, num_classes=self.max_atomic_number)
        h = self.embedding(z_one_hot) * atom_mask

        x_i = x_prep.reshape(B, N, 1, 3)
        x_j = x_prep.reshape(B, 1, N, 3)
        diff = x_i - x_j
        d_sq = (diff * diff).sum(axis=-1, keepdim=True)

        for layer in self.layers:
            h = layer(h, d_sq, edge_mask, atom_mask)

        # 1. Energy readout
        eps_i = self.readout_mlp[0](h)
        eps_i = self.readout_mlp[1](eps_i)
        eps_i = self.readout_mlp[2](eps_i) * atom_mask
        u_total = eps_i.sum(axis=(1, 2)) * molecule_mask

        # 2. Charge readout with residual electronegativity superposition & formal charge conservation
        from dens_city.cdft.generalized_born import GeneralizedBornSolvation

        gb = GeneralizedBornSolvation()
        sf = gb.compute_solvent_descriptors(x_prep, atomic_numbers, atom_mask, base_charges=base_charges)
        node_inputs = Tensor.cat(h, sf, dim=-1)

        delta_q_raw = self.charge_mlp[0](node_inputs)
        delta_q_raw = self.charge_mlp[1](delta_q_raw)
        delta_q_raw = self.charge_mlp[2](delta_q_raw)
        delta_q = self.max_delta_q * (delta_q_raw / self.max_delta_q).tanh() * atom_mask

        if base_charges is not None:
            if len(base_charges.shape) == 2:
                bq = base_charges.reshape(B, N, 1)
            else:
                bq = base_charges.reshape(B, N, 1)
            q_raw = (bq + delta_q) * atom_mask
        else:
            q_raw = delta_q

        num_real = atom_mask.sum(axis=1, keepdim=True).maximum(1.0)
        q_sum = q_raw.sum(axis=1, keepdim=True)
        if total_charge is not None:
            if isinstance(total_charge, (int, float)):
                q_target = Tensor.full((B, 1, 1), float(total_charge), dtype=dtypes.float32)
            else:
                q_target = total_charge.reshape(B, 1, 1).cast(dtypes.float32)
        else:
            q_target = Tensor.zeros(B, 1, 1, dtype=dtypes.float32)

        q_shift = (q_sum - q_target) / num_real
        q_masked = ((q_raw - q_shift) * atom_mask).reshape(B, N) * molecule_mask.reshape(B, 1)
        q_final = q_masked if Tensor.training else q_masked.realize()

        # 3. Forces via reverse-mode autograd
        loss = u_total.sum()
        loss.backward()

        grad = x_in.grad if x_in.grad is not None else Tensor.zeros_like(x_in)
        forces = (-grad * atom_mask).realize()
        return u_total.realize(), forces, q_final

    def get_jit_evaluator(self) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns a lazily-instantiated TinyJit compiled evaluator function
        mapping (x, atomic_numbers, atom_mask, molecule_mask) -> (u_total, forces).
        Traces the execution graph once and replays directly from GPU command queue.
        """
        if getattr(self, "_jit_evaluator", None) is None:
            # Ensure parameters are realized on device before tracing
            for p in nn.state.get_parameters(self):
                p.realize()

            def _step(x_in: Tensor, z_in: Tensor, a_mask: Tensor, m_mask: Tensor) -> Tuple[Tensor, Tensor]:
                u = self.compute_energy(x=x_in, atomic_numbers=z_in, atom_mask=a_mask, molecule_mask=m_mask)
                u.sum().backward()
                grad = x_in.grad if x_in.grad is not None else Tensor.zeros_like(x_in)
                f = (-grad * a_mask).realize()
                return u.realize(), f

            self._jit_evaluator = TinyJit(_step)
        return self._jit_evaluator

    def get_jit_relaxation_evaluator(
        self,
        relax_steps: int = 50,
        lr: float = 0.008,
        momentum: float = 0.85,
        force_tol: float = 5.0,
        max_disp: float = 0.15,
    ) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Returns a lazily-instantiated TinyJit compiled relaxation and evaluator function
        mapping (x_init, atomic_numbers, atom_mask, molecule_mask) -> (x_relaxed, u_final, forces_final).
        Performs an unrolled K-step Heavy-Ball momentum accelerated relaxation with adaptive displacement
        clamping and masked gradient-norm convergence on GPU.
        """
        attr_name = f"_jit_relax_evaluator_{relax_steps}_{lr}_{momentum}_{force_tol}_{max_disp}"
        if getattr(self, attr_name, None) is None:
            for p in nn.state.get_parameters(self):
                p.realize()

            force_tol_sq = float(force_tol * force_tol)

            def _step_relax(
                x_in: Tensor, z_in: Tensor, a_mask: Tensor, m_mask: Tensor
            ) -> Tuple[Tensor, Tensor, Tensor]:
                x_curr = x_in
                v_curr = Tensor.zeros_like(x_in)
                decay = 0.98

                for s in range(relax_steps):
                    x_curr.requires_grad = True
                    u = self.compute_energy(x=x_curr, atomic_numbers=z_in, atom_mask=a_mask, molecule_mask=m_mask)
                    u.sum().backward()
                    grad = x_curr.grad if x_curr.grad is not None else Tensor.zeros_like(x_curr)
                    forces = -grad * a_mask

                    # 1. Clip extreme gradient spikes on steric overlap
                    clipped_grad = (grad * a_mask).clip(-100.0, 100.0)

                    # 2. Heavy-Ball momentum velocity update with step decay
                    current_lr = lr * (decay**s)
                    v_next = momentum * v_curr + current_lr * clipped_grad

                    # 3. Adaptive displacement limit (starts at max_disp=0.15 A, decays to 0.05 A near minima)
                    disp_limit = max(0.05, max_disp * (decay**s))
                    step = v_next.clip(-disp_limit, disp_limit) * a_mask
                    x_cand = x_curr - step

                    # 4. Per-molecule maximum force magnitude squared
                    f_norm_sq = (forces * forces).sum(axis=-1, keepdim=True)
                    f_max_sq = (f_norm_sq * a_mask).max(axis=1, keepdim=True)

                    # 5. Masked convergence freezing: freeze coordinates and zero velocity for equilibrium molecules
                    converged_mask = f_max_sq < force_tol_sq
                    x_next = converged_mask.where(x_curr, x_cand).realize()
                    v_next_masked = converged_mask.where(Tensor.zeros_like(v_curr), v_next).realize()

                    x_curr = x_next.detach()
                    v_curr = v_next_masked.detach()

                x_curr.requires_grad = True
                u_final = self.compute_energy(x=x_curr, atomic_numbers=z_in, atom_mask=a_mask, molecule_mask=m_mask)
                u_final.sum().backward()
                grad_final = x_curr.grad if x_curr.grad is not None else Tensor.zeros_like(x_curr)
                forces_final = (-grad_final * a_mask).realize()
                return x_curr.realize(), u_final.realize(), forces_final.realize()

            setattr(self, attr_name, TinyJit(_step_relax))
        return getattr(self, attr_name)
