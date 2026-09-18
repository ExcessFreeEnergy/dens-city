"""
End-to-End Differentiable Quantum Charge Trainer.
Optimizes the EGNN dynamic charge readout head (charge_mlp) and message-passing trunk
directly against experimental FreeSolv hydration free energies (ΔG_expt) using Huber loss,
physical hyperbolic tangent (tanh) charge bounds, and L2 perturbation regularization.

Adheres strictly to canonical tinygrad beautiful_mnist.py and tinyspec.tex standards:
1. Pure @TinyJit single-graph GPU fusion: forward prediction, Generalized Born dielectric
   solvation, Huber loss, backward autograd, and optimizer updates compile into a static
   hardware command buffer.
2. Static contiguous dataset packing: all 643 FreeSolv molecules (padded to static batch slots)
   are loaded onto GPU memory once (totaling <4 MB VRAM).
3. On-device Threefry PRNG sampling: mini-batches are sampled via Tensor.randint directly in GPU
   registers, preserving 100% input buffer identity invariance across training steps.
4. Sequential JIT evaluation: validation sweeps across 21 static slices via static_eval_idx.assign(...)
   to cover 100% of the dataset deterministically without random dropout, shape recompilation, or VRAM explosion.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tinygrad import GlobalCounters, Tensor, TinyJit, dtypes, nn
from tinygrad.helpers import colored

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.cdft.generalized_born import GeneralizedBornSolvation
from dens_city.utils.materials import MaterialLoader

# Authoritative feature scaling constants for Delta-KRR joint representations
PHYSICAL_DESCRIPTOR_WEIGHT: float = 2.0
SOLVENT_DESCRIPTOR_WEIGHT: float = 2.5


@dataclass
class ChargeTrainingConfig:
    epochs: int = 1000
    warmup_epochs: int = 50
    lr_head: float = 8e-4
    lr_trunk: float = 2e-5
    lr_min: float = 1e-7
    batch_size: int = 32
    huber_delta: float = 2.5  # kcal/mol (expanded quadratic MSE basin)
    lambda_l2: float = 0.02  # Penalty on (Δq)^2
    lambda_vdw: float = 0.002  # Penalty on (Δg_vdw)^2 (calibrated for max_delta_vdw=3.5)
    lambda_global: float = 0.0005  # Penalty on (Δg_coop)^2
    max_delta_q: float = 0.25  # Max allowed perturbation |Δq| <= 0.25e
    max_delta_vdw: float = 3.5  # Max allowed atomic nonpolar perturbation |Δg_vdw| <= 3.5 kcal/mol
    max_delta_global: float = 25.0  # Max allowed molecular cooperative perturbation |ΔG_coop| <= 25.0 kcal/mol
    n_particles: int = 128
    hidden_dim: int = 128
    num_layers: int = 7
    n_conformers: int = 8  # Number of ensemble conformers per molecule during ensembled training
    dielectric_constant: float = 78.4
    weights_out: str = "data/checkpoints/egnn_charges_trained.npz"
    database_path: str = "FreeSolv/database.pickle"
    mol2_dir: str = "data/mol2files_gaff"


@dataclass
class PreprocessedBatch:
    coords: Tensor  # (B, N, 3)
    atomic_numbers: Tensor  # (B, N)
    atom_mask: Tensor  # (B, N, 1)
    base_charges: Tensor  # (B, N)
    total_charges: Tensor  # (B, 1, 1)
    vdw_energies: Tensor  # (B,)
    expt_energies: Tensor  # (B,)
    material_names: List[str]
    num_real_atoms: List[int]
    cached_h: Optional[Tensor] = None  # (B, N, 128) - discarded in Phase 2
    cached_solvent_features: Optional[Tensor] = None  # (B, N, 4) - detached 3D solvent descriptors


@dataclass
class ContiguousPackedDataset:
    coords: Tensor  # (TOTAL_PADDED, N, 3)
    atomic_numbers: Tensor  # (TOTAL_PADDED, N)
    atom_mask: Tensor  # (TOTAL_PADDED, N, 1)
    base_charges: Tensor  # (TOTAL_PADDED, N)
    total_charges: Tensor  # (TOTAL_PADDED, 1, 1)
    vdw_energies: Tensor  # (TOTAL_PADDED,)
    expt_energies: Tensor  # (TOTAL_PADDED,)
    solvent_features: Tensor  # (TOTAL_PADDED, N, 4)
    cached_h: Optional[Tensor] = None  # (TOTAL_PADDED, N, 128)
    coords_ensemble: Optional[Tensor] = None  # (TOTAL_PADDED, s, N, 3)
    material_names: List[str] = field(default_factory=list)
    num_real_molecules: int = 0
    total_padded_molecules: int = 0
    s_conformers: int = 1


class QuantumChargeTrainer:
    """
    End-to-End Differentiable Quantum Charge Trainer.
    Trains EGNN on FreeSolv database with single-sweep hardware realization,
    on-device Threefry PRNG sampling, and sequential JIT evaluation.
    """

    def __init__(self, config: Optional[ChargeTrainingConfig] = None):
        self.config = config or ChargeTrainingConfig()
        self.loader = MaterialLoader()

        # Initialize EGNN with untrained/zero-init charge head
        self.ff = EGNNForceField(
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            n_particles=self.config.n_particles,
            load_default_weights=False,
        )

        # Generalized Born continuous dielectric solver
        self.gb = GeneralizedBornSolvation(dielectric_constant=self.config.dielectric_constant)

        # Parameters: triple readout heads (charge_mlp + vdw_mlp + global_mlp)
        self.head_params = (
            nn.state.get_parameters(self.ff.charge_mlp)
            + nn.state.get_parameters(self.ff.vdw_mlp)
            + nn.state.get_parameters(self.ff.global_mlp)
        )
        self.charge_params = self.head_params  # Maintain alias for backwards compatibility

        trunk_layers = [self.ff.embedding] + list(self.ff.layers)
        self.trunk_params = []
        for lyr in trunk_layers:
            self.trunk_params.extend(nn.state.get_parameters(lyr))

        # Optimizers: dual learning rates combined in OptimizerGroup
        self.opt_head = nn.optim.Adam(self.head_params, lr=self.config.lr_head)
        self.opt_trunk = nn.optim.Adam(self.trunk_params, lr=self.config.lr_trunk)
        self.opt_group = nn.optim.OptimizerGroup(self.opt_head, self.opt_trunk)

        # Static index buffer for sequential deterministic evaluation
        self.static_eval_idx = Tensor(np.arange(self.config.batch_size, dtype=np.int32)).realize()
        self.dataset: Optional[ContiguousPackedDataset] = None

    def load_static_dataset(self, dataset_provider: Optional[Any] = None) -> ContiguousPackedDataset:
        """Loads and packs molecular dataset into static contiguous device tensors."""
        records = []
        if dataset_provider is not None:
            entries = dataset_provider.load_entries()
            for entry in entries:
                try:
                    mat = dataset_provider.get_material(entry.solute_id) or self.loader.load_material(entry.solute_id)
                    if mat is None or mat.num_sites == 0:
                        continue
                    mat.compute_topological_base_charges(kappa=0.10, q_max=0.50)
                    vdw = float(
                        entry.calc_dG_solv
                        if entry.calc_dG_solv is not None
                        else getattr(mat, "solvation_free_energy_kcal_mol", 0.0)
                    )
                    expt = float(entry.expt_dG_solv)
                    records.append((mat, vdw, expt))
                except Exception:
                    continue
        else:
            db_path = Path(self.config.database_path)
            mol2_dir = Path(self.config.mol2_dir)

            if not db_path.exists():
                raise FileNotFoundError(f"Database not found at: {db_path}")

            with open(db_path, "rb") as f:
                fs_db: Dict[str, Dict] = pickle.load(f, encoding="latin1")

            mol2_files = sorted(list(mol2_dir.glob("*.mol2")))

            for p in mol2_files:
                stem = p.stem
                if stem not in fs_db:
                    continue
                entry = fs_db[stem]
                expt = float(entry.get("expt", 0.0))
                vdw = float(entry.get("calc_vdw", 0.0))

                try:
                    mat = self.loader.load_material(str(p))
                    if mat.num_sites == 0:
                        continue
                    mat.compute_topological_base_charges(kappa=0.10, q_max=0.50)
                    records.append((mat, vdw, expt))
                except Exception:
                    continue

        # Add pure water anchor if not present
        if not any(mat.name == "water" for mat, _, _ in records):
            try:
                mat_water = self.loader.load_material("water")
                mat_water.compute_topological_base_charges(kappa=0.10, q_max=0.50)
                records.append((mat_water, 4.02, -6.30))
            except Exception:
                pass

        B = self.config.batch_size
        N = self.config.n_particles
        num_real = len(records)
        total_padded = ((num_real + B - 1) // B) * B

        coords_np = np.zeros((total_padded, N, 3), dtype=np.float32)
        z_np = np.zeros((total_padded, N), dtype=np.float32)
        mask_np = np.zeros((total_padded, N, 1), dtype=np.float32)
        bq_np = np.zeros((total_padded, N), dtype=np.float32)
        tot_q_np = np.zeros((total_padded, 1, 1), dtype=np.float32)
        vdw_np = np.zeros((total_padded,), dtype=np.float32)
        expt_np = np.zeros((total_padded,), dtype=np.float32)
        names: List[str] = []

        for i, (mat, vdw, expt) in enumerate(records):
            n_sites = min(N, mat.num_sites)
            for s_idx, site in enumerate(mat.sites[:n_sites]):
                coords_np[i, s_idx] = [site.x, site.y, site.z]
                z_np[i, s_idx] = getattr(site, "atomic_number", 6)
                mask_np[i, s_idx, 0] = 1.0

            if mat.base_charges:
                bq_np[i, :n_sites] = mat.base_charges[:n_sites]

            tot_q_np[i, 0, 0] = float(mat.total_charge)
            vdw_np[i] = vdw
            expt_np[i] = expt
            names.append(mat.name)

        # Pad dummy names for remaining slots
        for pad_i in range(num_real, total_padded):
            names.append(f"pad_dummy_{pad_i}")

        coords_t = Tensor(coords_np).contiguous().realize()
        z_t = Tensor(z_np).contiguous().realize()
        mask_t = Tensor(mask_np).contiguous().realize()
        bq_t = Tensor(bq_np).contiguous().realize()
        tot_q_t = Tensor(tot_q_np).contiguous().realize()
        vdw_t = Tensor(vdw_np).contiguous().realize()
        expt_t = Tensor(expt_np).contiguous().realize()

        # Precompute solvent descriptors and Phase 1 cached_h in static chunks
        Tensor.training = False
        sf_chunks = []
        h_chunks = []
        for start_idx in range(0, total_padded, B):
            c_chunk = coords_t[start_idx : start_idx + B]
            z_chunk = z_t[start_idx : start_idx + B]
            m_chunk = mask_t[start_idx : start_idx + B]
            bq_chunk = bq_t[start_idx : start_idx + B]

            sf_chunk = (
                self.gb.compute_solvent_descriptors(c_chunk, z_chunk, m_chunk, base_charges=bq_chunk).detach().realize()
            )
            sf_chunks.append(sf_chunk)

            x_prep, z_prep, atom_mask, _, edge_mask = self.ff._prepare_inputs(c_chunk, z_chunk, m_chunk, None)
            z_clamped = z_prep.cast(dtypes.int32)
            z_one_hot = Tensor.one_hot(z_clamped, num_classes=self.ff.max_atomic_number)
            h = self.ff.embedding(z_one_hot) * atom_mask
            x_i = x_prep.reshape(B, N, 1, 3)
            x_j = x_prep.reshape(B, 1, N, 3)
            diff = x_i - x_j
            d_sq = (diff * diff).sum(axis=-1, keepdim=True)
            for layer in self.ff.layers:
                h = layer(h, d_sq, edge_mask, atom_mask)
            h_chunks.append(h.detach().realize())

        sf_t = Tensor.cat(*sf_chunks, dim=0).contiguous().realize()
        h_t = Tensor.cat(*h_chunks, dim=0).contiguous().realize()

        # Multi-conformer ensemble generation (Weinreich FML principle):
        # Conformer 0 is the exact ground state; conformers 1..s-1 sample the thermal conformational basin
        s_conf = getattr(self.config, "n_conformers", 8)
        coords_ens_np = np.zeros((total_padded, s_conf, N, 3), dtype=np.float32)
        rng = np.random.default_rng(42)
        coords_ens_np[:, 0] = coords_np
        for k in range(1, s_conf):
            coords_ens_np[:, k] = coords_np + rng.normal(0.0, 0.05, (total_padded, N, 3)).astype(np.float32) * mask_np
        coords_ens_t = Tensor(coords_ens_np).contiguous().realize()

        ds = ContiguousPackedDataset(
            coords=coords_t,
            atomic_numbers=z_t,
            atom_mask=mask_t,
            base_charges=bq_t,
            total_charges=tot_q_t,
            vdw_energies=vdw_t,
            expt_energies=expt_t,
            solvent_features=sf_t,
            cached_h=h_t,
            coords_ensemble=coords_ens_t,
            material_names=names,
            num_real_molecules=num_real,
            total_padded_molecules=total_padded,
            s_conformers=s_conf,
        )
        self.dataset = ds
        return ds

    def load_dataset(self) -> List[PreprocessedBatch]:
        """Loads legacy PreprocessedBatch list for compatibility with existing unit tests."""
        static_ds = self.load_static_dataset()
        B = self.config.batch_size
        batches: List[PreprocessedBatch] = []
        for start_idx in range(0, static_ds.total_padded_molecules, B):
            c_b = static_ds.coords[start_idx : start_idx + B].realize()
            z_b = static_ds.atomic_numbers[start_idx : start_idx + B].realize()
            m_b = static_ds.atom_mask[start_idx : start_idx + B].realize()
            bq_b = static_ds.base_charges[start_idx : start_idx + B].realize()
            tq_b = static_ds.total_charges[start_idx : start_idx + B].realize()
            v_b = static_ds.vdw_energies[start_idx : start_idx + B].realize()
            e_b = static_ds.expt_energies[start_idx : start_idx + B].realize()
            sf_b = static_ds.solvent_features[start_idx : start_idx + B].realize()
            h_b = static_ds.cached_h[start_idx : start_idx + B].realize() if static_ds.cached_h is not None else None
            names_b = static_ds.material_names[start_idx : start_idx + B]
            n_real = [int(m_b[i].sum().item()) for i in range(B)]
            batches.append(
                PreprocessedBatch(
                    coords=c_b,
                    atomic_numbers=z_b,
                    atom_mask=m_b,
                    base_charges=bq_b,
                    total_charges=tq_b,
                    vdw_energies=v_b,
                    expt_energies=e_b,
                    material_names=names_b,
                    num_real_atoms=n_real,
                    cached_h=h_b,
                    cached_solvent_features=sf_b,
                )
            )
        return batches

    @TinyJit
    def _train_step_p1(
        self,
        coords: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Tensor,
        base_charges: Tensor,
        total_charges: Tensor,
        vdw_energies: Tensor,
        expt_energies: Tensor,
        solvent_features: Tensor,
        cached_h: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Phase 1: High-throughput JIT step optimizing charge_mlp with frozen cached_h."""
        self.opt_head.zero_grad()
        B = self.config.batch_size
        N = self.config.n_particles
        high_val = self.dataset.num_real_molecules if self.dataset is not None else coords.shape[0]
        idx = Tensor.randint(B, high=high_val)

        c = coords[idx]
        z = atomic_numbers[idx]
        m = atom_mask[idx]
        bq = base_charges[idx]
        tq = total_charges[idx]
        v = vdw_energies[idx]
        e = expt_energies[idx]
        sf = solvent_features[idx]
        h = cached_h[idx]

        node_inputs = Tensor.cat(h, sf, dim=-1)

        # Head 1: Neural Charge Readout
        delta_q_raw = self.ff.charge_mlp[0](node_inputs)
        delta_q_raw = self.ff.charge_mlp[1](delta_q_raw)
        delta_q_raw = self.ff.charge_mlp[2](delta_q_raw)
        delta_q = self.config.max_delta_q * (delta_q_raw / self.config.max_delta_q).tanh() * m

        q_raw = (bq.reshape(B, N, 1) + delta_q) * m
        num_real = m.sum(axis=1, keepdim=True).maximum(1.0)
        q_sum = q_raw.sum(axis=1, keepdim=True)
        q_shift = (q_sum - tq) / num_real
        q_pred = ((q_raw - q_shift) * m).reshape(B, N)

        # Head 2: Volumetric Nonpolar Cavitation Readout
        delta_vdw_raw = self.ff.vdw_mlp[0](node_inputs)
        delta_vdw_raw = self.ff.vdw_mlp[1](delta_vdw_raw)
        delta_vdw_raw = self.ff.vdw_mlp[2](delta_vdw_raw)
        delta_vdw_atomic = self.config.max_delta_vdw * (delta_vdw_raw / self.config.max_delta_vdw).tanh() * m
        delta_vdw_mol_atomic = delta_vdw_atomic.sum(axis=(1, 2))  # (B,)

        # Head 3: Multi-Scale Graph Pooling & Cooperative Readout
        num_real_nodes = m.sum(axis=1).maximum(1.0)
        mean_pool = (h * m).sum(axis=1) / num_real_nodes
        h_masked = h * m - (1.0 - m) * 1e4
        max_pool = h_masked.max(axis=1)
        h_diff = (h - mean_pool.reshape(B, 1, self.config.hidden_dim)) * m
        var_pool = (h_diff * h_diff).sum(axis=1) / num_real_nodes
        std_pool = (var_pool + 1e-6).sqrt()
        graph_features = Tensor.cat(mean_pool, max_pool, std_pool, dim=-1)
        delta_coop_raw = self.ff.global_mlp[0](graph_features)
        delta_coop_raw = self.ff.global_mlp[1](delta_coop_raw)
        delta_coop_raw = self.ff.global_mlp[2](delta_coop_raw).reshape(B)
        delta_g_coop = self.ff.max_delta_global * (delta_coop_raw / self.ff.max_delta_global).tanh()
        delta_vdw_mol = delta_vdw_mol_atomic + delta_g_coop

        dg_gb = self.gb.compute_solvation_free_energy(
            c, q_pred, z, m, dielectric_constant=self.config.dielectric_constant
        )
        dg_calc = v + delta_vdw_mol + dg_gb
        valid_mol = (m.sum(axis=(1, 2)) > 0).cast(dtypes.float32)
        num_valid = valid_mol.sum().maximum(1.0)
        err = (dg_calc - e) * valid_mol
        abs_err = err.abs() * valid_mol

        delta = self.config.huber_delta
        huber_terms = (abs_err <= delta).where(0.5 * err * err, delta * (abs_err - 0.5 * delta)) * valid_mol
        huber_loss = huber_terms.sum() / num_valid

        num_real_total = m.sum().maximum(1.0)
        l2_q = self.config.lambda_l2 * (delta_q * delta_q).sum() / num_real_total
        l2_vdw = self.config.lambda_vdw * (delta_vdw_atomic * delta_vdw_atomic).sum() / num_real_total
        l2_coop = self.config.lambda_global * (delta_g_coop * delta_g_coop).sum() / num_valid

        loss = (huber_loss + l2_q + l2_vdw + l2_coop).reshape(())
        loss.backward()

        mae_metric = abs_err.sum() / num_valid
        max_dq_metric = delta_q.abs().max()

        Tensor.realize(loss, mae_metric, max_dq_metric, *self.opt_head.schedule_step())
        return loss, mae_metric, max_dq_metric

    @TinyJit
    def _train_step_p2(
        self,
        coords: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Tensor,
        base_charges: Tensor,
        total_charges: Tensor,
        vdw_energies: Tensor,
        expt_energies: Tensor,
        solvent_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Phase 2: End-to-end JIT step optimizing both trunk and dual heads via OptimizerGroup."""
        self.opt_group.zero_grad()
        B = self.config.batch_size
        N = self.config.n_particles
        high_val = self.dataset.num_real_molecules if self.dataset is not None else coords.shape[0]
        idx = Tensor.randint(B, high=high_val)

        c = coords[idx]
        z = atomic_numbers[idx]
        m = atom_mask[idx]
        bq = base_charges[idx]
        tq = total_charges[idx]
        v = vdw_energies[idx]
        e = expt_energies[idx]
        sf = solvent_features[idx]

        q_pred, delta_vdw_mol, delta_vdw_atomic = self.ff.compute_solvation_readouts(
            x=c,
            atomic_numbers=z,
            atom_mask=m,
            total_charge=tq,
            base_charges=bq,
            solvent_features=sf,
            detach_trunk=False,
        )
        delta_q = (q_pred - bq) * m.reshape(B, N)

        dg_gb = self.gb.compute_solvation_free_energy(
            c, q_pred, z, m, dielectric_constant=self.config.dielectric_constant
        )
        dg_calc = v + delta_vdw_mol + dg_gb
        valid_mol = (m.sum(axis=(1, 2)) > 0).cast(dtypes.float32)
        num_valid = valid_mol.sum().maximum(1.0)
        err = (dg_calc - e) * valid_mol
        abs_err = err.abs() * valid_mol

        delta = self.config.huber_delta
        huber_terms = (abs_err <= delta).where(0.5 * err * err, delta * (abs_err - 0.5 * delta)) * valid_mol
        huber_loss = huber_terms.sum() / num_valid

        num_real_total = m.sum().maximum(1.0)
        l2_q = self.config.lambda_l2 * (delta_q * delta_q).sum() / num_real_total
        l2_vdw = self.config.lambda_vdw * (delta_vdw_atomic * delta_vdw_atomic).sum() / num_real_total

        loss = (huber_loss + l2_q + l2_vdw).reshape(())
        loss.backward()

        mae_metric = abs_err.sum() / num_valid
        max_dq_metric = delta_q.abs().max()

        Tensor.realize(loss, mae_metric, max_dq_metric, *self.opt_group.schedule_step())
        return loss, mae_metric, max_dq_metric

    def evaluate(
        self, batches: Optional[List[PreprocessedBatch]] = None
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Evaluates MAE, RMSE, and individual predictions deterministically across 100% of the dataset.
        Executes sequential batch slices across all molecules without JIT return-buffer collisions.
        """
        Tensor.training = False

        if self.dataset is not None:
            B = self.config.batch_size
            N = self.config.n_particles
            num_batches = self.dataset.total_padded_molecules // B
            all_errors: List[float] = []
            preds: Dict[str, float] = {}

            for b_idx in range(num_batches):
                start = b_idx * B
                c = self.dataset.coords[start : start + B]
                z = self.dataset.atomic_numbers[start : start + B]
                m = self.dataset.atom_mask[start : start + B]
                bq = self.dataset.base_charges[start : start + B]
                tq = self.dataset.total_charges[start : start + B]
                v = self.dataset.vdw_energies[start : start + B]
                e = self.dataset.expt_energies[start : start + B]
                sf = self.dataset.solvent_features[start : start + B]

                if self.dataset.cached_h is not None:
                    # Phase 1 fast evaluation: use cached_h through charge_mlp and vdw_mlp
                    h = self.dataset.cached_h[start : start + B]
                    node_inputs = Tensor.cat(h, sf, dim=-1)
                    delta_q_raw = self.ff.charge_mlp[0](node_inputs)
                    delta_q_raw = self.ff.charge_mlp[1](delta_q_raw)
                    delta_q_raw = self.ff.charge_mlp[2](delta_q_raw)
                    delta_q = self.config.max_delta_q * (delta_q_raw / self.config.max_delta_q).tanh() * m
                    q_raw = (bq.reshape(B, N, 1) + delta_q) * m
                    num_real = m.sum(axis=1, keepdim=True).maximum(1.0)
                    q_sum = q_raw.sum(axis=1, keepdim=True)
                    q_shift = (q_sum - tq) / num_real
                    q_pred = ((q_raw - q_shift) * m).reshape(B, N)

                    delta_vdw_raw = self.ff.vdw_mlp[0](node_inputs)
                    delta_vdw_raw = self.ff.vdw_mlp[1](delta_vdw_raw)
                    delta_vdw_raw = self.ff.vdw_mlp[2](delta_vdw_raw)
                    delta_vdw_atomic = (
                        self.config.max_delta_vdw * (delta_vdw_raw / self.config.max_delta_vdw).tanh() * m
                    )
                    delta_vdw_mol_atomic = delta_vdw_atomic.sum(axis=(1, 2))

                    # Molecular cooperative free energy readout in Phase 1
                    num_real_nodes = m.sum(axis=1).maximum(1.0)
                    mean_pool = (h * m).sum(axis=1) / num_real_nodes
                    h_masked = h * m - (1.0 - m) * 1e4
                    max_pool = h_masked.max(axis=1)
                    h_diff = (h - mean_pool.reshape(B, 1, self.config.hidden_dim)) * m
                    var_pool = (h_diff * h_diff).sum(axis=1) / num_real_nodes
                    std_pool = (var_pool + 1e-6).sqrt()
                    graph_features = Tensor.cat(mean_pool, max_pool, std_pool, dim=-1)
                    delta_coop_raw = self.ff.global_mlp[0](graph_features)
                    delta_coop_raw = self.ff.global_mlp[1](delta_coop_raw)
                    delta_coop_raw = self.ff.global_mlp[2](delta_coop_raw).reshape(B)
                    delta_g_coop = self.ff.max_delta_global * (delta_coop_raw / self.ff.max_delta_global).tanh()
                    delta_vdw_mol = delta_vdw_mol_atomic + delta_g_coop
                else:
                    # Phase 2 evaluation: full forward pass through EGNN trunk and all three heads
                    q_pred, delta_vdw_mol, _ = self.ff.compute_solvation_readouts(
                        x=c,
                        atomic_numbers=z,
                        atom_mask=m,
                        total_charge=tq,
                        base_charges=bq,
                        solvent_features=sf,
                        detach_trunk=True,
                    )

                dg_gb = self.gb.compute_solvation_free_energy(
                    c, q_pred, z, m, dielectric_constant=self.config.dielectric_constant
                )
                dg_calc = v + delta_vdw_mol + dg_gb
                valid_mol = (m.sum(axis=(1, 2)) > 0).cast(dtypes.float32)
                err = ((dg_calc - e) * valid_mol).realize()
                calc_realized = dg_calc.realize()

                n_valid = max(0, min(B, self.dataset.num_real_molecules - start))
                err_np = err.numpy()[:n_valid]
                calc_np = calc_realized.numpy()[:n_valid]

                for i in range(n_valid):
                    mol_idx = start + i
                    mol_name = self.dataset.material_names[mol_idx]
                    all_errors.append(abs(float(err_np[i])))
                    preds[mol_name] = float(calc_np[i])

            mae = float(np.mean(all_errors))
            rmse = float(np.sqrt(np.mean(np.square(all_errors))))
            max_err = float(np.max(all_errors))
            return mae, rmse, max_err, preds
        else:
            # Fallback for manual batches argument
            assert batches is not None, "Neither self.dataset nor batches provided to evaluate()"
            errors: List[float] = []
            preds_fb: Dict[str, float] = {}
            for b in batches:
                q_pred = self.ff.compute_charges(
                    x=b.coords,
                    atomic_numbers=b.atomic_numbers,
                    atom_mask=b.atom_mask,
                    total_charge=b.total_charges,
                    base_charges=b.base_charges,
                    solvent_features=b.cached_solvent_features,
                    detach_trunk=True,
                )
                dg_gb = self.gb.compute_solvation_free_energy(
                    x=b.coords,
                    charges=q_pred,
                    atomic_numbers=b.atomic_numbers,
                    atom_mask=b.atom_mask,
                    dielectric_constant=self.config.dielectric_constant,
                )
                dg_calc = (b.vdw_energies + dg_gb).numpy()
                dg_expt = b.expt_energies.numpy()
                for name, calc, expt in zip(b.material_names, dg_calc, dg_expt):
                    err = abs(calc - expt)
                    errors.append(err)
                    preds_fb[name] = float(calc)
            mae = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            max_err = float(np.max(errors))
            return mae, rmse, max_err, preds_fb

    def fit_krr_head(
        self,
        sigma: float = 10.0,
        reg_lambda: float = 0.1,
        save_path: Optional[str] = None,
        solvent_descriptors: Optional[np.ndarray] = None,
        include_solvent_descriptors: bool = True,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Fits an analytical Delta-KRR residual stacking model on top of the trained EGNN readouts.
        Features:
          Z: standardized 384-dimensional multi-scale pooled graph embeddings (mean, max, std).
          D_phys: standardized 6-dimensional physical descriptors (N_heavy, N_O, N_N, N_hal, ΔG_GB, v_vdw).
        Target:
          y_res = ΔG_expt - ΔG_EGNN.
        Computes exact closed-form matrix inversion and Sherman-Morrison LOOCV predictions.
        Saves kernel weights, training representations, and normalization statistics to save_path.
        Returns:
            mae_loo: float - Leave-one-out cross-validation MAE (kcal/mol)
            rmse_loo: float - Leave-one-out cross-validation RMSE (kcal/mol)
            preds_loo: Dict[str, float] - LOOCV prediction per molecule
        """
        if self.dataset is None:
            self.dataset = self.load_static_dataset()
        Tensor.training = False

        B = self.config.batch_size
        num_real = self.dataset.num_real_molecules
        num_batches = self.dataset.total_padded_molecules // B

        pooled_list = []
        base_calc_list = []
        expt_list = []
        desc_list = []
        names_list = []

        for b_idx in range(num_batches):
            start = b_idx * B
            c = self.dataset.coords[start : start + B]
            z = self.dataset.atomic_numbers[start : start + B]
            m = self.dataset.atom_mask[start : start + B]
            bq = self.dataset.base_charges[start : start + B]
            tq = self.dataset.total_charges[start : start + B]
            v = self.dataset.vdw_energies[start : start + B]
            e = self.dataset.expt_energies[start : start + B]
            sf = self.dataset.solvent_features[start : start + B]

            q_pred, delta_vdw_mol, delta_vdw_atomic, delta_g_coop, graph_features = self.ff.compute_solvation_readouts(
                x=c,
                atomic_numbers=z,
                atom_mask=m,
                total_charge=tq,
                base_charges=bq,
                solvent_features=sf,
                detach_trunk=True,
                return_global=True,
            )
            dg_gb = self.gb.compute_solvation_free_energy(
                c, q_pred, z, m, dielectric_constant=self.config.dielectric_constant
            )
            egnn_pred = v + delta_vdw_mol + dg_gb

            # Physical descriptor counts: O, N, halogens, heavy atoms, dg_gb, v
            z_np = z.numpy()
            m_np = m.numpy().squeeze(-1)
            n_heavy = np.sum((z_np > 1) * m_np, axis=1, keepdims=True)
            n_o = np.sum((z_np == 8) * m_np, axis=1, keepdims=True)
            n_n = np.sum((z_np == 7) * m_np, axis=1, keepdims=True)
            n_hal = np.sum(np.isin(z_np, [9, 17, 35, 53]) * m_np, axis=1, keepdims=True)
            dg_gb_np = dg_gb.numpy().reshape(-1, 1)
            v_np = v.numpy().reshape(-1, 1)
            phys_desc = np.concatenate([n_heavy, n_o, n_n, n_hal, dg_gb_np, v_np], axis=1)

            n_valid = max(0, min(B, num_real - start))
            if n_valid > 0:
                pooled_list.append(graph_features.numpy()[:n_valid])
                base_calc_list.append(egnn_pred.numpy()[:n_valid])
                expt_list.append(e.numpy()[:n_valid])
                desc_list.append(phys_desc[:n_valid])
                names_list.extend(self.dataset.material_names[start : start + n_valid])

        Z = np.concatenate(pooled_list, axis=0)  # (N_real, 384)
        D_phys = np.concatenate(desc_list, axis=0)  # (N_real, 6)
        y_egnn = np.concatenate(base_calc_list, axis=0)  # (N_real,)
        y_expt = np.concatenate(expt_list, axis=0)  # (N_real,)
        y_res = y_expt - y_egnn  # Residual target

        # Standardize features
        z_mean = np.mean(Z, axis=0, keepdims=True)
        z_std = np.std(Z, axis=0, keepdims=True) + 1e-6
        d_mean = np.mean(D_phys, axis=0, keepdims=True)
        d_std = np.std(D_phys, axis=0, keepdims=True) + 1e-6

        Z_norm = (Z - z_mean) / z_std
        D_norm = (D_phys - d_mean) / d_std

        if include_solvent_descriptors:
            if solvent_descriptors is None:
                from dens_city.utils.solvents import get_solvent_descriptors_vector

                s_water = get_solvent_descriptors_vector("water")
                S_solv = np.tile(s_water, (num_real, 1))
            else:
                S_solv = np.asarray(solvent_descriptors, dtype=np.float32)

            s_mean = np.mean(S_solv, axis=0, keepdims=True)
            s_std = np.std(S_solv, axis=0, keepdims=True) + 1e-6
            S_norm = (S_solv - s_mean) / s_std
            Z_comb = np.concatenate(
                [Z_norm, D_norm * PHYSICAL_DESCRIPTOR_WEIGHT, S_norm * SOLVENT_DESCRIPTOR_WEIGHT], axis=1
            )  # (N_real, 397)
        else:
            s_mean = None
            s_std = None
            Z_comb = np.concatenate([Z_norm, D_norm * PHYSICAL_DESCRIPTOR_WEIGHT], axis=1)  # (N_real, 390)

        # Pairwise squared Euclidean distances between representations
        z_sq = np.sum(Z_comb**2, axis=1, keepdims=True)
        D2 = np.maximum(z_sq + z_sq.T - 2.0 * np.dot(Z_comb, Z_comb.T), 0.0)

        # Gaussian RBF Kernel matrix: K_ij = exp(-D2_ij / (2 * sigma^2))
        K = np.exp(-D2 / (2.0 * (sigma**2)))
        A = K + reg_lambda * np.eye(num_real, dtype=np.float64)

        # Numerically stable Cholesky decomposition for Symmetric Positive-Definite (SPD) Gram matrix
        import scipy.linalg

        c_and_lower = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
        alpha = scipy.linalg.cho_solve(c_and_lower, y_res.astype(np.float64), check_finite=False).astype(np.float32)

        # Analytical Sherman-Morrison LOOCV via triangular inverse: diag(A^-1)_i = ||V_{:, i}||^2
        L = c_and_lower[0]
        V = scipy.linalg.solve_triangular(L, np.eye(num_real, dtype=np.float64), lower=True, check_finite=False)
        diag_A_inv = np.sum(V**2, axis=0).astype(np.float32)
        y_loo_res = y_res - (alpha / diag_A_inv)
        y_loo_pred = y_egnn + y_loo_res

        loo_errors = np.abs(y_loo_pred - y_expt)
        mae_loo = float(np.mean(loo_errors))
        rmse_loo = float(np.sqrt(np.mean(loo_errors**2)))
        preds_loo = {name: float(pred) for name, pred in zip(names_list, y_loo_pred)}

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            save_dict = {
                "alpha": alpha.astype(np.float32),
                "z_train": Z_comb.astype(np.float32),
                "z_mean": z_mean.astype(np.float32),
                "z_std": z_std.astype(np.float32),
                "d_mean": d_mean.astype(np.float32),
                "d_std": d_std.astype(np.float32),
                "sigma": float(sigma),
                "reg_lambda": float(reg_lambda),
                "names": np.array(names_list),
            }
            if s_mean is not None and s_std is not None:
                save_dict["s_mean"] = s_mean.astype(np.float32)
                save_dict["s_std"] = s_std.astype(np.float32)
            np.savez_compressed(p, **save_dict)

        return mae_loo, rmse_loo, preds_loo

    def train_epoch(
        self,
        batches: List[PreprocessedBatch],
        lr_head: float,
        lr_trunk: float,
        phase: int,
    ) -> Tuple[float, float, float]:
        """Legacy epoch training method maintained for unit test verification."""
        self.opt_head.lr.assign(Tensor([lr_head], dtype=dtypes.float32, device=self.opt_head.device)).realize()
        if phase == 2:
            self.opt_trunk.lr.assign(Tensor([lr_trunk], dtype=dtypes.float32, device=self.opt_trunk.device)).realize()

        total_loss = 0.0
        total_mae = 0.0
        total_max_dq = 0.0
        total_molecules = 0

        delta = self.config.huber_delta
        lambda_l2 = self.config.lambda_l2
        lambda_vdw = self.config.lambda_vdw
        max_dq = self.config.max_delta_q
        max_vdw = self.config.max_delta_vdw

        Tensor.training = True
        for b in batches:
            self.opt_head.zero_grad()
            if phase == 2:
                self.opt_trunk.zero_grad()

            b_size, n_atoms, _ = b.atom_mask.shape
            valid_mol = (b.atom_mask.sum(axis=(1, 2)) > 0).cast(dtypes.float32)
            num_valid_mols = valid_mol.sum().maximum(1.0)

            if phase == 1 and b.cached_h is not None:
                node_inputs = Tensor.cat(b.cached_h, b.cached_solvent_features, dim=-1)
                delta_q_raw = self.ff.charge_mlp[0](node_inputs)
                delta_q_raw = self.ff.charge_mlp[1](delta_q_raw)
                delta_q_raw = self.ff.charge_mlp[2](delta_q_raw)
                delta_q = max_dq * (delta_q_raw / max_dq).tanh() * b.atom_mask

                q_raw = (b.base_charges.reshape(b_size, n_atoms, 1) + delta_q) * b.atom_mask
                num_real = b.atom_mask.sum(axis=1, keepdim=True).maximum(1.0)
                q_sum = q_raw.sum(axis=1, keepdim=True)
                q_shift = (q_sum - b.total_charges) / num_real
                q_pred = ((q_raw - q_shift) * b.atom_mask).reshape(b_size, n_atoms)

                delta_vdw_raw = self.ff.vdw_mlp[0](node_inputs)
                delta_vdw_raw = self.ff.vdw_mlp[1](delta_vdw_raw)
                delta_vdw_raw = self.ff.vdw_mlp[2](delta_vdw_raw)
                delta_vdw_atomic = max_vdw * (delta_vdw_raw / max_vdw).tanh() * b.atom_mask
                delta_vdw_mol_atomic = delta_vdw_atomic.sum(axis=(1, 2))

                # Multi-scale graph pooling & cooperative readout in Phase 1
                mean_pool = (b.cached_h * b.atom_mask).sum(axis=1) / num_real.reshape(b_size, 1)
                h_masked = b.cached_h * b.atom_mask - (1.0 - b.atom_mask) * 1e4
                max_pool = h_masked.max(axis=1)
                h_diff = (b.cached_h - mean_pool.reshape(b_size, 1, self.config.hidden_dim)) * b.atom_mask
                var_pool = (h_diff * h_diff).sum(axis=1) / num_real.reshape(b_size, 1)
                std_pool = (var_pool + 1e-6).sqrt()
                graph_features = Tensor.cat(mean_pool, max_pool, std_pool, dim=-1)
                delta_coop_raw = self.ff.global_mlp[0](graph_features)
                delta_coop_raw = self.ff.global_mlp[1](delta_coop_raw)
                delta_coop_raw = self.ff.global_mlp[2](delta_coop_raw).reshape(b_size)
                delta_g_coop = self.ff.max_delta_global * (delta_coop_raw / self.ff.max_delta_global).tanh()
                delta_vdw_mol = delta_vdw_mol_atomic + delta_g_coop
            else:
                q_pred, delta_vdw_mol, delta_vdw_atomic = self.ff.compute_solvation_readouts(
                    x=b.coords,
                    atomic_numbers=b.atomic_numbers,
                    atom_mask=b.atom_mask,
                    total_charge=b.total_charges,
                    base_charges=b.base_charges,
                    solvent_features=b.cached_solvent_features,
                    detach_trunk=False,
                )
                delta_q = (q_pred - b.base_charges) * b.atom_mask.reshape(b_size, n_atoms)

            dg_gb = self.gb.compute_solvation_free_energy(
                x=b.coords,
                charges=q_pred,
                atomic_numbers=b.atomic_numbers,
                atom_mask=b.atom_mask,
                dielectric_constant=self.config.dielectric_constant,
            )
            dg_calc = b.vdw_energies + delta_vdw_mol + dg_gb
            err = (dg_calc - b.expt_energies) * valid_mol

            abs_err = err.abs() * valid_mol
            huber_terms = (abs_err <= delta).where(0.5 * err * err, delta * (abs_err - 0.5 * delta)) * valid_mol
            huber_loss = huber_terms.sum() / num_valid_mols

            num_real_total = b.atom_mask.sum().maximum(1.0)
            l2_q = lambda_l2 * (delta_q * delta_q).sum() / num_real_total
            l2_vdw = lambda_vdw * (delta_vdw_atomic * delta_vdw_atomic).sum() / num_real_total
            l2_coop = (
                getattr(self.config, "lambda_global", 0.0005) * (delta_vdw_mol * delta_vdw_mol).sum() / num_valid_mols
            )

            loss = (huber_loss + l2_q + l2_vdw + l2_coop).reshape(())
            mae_metric = abs_err.sum() / num_valid_mols
            max_dq_metric = delta_q.abs().max()

            loss.backward()

            if phase == 1:
                Tensor.realize(loss, mae_metric, max_dq_metric, *self.opt_head.schedule_step())
            else:
                Tensor.realize(
                    loss, mae_metric, max_dq_metric, *self.opt_head.schedule_step(), *self.opt_trunk.schedule_step()
                )

            self.opt_head.zero_grad()
            if phase == 2:
                self.opt_trunk.zero_grad()

            b_len = len(b.material_names)
            total_loss += float(loss.item()) * b_len
            total_mae += float(mae_metric.item()) * b_len
            total_max_dq = max(total_max_dq, float(max_dq_metric.item()))
            total_molecules += b_len

        avg_loss = total_loss / max(1, total_molecules)
        avg_mae = total_mae / max(1, total_molecules)
        return avg_loss, avg_mae, total_max_dq

    def train(self) -> Dict[str, float]:
        """
        Executes the canonical beautiful_mnist static JIT training loop.
        Fuses forward, backward, Generalized Born solver, and optimizer updates on hardware.
        """
        print(colored("==========================================================================", "cyan"))
        print(colored("  QuantumChargeTrainer: Dual-Headed Volumetric Static Training           ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"  Warmup Epochs     : {self.config.warmup_epochs} (Phase 1: Cached Head Alignment)")
        print(f"  Total Epochs      : {self.config.epochs} (Phase 2: Full End-to-End Unfreeze)")
        print(f"  Head LR           : {self.config.lr_head:.1e}")
        print(f"  Trunk LR          : {self.config.lr_trunk:.1e}")
        print(f"  Max |Δq| Limit    : ±{self.config.max_delta_q:.2f}e (Physical Tanh Squashing)")
        print(f"  Max |Δg_vdw| Limit: ±{self.config.max_delta_vdw:.2f} kcal/mol (Volumetric Cavitation)")
        print(f"  L2 Lambda (q/vdw) : {self.config.lambda_l2} / {self.config.lambda_vdw}")
        print("  Static Arch       : Pure @TinyJit Single-Graph GPU Fusion (Zero Leak)")
        print("-" * 74)

        GlobalCounters.reset()
        t_data_start = time.perf_counter()
        self.dataset = self.load_static_dataset()
        t_data = time.perf_counter() - t_data_start
        print(
            f"  Dataset Loaded    : {self.dataset.num_real_molecules} molecules "
            f"({self.dataset.total_padded_molecules} padded slots) ({t_data:.2f}s)"
        )
        print("-" * 74)

        # Baseline evaluation before any training
        init_mae, init_rmse, init_max, _ = self.evaluate()
        print(
            f"  Initial Baseline  : MAE = {init_mae:.3f} kcal/mol | RMSE = {init_rmse:.3f} kcal/mol | Max Err = {init_max:.2f}"
        )
        print("-" * 74)

        best_mae = init_mae
        out_path = Path(self.config.weights_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        B = self.config.batch_size
        steps_per_epoch = max(1, self.dataset.num_real_molecules // B)

        for epoch in range(1, self.config.epochs + 1):
            phase = 1 if epoch <= self.config.warmup_epochs else 2

            # Cache Invalidation at Phase 2 Transition
            if epoch == self.config.warmup_epochs + 1:
                print(colored("\n" + "=" * 74, "magenta"))
                print(colored("  [PHASE 2 TRANSITION] Unfreezing EGNN Trunk for End-to-End Fine-Tuning", "magenta"))
                print(colored("  Switching to JIT Phase 2 graph with full trunk message-passing autograd", "magenta"))
                print(colored("=" * 74 + "\n", "magenta"))
                self.dataset.cached_h = None

            # Cosine learning rate scheduling
            if phase == 1:
                progress = (epoch - 1) / max(1, self.config.warmup_epochs)
                cur_lr_head = self.config.lr_min + 0.5 * (self.config.lr_head - self.config.lr_min) * (
                    1.0 + math.cos(math.pi * progress)
                )
                cur_lr_trunk = 0.0
            else:
                remaining_epochs = self.config.epochs - self.config.warmup_epochs
                progress = (epoch - 1 - self.config.warmup_epochs) / max(1, remaining_epochs)
                cur_lr_head = self.config.lr_min + 0.5 * (self.config.lr_head * 0.4 - self.config.lr_min) * (
                    1.0 + math.cos(math.pi * progress)
                )
                cur_lr_trunk = self.config.lr_min + 0.5 * (self.config.lr_trunk - self.config.lr_min) * (
                    1.0 + math.cos(math.pi * progress)
                )

            # Update learning rate tensors in-place in optimizer state
            self.opt_head.lr.assign(Tensor([cur_lr_head], dtype=dtypes.float32, device=self.opt_head.device)).realize()
            self.opt_trunk.lr.assign(
                Tensor([cur_lr_trunk], dtype=dtypes.float32, device=self.opt_trunk.device)
            ).realize()

            t_ep_start = time.perf_counter()
            Tensor.training = True

            ep_loss = 0.0
            ep_mae = 0.0
            ep_max_dq = 0.0

            for _ in range(steps_per_epoch):
                if phase == 1:
                    loss, mae_metric, max_dq_metric = self._train_step_p1(
                        self.dataset.coords,
                        self.dataset.atomic_numbers,
                        self.dataset.atom_mask,
                        self.dataset.base_charges,
                        self.dataset.total_charges,
                        self.dataset.vdw_energies,
                        self.dataset.expt_energies,
                        self.dataset.solvent_features,
                        self.dataset.cached_h,
                    )
                else:
                    loss, mae_metric, max_dq_metric = self._train_step_p2(
                        self.dataset.coords,
                        self.dataset.atomic_numbers,
                        self.dataset.atom_mask,
                        self.dataset.base_charges,
                        self.dataset.total_charges,
                        self.dataset.vdw_energies,
                        self.dataset.expt_energies,
                        self.dataset.solvent_features,
                    )

                ep_loss += float(loss.item())
                ep_mae += float(mae_metric.item())
                ep_max_dq = max(ep_max_dq, float(max_dq_metric.item()))

            t_ep = time.perf_counter() - t_ep_start
            train_loss = ep_loss / steps_per_epoch
            train_mae = ep_mae / steps_per_epoch

            # Evaluate test metrics deterministically across 100% of the dataset
            eval_interval = 25 if self.config.epochs >= 100 else 5
            if epoch % eval_interval == 0 or epoch == self.config.epochs or epoch == 1:
                val_mae, val_rmse, val_max, _ = self.evaluate()
                status_color = "green" if val_mae < best_mae else "yellow"
                print(
                    colored(
                        f"  Epoch {epoch:4d}/{self.config.epochs:4d} (P{phase}) [{t_ep:5.2f}s] | "
                        f"Loss: {train_loss:7.4f} | MAE: {val_mae:6.3f} kcal/mol | "
                        f"RMSE: {val_rmse:6.3f} | Max |Δq|: {ep_max_dq:6.4f}e",
                        status_color,
                    )
                )
                if val_mae <= best_mae:
                    best_mae = val_mae
                    self.ff.save_weights(out_path)
            else:
                print(
                    f"  Epoch {epoch:4d}/{self.config.epochs:4d} (P{phase}) [{t_ep:5.2f}s] | "
                    f"Loss: {train_loss:7.4f} | Train MAE: {train_mae:6.3f} kcal/mol | Max |Δq|: {ep_max_dq:6.4f}e"
                )

        # Final checkpoint and evaluation
        if out_path.exists():
            self.ff.load_weights(out_path)
        final_mae, final_rmse, final_max, _ = self.evaluate()

        print("=" * 74)
        print(colored("  Training Completed Successfully!", "green"))
        print(f"  Best MAE Achieved : {best_mae:.3f} kcal/mol (Δ = {init_mae - best_mae:+.3f} kcal/mol)")
        print(f"  Final RMSE        : {final_rmse:.3f} kcal/mol")
        print(f"  Checkpoint Saved  : {out_path}")
        print("=" * 74)

        return {
            "initial_mae": init_mae,
            "final_mae": final_mae,
            "final_rmse": final_rmse,
            "best_mae": best_mae,
            "checkpoint": str(out_path),
        }


def run_train_charges(
    epochs: int = 1000,
    lr: float = 8e-4,
    batch_size: int = 32,
    huber_delta: float = 2.5,
    lambda_l2: float = 0.02,
    lambda_vdw: float = 0.002,
    lambda_global: float = 0.0005,
    max_delta_vdw: float = 3.5,
    max_delta_global: float = 12.0,
    n_conformers: int = 8,
    weights_out: str = "data/checkpoints/egnn_charges_trained.npz",
) -> Dict[str, float]:
    """Convenience entry point for training dynamic quantum charges and cooperative solvation."""
    cfg = ChargeTrainingConfig(
        epochs=epochs,
        lr_head=lr,
        batch_size=batch_size,
        huber_delta=huber_delta,
        lambda_l2=lambda_l2,
        lambda_vdw=lambda_vdw,
        lambda_global=lambda_global,
        max_delta_vdw=max_delta_vdw,
        max_delta_global=max_delta_global,
        n_conformers=n_conformers,
        weights_out=weights_out,
    )
    trainer = QuantumChargeTrainer(config=cfg)
    return trainer.train()


_KRR_WEIGHTS_CACHE: Dict[str, Dict[str, np.ndarray]] = {}
_KRR_DEVICE_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_krr_cache():
    """Clears both host and device KRR weights caches."""
    _KRR_WEIGHTS_CACHE.clear()
    _KRR_DEVICE_CACHE.clear()


def load_krr_weights(
    weights: Union[str, Path, Dict[str, np.ndarray]] = "data/checkpoints/krr_residual_weights.npz",
) -> Optional[Dict[str, np.ndarray]]:
    """Loads and caches fitted Delta-KRR residual model parameters on host CPU."""
    if isinstance(weights, dict):
        return weights
    p_str = str(weights)
    if p_str in _KRR_WEIGHTS_CACHE:
        return _KRR_WEIGHTS_CACHE[p_str]
    p = Path(weights)
    if not p.exists():
        return None
    data = dict(np.load(p, allow_pickle=True))
    _KRR_WEIGHTS_CACHE[p_str] = data
    return data


def load_krr_device_tensors(
    weights: Union[str, Path, Dict[str, np.ndarray]] = "data/checkpoints/krr_residual_weights.npz",
) -> Optional[Dict[str, Any]]:
    """
    Loads and caches immutable, persistent device tensors for pure GPU RBF kernel inference.
    Eliminates host-device synchronization roundtrips during @TinyJit single-sweep pipelines.
    """
    p_str = str(weights) if not isinstance(weights, dict) else "in_memory_dict"
    if p_str in _KRR_DEVICE_CACHE:
        return _KRR_DEVICE_CACHE[p_str]

    w_dict = load_krr_weights(weights)
    if w_dict is None:
        return None

    z_train_raw = np.asarray(w_dict["z_train"], dtype=np.float32)  # (N, D)
    alpha_raw = np.asarray(w_dict["alpha"], dtype=np.float32).reshape(-1, 1)  # (N, 1)
    z_mean = np.asarray(w_dict["z_mean"], dtype=np.float32).reshape(1, -1)  # (1, 384)
    z_std = np.asarray(w_dict["z_std"], dtype=np.float32).reshape(1, -1)  # (1, 384)
    d_mean = np.asarray(w_dict["d_mean"], dtype=np.float32).reshape(1, -1)  # (1, 6)
    d_std = np.asarray(w_dict["d_std"], dtype=np.float32).reshape(1, -1)  # (1, 6)

    s_mean = (
        np.asarray(w_dict["s_mean"], dtype=np.float32).reshape(1, -1)
        if "s_mean" in w_dict
        else np.zeros((1, 7), dtype=np.float32)
    )
    s_std = (
        np.asarray(w_dict["s_std"], dtype=np.float32).reshape(1, -1)
        if "s_std" in w_dict
        else np.ones((1, 7), dtype=np.float32)
    )
    sigma = float(w_dict["sigma"])

    n_train_real, orig_feat = z_train_raw.shape
    n_padded = ((n_train_real + 15) // 16) * 16  # 5472 (multiple of 16 for optimal SIMD/warp alignment)

    # Pad features to pure power-of-2 (512 features: 384 + 8 + 8 + 112)
    z_train_pad = np.zeros((n_padded, 512), dtype=np.float32)
    z_train_pad[:n_train_real, :384] = z_train_raw[:, :384]
    z_train_pad[:n_train_real, 384:390] = z_train_raw[:, 384:390]
    z_train_pad[:n_train_real, 392:399] = z_train_raw[:, 390:397]
    z_train_sq_pad = np.sum(z_train_pad**2, axis=1, keepdims=True).T.astype(np.float32)  # (1, N_padded)

    alpha_pad = np.zeros((n_padded, 1), dtype=np.float32)
    alpha_pad[:n_train_real, :] = alpha_raw

    device_dict = {
        "z_train": Tensor(z_train_pad, dtype=dtypes.float32).realize(),
        "z_train_sq": Tensor(z_train_sq_pad, dtype=dtypes.float32).realize(),
        "alpha": Tensor(alpha_pad, dtype=dtypes.float32).realize(),
        "z_mean": Tensor(z_mean, dtype=dtypes.float32).realize(),
        "z_std": Tensor(z_std, dtype=dtypes.float32).realize(),
        "d_mean": Tensor(d_mean, dtype=dtypes.float32).realize(),
        "d_std": Tensor(d_std, dtype=dtypes.float32).realize(),
        "s_mean": Tensor(s_mean, dtype=dtypes.float32).realize(),
        "s_std": Tensor(s_std, dtype=dtypes.float32).realize(),
        "sigma": sigma,
        "n_train_real": n_train_real,
        "n_features": 512,
    }
    _KRR_DEVICE_CACHE[p_str] = device_dict
    return device_dict


def predict_krr_residual_tensor(
    z_mol: Tensor,
    d_phys: Tensor,
    s_solv: Optional[Union[str, np.ndarray, Tensor]] = None,
    weights: Union[str, Path, Dict[str, np.ndarray]] = "data/checkpoints/krr_residual_weights.npz",
) -> Tuple[Tensor, Tensor]:
    """
    Pure tinygrad GPU tensor kernel predicting Delta-KRR residual and epistemic similarity density.
    Operates directly on device buffers with zero CPU host synchronization stalls.
    Guarantees strict power-of-2 dimension alignments (512 features, N % 16 == 0) for vectorized float4 loads.

    Args:
        z_mol: Tensor shape (B, 384) or (384,)
        d_phys: Tensor shape (B, 6), (B, 8), (6,), or (8,)
        s_solv: Solvent name string, numpy array (B, 7/8), or Tensor (B, 7/8)
        weights: Checkpoint path or dictionary

    Returns:
        (pred_tensor, epistemic_density_tensor): Each of shape (B, 1)
    """
    dev = load_krr_device_tensors(weights)
    if dev is None:
        b_sz = z_mol.shape[0] if len(z_mol.shape) > 1 else 1
        return (
            Tensor.zeros(b_sz, 1, dtype=dtypes.float32).realize(),
            Tensor.zeros(b_sz, 1, dtype=dtypes.float32).realize(),
        )

    z_q = z_mol if len(z_mol.shape) == 2 else z_mol.reshape(1, -1)
    d_q = d_phys if len(d_phys.shape) == 2 else d_phys.reshape(1, -1)
    b_sz = z_q.shape[0]

    # Resolve solvent representation on device
    from dens_city.utils.solvents import get_solvent_descriptors_vector

    if s_solv is None:
        s_vec = get_solvent_descriptors_vector("water")
        s_t = Tensor(np.tile(s_vec, (b_sz, 1)).astype(np.float32))
    elif isinstance(s_solv, str):
        s_vec = get_solvent_descriptors_vector(s_solv)
        s_t = Tensor(np.tile(s_vec, (b_sz, 1)).astype(np.float32))
    elif isinstance(s_solv, np.ndarray):
        s_arr = s_solv if len(s_solv.shape) == 2 else s_solv.reshape(1, -1)
        if s_arr.shape[0] == 1 and b_sz > 1:
            s_arr = np.tile(s_arr, (b_sz, 1))
        s_t = Tensor(s_arr.astype(np.float32))
    elif isinstance(s_solv, Tensor):
        s_t = s_solv if len(s_solv.shape) == 2 else s_solv.reshape(1, -1)
        if s_t.shape[0] == 1 and b_sz > 1:
            s_t = s_t.repeat((b_sz, 1))
    else:
        s_vec = get_solvent_descriptors_vector("water")
        s_t = Tensor(np.tile(s_vec, (b_sz, 1)).astype(np.float32))

    # Standardize features (extract real 6 and 7 active channels)
    z_norm = (z_q - dev["z_mean"]) / dev["z_std"]
    d_norm = (d_q[:, :6] - dev["d_mean"]) / dev["d_std"]
    s_norm = (s_t[:, :7] - dev["s_mean"]) / dev["s_std"]

    pad_d = Tensor.zeros(b_sz, 2, dtype=dtypes.float32)
    pad_s = Tensor.zeros(b_sz, 1, dtype=dtypes.float32)
    pad_tail = Tensor.zeros(b_sz, 112, dtype=dtypes.float32)

    # Padded feature tensor: 384 + 6 + 2 + 7 + 1 + 112 = 512 (power-of-2: 2^9)
    z_comb = Tensor.cat(
        z_norm,
        d_norm * PHYSICAL_DESCRIPTOR_WEIGHT,
        pad_d,
        s_norm * SOLVENT_DESCRIPTOR_WEIGHT,
        pad_s,
        pad_tail,
        dim=1,
    )

    # Vectorized pairwise RBF kernel computation: D^2 = ||z_q||^2 + ||z_train||^2 - 2 z_q z_train^T
    q_sq = (z_comb * z_comb).sum(axis=1, keepdim=True)  # (B, 1)
    d2 = (q_sq + dev["z_train_sq"] - 2.0 * z_comb.matmul(dev["z_train"].transpose())).maximum(0.0)  # (B, N_padded)
    k_mat = (-d2 / (2.0 * (dev["sigma"] ** 2))).exp()  # (B, N_padded)

    # Linear dual regression prediction and epistemic density metric
    pred = k_mat.matmul(dev["alpha"])  # (B, 1)
    density = k_mat[:, : dev["n_train_real"]].sum(axis=1, keepdim=True)  # (B, 1)
    return pred, density


def predict_krr_residual(
    z_mol: Union[np.ndarray, Tensor],
    d_phys: Union[np.ndarray, Tensor],
    weights: Union[str, Path, Dict[str, np.ndarray]] = "data/checkpoints/krr_residual_weights.npz",
    s_solv: Optional[Union[str, np.ndarray, Tensor]] = None,
) -> Union[float, np.ndarray]:
    """
    Evaluates the trained Delta-KRR residual model for query molecular embeddings and physical descriptors.
    Uses pure on-device tinygrad tensor operations and returns float or numpy array for backwards compatibility.
    """
    is_single = (isinstance(z_mol, Tensor) and len(z_mol.shape) == 1) or (
        isinstance(z_mol, np.ndarray) and len(z_mol.shape) == 1
    )

    z_t = z_mol if isinstance(z_mol, Tensor) else Tensor(np.asarray(z_mol, dtype=np.float32))
    d_t = d_phys if isinstance(d_phys, Tensor) else Tensor(np.asarray(d_phys, dtype=np.float32))

    pred_t, _ = predict_krr_residual_tensor(z_t, d_t, s_solv=s_solv, weights=weights)
    Tensor.realize(pred_t)
    pred_np = pred_t.numpy()

    if is_single:
        return float(pred_np[0, 0])
    return pred_np.reshape(-1)


def recalibrate_universal_krr(
    sigma: float = 25.0,
    reg_lambda: float = 1e-3,
    save_path: str = "data/checkpoints/krr_residual_weights.npz",
    eval_solvatum_log: str = "runs/test_solvatum_bs128_eval/pipeline_summary.jsonl",
    eval_freesolv_log: str = "runs/test_freesolv_bs128_verify/pipeline_summary.jsonl",
    deduplicate: bool = True,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    High-Throughput Universal Delta-KRR Recalibration Engine.
    Combines FreeSolv and Solvatum evaluated pairs, enforces canonical SMILES and feature-space
    deduplication (preventing Gram singularity and LOOCV data leakage), solves regularized dual
    weights via Cholesky decomposition in float64, and extracts exact Sherman-Morrison LOOCV.

    Returns:
        (overall_mae_loo, solvatum_mae_loo, freesolv_mae_loo)
    """
    import scipy.linalg
    from rdkit import Chem

    from dens_city.utils.benchmark_dataset import FreeSolvDataset, SolvatumDataset
    from dens_city.utils.materials import MaterialLoader
    from dens_city.utils.pipeline import get_global_egnn_model
    from dens_city.utils.solvents import get_solvent_descriptors_vector, normalize_solvent_name

    loader = MaterialLoader()
    candidate_pairs = []

    # 1. Load Solvatum evaluated pairs
    sv = SolvatumDataset()
    sv_entries = sv.load_entries()
    sv_map = {(str(e.solute_id).strip(), normalize_solvent_name(e.solvent_name).upper()): e for e in sv_entries}

    log_solv = Path(eval_solvatum_log)
    if log_solv.exists():
        with open(log_solv, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sid = str(r.get("solute_id") or "").strip()
                sname = normalize_solvent_name(str(r.get("solvent_name") or "")).upper()
                if (sid, sname) in sv_map:
                    e = sv_map[(sid, sname)]
                    candidate_pairs.append(
                        {
                            "solute_key": sid,
                            "solute_name": e.solute_name,
                            "smiles": e.smiles,
                            "solvent_name": sname,
                            "calc": float(r["solvation_free_energy_kcal_mol"]),
                            "born": float(r.get("born_solvation_kcal_mol") or 0.0),
                            "expt": float(e.expt_dG_solv),
                            "dataset": "solvatum",
                        }
                    )

    # 2. Load FreeSolv evaluated pairs
    fs = FreeSolvDataset()
    fs_entries = fs.load_entries()
    name_to_fs = {e.properties.get("mobley_id", e.solute_id): e for e in fs_entries}
    iupac_to_fs = {e.solute_name.lower().replace(" ", "_"): e for e in fs_entries}

    log_fs = Path(eval_freesolv_log)
    if log_fs.exists():
        with open(log_fs, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                m_name = Path(r["material_name"]).stem.lower()
                e = name_to_fs.get(m_name) or iupac_to_fs.get(m_name)
                if e is not None and e.smiles:
                    candidate_pairs.append(
                        {
                            "solute_key": m_name,
                            "solute_name": e.solute_name,
                            "smiles": e.smiles,
                            "solvent_name": "WATER",
                            "calc": float(r["solvation_free_energy_kcal_mol"]),
                            "born": float(r.get("born_solvation_kcal_mol") or 0.0),
                            "expt": float(e.expt_dG_solv),
                            "dataset": "freesolv",
                        }
                    )

    if not candidate_pairs:
        raise RuntimeError("No evaluated candidate pairs loaded for KRR recalibration!")

    if verbose:
        print(f"[KRR-RECALIBRATE] Loaded {len(candidate_pairs)} candidate pairs from evaluation logs.")

    # 3. Canonical Non-Isomeric SMILES deduplication
    if deduplicate:
        dedup_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for p in candidate_pairs:
            m = Chem.MolFromSmiles(p["smiles"])
            csmi = Chem.MolToSmiles(m, isomericSmiles=False) if m else p["smiles"]
            solv = p["solvent_name"]
            k = (csmi, solv)
            if k not in dedup_map:
                dedup_map[k] = []
            dedup_map[k].append(p)

        deduped_pairs = []
        for (csmi, solv), plist in dedup_map.items():
            entry = dict(plist[0])
            entry["canon_smiles"] = csmi
            entry["expt"] = float(np.mean([x["expt"] for x in plist]))
            entry["calc"] = float(np.mean([x["calc"] for x in plist]))
            entry["born"] = float(np.mean([x["born"] for x in plist]))
            deduped_pairs.append(entry)
        if verbose:
            print(
                f"[KRR-RECALIBRATE] Canonical SMILES deduplication: {len(candidate_pairs)} -> {len(deduped_pairs)} pairs (merged {len(candidate_pairs) - len(deduped_pairs)} entries)."
            )
    else:
        deduped_pairs = candidate_pairs

    # 4. Extract materials and graph embeddings
    solute_mats: Dict[str, Any] = {}
    for p in deduped_pairs:
        k = p["solute_key"]
        if k not in solute_mats:
            if p["dataset"] == "solvatum":
                solute_mats[k] = sv.get_material(k)
            else:
                try:
                    solute_mats[k] = loader.load_material(f"data/test_data/{k}.mol2")
                except Exception:
                    solute_mats[k] = None

    valid_pairs = [p for p in deduped_pairs if solute_mats.get(p["solute_key"]) is not None]
    unique_keys = sorted(list(set(p["solute_key"] for p in valid_pairs)))

    if verbose:
        print(f"[KRR-RECALIBRATE] Extracting 384D EGNN embeddings for {len(unique_keys)} unique solutes...")

    egnn = get_global_egnn_model()
    N_sites = 128
    solute_gf: Dict[str, np.ndarray] = {}
    solute_phys: Dict[str, List[float]] = {}

    B_chunk = 32
    for idx in range(0, len(unique_keys), B_chunk):
        chunk_keys = unique_keys[idx : idx + B_chunk]
        mats_chunk = [solute_mats[k] for k in chunk_keys]
        B = len(mats_chunk)
        c_np = np.zeros((B, N_sites, 3), dtype=np.float32)
        z_np = np.zeros((B, N_sites), dtype=np.float32)
        m_np = np.zeros((B, N_sites, 1), dtype=np.float32)
        bq_np = np.zeros((B, N_sites), dtype=np.float32)

        for i, m in enumerate(mats_chunk):
            n = min(N_sites, m.num_sites)
            for s_i, s in enumerate(m.sites[:n]):
                c_np[i, s_i] = [s.x, s.y, s.z]
                z_np[i, s_i] = getattr(s, "atomic_number", 6)
                m_np[i, s_i, 0] = 1.0

        c_t = Tensor(c_np, dtype=dtypes.float32)
        z_t = Tensor(z_np, dtype=dtypes.float32)
        m_t = Tensor(m_np, dtype=dtypes.float32)
        bq_t = Tensor(bq_np, dtype=dtypes.float32)

        _, _, _, _, gf = egnn.compute_solvation_readouts(
            x=c_t,
            atomic_numbers=z_t,
            atom_mask=m_t,
            total_charge=0.0,
            base_charges=bq_t,
            detach_trunk=True,
            return_global=True,
        )
        Tensor.realize(gf)
        gf_arr = gf.numpy()

        for i, k in enumerate(chunk_keys):
            m = mats_chunk[i]
            solute_gf[k] = gf_arr[i]
            z_m = np.array([getattr(s, "atomic_number", 6) for s in m.sites])
            n_heavy = float(np.sum(z_m > 1))
            n_o = float(np.sum(z_m == 8))
            n_n = float(np.sum(z_m == 7))
            n_hal = float(np.sum(np.isin(z_m, [9, 17, 35, 53])))
            solute_phys[k] = [n_heavy, n_o, n_n, n_hal]

    # 5. Assemble and standardize joint representation tensors
    Z_list, D_list, S_list, y_res_list, y_expt_list, y_calc_list, names_list = [], [], [], [], [], [], []
    solv_cache: Dict[str, np.ndarray] = {}

    for p in valid_pairs:
        k = p["solute_key"]
        sname = p["solvent_name"]
        gf = solute_gf[k]
        born = p["born"]
        vdw = p["calc"] - born
        phys = solute_phys[k] + [born, vdw]
        if sname not in solv_cache:
            solv_cache[sname] = get_solvent_descriptors_vector(sname)
        s_vec = solv_cache[sname]

        Z_list.append(gf)
        D_list.append(phys)
        S_list.append(s_vec)
        y_res_list.append(p["expt"] - p["calc"])
        y_expt_list.append(p["expt"])
        y_calc_list.append(p["calc"])
        names_list.append(f"{k}::{sname}")

    Z = np.array(Z_list, dtype=np.float32)
    D = np.array(D_list, dtype=np.float32)
    S = np.array(S_list, dtype=np.float32)
    y_res = np.array(y_res_list, dtype=np.float32)
    y_expt = np.array(y_expt_list, dtype=np.float32)
    y_calc = np.array(y_calc_list, dtype=np.float32)

    z_mean = np.mean(Z, axis=0, keepdims=True)
    z_std = np.std(Z, axis=0, keepdims=True) + 1e-6
    d_mean = np.mean(D, axis=0, keepdims=True)
    d_std = np.std(D, axis=0, keepdims=True) + 1e-6
    s_mean = np.mean(S, axis=0, keepdims=True)
    s_std = np.std(S, axis=0, keepdims=True) + 1e-6

    Z_norm = (Z - z_mean) / z_std
    D_norm = (D - d_mean) / d_std
    S_norm = (S - s_mean) / s_std

    Z_comb = np.concatenate(
        [Z_norm, D_norm * PHYSICAL_DESCRIPTOR_WEIGHT, S_norm * SOLVENT_DESCRIPTOR_WEIGHT], axis=1
    ).astype(np.float64)

    # 6. Feature-space deduplication / clustering (threshold epsilon = 1e-4) to merge electronic duplicate vectors (e.g. H2 vs D2)
    z_sq = np.sum(Z_comb**2, axis=1, keepdims=True)
    D2 = np.maximum(z_sq + z_sq.T - 2.0 * np.dot(Z_comb, Z_comb.T), 0.0)

    # Find near-duplicate pairs (D2 < 1e-6)
    merged_indices = set()
    clusters: List[List[int]] = []
    N_raw = len(Z_comb)
    for i in range(N_raw):
        if i in merged_indices:
            continue
        cluster = [i]
        for j in range(i + 1, N_raw):
            if j not in merged_indices and D2[i, j] < 1e-6:
                cluster.append(j)
                merged_indices.add(j)
        clusters.append(cluster)

    if len(clusters) < N_raw:
        if verbose:
            print(
                f"[KRR-RECALIBRATE] Merged {N_raw - len(clusters)} feature-space near-duplicate vectors (e.g. isotopic/stereoisomer support collisions)."
            )
        clustered_Z = []
        clustered_y_res = []
        clustered_y_calc = []
        clustered_y_expt = []
        clustered_names = []
        clustered_datasets = []

        for cl in clusters:
            clustered_Z.append(np.mean(Z_comb[cl], axis=0))
            clustered_y_res.append(float(np.mean(y_res[cl])))
            clustered_y_calc.append(float(np.mean(y_calc[cl])))
            clustered_y_expt.append(float(np.mean(y_expt[cl])))
            clustered_names.append(names_list[cl[0]])
            clustered_datasets.append(valid_pairs[cl[0]]["dataset"])

        Z_comb = np.array(clustered_Z, dtype=np.float64)
        y_res = np.array(clustered_y_res, dtype=np.float32)
        y_calc = np.array(clustered_y_calc, dtype=np.float32)
        y_expt = np.array(clustered_y_expt, dtype=np.float32)
        names_list = clustered_names
        is_fs = np.array([ds == "freesolv" for ds in clustered_datasets])
    else:
        is_fs = np.array([p["dataset"] == "freesolv" for p in valid_pairs])

    N_train = len(Z_comb)
    z_sq = np.sum(Z_comb**2, axis=1, keepdims=True)
    D2 = np.maximum(z_sq + z_sq.T - 2.0 * np.dot(Z_comb, Z_comb.T), 0.0)

    # Verify linear independence
    np.fill_diagonal(D2, 1e9)
    min_off_diag = np.sqrt(np.min(D2))
    np.fill_diagonal(D2, 0.0)
    if verbose:
        print(
            f"[KRR-RECALIBRATE] Verified feature space linear independence: min pairwise off-diagonal distance = {min_off_diag:.4f}."
        )

    # 7. Stable Cholesky factorization in float64 and Sherman-Morrison LOOCV
    K = np.exp(-D2 / (2.0 * (sigma**2)))
    A = K + reg_lambda * np.eye(N_train, dtype=np.float64)

    c_and_lower = scipy.linalg.cho_factor(A, lower=True, check_finite=False)
    alpha = scipy.linalg.cho_solve(c_and_lower, y_res.astype(np.float64), check_finite=False).astype(np.float32)

    L = c_and_lower[0]
    V = scipy.linalg.solve_triangular(L, np.eye(N_train, dtype=np.float64), lower=True, check_finite=False)
    diag_A_inv = np.sum(V**2, axis=0)

    y_loo_res = y_res.astype(np.float64) - (alpha / diag_A_inv)
    y_loo_pred = y_calc + y_loo_res

    loo_err = np.abs(y_loo_pred - y_expt)
    mae_overall = float(np.mean(loo_err))
    rmse_overall = float(np.sqrt(np.mean(loo_err**2)))
    bias_overall = float(np.mean(y_loo_pred - y_expt))
    r_overall = float(np.corrcoef(y_expt, y_loo_pred)[0, 1])

    mae_fs = float(np.mean(loo_err[is_fs])) if np.any(is_fs) else 0.0
    mae_sv = float(np.mean(loo_err[~is_fs])) if np.any(~is_fs) else 0.0

    if verbose:
        print("=" * 88)
        print(f"[KRR-RECALIBRATE] Completed Closed-Form Cholesky Calibration (N={N_train}, D={Z_comb.shape[1]}):")
        print(
            f"  Overall LOOCV MAE : {mae_overall:.4f} kcal/mol (RMSE: {rmse_overall:.4f}, Bias: {bias_overall:+.4f}, R: {r_overall:.4f})"
        )
        print(f"  Solvatum LOOCV MAE: {mae_sv:.4f} kcal/mol")
        print(f"  FreeSolv LOOCV MAE: {mae_fs:.4f} kcal/mol")
        print("=" * 88)

    # 8. Save compressed checkpoint
    p_out = Path(save_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "alpha": alpha.astype(np.float32),
        "z_train": Z_comb.astype(np.float32),
        "z_mean": z_mean.astype(np.float32),
        "z_std": z_std.astype(np.float32),
        "d_mean": d_mean.astype(np.float32),
        "d_std": d_std.astype(np.float32),
        "s_mean": s_mean.astype(np.float32),
        "s_std": s_std.astype(np.float32),
        "sigma": float(sigma),
        "reg_lambda": float(reg_lambda),
        "names": np.array(names_list),
    }
    np.savez_compressed(p_out, **save_dict)
    clear_krr_cache()

    if verbose:
        print(f"[KRR-RECALIBRATE] Saved recalibrated KRR checkpoint to: {p_out}")

    return mae_overall, mae_sv, mae_fs
