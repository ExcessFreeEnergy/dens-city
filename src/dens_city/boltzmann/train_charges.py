"""
End-to-End Differentiable Quantum Charge Trainer.
Optimizes the EGNN dynamic charge readout head (charge_mlp) and message-passing trunk
directly against experimental FreeSolv hydration free energies (ΔG_expt) using Huber loss,
physical hyperbolic tangent (tanh) charge bounds, and L2 perturbation regularization.

In accordance with data/tinyspec.tex:
1. Realizes forward loss, backward gradients, and optimizer momentum in a single hardware sweep
   via Tensor.realize(loss, *opt.schedule_step()).
2. Two-Phase Discriminative Fine-Tuning:
   - Phase 1 (Epochs 1-15): High-throughput head warmup with detached cached_h.
   - Phase 2 (Epochs 16-60): Cache invalidation (b.cached_h = None) and end-to-end fine-tuning
     with differential learning rates (lr_trunk = 1e-5, lr_head = 2e-4).
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tinygrad import GlobalCounters, Tensor, dtypes, nn
from tinygrad.helpers import colored

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.cdft.generalized_born import GeneralizedBornSolvation
from dens_city.utils.materials import MaterialLoader


@dataclass
class ChargeTrainingConfig:
    epochs: int = 60
    warmup_epochs: int = 15
    lr_head: float = 5e-4
    lr_trunk: float = 1e-5
    lr_min: float = 1e-6
    batch_size: int = 32
    huber_delta: float = 1.0  # kcal/mol
    lambda_l2: float = 0.05  # Penalty on (Δq)^2
    max_delta_q: float = 0.25  # Max allowed perturbation |Δq| <= 0.25e
    n_particles: int = 128
    hidden_dim: int = 128
    num_layers: int = 7
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


class QuantumChargeTrainer:
    """
    End-to-End Differentiable Quantum Charge Trainer.
    Trains EGNN on FreeSolv database with single-sweep hardware realization and discriminative fine-tuning.
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

        # Parameters
        self.charge_params = nn.state.get_parameters(self.ff.charge_mlp)

        trunk_layers = [self.ff.embedding] + list(self.ff.layers)
        self.trunk_params = []
        for lyr in trunk_layers:
            self.trunk_params.extend(nn.state.get_parameters(lyr))

        # Dual Optimizers for discriminative fine-tuning
        self.opt_head = nn.optim.Adam(self.charge_params, lr=self.config.lr_head)
        self.opt_trunk = nn.optim.Adam(self.trunk_params, lr=self.config.lr_trunk)

    def load_dataset(self) -> List[PreprocessedBatch]:
        """Loads, pre-computes detached trunk embeddings and solvent descriptors, and batches molecules."""
        db_path = Path(self.config.database_path)
        mol2_dir = Path(self.config.mol2_dir)

        if not db_path.exists():
            raise FileNotFoundError(f"FreeSolv database not found at: {db_path}")

        with open(db_path, "rb") as f:
            fs_db: Dict[str, Dict] = pickle.load(f, encoding="latin1")

        mol2_files = sorted(list(mol2_dir.glob("*.mol2")))
        records = []

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
                # Ensure 2D topological baseline charges are precomputed
                mat.compute_topological_base_charges(kappa=0.10, q_max=0.50)
                records.append((mat, vdw, expt))
            except Exception:
                continue

        # Add pure water anchor if not present
        if not any(mat.name == "water" for mat, _, _ in records):
            try:
                mat_water = self.loader.load_material("water")
                mat_water.compute_topological_base_charges(kappa=0.10, q_max=0.50)
                # Water: expt = -6.30 kcal/mol, cavity vdw ~ +4.02 kcal/mol
                records.append((mat_water, 4.02, -6.30))
            except Exception:
                pass

        B = self.config.batch_size
        N = self.config.n_particles
        batches: List[PreprocessedBatch] = []

        # Forward through EGNN trunk in inference mode
        Tensor.training = False

        for start_idx in range(0, len(records), B):
            chunk = records[start_idx : start_idx + B]

            # Pad all batches to static shape B (eliminating dynamic shape JIT recompilations)
            coords_np = np.zeros((B, N, 3), dtype=np.float32)
            z_np = np.zeros((B, N), dtype=np.float32)
            mask_np = np.zeros((B, N, 1), dtype=np.float32)
            bq_np = np.zeros((B, N), dtype=np.float32)
            tot_q_np = np.zeros((B, 1, 1), dtype=np.float32)
            vdw_np = np.zeros((B,), dtype=np.float32)
            expt_np = np.zeros((B,), dtype=np.float32)
            names: List[str] = []
            n_real: List[int] = []

            for i, (mat, vdw, expt) in enumerate(chunk):
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
                n_real.append(n_sites)

            # In accordance with data/tinyspec.tex, static inputs are pre-realized leaf buffers
            coords_t = Tensor(coords_np).realize()
            z_t = Tensor(z_np).realize()
            mask_t = Tensor(mask_np).realize()
            bq_t = Tensor(bq_np).realize()
            tot_q_t = Tensor(tot_q_np).realize()
            vdw_t = Tensor(vdw_np).realize()
            expt_t = Tensor(expt_np).realize()

            # Precompute static EGNN node embeddings and 4-channel solvent descriptors once
            x_prep, z_prep, atom_mask, molecule_mask, edge_mask = self.ff._prepare_inputs(coords_t, z_t, mask_t, None)
            z_clamped = z_prep.cast(dtypes.int32)
            z_one_hot = Tensor.one_hot(z_clamped, num_classes=self.ff.max_atomic_number)
            h = self.ff.embedding(z_one_hot) * atom_mask
            x_i = x_prep.reshape(B, N, 1, 3)
            x_j = x_prep.reshape(B, 1, N, 3)
            diff = x_i - x_j
            d_sq = (diff * diff).sum(axis=-1, keepdim=True)
            for layer in self.ff.layers:
                h = layer(h, d_sq, edge_mask, atom_mask)

            # Explicitly detach and realize per tinyspec.tex \op{Detach}
            h_cached = h.detach().realize()
            sf_cached = self.gb.compute_solvent_descriptors(coords_t, z_t, mask_t, base_charges=bq_t).detach().realize()

            batch = PreprocessedBatch(
                coords=coords_t,
                atomic_numbers=z_t,
                atom_mask=mask_t,
                base_charges=bq_t,
                total_charges=tot_q_t,
                vdw_energies=vdw_t,
                expt_energies=expt_t,
                material_names=names,
                num_real_atoms=n_real,
                cached_h=h_cached,
                cached_solvent_features=sf_cached,
            )
            batches.append(batch)

        return batches

    def train_epoch(
        self,
        batches: List[PreprocessedBatch],
        lr_head: float,
        lr_trunk: float,
        phase: int,
    ) -> Tuple[float, float, float]:
        """Runs a single training epoch across all batches with single-sweep hardware realization."""
        self.opt_head.lr = lr_head
        if phase == 2:
            self.opt_trunk.lr = lr_trunk

        total_loss = 0.0
        total_mae = 0.0
        total_max_dq = 0.0
        total_molecules = 0

        delta = self.config.huber_delta
        lambda_l2 = self.config.lambda_l2
        max_dq = self.config.max_delta_q

        Tensor.training = True
        for b in batches:
            self.opt_head.zero_grad()
            if phase == 2:
                self.opt_trunk.zero_grad()

            b_size, n_atoms, _ = b.atom_mask.shape
            # Identify valid active molecules vs padded dummy slots
            valid_mol = (b.atom_mask.sum(axis=(1, 2)) > 0).cast(dtypes.float32)
            num_valid_mols = valid_mol.sum().maximum(1.0)

            if phase == 1 and b.cached_h is not None:
                # Phase 1: High-throughput cached execution through charge_mlp only
                node_inputs = Tensor.cat(b.cached_h, b.cached_solvent_features, dim=-1)
                delta_q_raw = self.ff.charge_mlp[0](node_inputs)
                delta_q_raw = self.ff.charge_mlp[1](delta_q_raw)
                delta_q_raw = self.ff.charge_mlp[2](delta_q_raw)
                delta_q = max_dq * (delta_q_raw / max_dq).tanh() * b.atom_mask  # (B, N, 1)

                q_raw = (b.base_charges.reshape(b_size, n_atoms, 1) + delta_q) * b.atom_mask
                num_real = b.atom_mask.sum(axis=1, keepdim=True).maximum(1.0)
                q_sum = q_raw.sum(axis=1, keepdim=True)
                q_shift = (q_sum - b.total_charges) / num_real
                q_pred = ((q_raw - q_shift) * b.atom_mask).reshape(b_size, n_atoms)
            else:
                # Phase 2: Full end-to-end forward pass through EGNN trunk from raw leaf buffers
                q_pred = self.ff.compute_charges(
                    x=b.coords,
                    atomic_numbers=b.atomic_numbers,
                    atom_mask=b.atom_mask,
                    total_charge=b.total_charges,
                    base_charges=b.base_charges,
                    solvent_features=b.cached_solvent_features,
                    detach_trunk=False,
                )
                delta_q = (q_pred - b.base_charges) * b.atom_mask.reshape(b_size, n_atoms)

            # 2. Generalized Born solvation free energy
            dg_gb = self.gb.compute_solvation_free_energy(
                x=b.coords,
                charges=q_pred,
                atomic_numbers=b.atomic_numbers,
                atom_mask=b.atom_mask,
                dielectric_constant=self.config.dielectric_constant,
            )

            # 3. Total calculated hydration free energy = Nonpolar VDW + Electrostatic GB
            dg_calc = b.vdw_energies + dg_gb
            err = (dg_calc - b.expt_energies) * valid_mol

            # 4. Huber Loss only over valid molecules
            abs_err = err.abs() * valid_mol
            huber_terms = (abs_err <= delta).where(0.5 * err * err, delta * (abs_err - 0.5 * delta)) * valid_mol
            huber_loss = huber_terms.sum() / num_valid_mols

            # 5. L2 Regularization on neural charge perturbation Δq
            num_real_total = b.atom_mask.sum().maximum(1.0)
            l2_reg = lambda_l2 * (delta_q * delta_q).sum() / num_real_total

            # Total Scalar Loss
            loss = (huber_loss + l2_reg).reshape(())

            # Fused scalar metrics for single-sweep hardware realization
            mae_metric = abs_err.sum() / num_valid_mols
            max_dq_metric = delta_q.abs().max()

            # Reverse-mode autograd backward pass
            loss.backward()

            # Single hardware sweep realization (tinyspec.tex + beautiful_mnist.py)
            if phase == 1:
                Tensor.realize(loss, mae_metric, max_dq_metric, *self.opt_head.schedule_step())
            else:
                Tensor.realize(
                    loss, mae_metric, max_dq_metric, *self.opt_head.schedule_step(), *self.opt_trunk.schedule_step()
                )

            # Immediate reference severing to eliminate Python GC cycle accumulation
            self.opt_head.zero_grad()
            if phase == 2:
                self.opt_trunk.zero_grad()

            # Track statistics strictly from pre-realized scalar buffers
            b_len = len(b.material_names)
            total_loss += float(loss.item()) * b_len
            total_mae += float(mae_metric.item()) * b_len
            total_max_dq = max(total_max_dq, float(max_dq_metric.item()))
            total_molecules += b_len

        avg_loss = total_loss / max(1, total_molecules)
        avg_mae = total_mae / max(1, total_molecules)
        return avg_loss, avg_mae, total_max_dq

    def evaluate(self, batches: List[PreprocessedBatch]) -> Tuple[float, float, float, Dict[str, float]]:
        """Evaluates MAE, RMSE, max error, and individual predictions in inference mode."""
        errors: List[float] = []
        preds: Dict[str, float] = {}

        Tensor.training = False
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
                preds[name] = float(calc)

        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        max_err = float(np.max(errors))
        return mae, rmse, max_err, preds

    def train(self) -> Dict[str, float]:
        """Executes the complete training loop with learning rate cosine decay and checkpointing."""
        print(colored("==========================================================================", "cyan"))
        print(colored("  QuantumChargeTrainer: End-to-End Differentiable Charge Head Training   ", "cyan"))
        print(colored("==========================================================================", "cyan"))
        print(f"  Warmup Epochs     : {self.config.warmup_epochs} (Phase 1: Cached Head Alignment)")
        print(f"  Total Epochs      : {self.config.epochs} (Phase 2: Full End-to-End Unfreeze)")
        print(f"  Head LR           : {self.config.lr_head:.1e}")
        print(f"  Trunk LR          : {self.config.lr_trunk:.1e}")
        print(f"  Max |Δq| Limit    : ±{self.config.max_delta_q:.2f}e (Physical Tanh Squashing)")
        print(f"  L2 Lambda         : {self.config.lambda_l2}")
        print("  Single Hardware   : Tensor.realize(loss, *opt.schedule_step()) [tinyspec.tex]")
        print("-" * 74)

        GlobalCounters.reset()
        t0 = time.perf_counter()
        batches = self.load_dataset()
        total_molecules = sum(len(b.material_names) for b in batches)
        print(
            f"  Dataset Loaded    : {total_molecules} molecules across {len(batches)} batches ({time.perf_counter() - t0:.2f}s)"
        )
        print("-" * 74)

        # Baseline evaluation before training
        init_mae, init_rmse, init_max, _ = self.evaluate(batches)
        print(
            f"  Initial Baseline  : MAE = {init_mae:.3f} kcal/mol | RMSE = {init_rmse:.3f} kcal/mol | Max Err = {init_max:.2f}"
        )
        print("-" * 74)

        best_mae = init_mae
        out_path = Path(self.config.weights_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            phase = 1 if epoch <= self.config.warmup_epochs else 2

            # Cache Invalidation at Epoch 16
            if epoch == self.config.warmup_epochs + 1:
                print(colored("\n" + "=" * 74, "magenta"))
                print(colored("  [PHASE 2 TRANSITION] Unfreezing EGNN Trunk for End-to-End Fine-Tuning", "magenta"))
                print(
                    colored("  Discarding Phase 1 cached_h buffers to re-evaluate raw leaf Ops.BUFFER DAG", "magenta")
                )
                print(colored("=" * 74 + "\n", "magenta"))
                for b in batches:
                    b.cached_h = None

            # Cosine learning rate schedules
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

            t_ep_start = time.perf_counter()
            loss, train_mae, max_dq = self.train_epoch(batches, cur_lr_head, cur_lr_trunk, phase)
            t_ep = time.perf_counter() - t_ep_start

            # Evaluate test metrics every 5 epochs or at the end
            if epoch % 5 == 0 or epoch == self.config.epochs or epoch == 1:
                val_mae, val_rmse, val_max, _ = self.evaluate(batches)
                status_color = "green" if val_mae < best_mae else "yellow"
                print(
                    colored(
                        f"  Epoch {epoch:3d}/{self.config.epochs:3d} (P{phase}) [{t_ep:5.2f}s] | "
                        f"Loss: {loss:7.4f} | MAE: {val_mae:6.3f} kcal/mol | "
                        f"RMSE: {val_rmse:6.3f} | Max |Δq|: {max_dq:6.4f}e",
                        status_color,
                    )
                )
                if val_mae <= best_mae:
                    best_mae = val_mae
                    self.ff.save_weights(out_path)
            else:
                print(
                    f"  Epoch {epoch:3d}/{self.config.epochs:3d} (P{phase}) [{t_ep:5.2f}s] | "
                    f"Loss: {loss:7.4f} | Train MAE: {train_mae:6.3f} kcal/mol | Max |Δq|: {max_dq:6.4f}e"
                )

        # Final checkpoint and evaluation
        if out_path.exists():
            self.ff.load_weights(out_path)
        final_mae, final_rmse, final_max, _ = self.evaluate(batches)

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
    epochs: int = 60,
    lr: float = 5e-4,
    batch_size: int = 32,
    huber_delta: float = 1.0,
    lambda_l2: float = 0.05,
    weights_out: str = "data/checkpoints/egnn_charges_trained.npz",
) -> Dict[str, float]:
    """Convenience entry point for training dynamic quantum charges."""
    cfg = ChargeTrainingConfig(
        epochs=epochs,
        lr_head=lr,
        batch_size=batch_size,
        huber_delta=huber_delta,
        lambda_l2=lambda_l2,
        weights_out=weights_out,
    )
    trainer = QuantumChargeTrainer(config=cfg)
    return trainer.train()
