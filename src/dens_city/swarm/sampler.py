"""
High-throughput RL Swarm Candidate Sampler for dens-city.
Executes parallel rollouts across C-FFI environments, extracts raw physical
parameter arrays directly from C memory (bypassing Python/file force-field parsing),
and streams contiguous NumPy tensor blocks into GPU memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem

from dens_city.swarm.env import CDFTSwarmEnv
from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.trainer import VectorizedSwarmEnv
from dens_city.utils.materials import (
    MolecularBatch,
    solve_bulk_density_from_chemical_potential,
)


@dataclass
class ContiguousCandidateBatch:
    """
    Encapsulates pre-padded, contiguous NumPy array blocks containing K candidate molecules.
    Enables zero-copy slicing straight into tinygrad.Tensor buffers without Python object fragmentation.
    """

    num_candidates: int
    n_particles: int
    coords: np.ndarray  # (K, N, 3) float32
    sigmas: np.ndarray  # (K, N) float32
    epsilons_k: np.ndarray  # (K, N) float32
    charges: np.ndarray  # (K, N) float32
    atomic_numbers: np.ndarray  # (K, N) int32
    atom_mask: np.ndarray  # (K, N) float32
    molecule_mask: np.ndarray  # (K,) float32
    temperature_k: np.ndarray  # (K,) float32
    bulk_density_a3: np.ndarray  # (K,) float32
    bulk_mu: np.ndarray  # (K,) float32
    slit_width_a: np.ndarray  # (K,) float32
    conditioning: np.ndarray  # (K, 5) float32
    exclusions: Optional[np.ndarray] = None  # (K, N, N) float32
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def slice_molecular_batch(
        self,
        start_idx: int,
        count: int,
        batch_size: int = 512,
    ) -> MolecularBatch:
        """
        Extracts a slice of up to count candidates and packs them into a fixed (batch_size, n_particles)
        MolecularBatch directly using zero-copy array views.
        """
        end_idx = min(self.num_candidates, start_idx + count)
        actual_count = max(0, end_idx - start_idx)

        b_sigmas = np.zeros((batch_size, self.n_particles), dtype=np.float32)
        b_epsilons = np.zeros((batch_size, self.n_particles), dtype=np.float32)
        b_charges = np.zeros((batch_size, self.n_particles), dtype=np.float32)
        b_atomic_numbers = np.zeros((batch_size, self.n_particles), dtype=np.int32)
        b_atom_mask = np.zeros((batch_size, self.n_particles), dtype=np.float32)
        b_molecule_mask = np.zeros(batch_size, dtype=np.float32)
        b_temperature_k = np.full(batch_size, 300.0, dtype=np.float32)
        b_bulk_density_a3 = np.zeros(batch_size, dtype=np.float32)
        b_bulk_mu = np.zeros(batch_size, dtype=np.float32)
        b_slit_width_a = np.full(batch_size, 40.0, dtype=np.float32)
        b_conditioning = np.zeros((batch_size, 5), dtype=np.float32)
        b_exclusions = np.zeros((batch_size, self.n_particles, self.n_particles), dtype=np.float32)

        if actual_count > 0:
            b_sigmas[:actual_count] = self.sigmas[start_idx:end_idx]
            b_epsilons[:actual_count] = self.epsilons_k[start_idx:end_idx]
            b_charges[:actual_count] = self.charges[start_idx:end_idx]
            b_atomic_numbers[:actual_count] = self.atomic_numbers[start_idx:end_idx]
            b_atom_mask[:actual_count] = self.atom_mask[start_idx:end_idx]
            b_molecule_mask[:actual_count] = 1.0
            b_temperature_k[:actual_count] = self.temperature_k[start_idx:end_idx]
            b_bulk_density_a3[:actual_count] = self.bulk_density_a3[start_idx:end_idx]
            b_bulk_mu[:actual_count] = self.bulk_mu[start_idx:end_idx]
            b_slit_width_a[:actual_count] = self.slit_width_a[start_idx:end_idx]
            b_conditioning[:actual_count] = self.conditioning[start_idx:end_idx]
            if self.exclusions is not None:
                b_exclusions[:actual_count] = self.exclusions[start_idx:end_idx]

        return MolecularBatch.from_contiguous_arrays(
            sigmas=b_sigmas,
            epsilons=b_epsilons,
            charges=b_charges,
            atomic_numbers=b_atomic_numbers,
            atom_mask=b_atom_mask,
            molecule_mask=b_molecule_mask,
            temperature_k=b_temperature_k,
            bulk_density_a3=b_bulk_density_a3,
            bulk_mu=b_bulk_mu,
            slit_width_a=b_slit_width_a,
            conditioning=b_conditioning,
            exclusions=b_exclusions,
        )


class SwarmCandidateSampler:
    """
    Executes parallel inference with a trained MolecularSwarmPolicy to generate
    thousands of likely molecular candidates directly into contiguous NumPy arrays.
    """

    def __init__(
        self,
        policy: MolecularSwarmPolicy,
        spec_path: str | Path,
        num_envs: int = 32,
        device: str = "cuda",
        target_n_particles: int = 128,
        default_temp_k: float = 300.0,
    ):
        self.policy = policy
        self.spec_path = Path(spec_path)
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        self.target_n_particles = target_n_particles
        self.default_temp_k = default_temp_k

        self.spec_data = SwarmSpecLoader.load_yaml(self.spec_path)
        self.target_spec = SwarmSpecLoader.derive_target_spec(self.spec_data)
        self.min_valency = int(self.target_spec.get("min_valency", 2))
        self.max_molecular_weight = float(self.target_spec.get("max_molecular_weight", 850.0))

    def sample_candidates(
        self,
        total_candidates: int = 1024,
        max_rollout_steps: int = 50000,
        temperature: float = 1.0,
    ) -> ContiguousCandidateBatch:
        """
        Executes parallel rollout inference over VectorizedSwarmEnv until total_candidates
        surviving, physically valid candidates are aggregated in contiguous memory.
        """
        K = total_candidates
        N = self.target_n_particles

        coords_arr = np.zeros((K, N, 3), dtype=np.float32)
        sigmas_arr = np.zeros((K, N), dtype=np.float32)
        epsilons_arr = np.zeros((K, N), dtype=np.float32)
        charges_arr = np.zeros((K, N), dtype=np.float32)
        atomic_numbers_arr = np.zeros((K, N), dtype=np.int32)
        atom_mask_arr = np.zeros((K, N), dtype=np.float32)
        molecule_mask_arr = np.ones(K, dtype=np.float32)
        temp_arr = np.full(K, self.default_temp_k, dtype=np.float32)
        rho_arr = np.zeros(K, dtype=np.float32)
        mu_arr = np.zeros(K, dtype=np.float32)
        slit_arr = np.full(K, 40.0, dtype=np.float32)
        cond_arr = np.zeros((K, 5), dtype=np.float32)
        exclusions_arr = np.zeros((K, N, N), dtype=np.float32)
        metadata_list: List[Dict[str, Any]] = []

        vec_env = VectorizedSwarmEnv(
            num_envs=self.num_envs,
            target_spec=self.target_spec,
            seed=999,
        )

        candidates_collected = 0
        hidden_state = None
        recurrent = getattr(self.policy, "recurrent", False)

        try:
            obs_t, masks_t = vec_env.reset()
            obs_t = obs_t.to(self.device)
            masks_t = masks_t.to(self.device)
            step_count = 0

            while candidates_collected < K and step_count < max_rollout_steps:
                with torch.no_grad():
                    if recurrent:
                        actions, _, _, hidden_state = self.policy.sample_action(
                            obs=obs_t,
                            action_masks=masks_t,
                            hidden_state=hidden_state,
                            deterministic=(temperature <= 0.01),
                        )
                    else:
                        actions, _, _ = self.policy.sample_action(
                            obs=obs_t,
                            action_masks=masks_t,
                            deterministic=(temperature <= 0.01),
                        )

                next_obs, rewards, terminals, next_masks, infos = vec_env.step(actions)
                step_count += self.num_envs

                for env_idx in range(self.num_envs):
                    if terminals[env_idx] > 0.5:
                        reward = float(rewards[env_idx])
                        single_env: CDFTSwarmEnv = vec_env.envs[env_idx]
                        raw_data = single_env.get_raw_atom_arrays()
                        n_atoms = raw_data["num_atoms"]

                        is_valid = (
                            reward > -30.0
                            and n_atoms >= 6
                            and raw_data["mw"] <= self.max_molecular_weight
                            and raw_data["p_wall"] > 0.0
                        )

                        if is_valid and candidates_collected < K:
                            idx = candidates_collected
                            n_copy = min(n_atoms, N)

                            valid_sigs = [float(s) for s in raw_data["sigmas"][:n_copy] if s > 0.0]
                            eff_sig = (
                                sum(s**3 for s in valid_sigs) ** (1.0 / 3.0)
                                if valid_sigs
                                else float(raw_data["sigmas"][0])
                            )

                            att_vol_sum = 0.0
                            for a1 in range(n_copy):
                                for a2 in range(n_copy):
                                    eps_12 = math.sqrt(
                                        max(0.0, float(raw_data["epsilons_k"][a1] * raw_data["epsilons_k"][a2]))
                                    )
                                    sig_12 = 0.5 * float(raw_data["sigmas"][a1] + raw_data["sigmas"][a2])
                                    att_vol_sum += eps_12 * (sig_12**3)
                            eff_eps = att_vol_sum / (eff_sig**3) if eff_sig > 0.0 else 120.0

                            rho_b = solve_bulk_density_from_chemical_potential(
                                mu_kbt=-8.0,
                                temp_k=self.default_temp_k,
                                sigma=eff_sig,
                                epsilon_k=eff_eps,
                            )
                            slit_w = max(40.0, 12.0 * eff_sig)

                            coords_centered = raw_data["coords"][:n_copy].copy()
                            # Center molecule in Z within the slit pore to avoid wall boundary overlap
                            z_min = float(coords_centered[:, 2].min()) if n_copy > 0 else 0.0
                            z_max = float(coords_centered[:, 2].max()) if n_copy > 0 else 0.0
                            z_mid = 0.5 * (z_min + z_max)
                            coords_centered[:, 2] += 0.5 * slit_w - z_mid

                            coords_arr[idx, :n_copy] = coords_centered
                            sig_raw = raw_data["sigmas"][:n_copy].copy()
                            eps_raw = raw_data["epsilons_k"][:n_copy].copy()
                            # Physical floor for active atoms (prevents zero-LJ Coulomb collapse)
                            sig_raw = np.where(sig_raw < 0.40, 1.06, sig_raw)
                            eps_raw = np.where(eps_raw < 1.0, 7.55, eps_raw)
                            sigmas_arr[idx, :n_copy] = sig_raw
                            epsilons_arr[idx, :n_copy] = eps_raw
                            charges_arr[idx, :n_copy] = raw_data["charges"][:n_copy]
                            atomic_numbers_arr[idx, :n_copy] = raw_data["atomic_numbers"][:n_copy]
                            atom_mask_arr[idx, :n_copy] = 1.0

                            if "exclusions" in raw_data:
                                exclusions_arr[idx, :n_copy, :n_copy] = raw_data["exclusions"][:n_copy, :n_copy]

                            rho_arr[idx] = rho_b
                            mu_arr[idx] = -8.0
                            slit_arr[idx] = slit_w
                            cond_arr[idx] = [eff_sig, eff_eps, self.default_temp_k, rho_b, -8.0]

                            cand_name = f"cand_{idx + 1:04d}_{self.spec_path.stem}"
                            mol2_str = single_env.export_mol2_string(cand_name)
                            rd_mol = single_env.get_current_rdkit_mol()
                            smiles_str = Chem.MolToSmiles(rd_mol, canonical=True) if rd_mol is not None else ""

                            metadata_list.append(
                                {
                                    "index": idx,
                                    "name": cand_name,
                                    "rl_reward": reward,
                                    "num_atoms": n_atoms,
                                    "mw": raw_data["mw"],
                                    "p_wall": raw_data["p_wall"],
                                    "contact_ratio": raw_data.get("contact_ratio", 1.0),
                                    "omega_solv": raw_data["omega_solv"],
                                    "pmi_linearity": raw_data["pmi_linearity"],
                                    "aromatic_density": raw_data["aromatic_density"],
                                    "rotatable_fraction": raw_data["rotatable_fraction"],
                                    "wl_hash": raw_data.get("wl_hash", 0),
                                    "smiles": smiles_str,
                                    "mol2": mol2_str,
                                    "effective_sigma": eff_sig,
                                    "effective_epsilon_k": eff_eps,
                                }
                            )

                            candidates_collected += 1

                obs_t = next_obs.to(self.device)
                masks_t = next_masks.to(self.device)

        finally:
            vec_env.close()

        actual_K = candidates_collected
        return ContiguousCandidateBatch(
            num_candidates=actual_K,
            n_particles=N,
            coords=coords_arr[:actual_K],
            sigmas=sigmas_arr[:actual_K],
            epsilons_k=epsilons_arr[:actual_K],
            charges=charges_arr[:actual_K],
            atomic_numbers=atomic_numbers_arr[:actual_K],
            atom_mask=atom_mask_arr[:actual_K],
            molecule_mask=molecule_mask_arr[:actual_K],
            temperature_k=temp_arr[:actual_K],
            bulk_density_a3=rho_arr[:actual_K],
            bulk_mu=mu_arr[:actual_K],
            slit_width_a=slit_arr[:actual_K],
            conditioning=cond_arr[:actual_K],
            exclusions=exclusions_arr[:actual_K],
            metadata=metadata_list,
        )
