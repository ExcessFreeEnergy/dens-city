"""
Integration tests for SwarmCandidateSampler and C-native contiguous array streaming.
"""

from pathlib import Path

from dens_city.swarm.policy import MolecularSwarmPolicy
from dens_city.swarm.sampler import SwarmCandidateSampler


def test_swarm_candidate_sampler_rollout():
    spec_path = Path(__file__).resolve().parent / "data" / "conjugated_oled_semiconductors.yaml"
    policy = MolecularSwarmPolicy(hidden_size=128, recurrent=False)

    sampler = SwarmCandidateSampler(
        policy=policy,
        spec_path=spec_path,
        num_envs=8,
        device="cpu",
        target_n_particles=128,
    )

    candidate_batch = sampler.sample_candidates(
        total_candidates=16,
        max_rollout_steps=5000,
        temperature=1.0,
    )

    assert candidate_batch.num_candidates > 0
    assert candidate_batch.coords.shape[1] == 128
    assert candidate_batch.coords.shape[2] == 3
    assert candidate_batch.sigmas.shape[1] == 128
    assert len(candidate_batch.metadata) == candidate_batch.num_candidates

    # Verify slicing into MolecularBatch
    mol_batch = candidate_batch.slice_molecular_batch(start_idx=0, count=8, batch_size=16)
    assert mol_batch.batch_size == 16
    assert mol_batch.sigmas.shape == (16, 128)
