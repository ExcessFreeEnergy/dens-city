"""
dens_city.swarm: Stage 1 Multi-Objective RL Swarm environment for molecular patterning and 5-stage generative funnel.
"""

from dens_city.swarm.env import CDFTSwarmEnv
from dens_city.swarm.evaluator import (
    compute_batch_diversity,
    evaluate_all_swarm_specs,
    evaluate_molecule_chemistry,
    run_spec_rl_stage,
)
from dens_city.swarm.funnel import (
    run_all_specs_funnel_benchmark,
    run_generative_funnel,
)
from dens_city.swarm.policy import (
    MolecularActionDecoder,
    MolecularPortEncoder,
    MolecularSwarmPolicy,
)
from dens_city.swarm.sampler import SwarmCandidateSampler
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.sweep import CurriculumSweepRunner, run_curriculum_sweep
from dens_city.swarm.trainer import (
    SwarmCurriculumManager,
    SwarmPuffeRLTrainer,
    VectorizedSwarmEnv,
    train_swarm_policy,
)

__all__ = [
    "CDFTSwarmEnv",
    "SwarmSpecLoader",
    "MolecularPortEncoder",
    "MolecularActionDecoder",
    "MolecularSwarmPolicy",
    "VectorizedSwarmEnv",
    "SwarmCurriculumManager",
    "SwarmPuffeRLTrainer",
    "CurriculumSweepRunner",
    "evaluate_molecule_chemistry",
    "compute_batch_diversity",
    "run_spec_rl_stage",
    "evaluate_all_swarm_specs",
    "run_curriculum_sweep",
    "train_swarm_policy",
    "run_generative_funnel",
    "run_all_specs_funnel_benchmark",
    "SwarmCandidateSampler",
]
