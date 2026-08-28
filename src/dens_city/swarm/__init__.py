"""
dens_city.swarm: Stage 1 Multi-Objective RL Swarm environment for molecular patterning.
"""

from dens_city.swarm.env import CDFTSwarmEnv
from dens_city.swarm.evaluator import (
    compute_batch_diversity,
    evaluate_molecule_chemistry,
    run_spec_rl_stage,
)
from dens_city.swarm.policy import (
    MolecularActionDecoder,
    MolecularPortEncoder,
    MolecularSwarmPolicy,
)
from dens_city.swarm.spec_loader import SwarmSpecLoader
from dens_city.swarm.sweep import CurriculumSweepRunner
from dens_city.swarm.trainer import (
    SwarmCurriculumManager,
    SwarmPuffeRLTrainer,
    VectorizedSwarmEnv,
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
]
