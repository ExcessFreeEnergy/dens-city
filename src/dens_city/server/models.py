"""
Pydantic data schemas for dens_city FastAPI & MCP Server integration.
Enforces strict chemical boundary validation, asynchronous job lifecycles,
and semantic LLM error definitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(str, Enum):
    FULL_PIPELINE = "FULL_PIPELINE"
    TRAIN_SWARM = "TRAIN_SWARM"
    SAMPLE_CANDIDATES = "SAMPLE_CANDIDATES"
    RUN_CDFT = "RUN_CDFT"
    RANK_PARETO = "RANK_PARETO"


class SemanticError(BaseModel):
    error_code: str = Field(..., description="Standard machine-readable error token")
    summary: str = Field(..., description="Clear explanation of why the simulation or stage failed")
    agent_action_required: str = Field(
        ...,
        description="Prescriptive instructions on what tool to call next or what parameter to adjust to recover",
    )
    technical_detail: Optional[str] = Field(
        None, description="Low-level traceback or raw exception message for debugging"
    )


class JobSubmissionResponse(BaseModel):
    job_id: str = Field(..., description="Unique asynchronous job tracking identifier")
    status: JobStatus = Field(JobStatus.PENDING, description="Initial job status")
    job_type: JobType = Field(..., description="Categorical type of stage/pipeline execution")
    created_at: str = Field(..., description="ISO timestamp of submission")
    message: str = Field(
        "Job queued successfully. Poll get_job_status with wait_for_completion=True to await completion.",
        description="User-friendly submission acknowledgment",
    )


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique asynchronous job tracking identifier")
    status: JobStatus = Field(..., description="Current lifecycle state")
    job_type: str = Field(..., description="Type of stage executed")
    progress_percent: float = Field(0.0, ge=0.0, le=100.0, description="Estimated completion percentage")
    step: int = Field(0, description="Current iteration or batch step")
    total_steps: int = Field(0, description="Total expected steps")
    latest_logs: str = Field("", description="Tail of stdout/stderr log stream (last 20 lines)")
    created_at: str = Field(..., description="ISO submission timestamp")
    started_at: Optional[str] = Field(None, description="ISO timestamp when execution began")
    finished_at: Optional[str] = Field(None, description="ISO timestamp when execution halted")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Structured output payload, pool IDs, and artifact paths upon COMPLETED"
    )
    semantic_error: Optional[SemanticError] = Field(None, description="Actionable LLM error explanation upon FAILED")


class RunFullPipelineRequest(BaseModel):
    target_spec: Dict[str, Any] | str = Field(
        ...,
        description="Material specification dictionary or path to a specification YAML (e.g. 'tests/data/conjugated_oled_semiconductors.yaml')",
    )
    train_steps: int = Field(25000, ge=100, description="PPO RL training steps for Stage 1 (default: 25,000)")
    num_candidates: int = Field(
        512, ge=1, le=50000, description="Number of candidate molecules to generate (default: 512)"
    )
    top_k: int = Field(20, ge=1, description="Number of top Pareto candidates to export to .mol2 (default: 20)")
    checkpoint: Optional[str] = Field(
        None, description="Optional path to existing trained_policy.pt to skip Stage 1 training"
    )
    solvent_id: str = Field("vacuum", description="Solvent environment (e.g. 'vacuum', 'water', 'ethanol')")
    temperature_k: float = Field(300.0, gt=0.0, description="Thermodynamic temperature in Kelvin")
    batch_size: Optional[int] = Field(
        None, description="Optional static GPU power-of-2 batch size (auto-detected if omitted)"
    )
    out_dir: Optional[str] = Field(None, description="Destination directory for output run artifacts")


class TrainSwarmRequest(BaseModel):
    target_spec: Dict[str, Any] | str = Field(
        ...,
        description="Material specification dictionary or path to a specification YAML",
    )
    total_timesteps: int = Field(25000, ge=100, description="Number of RL environment steps to train (default: 25,000)")
    num_envs: int = Field(16, ge=1, le=128, description="Number of parallel C-FFI environment workers (default: 16)")
    horizon: int = Field(16, ge=4, le=64, description="Rollout horizon per environment step")
    learning_rate: float = Field(3e-4, gt=0.0, description="PPO learning rate")
    use_curriculum: bool = Field(True, description="Enable 3-stage metric-driven curriculum")
    entropy_decay: float = Field(0.999, ge=0.5, le=1.0, description="Exploration entropy annealing factor")
    sa_penalty: bool = Field(True, description="Enable in-the-loop synthetic accessibility penalty")
    checkpoint_dir: Optional[str] = Field(None, description="Directory to save trained_policy.pt weights")


class SampleCandidatesRequest(BaseModel):
    model_weights_path: str = Field(..., description="Path to trained policy weights file (.pt)")
    target_spec: Optional[Dict[str, Any] | str] = Field(
        None, description="Optional target specification dictionary or YAML path (defaults to matching training spec)"
    )
    num_samples: int = Field(512, ge=1, le=50000, description="Total number of candidates to sample (default: 512)")
    temperature: float = Field(
        1.0, gt=0.0, le=5.0, description="Action sampling temperature (higher = greater diversity)"
    )
    continue_pipeline: bool = Field(
        False, description="If True, automatically continues execution through Stages 3, 4, and 5"
    )
    top_k: int = Field(20, ge=1, description="Number of Pareto candidates to export if continuing pipeline")


class CDFTThermoRequest(BaseModel):
    candidate_pool_id: Optional[str] = Field(
        None, description="Pointer handle to a candidate pool generated by sample_candidates"
    )
    smiles_list: Optional[List[str]] = Field(
        None, description="Direct list of canonical SMILES strings to screen (validated synchronously)"
    )
    candidates_dir: Optional[str] = Field(
        None, description="Directory containing external 3D .mol2 or .sdf candidate files"
    )
    target_spec: Optional[Dict[str, Any] | str] = Field(
        None, description="Target specification constraints dictionary or YAML path"
    )
    solvent_id: str = Field("vacuum", description="Solvent environment (e.g. 'vacuum', 'water', 'ethanol')")
    temperature_k: float = Field(300.0, gt=0.0, description="Temperature in Kelvin (default: 300.0)")
    cdft_steps: int = Field(50, ge=10, le=500, description="cDFT Euler-Lagrange variational steps")
    bg_steps: int = Field(30, ge=0, le=200, description="Boltzmann Generator normalizing flow steps")
    batch_size: Optional[int] = Field(None, description="GPU batch size (auto-throttled to 32/64/128)")
    continue_pipeline: bool = Field(
        False, description="If True, automatically continues through Stage 4 (Pareto ranking)"
    )
    top_k: int = Field(20, ge=1, description="Number of Pareto candidates to export if continuing pipeline")


class ParetoRankingRequest(BaseModel):
    thermo_pool_id: Optional[str] = Field(None, description="Pointer handle to a screened pool from run_cdft_thermo")
    scored_pool_id: Optional[str] = Field(None, description="Alias for thermo_pool_id for backward compatibility")
    candidates_dir: Optional[str] = Field(
        None, description="Directory containing candidate .mol2 files or previous pipeline results"
    )
    target_spec: Optional[Dict[str, Any] | str] = Field(
        None, description="Target specification constraints dictionary or YAML path"
    )
    ranking_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional importance weights dict: {'w_rl': 0.3, 'w_cdft': 0.3, 'w_boltzmann': 0.2, 'w_egnn': 0.2}",
    )
    max_sa_score: float = Field(6.0, ge=1.0, le=10.0, description="Synthesizability SA Score ceiling (default: 6.0)")
    disable_sa_filter: bool = Field(False, description="If True, disables the SA Score synthesizability filter")
    top_k: int = Field(20, ge=1, description="Number of top non-dominated candidates to export (default: 20)")


class JobPollRequest(BaseModel):
    wait_for_completion: bool = Field(
        True,
        description="If True, uses server-side long-polling (holds request open until COMPLETED, FAILED, or timeout)",
    )
    timeout_seconds: int = Field(
        300, ge=1, le=600, description="Maximum seconds to hold long-poll request open (default: 300s)"
    )


class CleanupMode(str, Enum):
    KEEP_PARETO_ONLY = "keep_pareto_only"
    POOL = "pool"
    FAILED = "failed"
    AGE = "age"
    ALL = "all"


class CleanupRequest(BaseModel):
    mode: CleanupMode = Field(
        CleanupMode.KEEP_PARETO_ONLY,
        description="Cleanup mode: 'keep_pareto_only' (prune intermediate 3D scratch files while preserving top-K), 'pool' (delete single pool_id), 'failed' (prune failed job scratch), 'age' (prune older than N hours), or 'all' (purge all pools)",
    )
    pool_id: Optional[str] = Field(None, description="Required when mode is 'pool'")
    older_than_hours: float = Field(24.0, ge=0.1, description="Threshold hours when mode is 'age' (default: 24h)")


class CleanupResponse(BaseModel):
    mode: str = Field(..., description="Applied cleanup mode")
    deleted_pools: List[str] = Field(default_factory=list, description="IDs of pruned artifact pools")
    freed_bytes: int = Field(0, description="Total storage bytes reclaimed")
    freed_mb: float = Field(0.0, description="Megabytes reclaimed")
    message: str = Field(..., description="Human-readable outcome summary")


class ValidationRequest(BaseModel):
    smiles_list: Optional[List[str]] = Field(None, description="List of SMILES to validate for valency and chemistry")
    target_spec: Optional[Dict[str, Any]] = Field(None, description="Specification dictionary to validate")


class InvalidSMILESEntry(BaseModel):
    index: int
    smiles: str
    reason: str
    atom_index: Optional[int] = None


class ValidationResponse(BaseModel):
    valid: bool = Field(..., description="True if all inputs passed chemical and structural validation")
    total_checked: int = Field(0, description="Total items inspected")
    invalid_smiles: List[InvalidSMILESEntry] = Field(
        default_factory=list, description="List of rejected SMILES with chemical root cause"
    )
    spec_errors: List[str] = Field(default_factory=list, description="Specification formatting or numeric bound errors")
    remediation: Optional[str] = Field(None, description="Actionable advice for the LLM on how to correct inputs")
