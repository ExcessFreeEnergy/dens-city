"""
Model Context Protocol (MCP) Server for dens-city.
Exposes modular molecular inverse design tools to autonomous LLM harnesses
via Stdio or SSE/HTTP transport with native progress reporting and context-window protection.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import Context, MCPServer

from dens_city.server.jobs import JobManager
from dens_city.server.models import (
    CleanupMode,
    CleanupRequest,
    JobType,
)
from dens_city.server.pools import ArtifactPoolStore
from dens_city.server.validators import run_intake_validation, validate_smiles_list

# Initialize MCP server
mcp = MCPServer(
    name="dens-city-molecular-engine",
    version="0.2.0",
    description="Statistical mechanics Classical Density Functional Theory, Boltzmann Generative Flows, and EGNN Quantum Machine Learning Platform.",
)

job_manager = JobManager()
pool_store = ArtifactPoolStore()


# =============================================================================
# 1. End-to-End Orchestration Tool
# =============================================================================


@mcp.tool()
async def run_full_pipeline(
    target_spec: Dict[str, Any] | str,
    train_steps: int = 25000,
    num_candidates: int = 512,
    top_k: int = 20,
    checkpoint: Optional[str] = None,
    solvent_id: str = "vacuum",
    temperature_k: float = 300.0,
    batch_size: Optional[int] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the complete 5-stage generative molecular funnel on a target material specification.

    Runs Stage 1 (RL Swarm PPO) -> Stage 2 (Candidate Sampling) -> Stage 3 (cDFT + L-BFGS + Boltzmann)
    -> Stage 4 (EGNN Quantum Screening) -> Stage 5 (Pareto Ranking & .mol2 Export).

    Returns immediately with a job_id. Call get_job_status(job_id, wait_for_completion=True) to await results.

    Args:
        target_spec: JSON dict containing physical targets or path to a specification YAML.
        train_steps: PPO training timesteps (default: 25,000).
        num_candidates: Number of candidates to sample and screen (default: 512).
        top_k: Number of Pareto-optimal candidates to export to .mol2 (default: 20).
        checkpoint: Optional path to pre-trained policy .pt weights to skip Stage 1 training.
        solvent_id: Solvent context (e.g. 'water', 'ethanol', 'vacuum').
        temperature_k: Reservoir temperature in Kelvin (default: 300.0).
        batch_size: GPU power-of-2 batch size (auto-throttled if omitted).
        out_dir: Destination folder for output artifacts.
    """
    params = {
        "target_spec": target_spec,
        "train_steps": train_steps,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "checkpoint": checkpoint,
        "solvent_id": solvent_id,
        "temperature_k": temperature_k,
        "batch_size": batch_size,
        "out_dir": out_dir,
    }
    job_id = job_manager.enqueue_job(JobType.FULL_PIPELINE, params)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": (
            f"Queued full 5-stage pipeline for spec '{target_spec}'. "
            f"Call get_job_status(job_id='{job_id}', wait_for_completion=True) to await completion."
        ),
    }


# =============================================================================
# 2. Granular Stage Execution Tools (Composability & Reinjection)
# =============================================================================


@mcp.tool()
async def train_swarm_agent(
    target_spec: Dict[str, Any] | str,
    total_timesteps: int = 25000,
    num_envs: int = 16,
    horizon: int = 16,
    learning_rate: float = 3e-4,
    use_curriculum: bool = True,
    entropy_decay: float = 0.999,
    sa_penalty: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage 1: Trains the PufferLib RL Molecular Swarm policy to assemble goal-directed molecular graphs.

    Learns discrete bond/atom placements matching the target physical constraints.
    Returns immediately with a job_id that resolves to model_weights_path upon completion.

    Args:
        target_spec: Target specification dictionary or YAML path.
        total_timesteps: Total environment timesteps to train (default: 25,000).
        num_envs: Parallel C-FFI environment workers (default: 16).
        horizon: Rollout horizon per environment step (default: 16).
        learning_rate: PPO policy learning rate (default: 3e-4).
        use_curriculum: Enable 3-stage metric-driven curriculum (default: True).
        entropy_decay: Annealing multiplier for exploration entropy (default: 0.999).
        sa_penalty: Penalize high synthetic accessibility difficulty in the loop (default: True).
        checkpoint_dir: Output directory for trained_policy.pt weights.
    """
    params = {
        "target_spec": target_spec,
        "total_timesteps": total_timesteps,
        "num_envs": num_envs,
        "horizon": horizon,
        "learning_rate": learning_rate,
        "use_curriculum": use_curriculum,
        "entropy_decay": entropy_decay,
        "sa_penalty": sa_penalty,
        "checkpoint_dir": checkpoint_dir,
    }
    job_id = job_manager.enqueue_job(JobType.TRAIN_SWARM, params)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": f"Queued Stage 1 swarm training. Call get_job_status('{job_id}', wait_for_completion=True).",
    }


@mcp.tool()
async def sample_candidates(
    model_weights_path: str,
    target_spec: Optional[Dict[str, Any] | str] = None,
    num_samples: int = 512,
    temperature: float = 1.0,
    continue_pipeline: bool = False,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Stage 2: Samples molecular candidates from a trained policy checkpoint.

    Extracts valid molecular graphs directly from C memory into a persistent candidate pool.

    Args:
        model_weights_path: Path to trained policy checkpoint (.pt file).
        target_spec: Target specification constraints or YAML path.
        num_samples: Number of candidate molecules to generate (default: 512; supports 1 to 50,000+).
        temperature: Action sampling temperature (higher = greater diversity, default: 1.0).
        continue_pipeline: If True, automatically chains through Stages 3, 4, and 5 to final Pareto export.
        top_k: Number of Pareto candidates to export if continuing pipeline (default: 20).
    """
    params = {
        "model_weights_path": model_weights_path,
        "target_spec": target_spec,
        "num_samples": num_samples,
        "temperature": temperature,
        "continue_pipeline": continue_pipeline,
        "top_k": top_k,
    }
    job_id = job_manager.enqueue_job(JobType.SAMPLE_CANDIDATES, params)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": (
            f"Queued Stage 2 candidate sampling ({num_samples} samples, continue={continue_pipeline}). "
            f"Call get_job_status('{job_id}', wait_for_completion=True)."
        ),
    }


@mcp.tool()
async def run_cdft_thermo(
    candidate_pool_id: Optional[str] = None,
    smiles_list: Optional[List[str]] = None,
    candidates_dir: Optional[str] = None,
    target_spec: Optional[Dict[str, Any] | str] = None,
    solvent_id: str = "vacuum",
    temperature_k: float = 300.0,
    cdft_steps: int = 50,
    bg_steps: int = 30,
    batch_size: Optional[int] = None,
    continue_pipeline: bool = False,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Stage 3: Evaluates Classical DFT thermodynamics, L-BFGS geometry relaxation, and Boltzmann Flows.

    Accepts a candidate_pool_id from Stage 2, or direct smiles_list (validated synchronously), or a directory of .mol2 files.
    Automatically batches large pools (e.g. 10,000+ molecules) in chunks of 1024 to prevent memory exhaustion.

    Args:
        candidate_pool_id: Pool handle from sample_candidates.
        smiles_list: Direct list of SMILES to screen (synchronously checked for chemical validity).
        candidates_dir: Path to directory of external 3D .mol2 files.
        target_spec: Target specification constraints or YAML path.
        solvent_id: Solvent environment (default: 'water').
        temperature_k: Temperature in Kelvin (default: 300.0).
        cdft_steps: Euler-Lagrange variational optimization steps (default: 50).
        bg_steps: Boltzmann Generator training steps (default: 30).
        batch_size: GPU power-of-2 batch size (e.g. 64, 128, 1024).
        continue_pipeline: If True, automatically continues through Stages 4 (EGNN) and 5 (Pareto ranking).
        top_k: Number of Pareto candidates to export if continuing pipeline (default: 20).
    """
    # Synchronous Chemical Intake Gate for direct SMILES
    if smiles_list:
        is_valid, invalid_entries = validate_smiles_list(smiles_list)
        if not is_valid:
            first_err = invalid_entries[0]
            return {
                "error": "CHEMICAL_INTAKE_REJECTED",
                "summary": f"Synchronous validation rejected SMILES #{first_err.index} ('{first_err.smiles}'): {first_err.reason}",
                "agent_action_required": "Fix chemical valency and structure before calling run_cdft_thermo.",
            }

    params = {
        "candidate_pool_id": candidate_pool_id,
        "smiles_list": smiles_list,
        "candidates_dir": candidates_dir,
        "target_spec": target_spec,
        "solvent_id": solvent_id,
        "temperature_k": temperature_k,
        "cdft_steps": cdft_steps,
        "bg_steps": bg_steps,
        "batch_size": batch_size,
        "continue_pipeline": continue_pipeline,
        "top_k": top_k,
    }
    job_id = job_manager.enqueue_job(JobType.RUN_CDFT, params)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": f"Queued Stage 3 cDFT thermodynamics. Call get_job_status('{job_id}', wait_for_completion=True).",
    }


@mcp.tool()
async def run_egnn_quantum(
    thermo_pool_id: Optional[str] = None,
    candidates_dir: Optional[str] = None,
    target_spec: Optional[Dict[str, Any] | str] = None,
    relax_steps: int = 50,
    egnn_layers: int = 7,
    batch_size: Optional[int] = None,
    continue_pipeline: bool = False,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Stage 4: Evaluates quantum ground-state energy, force residuals, and electrostatic solvation via 7-layer EGNN.

    Evaluates 3D coordinates from thermo_pool_id or an external candidates_dir.

    Args:
        thermo_pool_id: Pool handle from run_cdft_thermo.
        candidates_dir: Directory containing external 3D .mol2 structures.
        target_spec: Target specification constraints or YAML path.
        relax_steps: Unrolled GPU quantum geometry relaxation steps (default: 50).
        egnn_layers: Number of EGNN message passing layers (default: 7).
        batch_size: EGNN GPU batch size (default: 32).
        continue_pipeline: If True, automatically continues to Stage 5 (Pareto ranking).
        top_k: Number of Pareto candidates to export if continuing pipeline (default: 20).
    """
    params = {
        "thermo_pool_id": thermo_pool_id,
        "candidates_dir": candidates_dir,
        "target_spec": target_spec,
        "relax_steps": relax_steps,
        "egnn_layers": egnn_layers,
        "batch_size": batch_size,
        "continue_pipeline": continue_pipeline,
        "top_k": top_k,
    }
    job_id = job_manager.enqueue_job(JobType.RUN_EGNN, params)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": f"Queued Stage 4 EGNN quantum evaluation. Call get_job_status('{job_id}', wait_for_completion=True).",
    }


@mcp.tool()
async def rank_pareto_frontier(
    scored_pool_id: Optional[str] = None,
    candidates_dir: Optional[str] = None,
    target_spec: Optional[Dict[str, Any] | str] = None,
    ranking_weights: Optional[Dict[str, float]] = None,
    max_sa_score: float = 6.0,
    disable_sa_filter: bool = False,
    top_k: int = 20,
) -> Dict[str, Any]:
    """Stage 5: Performs topological deduplication, SA Score synthesizability safety gating, and non-dominated sorting.

    Exports top-K candidates to 3D .mol2 files, CSV summary, JSON summary, and Markdown report.
    Supports single molecules up to 10,000+ candidates.

    Args:
        scored_pool_id: Pool handle from run_egnn_quantum.
        candidates_dir: Directory containing candidate .mol2 files.
        target_spec: Target specification constraints or YAML path.
        ranking_weights: Importance weights dict (e.g. {'w_rl': 0.3, 'w_cdft': 0.3, 'w_boltzmann': 0.2, 'w_egnn': 0.2}).
        max_sa_score: RDKit Synthesizability (SA) Score threshold (default: 6.0; lower = easier to synthesize).
        disable_sa_filter: If True, disables dropping high-difficulty candidates.
        top_k: Number of top Pareto candidates to export to .mol2 (default: 20).
    """
    params = {
        "scored_pool_id": scored_pool_id,
        "candidates_dir": candidates_dir,
        "target_spec": target_spec,
        "ranking_weights": ranking_weights,
        "max_sa_score": max_sa_score,
        "disable_sa_filter": disable_sa_filter,
        "top_k": top_k,
    }
    job_id = job_manager.enqueue_job(JobType.RANK_PARETO, params)
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": f"Queued Stage 5 Pareto ranking. Call get_job_status('{job_id}', wait_for_completion=True).",
    }


# =============================================================================
# 3. Agentic Observability & Long-Polling Status Tool
# =============================================================================


@mcp.tool()
async def get_job_status(
    job_id: str,
    wait_for_completion: bool = True,
    timeout_seconds: int = 300,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Polls the status of an asynchronous job with server-side long-polling and native progress streaming.

    CRITICAL FOR TOKEN CONSERVATION:
    Keep wait_for_completion=True (default). The server will hold the connection open for up to
    timeout_seconds (default: 300s) until the job finishes or fails, compressing 30 polling turns into 1!

    Args:
        job_id: Asynchronous job ID returned by any stage or pipeline tool call.
        wait_for_completion: If True, holds the tool call open until terminal status (COMPLETED/FAILED) or timeout.
        timeout_seconds: Maximum seconds to hold long-polling connection open (default: 300s).
    """
    if wait_for_completion:
        try:
            job = await job_manager.wait_for_job(job_id, timeout_seconds=timeout_seconds)
        except KeyError:
            return {"status": "FAILED", "error": f"Job ID '{job_id}' not found."}
    else:
        job = job_manager.get_job(job_id)
        if not job:
            return {"status": "FAILED", "error": f"Job ID '{job_id}' not found."}

    # Stream native progress notification to MCP client if supported
    if ctx and hasattr(ctx, "report_progress"):
        try:
            await ctx.report_progress(job.progress_percent, 100.0)
        except Exception:
            pass

    res = job.model_dump()
    return res


@mcp.tool()
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cooperatively cancels an active or queued job.

    Args:
        job_id: Target job ID to cancel.
    """
    success = job_manager.cancel_job(job_id)
    return {
        "job_id": job_id,
        "cancelled": success,
        "message": f"Job {job_id} cancelled."
        if success
        else f"Could not cancel job {job_id} (already completed or not found).",
    }


# =============================================================================
# 4. Synchronous Validation & Chemical Gate Tool
# =============================================================================


@mcp.tool()
async def validate_spec(
    smiles_list: Optional[List[str]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synchronously validates chemical SMILES strings and target specification parameters.

    Call this tool BEFORE queueing long simulations to ensure chemical valency (carbon valency <= 4),
    ring closure, supported element symbols, and physical numeric ranges are satisfied.

    Args:
        smiles_list: Optional list of candidate SMILES strings to validate.
        target_spec: Optional target specification dictionary to validate.
    """
    val_res = run_intake_validation(smiles_list=smiles_list, target_spec=target_spec)
    return val_res.model_dump()


# =============================================================================
# 5. Artifact & Disk Creep Cleanup Tool
# =============================================================================


@mcp.tool()
async def cleanup_artifacts(
    mode: str = "keep_pareto_only",
    pool_id: Optional[str] = None,
    older_than_hours: float = 24.0,
) -> Dict[str, Any]:
    """Prunes intermediate candidate and scratch files to prevent disk space creep.

    Args:
        mode: Cleanup policy:
            - 'keep_pareto_only': (Default) Removes heavy intermediate Stage 2/3/4 pools while retaining final Pareto .mol2 and CSV summaries.
            - 'pool': Deletes a specific pool_id.
            - 'age': Prunes pools older than older_than_hours.
            - 'all': Purges all artifact pools.
        pool_id: Specific pool ID to remove (required when mode='pool').
        older_than_hours: Pruning threshold in hours when mode='age' (default: 24.0).
    """
    try:
        c_mode = CleanupMode(mode)
    except ValueError:
        return {"error": f"Invalid mode '{mode}'. Choose from: keep_pareto_only, pool, age, all."}

    req = CleanupRequest(mode=c_mode, pool_id=pool_id, older_than_hours=older_than_hours)
    if req.mode == CleanupMode.POOL:
        if not req.pool_id:
            return {"error": "pool_id is required when mode='pool'."}
        freed = pool_store.cleanup_pool(req.pool_id)
        return {"mode": req.mode.value, "deleted_pools": [req.pool_id], "freed_mb": freed / (1024 * 1024)}
    elif req.mode == CleanupMode.KEEP_PARETO_ONLY:
        deleted, freed = pool_store.prune_intermediate_pools(keep_pareto=True)
        return {"mode": req.mode.value, "deleted_pools": deleted, "freed_mb": freed / (1024 * 1024)}
    elif req.mode == CleanupMode.AGE:
        deleted, freed = pool_store.prune_older_than(hours=req.older_than_hours)
        return {"mode": req.mode.value, "deleted_pools": deleted, "freed_mb": freed / (1024 * 1024)}
    elif req.mode == CleanupMode.ALL:
        deleted, freed = pool_store.prune_older_than(hours=0.0)
        return {"mode": req.mode.value, "deleted_pools": deleted, "freed_mb": freed / (1024 * 1024)}


def main():
    """CLI entrypoint for running the MCP server over Stdio or SSE."""
    parser = argparse.ArgumentParser(description="dens-city Model Context Protocol (MCP) Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE transport (default: 8001)")
    args = parser.parse_args()

    print(f"[dens-city MCP] Starting server (transport={args.transport})...", file=sys.stderr)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
