"""
FastAPI application for dens-city self-driving autonomous LLM harness.
Exposes non-blocking REST endpoints, server-side long-polling, synchronous chemical intake gate,
and artifact disk creep cleanup.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from dens_city.server.jobs import JobManager
from dens_city.server.models import (
    CDFTThermoRequest,
    CleanupMode,
    CleanupRequest,
    CleanupResponse,
    EGNNQuantumRequest,
    JobStatusResponse,
    JobSubmissionResponse,
    JobType,
    ParetoRankingRequest,
    RunFullPipelineRequest,
    SampleCandidatesRequest,
    TrainSwarmRequest,
    ValidationRequest,
    ValidationResponse,
)
from dens_city.server.pools import ArtifactPoolStore
from dens_city.server.validators import run_intake_validation, validate_smiles_list
from dens_city.server.worker import start_worker_process

job_manager = JobManager()
pool_store = ArtifactPoolStore()
worker_proc: Optional[mp.Process] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_proc
    # Spawn dedicated background GPU worker process on server startup
    print("[Server Lifecycle] Initializing SQLite job manager and spawning GPU worker...")
    worker_proc = start_worker_process(job_manager.db_path)
    print(f"[Server Lifecycle] Dedicated GPU worker running (PID={worker_proc.pid}).")
    yield
    # Shutdown worker
    if worker_proc and worker_proc.is_alive():
        print("[Server Lifecycle] Terminating GPU worker process...")
        worker_proc.terminate()
        worker_proc.join(timeout=5.0)


app = FastAPI(
    title="dens-city Autonomous Material Design Server",
    version="0.2.0",
    description="High-performance asynchronous molecular cDFT, Boltzmann Flows, and EGNN server for LLM harnesses.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", summary="Service Health & Worker Probe")
async def healthz():
    worker_alive = bool(worker_proc and worker_proc.is_alive())
    return {
        "status": "HEALTHY" if worker_alive else "DEGRADED",
        "worker_pid": worker_proc.pid if worker_proc else None,
        "worker_alive": worker_alive,
        "timestamp": time.time(),
    }


# =========================================================================
# 1. Pipeline & Stage Submissions (All Return job_id Immediately)
# =========================================================================


@app.post(
    "/api/v1/pipeline/full",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run Complete 5-Stage Molecular Funnel",
)
async def submit_full_pipeline(req: RunFullPipelineRequest):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job_id = job_manager.enqueue_job(JobType.FULL_PIPELINE, req.model_dump())
    return JobSubmissionResponse(
        job_id=job_id,
        job_type=JobType.FULL_PIPELINE,
        created_at=now,
    )


@app.post(
    "/api/v1/stages/train-swarm",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stage 1: Train RL Swarm Policy",
)
async def submit_train_swarm(req: TrainSwarmRequest):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job_id = job_manager.enqueue_job(JobType.TRAIN_SWARM, req.model_dump())
    return JobSubmissionResponse(
        job_id=job_id,
        job_type=JobType.TRAIN_SWARM,
        created_at=now,
    )


@app.post(
    "/api/v1/stages/sample-candidates",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stage 2: Sample Molecular Candidates from Policy",
)
async def submit_sample_candidates(req: SampleCandidatesRequest):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job_id = job_manager.enqueue_job(JobType.SAMPLE_CANDIDATES, req.model_dump())
    return JobSubmissionResponse(
        job_id=job_id,
        job_type=JobType.SAMPLE_CANDIDATES,
        created_at=now,
    )


@app.post(
    "/api/v1/stages/run-cdft",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stage 3: Run cDFT & Boltzmann Thermodynamics",
)
async def submit_run_cdft(req: CDFTThermoRequest):
    # Synchronous Chemical Intake Gate: Validate any direct SMILES inputs immediately
    if req.smiles_list:
        is_valid, invalid_entries = validate_smiles_list(req.smiles_list)
        if not is_valid:
            first_err = invalid_entries[0]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "CHEMICAL_INTAKE_REJECTED",
                    "summary": f"Synchronous validation failed for SMILES at index {first_err.index} ('{first_err.smiles}').",
                    "reason": first_err.reason,
                    "agent_action_required": "Correct chemical valency before queueing GPU simulation.",
                },
            )

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job_id = job_manager.enqueue_job(JobType.RUN_CDFT, req.model_dump())
    return JobSubmissionResponse(
        job_id=job_id,
        job_type=JobType.RUN_CDFT,
        created_at=now,
    )


@app.post(
    "/api/v1/stages/run-egnn",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stage 4: Run EGNN Quantum Surrogate Screening",
)
async def submit_run_egnn(req: EGNNQuantumRequest):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job_id = job_manager.enqueue_job(JobType.RUN_EGNN, req.model_dump())
    return JobSubmissionResponse(
        job_id=job_id,
        job_type=JobType.RUN_EGNN,
        created_at=now,
    )


@app.post(
    "/api/v1/stages/rank-pareto",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Stage 5: Multi-Objective Pareto Ranking & Export",
)
async def submit_rank_pareto(req: ParetoRankingRequest):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    job_id = job_manager.enqueue_job(JobType.RANK_PARETO, req.model_dump())
    return JobSubmissionResponse(
        job_id=job_id,
        job_type=JobType.RANK_PARETO,
        created_at=now,
    )


# =========================================================================
# 2. Agentic Observability & Server-Side Long-Polling
# =========================================================================


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get Job Status (Supports Server-Side Long-Polling)",
)
async def get_job_status(
    job_id: str,
    wait_for_completion: bool = Query(
        True,
        description="If True, server holds connection open until job reaches terminal state or timeout (compresses polling turns).",
    ),
    timeout_seconds: int = Query(
        300,
        ge=1,
        le=600,
        description="Maximum seconds to hold long-polling connection open.",
    ),
):
    if wait_for_completion:
        try:
            return await job_manager.wait_for_job(job_id, timeout_seconds=timeout_seconds)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.post("/api/v1/jobs/{job_id}/cancel", summary="Cancel Running Job")
async def cancel_job(job_id: str):
    cancelled = job_manager.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail=f"Could not cancel job {job_id} (already finished or not found).")
    return {"message": f"Job {job_id} cancelled successfully."}


@app.get("/api/v1/jobs", summary="List Recent Jobs")
async def list_jobs(limit: int = Query(20, ge=1, le=100)):
    return job_manager.list_jobs(limit=limit)


# =========================================================================
# 3. Synchronous Validation & Chemical Intake Gate
# =========================================================================


@app.post(
    "/api/v1/spec/validate",
    response_model=ValidationResponse,
    summary="Synchronous Chemical & Specification Validation Gate",
)
async def validate_spec(req: ValidationRequest):
    return run_intake_validation(
        smiles_list=req.smiles_list,
        target_spec=req.target_spec,
    )


# =========================================================================
# 4. Artifact & Disk Creep Management
# =========================================================================


@app.post(
    "/api/v1/pools/cleanup",
    response_model=CleanupResponse,
    summary="Prune Intermediate Artifact Pools to Prevent Disk Creep",
)
async def cleanup_pools(req: CleanupRequest):
    if req.mode == CleanupMode.POOL:
        if not req.pool_id:
            raise HTTPException(status_code=400, detail="pool_id required when mode='pool'")
        freed = pool_store.cleanup_pool(req.pool_id)
        return CleanupResponse(
            mode=req.mode.value,
            deleted_pools=[req.pool_id],
            freed_bytes=freed,
            freed_mb=freed / (1024 * 1024),
            message=f"Deleted pool {req.pool_id} ({freed / (1024 * 1024):.2f} MB freed).",
        )
    elif req.mode == CleanupMode.KEEP_PARETO_ONLY:
        deleted, freed = pool_store.prune_intermediate_pools(keep_pareto=True)
        return CleanupResponse(
            mode=req.mode.value,
            deleted_pools=deleted,
            freed_bytes=freed,
            freed_mb=freed / (1024 * 1024),
            message=f"Pruned {len(deleted)} intermediate pool(s), keeping Pareto exports ({freed / (1024 * 1024):.2f} MB freed).",
        )
    elif req.mode == CleanupMode.AGE:
        deleted, freed = pool_store.prune_older_than(hours=req.older_than_hours)
        return CleanupResponse(
            mode=req.mode.value,
            deleted_pools=deleted,
            freed_bytes=freed,
            freed_mb=freed / (1024 * 1024),
            message=f"Pruned {len(deleted)} pool(s) older than {req.older_than_hours}h ({freed / (1024 * 1024):.2f} MB freed).",
        )
    elif req.mode == CleanupMode.ALL:
        deleted, freed = pool_store.prune_older_than(hours=0.0)
        return CleanupResponse(
            mode=req.mode.value,
            deleted_pools=deleted,
            freed_bytes=freed,
            freed_mb=freed / (1024 * 1024),
            message=f"Purged all {len(deleted)} pool(s) ({freed / (1024 * 1024):.2f} MB freed).",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")


@app.get("/api/v1/pools/stats", summary="Get Artifact Disk Storage Usage")
async def get_pool_stats():
    return pool_store.get_storage_stats()


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for CLI or `dens-city-server` command."""
    import uvicorn

    uvicorn.run("dens_city.server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
