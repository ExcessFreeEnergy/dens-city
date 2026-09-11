"""
SQLite-backed Asynchronous Job Queue and Long-Polling Manager for dens-city server.
Enforces non-blocking execution, context-window token compression, and cooperative job cancellation.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dens_city.server.models import (
    JobStatus,
    JobStatusResponse,
    JobType,
    SemanticError,
)


class JobManager:
    """
    Manages persistent job records in SQLite with long-polling and log capture.
    """

    def __init__(self, db_dir: str | Path = "runs/jobs"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "jobs.db"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    progress_percent REAL DEFAULT 0.0,
                    step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    log_file TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_summary TEXT,
                    agent_action_required TEXT,
                    technical_detail TEXT
                )
                """
            )
            conn.commit()

    def enqueue_job(self, job_type: JobType, params: Dict[str, Any]) -> str:
        """
        Creates a new PENDING job and returns its unique job_id.
        """
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        rand = time.strftime("%f")[:5]
        job_id = f"job_{ts_str}_{rand}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        log_file = str((self.db_dir / f"{job_id}.log").resolve())
        # Touch log file
        Path(log_file).touch()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, status, params_json, progress_percent,
                    step, total_steps, created_at, log_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type.value,
                    JobStatus.PENDING.value,
                    json.dumps(params),
                    0.0,
                    0,
                    0,
                    now,
                    log_file,
                ),
            )
            conn.commit()

        return job_id

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Retrieves current status and log tail for a job.
        """
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None

        # Fetch latest 20 log lines
        log_path = Path(row["log_file"])
        latest_logs = ""
        if log_path.exists():
            try:
                lines = log_path.read_text(errors="replace").splitlines()
                latest_logs = "\n".join(lines[-20:]) if lines else ""
            except Exception:
                pass

        result_dict = json.loads(row["result_json"]) if row["result_json"] else None

        semantic_err = None
        if row["error_code"]:
            semantic_err = SemanticError(
                error_code=row["error_code"],
                summary=row["error_summary"] or "Stage execution failed",
                agent_action_required=row["agent_action_required"] or "Inspect logs",
                technical_detail=row["technical_detail"],
            )

        return JobStatusResponse(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            job_type=row["job_type"],
            progress_percent=float(row["progress_percent"] or 0.0),
            step=int(row["step"] or 0),
            total_steps=int(row["total_steps"] or 0),
            latest_logs=latest_logs,
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result=result_dict,
            semantic_error=semantic_err,
        )

    async def wait_for_job(
        self,
        job_id: str,
        timeout_seconds: int = 300,
        poll_interval: float = 1.0,
    ) -> JobStatusResponse:
        """
        Server-side long-polling implementation.
        Holds connection open until job reaches terminal state or timeout_seconds elapses.
        Compresses dozens of LLM client round-trips into a single tool invocation.
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            job = self.get_job(job_id)
            if not job:
                raise KeyError(f"Job {job_id} not found.")

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job

            await asyncio.sleep(poll_interval)

        # Timeout reached: return current status (e.g. RUNNING with latest progress & logs)
        return self.get_job(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Signals cooperative cancellation of a job.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, error_code = ?, error_summary = ?, agent_action_required = ?
                WHERE job_id = ? AND status IN (?, ?)
                """,
                (
                    JobStatus.CANCELLED.value,
                    now,
                    "JOB_CANCELLED_BY_AGENT",
                    "Job was cancelled via agent request.",
                    "No action required. Submit a new job when ready.",
                    job_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_jobs(self, limit: int = 50) -> List[JobStatusResponse]:
        """Lists recent jobs."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT job_id FROM jobs ORDER BY rowid DESC LIMIT ?", (limit,))
            job_ids = [row["job_id"] for row in cur.fetchall()]

        res = []
        for j_id in job_ids:
            j = self.get_job(j_id)
            if j:
                res.append(j)
        return res
