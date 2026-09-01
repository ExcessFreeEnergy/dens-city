"""
Raw Trace Recorder: Immutable execution trace storage for WikiSkill.

Records execution logs, command outputs, pytest test runs, and physics simulation
failures into immutable structured files under .agents/wikiskill/raw/traces/.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionTrace:
    trace_id: str
    timestamp: str
    command: str
    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str
    passed: bool
    summary: str
    test_results: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionTrace:
        return cls(**data)


class RawTraceRecorder:
    """Manages the immutable Raw Layer of execution traces."""

    def __init__(self, workspace_root: Optional[Path | str] = None) -> None:
        if workspace_root is None:
            self.workspace_root = Path.cwd()
        else:
            self.workspace_root = Path(workspace_root)
        self.raw_dir = self.workspace_root / ".agents" / "wikiskill" / "raw"
        self.traces_dir = self.raw_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def record_command(
        self,
        command: str | List[str],
        cwd: Optional[Path | str] = None,
        env: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """Executes a command, records full stdout/stderr, and saves an immutable trace."""
        cwd_path = Path(cwd) if cwd else self.workspace_root
        cmd_str = command if isinstance(command, str) else " ".join(command)
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        trace_id = f"trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"

        # Inherit PATH with venv priority
        run_env = os.environ.copy()
        venv_bin = self.workspace_root / ".venv" / "bin"
        if venv_bin.exists():
            run_env["PATH"] = f"{venv_bin}:{run_env.get('PATH', '')}"
        if env:
            run_env.update(env)

        try:
            res = subprocess.run(
                cmd_str,
                shell=True,
                cwd=str(cwd_path),
                env=run_env,
                capture_output=True,
                text=True,
            )
            exit_code = res.returncode
            stdout = res.stdout
            stderr = res.stderr
        except Exception as e:
            exit_code = -1
            stdout = ""
            stderr = f"Execution exception: {e}"

        duration = time.time() - start_time
        passed = exit_code == 0

        # Parse test results if pytest was run
        test_results = self._parse_pytest_output(stdout, stderr)
        summary = self._generate_summary(cmd_str, passed, exit_code, test_results, stdout, stderr)

        trace = ExecutionTrace(
            trace_id=trace_id,
            timestamp=timestamp,
            command=cmd_str,
            exit_code=exit_code,
            duration_sec=round(duration, 3),
            stdout=stdout,
            stderr=stderr,
            passed=passed,
            summary=summary,
            test_results=test_results,
            tags=tags or [],
            metadata=metadata or {},
        )

        self.save_trace(trace)
        return trace

    def save_trace(self, trace: ExecutionTrace) -> Path:
        """Saves a trace to the traces directory. File is immutable."""
        trace_file = self.traces_dir / f"{trace.trace_id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2)
        return trace_file

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Loads a trace by ID."""
        clean_id = trace_id.replace(".json", "")
        trace_file = self.traces_dir / f"{clean_id}.json"
        if not trace_file.exists():
            return None
        with open(trace_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExecutionTrace.from_dict(data)

    def list_traces(self, limit: int = 50) -> List[ExecutionTrace]:
        """Lists recent traces in chronological order."""
        files = sorted(self.traces_dir.glob("trace_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        traces = []
        for f in files[:limit]:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                traces.append(ExecutionTrace.from_dict(data))
            except Exception:
                continue
        return traces

    def sample_traces(self, max_failing: int = 5, max_passing: int = 3) -> List[ExecutionTrace]:
        """Stratified trace sampling for Wiki Maintainer (failures prioritized for root cause)."""
        recent = self.list_traces(limit=50)
        failing = [t for t in recent if not t.passed][:max_failing]
        passing = [t for t in recent if t.passed][:max_passing]
        return failing + passing

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parses pytest summary statistics from output."""
        res: Dict[str, Any] = {"passed": 0, "failed": 0, "errors": 0, "failed_tests": []}
        lines = (stdout + "\n" + stderr).splitlines()
        for line in lines:
            if " FAILED " in line:
                test_name = line.split(" FAILED ")[0].strip()
                res["failed_tests"].append(test_name)
            if " passed" in line or " failed" in line or " error" in line:
                import re

                p_match = re.search(r"(\d+)\s+passed", line)
                if p_match:
                    res["passed"] = int(p_match.group(1))
                f_match = re.search(r"(\d+)\s+failed", line)
                if f_match:
                    res["failed"] = int(f_match.group(1))
                e_match = re.search(r"(\d+)\s+error", line)
                if e_match:
                    res["errors"] = int(e_match.group(1))
        return res

    def _generate_summary(
        self,
        cmd: str,
        passed: bool,
        exit_code: int,
        test_results: Dict[str, Any],
        stdout: str,
        stderr: str,
    ) -> str:
        """Generates a high-level summary string."""
        if test_results.get("failed", 0) > 0 or test_results.get("passed", 0) > 0:
            p_cnt = test_results.get("passed", 0)
            f_cnt = test_results.get("failed", 0)
            e_cnt = test_results.get("errors", 0)
            status = "PASSED" if (f_cnt == 0 and e_cnt == 0 and passed) else "FAILED"
            failed_tests = test_results.get("failed_tests", [])
            f_str = f" (Failed: {', '.join(failed_tests[:3])})" if failed_tests else ""
            return f"Pytest {status}: {p_cnt} passed, {f_cnt} failed, {e_cnt} errors{f_str}"

        status = "PASSED" if passed else f"FAILED (exit code {exit_code})"
        # Extract last non-empty line of output
        out_lines = [line.strip() for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
        last_line = out_lines[-1] if out_lines else "No output"
        return f"Command `{cmd}` {status}: {last_line[:120]}"
