#!/usr/bin/env python3
"""
Antigravity PostToolUse Hook for WikiSkill.

Triggered after tool executions (specifically run_command).
If a command produces an error or failure, this hook automatically captures
an immutable execution trace in .agents/wikiskill/raw/traces/.
"""

import json
import sys
from pathlib import Path


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print("{}")
            return

        payload = json.loads(raw_input)
    except Exception:
        print("{}")
        return

    try:
        # Determine workspace root
        workspace_paths = payload.get("workspacePaths", [])
        if workspace_paths:
            workspace_root = Path(workspace_paths[0])
        else:
            # Fall back to parent directory of .agents
            workspace_root = Path(__file__).resolve().parent.parent.parent

        # Add src/ to sys.path so dens_city can be imported
        src_path = workspace_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        error = payload.get("error")
        tool_output = payload.get("toolOutput", "")

        # Only process run_command when an error or non-zero exit is indicated
        if tool_name == "run_command" and error:
            cmd = tool_args.get("CommandLine", "")
            if cmd:
                from dens_city.wikiskill.trace_recorder import ExecutionTrace, RawTraceRecorder

                recorder = RawTraceRecorder(workspace_root=workspace_root)
                import time
                from datetime import datetime, timezone

                trace_id = (
                    f"trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
                )
                test_results = recorder._parse_pytest_output(str(tool_output), str(error))
                summary = recorder._generate_summary(
                    cmd,
                    passed=False,
                    exit_code=1,
                    test_results=test_results,
                    stdout=str(tool_output),
                    stderr=str(error),
                )

                trace = ExecutionTrace(
                    trace_id=trace_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    command=cmd,
                    exit_code=1,
                    duration_sec=0.0,
                    stdout=str(tool_output),
                    stderr=str(error),
                    passed=False,
                    summary=summary,
                    test_results=test_results,
                    tags=["antigravity-hook", "auto-recorded"],
                    metadata={"stepIdx": payload.get("stepIdx")},
                )
                recorder.save_trace(trace)
    except Exception:
        pass

    # PostToolUse contract requires {}
    print("{}")


if __name__ == "__main__":
    main()
