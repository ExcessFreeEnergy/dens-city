"""
Wiki Maintainer Agent: Pattern Consolidation for WikiSkill.

Consolidates raw execution traces into structured, persistent knowledge:
- Performs root-cause analysis on failure traces (identifying WHY failures occur).
- Extracts effective strategies from passing traces.
- Creates new pattern pages under wiki/patterns/ or updates existing ones via patch edits.
- Maintains and refreshes wiki/index.md and wiki/logs.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dens_city.wikiskill.trace_recorder import ExecutionTrace, RawTraceRecorder
from dens_city.wikiskill.wiki_manager import PatchOperation, WikiManager, WikiPattern


@dataclass
class MaintenanceReport:
    iteration: int
    patterns_created: List[str]
    patterns_updated: List[str]
    log_summary: str
    diagnosed_failures: List[str]


class WikiMaintainer:
    """Orchestrates pattern consolidation from execution traces into the Wiki."""

    def __init__(self, wiki_manager: WikiManager, trace_recorder: RawTraceRecorder) -> None:
        self.wiki = wiki_manager
        self.traces = trace_recorder

    def consolidate_traces(
        self,
        iteration: int = 1,
        sampled_traces: Optional[List[ExecutionTrace]] = None,
    ) -> MaintenanceReport:
        """Analyzes sampled traces and updates wiki patterns, index, and logs."""
        if sampled_traces is None:
            sampled_traces = self.traces.sample_traces(max_failing=5, max_passing=3)

        created_patterns: List[str] = []
        updated_patterns: List[str] = []
        diagnosed: List[str] = []

        if not sampled_traces:
            summary = "No new traces available for consolidation."
            self.wiki.append_log(summary, iteration=iteration)
            return MaintenanceReport(
                iteration=iteration,
                patterns_created=[],
                patterns_updated=[],
                log_summary=summary,
                diagnosed_failures=[],
            )

        failing = [t for t in sampled_traces if not t.passed]
        passing = [t for t in sampled_traces if t.passed]

        # Analyze failing traces
        for trace in failing:
            diagnosis = self._diagnose_trace(trace)
            if diagnosis:
                diagnosed.append(diagnosis["title"])
                pattern_name = diagnosis["name"]
                existing = self.wiki.get_pattern(pattern_name)

                if existing:
                    # Patch existing pattern with new trace evidence
                    evidence_text = (
                        f"\n### Evidence from Trace `{trace.trace_id}` ({trace.timestamp})\n"
                        f"- Command: `{trace.command}`\n"
                        f"- Observed Error: {diagnosis['observed_error']}\n"
                    )
                    self.wiki.patch_pattern(
                        pattern_name,
                        [PatchOperation(op="append", content=evidence_text)],
                    )
                    updated_patterns.append(pattern_name)
                else:
                    # Create new pattern
                    pat = WikiPattern(
                        name=pattern_name,
                        title=diagnosis["title"],
                        problem=diagnosis["problem"],
                        root_cause=diagnosis["root_cause"],
                        actionable_fix=diagnosis["actionable_fix"],
                        anti_patterns=diagnosis.get("anti_patterns", []),
                        code_example=diagnosis.get("code_example"),
                        related_skills=diagnosis.get("related_skills", []),
                        tags=diagnosis.get("tags", []),
                    )
                    self.wiki.save_pattern(pat)
                    created_patterns.append(pattern_name)

        # Log consolidation
        log_summary = (
            f"Consolidated {len(sampled_traces)} traces ({len(failing)} failing, {len(passing)} passing). "
            f"Created {len(created_patterns)} patterns, updated {len(updated_patterns)}."
        )
        findings = "\n".join(f"- {d}" for d in diagnosed) if diagnosed else "All analyzed traces passed."
        self.wiki.append_log(log_summary, iteration=iteration, findings=findings)
        self.wiki.rebuild_index()

        return MaintenanceReport(
            iteration=iteration,
            patterns_created=created_patterns,
            patterns_updated=updated_patterns,
            log_summary=log_summary,
            diagnosed_failures=diagnosed,
        )

    def _diagnose_trace(self, trace: ExecutionTrace) -> Optional[Dict[str, Any]]:
        """Extracts failure diagnoses and patterns from command output."""
        output = trace.stdout + "\n" + trace.stderr

        # 1. NaN in Density Profile / Optimization divergence
        if "NaN" in output or "nan" in output.lower() or "FloatingPointError" in output:
            if "log" in output.lower() or "rho" in output.lower():
                return {
                    "name": "pattern_log_free_latent_density",
                    "title": "Log-Free Latent Density Field Parameterization",
                    "problem": "Optimization producing NaNs or negative densities when updating density field directly.",
                    "root_cause": (
                        "Direct optimization of density rho(z) can venture into negative space, causing ln(rho) singularities "
                        "and gradient NaN explosion."
                    ),
                    "actionable_fix": (
                        "Parameterize density as rho(z) = rho_bulk * exp(psi(z)) and optimize the latent potential psi(z). "
                        "This mathematically guarantees rho(z) > 0 at all spatial coordinates."
                    ),
                    "anti_patterns": [
                        "Optimizing rho(z) directly with gradient descent",
                        "Using np.clip(rho, 1e-12, None) to suppress negative values",
                    ],
                    "observed_error": "NaN encountered during cDFT functional minimization",
                    "related_skills": ["cdft-physics", "cdft-solver"],
                    "tags": ["physics", "cdft", "nan-trap"],
                }

        # 2. Virial Contact Pressure Slicing Bug
        if "contact_pressure" in output.lower() or "wall_pressure" in output.lower():
            return {
                "name": "pattern_irving_kirkwood_virial_pressure",
                "title": "Exact Irving-Kirkwood Virial Contact Pressure Integral",
                "problem": "Wall contact pressure differs from bulk thermodynamic pressure at equilibrium.",
                "root_cause": (
                    "Using ad-hoc spatial slices (e.g. rho[0:5] or rho[mid]) to estimate contact density breaks momentum conservation "
                    "under continuous external potentials."
                ),
                "actionable_fix": (
                    "Evaluate wall contact pressure via the exact Irving-Kirkwood momentum balance integral: "
                    "P_wall = - \\int_0^{L_z/2} rho(z) (dV_ext/dz) dz."
                ),
                "anti_patterns": [
                    "Using rho[0] * k_B * T for wall contact pressure with soft potentials",
                    "Hardcoding spatial slice indices [0:15]",
                ],
                "observed_error": "Contact pressure deviation from exact virial theorem",
                "related_skills": ["cdft-physics", "cdft-observables"],
                "tags": ["physics", "observables", "virial"],
            }

        # 3. Tinygrad JIT realization / compilation issues
        if "tinygrad" in output.lower() and ("jit" in output.lower() or "realize" in output.lower()):
            return {
                "name": "pattern_tinygrad_jit_graph_caching",
                "title": "Tinygrad JIT Graph Caching & Realization Safe Practices",
                "problem": "Tinygrad recompiles kernel graphs repeatedly or hangs on un-realized tensors.",
                "root_cause": (
                    "Dynamically sized inputs or intermediate tensors passed to @TinyJit functions invalidate graph cache."
                ),
                "actionable_fix": (
                    "Ensure tensor shapes and memory strides are fixed and contiguous before invoking JIT-compiled functions. "
                    "Call .realize() on outputs before converting to numpy."
                ),
                "anti_patterns": [
                    "Passing variable-length batches to a static TinyJit function",
                    "Accessing .numpy() on lazy tensors without realization",
                ],
                "observed_error": "Tinygrad JIT graph invalidation or performance drop",
                "related_skills": ["tinygrad-jit", "boltzmann-flow"],
                "tags": ["tinygrad", "jit", "performance"],
            }

        # 4. General Test Failure
        if trace.test_results.get("failed_tests"):
            failed = trace.test_results["failed_tests"]
            first_fail = failed[0]
            clean_stem = re.sub(r"[^a-zA-Z0-9_]", "_", first_fail).lower()
            return {
                "name": f"pattern_test_failure_{clean_stem[:40]}",
                "title": f"Test Regression in {first_fail}",
                "problem": f"Automated test `{first_fail}` failed during execution.",
                "root_cause": f"Regression detected in test suite: {trace.summary}",
                "actionable_fix": "Inspect the failure traceback in raw traces and ensure physical invariants are upheld.",
                "anti_patterns": ["Disabling the test or loosening assertions without physical justification"],
                "observed_error": trace.summary,
                "related_skills": ["cdft-physics"],
                "tags": ["test-failure", "regression"],
            }

        # 5. Non-zero exit command
        if trace.exit_code != 0:
            cmd_stem = re.sub(r"[^a-zA-Z0-9_]", "_", trace.command).lower()
            return {
                "name": f"pattern_command_error_{cmd_stem[:30]}",
                "title": f"Command Failure: {trace.command[:40]}",
                "problem": f"Command `{trace.command}` failed with exit code {trace.exit_code}.",
                "root_cause": f"Command execution error: {trace.summary}",
                "actionable_fix": "Check environment dependencies, file paths, and syntax.",
                "anti_patterns": ["Ignoring command exit codes in automation scripts"],
                "observed_error": trace.summary,
                "related_skills": ["tooling"],
                "tags": ["command-error"],
            }

        return None
