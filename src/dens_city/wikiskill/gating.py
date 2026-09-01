"""
Gating and Rollback Harness for WikiSkill.

Enforces strict empirical validation of proposed skills, rules, or code modifications:
- Tests candidate proposals against test suites (e.g., pytest, physical invariants).
- If tests pass: accepts the modification as the new active configuration.
- If tests fail: rolls back the modification to the previous clean state.
- Crucially, the Wiki (patterns, index, logs) and skill-impact.md are NEVER rolled back,
  recording the rejection and reason to prevent repeating failed interventions.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dens_city.wikiskill.proposer import Proposal, SkillProposer
from dens_city.wikiskill.trace_recorder import ExecutionTrace, RawTraceRecorder
from dens_city.wikiskill.wiki_manager import WikiManager


@dataclass
class GatingResult:
    proposal_id: str
    target: str
    accepted: bool
    score_before: float
    score_after: float
    validation_trace: Optional[ExecutionTrace]
    rationale: str


class GatingHarness:
    """Evaluates candidate proposals, executing validation tests and handling rollback."""

    def __init__(
        self,
        wiki_manager: WikiManager,
        trace_recorder: RawTraceRecorder,
        proposer: SkillProposer,
        workspace_root: Optional[Path | str] = None,
    ) -> None:
        self.wiki = wiki_manager
        self.traces = trace_recorder
        self.proposer = proposer
        self.workspace_root = Path(workspace_root) if workspace_root else wiki_manager.workspace_root

    def evaluate_proposal(
        self,
        proposal: Proposal,
        validation_command: str = "pytest tests/ -k 'test_tiny_cdft or test_batched_cdft' -q",
        score_before: float = 1.0,
    ) -> GatingResult:
        """Applies proposal, runs validation tests, and accepts or rolls back."""
        target_file = proposal.target_file
        backup_content: Optional[str] = None
        backup_purpose: Optional[str] = None
        purpose_file = target_file.parent / "PURPOSE.md"

        # 1. Create backup
        if target_file.exists():
            backup_content = target_file.read_text(encoding="utf-8")
        if purpose_file.exists():
            backup_purpose = purpose_file.read_text(encoding="utf-8")

        # 2. Apply proposal
        self.proposer.apply_proposal(proposal)

        # 3. Run validation command
        trace = self.traces.record_command(
            command=validation_command,
            tags=["wikiskill-gating", proposal.proposal_id],
            metadata={"target": proposal.target_name, "proposal_id": proposal.proposal_id},
        )

        passed = trace.passed
        failed_count = trace.test_results.get("failed", 0) + trace.test_results.get("errors", 0)
        passed_count = trace.test_results.get("passed", 0)
        total_tests = passed_count + failed_count

        if total_tests > 0:
            score_after = passed_count / total_tests
        else:
            score_after = 1.0 if passed else 0.0

        # Acceptance criterion: tests must pass and score must not regress
        accepted = passed and (score_after >= score_before)

        if accepted:
            outcome = "Accepted"
            rationale = (
                f"Validation passed ({passed_count}/{total_tests} tests). "
                f"Score {score_before:.3f} -> {score_after:.3f}."
            )
        else:
            outcome = "Rejected"
            rationale = (
                f"Validation failed (exit code {trace.exit_code}, {failed_count} failures). "
                f"Reverted candidate change to preserve system stability."
            )
            # 4. Rollback target files
            if backup_content is not None:
                target_file.write_text(backup_content, encoding="utf-8")
            elif target_file.exists():
                target_file.unlink()

            if backup_purpose is not None:
                purpose_file.write_text(backup_purpose, encoding="utf-8")
            elif purpose_file.exists():
                purpose_file.unlink()

            # If directory is now empty, remove it
            if target_file.parent.exists() and not any(target_file.parent.iterdir()):
                shutil.rmtree(target_file.parent)

        # 5. ALWAYS record outcome in persistent skill-impact.md
        self.wiki.record_skill_impact(
            proposal_id=proposal.proposal_id,
            target_skill_or_file=proposal.target_name,
            diff=proposal.diff,
            score_before=score_before,
            score_after=score_after,
            outcome=outcome,
            rationale=rationale,
        )

        return GatingResult(
            proposal_id=proposal.proposal_id,
            target=proposal.target_name,
            accepted=accepted,
            score_before=score_before,
            score_after=score_after,
            validation_trace=trace,
            rationale=rationale,
        )
