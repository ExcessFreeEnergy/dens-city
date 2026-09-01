"""
Unit and Integration Tests for WikiSkill Persistent Knowledge Co-Evolution System.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from dens_city.ui.cli import main
from dens_city.wikiskill import (
    ExecutionTrace,
    GatingHarness,
    PatchOperation,
    Proposal,
    ProposalAction,
    RawTraceRecorder,
    SkillProposer,
    WikiMaintainer,
    WikiManager,
    WikiPattern,
)


@pytest.fixture
def temp_workspace():
    """Creates an isolated temporary workspace directory for testing WikiSkill."""
    tmp_dir = tempfile.mkdtemp(prefix="wikiskill_test_")
    ws = Path(tmp_dir)
    yield ws
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_raw_trace_recorder(temp_workspace: Path):
    recorder = RawTraceRecorder(workspace_root=temp_workspace)
    assert recorder.traces_dir.exists()

    # Record passing command
    trace_pass = recorder.record_command("echo 'Physics test pass'", cwd=temp_workspace, tags=["test-pass"])
    assert trace_pass.passed is True
    assert trace_pass.exit_code == 0
    assert "Physics test pass" in trace_pass.stdout
    assert "echo 'Physics test pass'" in trace_pass.command

    # Record failing command
    trace_fail = recorder.record_command("sh -c 'exit 42'", cwd=temp_workspace, tags=["test-fail"])
    assert trace_fail.passed is False
    assert trace_fail.exit_code == 42

    # List traces
    traces = recorder.list_traces()
    assert len(traces) == 2
    assert traces[0].trace_id == trace_fail.trace_id or traces[1].trace_id == trace_fail.trace_id

    # Stratified sampling
    sampled = recorder.sample_traces(max_failing=1, max_passing=1)
    assert len(sampled) == 2
    assert any(not t.passed for t in sampled)
    assert any(t.passed for t in sampled)


def test_wiki_manager_crud_and_patching(temp_workspace: Path):
    wiki = WikiManager(workspace_root=temp_workspace)
    assert wiki.index_file.exists()
    assert wiki.log_file.exists()
    assert wiki.impact_file.exists()

    # Create pattern
    pat = WikiPattern(
        name="test_pattern_virial",
        title="Test Virial Invariant",
        problem="Contact pressure mismatch due to slicing",
        root_cause="Spatial slicing breaks momentum conservation",
        actionable_fix="Integrate -rho * dV/dz across wall half-box",
        anti_patterns=["Using rho[0] directly"],
        related_skills=["cdft-physics"],
    )
    wiki.save_pattern(pat)
    assert (wiki.patterns_dir / "test_pattern_virial.md").exists()

    # Verify index contains pattern
    index_content = wiki.get_index()
    assert "test_pattern_virial" in index_content
    assert "Contact pressure mismatch" in index_content

    # Apply patch: append
    wiki.patch_pattern(
        "test_pattern_virial",
        [PatchOperation(op="append", content="### Additional Note\nVerified on 1D pore grid.")],
    )
    updated_content = wiki.get_pattern("test_pattern_virial")
    assert updated_content is not None
    assert "Additional Note" in updated_content

    # Apply patch: replace
    wiki.patch_pattern(
        "test_pattern_virial",
        [PatchOperation(op="replace", target="1D pore grid.", content="1D pore grid with dz=0.05.")],
    )
    updated_content2 = wiki.get_pattern("test_pattern_virial")
    assert updated_content2 is not None
    assert "dz=0.05" in updated_content2


def test_wiki_manager_skill_impact_tracker(temp_workspace: Path):
    wiki = WikiManager(workspace_root=temp_workspace)

    # Record accepted proposal
    wiki.record_skill_impact(
        proposal_id="prop_001",
        target_skill_or_file="cdft-solver",
        diff="+ def solve_psi(): pass",
        score_before=0.8,
        score_after=1.0,
        outcome="Accepted",
        rationale="Tests passed successfully",
    )

    # Record rejected proposal
    wiki.record_skill_impact(
        proposal_id="prop_002",
        target_skill_or_file="cdft-wall-clamp",
        diff="- v_ext = np.clip(v_ext, -500, 1000)\n+ v_ext = np.clip(v_ext, -100, 100)",
        score_before=1.0,
        score_after=0.2,
        outcome="Rejected",
        rationale="Caused severe fluid penetration into wall",
    )

    history = wiki.get_skill_impact_history()
    assert "prop_001" in history
    assert "Accepted" in history
    assert "prop_002" in history
    assert "Rejected" in history

    # Check anti-pattern / rejected re-proposal detection
    warning = wiki.is_proposal_previously_rejected(
        target_name="cdft-wall-clamp",
        candidate_text="v_ext = np.clip(v_ext, -100, 100) and soft boundary clamping",
    )
    assert warning is not None
    assert "prop_002" in warning


def test_wiki_maintainer_consolidation(temp_workspace: Path):
    wiki = WikiManager(workspace_root=temp_workspace)
    recorder = RawTraceRecorder(workspace_root=temp_workspace)
    maintainer = WikiMaintainer(wiki_manager=wiki, trace_recorder=recorder)

    # Simulate trace with NaN failure
    trace = ExecutionTrace(
        trace_id="trace_nan_001",
        timestamp="2026-08-31T00:00:00Z",
        command="dens-city --materials argon",
        exit_code=1,
        duration_sec=1.5,
        stdout="Optimizing rho directly...\nFloatingPointError: NaN encountered in ln(rho)",
        stderr="",
        passed=False,
        summary="Failed with NaN in ln(rho)",
    )
    recorder.save_trace(trace)

    report = maintainer.consolidate_traces(iteration=1)
    assert len(report.patterns_created) > 0
    assert "pattern_log_free_latent_density" in report.patterns_created
    assert (wiki.patterns_dir / "pattern_log_free_latent_density.md").exists()


def test_skill_proposer_and_gating_accepted(temp_workspace: Path):
    wiki = WikiManager(workspace_root=temp_workspace)
    recorder = RawTraceRecorder(workspace_root=temp_workspace)
    proposer = SkillProposer(wiki_manager=wiki, workspace_root=temp_workspace)
    gating = GatingHarness(
        wiki_manager=wiki,
        trace_recorder=recorder,
        proposer=proposer,
        workspace_root=temp_workspace,
    )

    # Seed pattern
    pat = WikiPattern(
        name="pattern_fmt_convolution",
        title="FMT Planar Convolution",
        problem="Aliasing in weighted densities",
        root_cause="Discrete point sampling of spherical weight functions",
        actionable_fix="Use analytical cell-integrated kernels",
    )
    wiki.save_pattern(pat)

    # Propose new skill
    proposal = proposer.propose_from_pattern("pattern_fmt_convolution")
    assert proposal is not None
    assert proposal.action == ProposalAction.CREATE
    assert proposal.target_name == "fmt-convolution"

    # Evaluate proposal with a passing validation command
    res = gating.evaluate_proposal(
        proposal=proposal,
        validation_command="echo 'All 5 tests passed' && exit 0",
        score_before=0.8,
    )

    assert res.accepted is True
    assert proposal.target_file.exists()
    assert (proposal.target_file.parent / "PURPOSE.md").exists()

    # Check skill-impact.md recorded the acceptance
    history = wiki.get_skill_impact_history()
    assert proposal.proposal_id in history
    assert "Accepted" in history


def test_skill_proposer_and_gating_rollback(temp_workspace: Path):
    wiki = WikiManager(workspace_root=temp_workspace)
    recorder = RawTraceRecorder(workspace_root=temp_workspace)
    proposer = SkillProposer(wiki_manager=wiki, workspace_root=temp_workspace)
    gating = GatingHarness(
        wiki_manager=wiki,
        trace_recorder=recorder,
        proposer=proposer,
        workspace_root=temp_workspace,
    )

    # Create pre-existing file
    test_skill_dir = temp_workspace / ".agents" / "skills" / "bad-skill"
    test_skill_dir.mkdir(parents=True, exist_ok=True)
    target_skill_file = test_skill_dir / "SKILL.md"
    target_skill_file.write_text("Original Valid Content\n", encoding="utf-8")

    # Propose a broken change
    broken_proposal = Proposal(
        proposal_id="prop_broken_001",
        action=ProposalAction.PATCH,
        target_name="bad-skill",
        target_file=target_skill_file,
        edits=[PatchOperation(op="append", content="BROKEN SYNTAX CODE")],
        diff="+ BROKEN SYNTAX CODE",
        rationale="Experimental unverified patch",
    )

    # Evaluate proposal with a failing validation command
    res = gating.evaluate_proposal(
        proposal=broken_proposal,
        validation_command="sh -c 'echo \"Test failure: SyntaxError\" && exit 1'",
        score_before=1.0,
    )

    # Gating should REJECT and ROLL BACK
    assert res.accepted is False
    assert target_skill_file.read_text(encoding="utf-8") == "Original Valid Content\n"

    # Crucially, wiki/skill-impact.md must retain the rejection record
    history = wiki.get_skill_impact_history()
    assert "prop_broken_001" in history
    assert "Rejected" in history


def test_cli_wikiskill_flags():
    """Verifies that the CLI flags for WikiSkill execute without error."""
    assert main(["--wikiskill-init"]) == 0
    assert main(["--wikiskill-status"]) == 0
    assert main(["--wikiskill-audit", "pattern_log_free_latent_density"]) == 0
    assert main(["--wikiskill-consolidate"]) == 0
