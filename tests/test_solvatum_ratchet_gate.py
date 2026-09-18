"""
Unit tests for the Solvatum E2E Pre-Commit & Pre-Merge Ratchet Gate.

Tests all invariant decision boundaries:
1. Equal or faster speed with preserved accuracy -> PASS (and ratchets).
2. Faster speed but degraded accuracy -> FAIL (blocks merge).
3. Hard 10% speed ceiling (>10% slower) -> FAIL (unconditionally blocked even if more accurate).
4. Slower within 10% with improved accuracy -> PASS (tradeoff allowed, ratchets accuracy).
5. Slower within 10% without improved accuracy -> FAIL (blocks merge).
6. Baseline persistence and ratchet history immutability.
7. Metric extraction from real run directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dens_city.utils.ratchet_gate import (
    DEFAULT_BASELINE_PATH,
    evaluate_solvatum_metrics,
    extract_metrics_from_results_dir,
    load_ratchet_baseline,
    ratchet_baseline,
    save_ratchet_baseline,
)


@pytest.fixture
def mock_baseline():
    return {
        "benchmark": "solvatum",
        "total_materials": 5952,
        "total_wall_time_seconds": 1656.11,
        "materials_per_second": 3.5939,
        "mae_kcal_mol": 0.7814,
        "rmse_kcal_mol": 1.1284,
        "max_allowed_slowdown_ratio": 1.10,
        "ratchet_history": [
            {
                "timestamp": "2026-09-17T16:55:52Z",
                "commit": "e242012",
                "total_wall_time_seconds": 1656.11,
                "mae_kcal_mol": 0.7814,
                "rmse_kcal_mol": 1.1284,
                "note": "Initial baseline",
            }
        ],
    }


def test_load_default_baseline():
    """Verify that the repository default baseline file exists and is well-formed."""
    baseline = load_ratchet_baseline(DEFAULT_BASELINE_PATH)
    assert baseline["total_materials"] == 5952
    assert baseline["total_wall_time_seconds"] == pytest.approx(1656.11, rel=1e-3)
    assert baseline["mae_kcal_mol"] == pytest.approx(0.7814, rel=1e-3)
    assert baseline["rmse_kcal_mol"] == pytest.approx(1.1284, rel=1e-3)
    assert baseline["max_allowed_slowdown_ratio"] == 1.10
    assert len(baseline["ratchet_history"]) >= 1


def test_ratchet_gate_passes_when_faster_and_accurate(mock_baseline):
    """Candidate run is faster (1500s < 1656s) with equal or better MAE."""
    candidate = {
        "total_wall_time_seconds": 1500.0,
        "mae_kcal_mol": 0.7750,
        "rmse_kcal_mol": 1.1200,
    }
    result = evaluate_solvatum_metrics(candidate, mock_baseline)
    assert result["passed"] is True
    assert result["verdict"] == "PASSED_EQUAL_OR_FASTER"
    assert result["should_ratchet"] is True
    assert result["speed_improved"] is True
    assert result["mae_improved"] is True


def test_ratchet_gate_fails_when_faster_but_accuracy_degrades(mock_baseline):
    """Candidate run is faster (1200s), but MAE degrades beyond tolerance."""
    candidate = {
        "total_wall_time_seconds": 1200.0,
        "mae_kcal_mol": 0.8200,  # Worse by > 0.005
        "rmse_kcal_mol": 1.2000,
    }
    result = evaluate_solvatum_metrics(candidate, mock_baseline)
    assert result["passed"] is False
    assert result["verdict"] == "REJECTED_ACCURACY_DEGRADED"
    assert result["should_ratchet"] is False


def test_ratchet_gate_fails_when_over_10_percent_slower(mock_baseline):
    """Candidate run is 15% slower (> 1.10 * 1656.11s = 1821.72s). Must be unconditionally blocked!"""
    candidate = {
        "total_wall_time_seconds": 1900.0,
        "mae_kcal_mol": 0.5000,  # Drastically better accuracy
        "rmse_kcal_mol": 0.7000,
    }
    result = evaluate_solvatum_metrics(candidate, mock_baseline)
    assert result["passed"] is False
    assert result["verdict"] == "REJECTED_SPEED_CEILING"
    assert (
        "hard 110% speed ceiling" in result["reason"].lower()
        or "hard 1.1" in result["reason"].lower()
        or "exceeding" in result["reason"].lower()
    )


def test_ratchet_gate_passes_when_slower_within_10_percent_if_more_accurate(mock_baseline):
    """Candidate run is 5% slower, but achieves superior accuracy (tradeoff allowed)."""
    candidate = {
        "total_wall_time_seconds": 1738.9,  # ~1.05 * 1656.11
        "mae_kcal_mol": 0.7500,  # Better than 0.7814
        "rmse_kcal_mol": 1.1000,
    }
    result = evaluate_solvatum_metrics(candidate, mock_baseline)
    assert result["passed"] is True
    assert result["verdict"] == "PASSED_ACCURACY_TRADEOFF"
    assert result["should_ratchet"] is True
    assert result["speed_improved"] is False
    assert result["mae_improved"] is True


def test_ratchet_gate_fails_when_slower_without_accuracy_improvement(mock_baseline):
    """Candidate run is 5% slower without improving accuracy -> REJECTED."""
    candidate = {
        "total_wall_time_seconds": 1738.9,
        "mae_kcal_mol": 0.7814,  # Equal, not improved
        "rmse_kcal_mol": 1.1284,
    }
    result = evaluate_solvatum_metrics(candidate, mock_baseline)
    assert result["passed"] is False
    assert result["verdict"] == "REJECTED_SLOWER_WITHOUT_ACCURACY_IMPROVEMENT"
    assert result["should_ratchet"] is False


def test_ratchet_baseline_persistence_and_ratcheting(mock_baseline):
    """Verify that ratchet_baseline updates numbers correctly and appends history."""
    candidate = {
        "total_wall_time_seconds": 1600.0,
        "mae_kcal_mol": 0.7600,
        "rmse_kcal_mol": 1.1000,
        "materials_per_second": 3.72,
    }
    updated, was_ratcheted = ratchet_baseline(
        baseline_data=mock_baseline,
        candidate_metrics=candidate,
        commit_hash="abc1234",
        note="Faster and higher accuracy test run",
    )
    assert was_ratcheted is True
    assert updated["total_wall_time_seconds"] == 1600.0
    assert updated["mae_kcal_mol"] == 0.7600
    assert updated["rmse_kcal_mol"] == 1.1000
    assert updated["materials_per_second"] == 3.72
    assert len(updated["ratchet_history"]) == 2
    assert updated["ratchet_history"][-1]["commit"] == "abc1234"
    assert updated["ratchet_history"][-1]["total_wall_time_seconds"] == 1600.0

    # Test saving and reloading from a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = Path(tf.name)

    try:
        save_ratchet_baseline(updated, temp_path)
        reloaded = load_ratchet_baseline(temp_path)
        assert reloaded["total_wall_time_seconds"] == 1600.0
        assert reloaded["mae_kcal_mol"] == 0.7600
        assert len(reloaded["ratchet_history"]) == 2
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_extract_metrics_from_baseline_results_dir():
    """Verify that extract_metrics_from_results_dir reads batch_metadata.json and Solvatum data."""
    results_dir = Path("runs/batch_20260917_162803")
    if not results_dir.exists():
        pytest.skip("Baseline results directory runs/batch_20260917_162803 not present")

    metrics = extract_metrics_from_results_dir(results_dir)
    assert metrics["total_wall_time_seconds"] == pytest.approx(1656.11, rel=1e-3)
    assert metrics["mae_kcal_mol"] == pytest.approx(0.7814, rel=1e-2)
    assert metrics["rmse_kcal_mol"] == pytest.approx(1.1284, rel=1e-2)
    assert metrics["total_materials"] == 5952
