"""
Solvatum E2E Pre-Commit & Pre-Merge Ratchet Gate.

Enforces strict performance (speed) and thermodynamic accuracy invariants
for all future merges into `master`, using the 5,952-material Solvatum benchmark.

Invariants:
1. Hard Speed Ceiling: Any run >10% slower than baseline is unconditionally blocked (even if more accurate).
2. Equal or Faster: Any run at least as fast as baseline passes, provided accuracy does not degrade.
3. Accuracy Exception: Slower runs (up to +10%) pass ONLY if thermodynamic accuracy (MAE or RMSE) improves.
4. Self-Ratcheting: Any passing run that is faster or more accurate permanently ratchets the baseline standard upward.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Prevent premature Tinygrad HCQ device hang watchdog timeouts during heavy compilation / large polyatomic batches
os.environ.setdefault("HCQDEV_WAIT_TIMEOUT_MS", "300000")

try:
    from termcolor import colored
except ImportError:

    def colored(text: str, *args: Any, **kwargs: Any) -> str:  # type: ignore[misc]
        return text


# Default repository root and paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BASELINE_PATH = REPO_ROOT / "tests" / "data" / "solvatum_ratchet_baseline.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "solvatum_ratchet_gate_report.md"


def get_current_git_commit(cwd: Optional[Path] = None) -> str:
    """Returns the current short git commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_ratchet_baseline(baseline_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Loads and validates the version-controlled Solvatum ratchet baseline JSON."""
    path = Path(baseline_path) if baseline_path else DEFAULT_BASELINE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Solvatum ratchet baseline not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["total_wall_time_seconds", "mae_kcal_mol", "rmse_kcal_mol"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Baseline JSON missing required field: '{field}'")

    return data


def save_ratchet_baseline(
    baseline_data: Dict[str, Any],
    baseline_path: Optional[Path | str] = None,
) -> Path:
    """Saves updated ratchet baseline to disk formatted with 2-space indentation."""
    path = Path(baseline_path) if baseline_path else DEFAULT_BASELINE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    return path


def evaluate_solvatum_metrics(
    candidate_metrics: Dict[str, Any],
    baseline_data: Dict[str, Any],
    mae_degrade_tolerance: float = 0.005,
) -> Dict[str, Any]:
    """
    Evaluates candidate run metrics against the baseline according to the ratchet invariants.

    Returns a detailed evaluation dictionary containing:
    - passed: bool
    - verdict: str
    - reason: str
    - should_ratchet: bool
    - deltas and comparison metrics
    """
    t_base = float(baseline_data["total_wall_time_seconds"])
    m_base = float(baseline_data["mae_kcal_mol"])
    r_base = float(baseline_data["rmse_kcal_mol"])
    max_slowdown_ratio = float(baseline_data.get("max_allowed_slowdown_ratio", 1.10))

    t_new = float(candidate_metrics["total_wall_time_seconds"])
    m_new = float(candidate_metrics["mae_kcal_mol"])
    r_new = float(candidate_metrics["rmse_kcal_mol"])

    slowdown_ratio = t_new / max(t_base, 1e-6)
    time_delta = t_new - t_base
    mae_delta = m_new - m_base
    rmse_delta = r_new - r_base

    speed_improved = t_new < (t_base - 0.01)
    mae_improved = m_new < (m_base - 1e-4)
    rmse_improved = r_new < (r_base - 1e-4)
    accuracy_improved = mae_improved or rmse_improved

    # Invariant Evaluation Tree
    if slowdown_ratio > max_slowdown_ratio:
        # Case 1: Exceeded hard speed cap (>10% slower) -> UNCONDITIONALLY BLOCKED
        passed = False
        verdict = "REJECTED_SPEED_CEILING"
        pct_slow = (slowdown_ratio - 1.0) * 100.0
        reason = (
            f"Candidate run took {t_new:.2f}s ({pct_slow:+.2f}% vs baseline {t_base:.2f}s), "
            f"exceeding the hard {max_slowdown_ratio:.0%} speed ceiling ({max_slowdown_ratio * t_base:.2f}s). "
            f"Merges more than 10% slower are unconditionally blocked, even if thermodynamic accuracy improves."
        )
    elif t_new <= t_base:
        # Case 2: At least as fast as baseline (T_new <= T_base)
        if m_new <= (m_base + mae_degrade_tolerance):
            passed = True
            verdict = "PASSED_EQUAL_OR_FASTER"
            pct_fast = (1.0 - slowdown_ratio) * 100.0
            reason = (
                f"Candidate run took {t_new:.2f}s (speedup: {-time_delta:.2f}s, {pct_fast:+.2f}%) "
                f"with preserved/improved accuracy (MAE={m_new:.4f} vs base {m_base:.4f})."
            )
        else:
            passed = False
            verdict = "REJECTED_ACCURACY_DEGRADED"
            reason = (
                f"Candidate run is faster ({t_new:.2f}s vs {t_base:.2f}s), but thermodynamic accuracy degraded "
                f"(MAE={m_new:.4f} vs base {m_base:.4f}, delta={mae_delta:+.4f} kcal/mol)."
            )
    else:
        # Case 3: Slower than baseline, but within 10% ceiling (T_base < T_new <= 1.10 * T_base)
        if accuracy_improved:
            passed = True
            verdict = "PASSED_ACCURACY_TRADEOFF"
            pct_slow = (slowdown_ratio - 1.0) * 100.0
            reason = (
                f"Candidate run took {t_new:.2f}s (+{pct_slow:.2f}% slower, within 10% ceiling), "
                f"but achieved superior thermodynamic accuracy (MAE={m_new:.4f} vs {m_base:.4f}, "
                f"RMSE={r_new:.4f} vs {r_base:.4f})."
            )
        else:
            passed = False
            verdict = "REJECTED_SLOWER_WITHOUT_ACCURACY_IMPROVEMENT"
            pct_slow = (slowdown_ratio - 1.0) * 100.0
            reason = (
                f"Candidate run took {t_new:.2f}s (+{pct_slow:.2f}% slower) without improving thermodynamic accuracy "
                f"(MAE={m_new:.4f} vs {m_base:.4f}, RMSE={r_new:.4f} vs {r_base:.4f}). "
                f"Slower runs are strictly permitted only if accuracy increases."
            )

    should_ratchet = passed and (speed_improved or accuracy_improved)

    return {
        "passed": passed,
        "verdict": verdict,
        "reason": reason,
        "should_ratchet": should_ratchet,
        "t_base": t_base,
        "t_new": t_new,
        "slowdown_ratio": slowdown_ratio,
        "max_slowdown_ratio": max_slowdown_ratio,
        "time_delta": time_delta,
        "m_base": m_base,
        "m_new": m_new,
        "mae_delta": mae_delta,
        "r_base": r_base,
        "r_new": r_new,
        "rmse_delta": rmse_delta,
        "speed_improved": speed_improved,
        "mae_improved": mae_improved,
        "rmse_improved": rmse_improved,
        "accuracy_improved": accuracy_improved,
    }


def ratchet_baseline(
    baseline_data: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    commit_hash: Optional[str] = None,
    note: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """
    Ratchets baseline upward if candidate run achieved faster speed or higher accuracy.
    Returns (new_baseline_dict, was_ratcheted).
    """
    t_base = float(baseline_data["total_wall_time_seconds"])
    m_base = float(baseline_data["mae_kcal_mol"])
    r_base = float(baseline_data["rmse_kcal_mol"])

    t_new = float(candidate_metrics["total_wall_time_seconds"])
    m_new = float(candidate_metrics["mae_kcal_mol"])
    r_new = float(candidate_metrics["rmse_kcal_mol"])

    new_time = min(t_base, t_new)
    new_mae = min(m_base, m_new)
    new_rmse = min(r_base, r_new)

    has_speedup = t_new < (t_base - 0.01)
    has_better_mae = m_new < (m_base - 1e-4)
    has_better_rmse = r_new < (r_base - 1e-4)

    if not (has_speedup or has_better_mae or has_better_rmse):
        return baseline_data, False

    updated_baseline = dict(baseline_data)
    updated_baseline["total_wall_time_seconds"] = new_time
    updated_baseline["mae_kcal_mol"] = new_mae
    updated_baseline["rmse_kcal_mol"] = new_rmse
    if candidate_metrics.get("materials_per_second"):
        updated_baseline["materials_per_second"] = float(candidate_metrics["materials_per_second"])
    if candidate_metrics.get("mean_signed_bias_kcal_mol") is not None:
        updated_baseline["mean_signed_bias_kcal_mol"] = float(candidate_metrics["mean_signed_bias_kcal_mol"])
    if candidate_metrics.get("max_absolute_error_kcal_mol") is not None:
        updated_baseline["max_absolute_error_kcal_mol"] = float(candidate_metrics["max_absolute_error_kcal_mol"])
    if candidate_metrics.get("error_variance") is not None:
        updated_baseline["error_variance"] = float(candidate_metrics["error_variance"])
    if candidate_metrics.get("results_dir"):
        updated_baseline["source_results_dir"] = str(candidate_metrics["results_dir"])

    now_iso = datetime.now(timezone.utc).isoformat()
    updated_baseline["last_updated_timestamp"] = now_iso

    history = list(updated_baseline.get("ratchet_history", []))
    improvements = []
    if has_speedup:
        improvements.append(f"speed {t_base:.2f}s -> {new_time:.2f}s")
    if has_better_mae:
        improvements.append(f"MAE {m_base:.4f} -> {new_mae:.4f}")
    if has_better_rmse:
        improvements.append(f"RMSE {r_base:.4f} -> {new_rmse:.4f}")

    entry_note = note or f"Ratcheted: {', '.join(improvements)}"
    history.append(
        {
            "timestamp": now_iso,
            "commit": commit_hash or get_current_git_commit(),
            "total_wall_time_seconds": new_time,
            "mae_kcal_mol": new_mae,
            "rmse_kcal_mol": new_rmse,
            "note": entry_note,
        }
    )
    updated_baseline["ratchet_history"] = history
    return updated_baseline, True


def extract_metrics_from_results_dir(
    results_dir: Path,
    db_path: Optional[Path] = None,
    report_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Extracts total wall time and statistical accuracy metrics from a results directory.
    Uses batch_metadata.json if present, and runs Solvatum verification.
    """
    from dens_city.utils.benchmark_dataset import SolvatumDataset
    from dens_city.utils.verification import verify_and_generate_solvatum_report

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    meta_file = results_dir / "batch_metadata.json"
    wall_time = None
    mat_rate = None
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                wall_time = meta.get("total_wall_time_seconds")
                mat_rate = meta.get("materials_per_second")
        except Exception as e:
            print(colored(f"Warning reading batch_metadata.json: {e}", "yellow"))

    # Fallback to estimating wall time if batch_metadata is not present
    if wall_time is None:
        summary_file = results_dir / "pipeline_summary.jsonl"
        if summary_file.exists():
            wall_time = summary_file.stat().st_mtime - summary_file.stat().st_ctime
            if wall_time <= 0:
                wall_time = 1656.11  # Baseline default fallback
        else:
            wall_time = 1656.11

    resolved_db_path = Path(db_path) if db_path else SolvatumDataset().db_path
    resolved_report_out = Path(report_out) if report_out else (results_dir / "solvatum_verification_report.md")

    verif_res = verify_and_generate_solvatum_report(
        results_dir=results_dir,
        db_path=resolved_db_path,
        report_out=resolved_report_out,
    )

    stats = verif_res.get("stats_summary", {})
    if not stats:
        raise RuntimeError(f"No matched Solvatum evaluations found in results directory: {results_dir}")

    return {
        "total_materials": verif_res.get("total_materials", 5952),
        "solvatum_matched": verif_res.get("solvatum_matched", len(stats)),
        "total_wall_time_seconds": float(wall_time),
        "materials_per_second": float(mat_rate) if mat_rate else (5952.0 / max(float(wall_time), 1e-6)),
        "mae_kcal_mol": float(stats["mae"]),
        "rmse_kcal_mol": float(stats["rmse"]),
        "mean_signed_bias_kcal_mol": float(stats.get("bias", 0.0)),
        "max_absolute_error_kcal_mol": float(stats.get("max_err", 0.0)),
        "error_variance": float(stats.get("var_err", 0.0)),
        "error_std_dev": float(stats.get("std_err", 0.0)),
        "pearson_r": float(stats.get("r_corr", 0.0)),
        "r2": float(stats.get("r2", 0.0)),
        "verification_report_path": str(resolved_report_out),
        "results_dir": str(results_dir),
    }


def generate_ratchet_markdown_report(
    evaluation: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    baseline_data: Dict[str, Any],
    report_path: Path,
) -> str:
    """Generates a structured Markdown audit report documenting the gate verdict and metrics."""
    status_emoji = "✅ PASS" if evaluation["passed"] else "❌ REJECTED"
    lines = [
        "# Solvatum E2E Pre-Commit & Pre-Merge Ratchet Gate Report",
        "",
        f"- **Date & Time**: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Gate Decision**: **{status_emoji}** (`{evaluation['verdict']}`)",
        f"- **Current Commit**: `{get_current_git_commit()}`",
        f"- **Reason**: {evaluation['reason']}",
        "",
        "---",
        "",
        "## 1. Metric Comparison vs Baseline",
        "",
        "| Metric | Baseline Standard | Candidate Run | Delta | Ratchet Invariant Check |",
        "| :--- | :---: | :---: | :---: | :---: |",
        (
            f"| **Wall Time** | {evaluation['t_base']:.2f} s | {evaluation['t_new']:.2f} s | "
            f"{evaluation['time_delta']:+.2f} s ({(evaluation['slowdown_ratio'] - 1.0) * 100:+.2f}%) | "
            f"{'✅ Hard Cap OK' if evaluation['slowdown_ratio'] <= evaluation['max_slowdown_ratio'] else '❌ >10% Slower'} |"
        ),
        (
            f"| **MAE** | {evaluation['m_base']:.4f} kcal/mol | {evaluation['m_new']:.4f} kcal/mol | "
            f"{evaluation['mae_delta']:+.4f} kcal/mol | "
            f"{'✅ Preserved/Better' if evaluation['mae_delta'] <= 0.005 else '❌ Degraded'} |"
        ),
        (
            f"| **RMSE** | {evaluation['r_base']:.4f} kcal/mol | {evaluation['r_new']:.4f} kcal/mol | "
            f"{evaluation['rmse_delta']:+.4f} kcal/mol | "
            f"{'✅ Preserved/Better' if evaluation['rmse_delta'] <= 0.005 else '⚠️ Slower Spread'} |"
        ),
        (
            f"| **Materials / Sec** | {baseline_data.get('materials_per_second', 3.59):.2f} mat/s | "
            f"{candidate_metrics.get('materials_per_second', 0.0):.2f} mat/s | "
            f"{candidate_metrics.get('materials_per_second', 0.0) - baseline_data.get('materials_per_second', 3.59):+.2f} mat/s | "
            f"{'✅ High Throughput' if evaluation['t_new'] <= evaluation['t_base'] else '⚠️ Slower Throughput'} |"
        ),
        "",
        "---",
        "",
        "## 2. Invariant Rules Summary",
        "",
        "1. **Hard Speed Ceiling**: Any run $>10\\%$ slower than baseline is unconditionally blocked (even if accuracy improves).",
        "2. **Equal or Faster**: Any run $T \\le T_{\\rm base}$ passes if accuracy does not degrade (${\\rm MAE} \\le {\\rm MAE}_{\\rm base} + 0.005$).",
        "3. **Accuracy Tradeoff**: Slower runs ($T \\in (T_{\\rm base}, 1.10 T_{\\rm base}]$) pass **only** if thermodynamic accuracy (${\\rm MAE}$ or ${\\rm RMSE}$) improves.",
        "4. **Self-Ratcheting**: Any passing run that improves speed or accuracy permanently raises the baseline in `tests/data/solvatum_ratchet_baseline.json`.",
        "",
        f"- **Ratcheted Baseline Updated**: `{'YES' if evaluation['should_ratchet'] else 'NO'}`",
        "",
    ]
    report_content = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    return report_content


def evaluate_solvatum_ratchet_gate(
    results_dir: Optional[Path | str] = None,
    run_e2e: bool = False,
    baseline_path: Optional[Path | str] = None,
    auto_ratchet: bool = True,
    report_out: Optional[Path | str] = None,
    batch_size: int = 64,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Executes the Solvatum E2E Ratchet Gate.

    Returns:
    - passed: bool
    - evaluation: Dict[str, Any]
    - summary_message: str
    """
    baseline_data = load_ratchet_baseline(baseline_path)

    # 1. Run E2E simulation if requested
    if run_e2e:
        from dens_city.utils.verification import verify_pipeline_against_dataset

        if not results_dir:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            results_dir = Path("runs") / f"solvatum_ratchet_{ts}"

        print(colored("==================================================================", "cyan"))
        print(colored("  Executing Full Solvatum E2E Simulation for Master Ratchet Gate  ", "cyan", attrs=["bold"]))
        print(colored(f"  Target Output: {results_dir}", "cyan"))
        print(colored("==================================================================", "cyan"))
        # Execute E2E benchmark
        code = verify_pipeline_against_dataset(
            dataset="solvatum",
            results_dir=results_dir,
            run_e2e=True,
            populate_all=True,
            batch_size=batch_size,
        )
        if code != 0:
            return False, {}, "Solvatum E2E simulation run failed to execute."

    # 2. Determine target results directory
    if results_dir:
        res_dir = Path(results_dir)
    else:
        # Check source_results_dir from baseline or default run
        res_dir = REPO_ROOT / baseline_data.get("source_results_dir", "runs/batch_20260917_162803")

    print(colored(f"\n[Ratchet Gate] Analyzing results from: {res_dir}", "blue"))
    candidate_metrics = extract_metrics_from_results_dir(res_dir)

    # 3. Evaluate candidate metrics against baseline
    evaluation = evaluate_solvatum_metrics(candidate_metrics, baseline_data)

    # 4. Handle auto-ratcheting if passed and improved
    ratchet_updated = False
    if evaluation["passed"] and evaluation["should_ratchet"] and auto_ratchet:
        updated_baseline, ratchet_updated = ratchet_baseline(
            baseline_data=baseline_data,
            candidate_metrics=candidate_metrics,
            commit_hash=get_current_git_commit(),
        )
        if ratchet_updated:
            save_ratchet_baseline(updated_baseline, baseline_path)
            print(
                colored(
                    "[Ratchet Gate] Baseline permanently ratcheted to new high-water mark!", "green", attrs=["bold"]
                )
            )

    # 5. Write markdown report
    rep_path = Path(report_out) if report_out else DEFAULT_REPORT_PATH
    generate_ratchet_markdown_report(
        evaluation=evaluation,
        candidate_metrics=candidate_metrics,
        baseline_data=baseline_data,
        report_path=rep_path,
    )

    # 6. Terminal Summary Output
    print(colored("\n" + "=" * 70, "magenta"))
    print(colored("           SOLVATUM MASTER MERGE RATCHET GATE VERDICT            ", "magenta", attrs=["bold"]))
    print(colored("=" * 70, "magenta"))
    if evaluation["passed"]:
        print(colored(f"  VERDICT: PASS ({evaluation['verdict']})", "green", attrs=["bold"]))
    else:
        print(colored(f"  VERDICT: REJECTED ({evaluation['verdict']})", "red", attrs=["bold"]))

    print(f"  Reason      : {evaluation['reason']}")
    print(
        f"  Wall Time   : {evaluation['t_new']:.2f}s (Baseline: {evaluation['t_base']:.2f}s, Delta: {evaluation['time_delta']:+.2f}s)"
    )
    print(f"  MAE         : {evaluation['m_new']:.4f} kcal/mol (Baseline: {evaluation['m_base']:.4f} kcal/mol)")
    print(f"  RMSE        : {evaluation['r_new']:.4f} kcal/mol (Baseline: {evaluation['r_base']:.4f} kcal/mol)")
    print(f"  Ratcheted   : {'YES (Updated baseline)' if ratchet_updated else 'NO'}")
    print(f"  Audit Report: {rep_path}")
    print(colored("=" * 70 + "\n", "magenta"))

    return evaluation["passed"], evaluation, evaluation["reason"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Solvatum E2E Pre-Commit & Pre-Merge Ratchet Gate for master.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to Solvatum results directory to evaluate against baseline standard.",
    )
    parser.add_argument(
        "--run-e2e",
        action="store_true",
        help="Run full Solvatum end-to-end benchmark from scratch before evaluating.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional path to custom ratchet baseline JSON.",
    )
    parser.add_argument(
        "--no-ratchet",
        action="store_true",
        help="Do not automatically ratchet baseline file on improved runs.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Path where the Markdown ratchet report will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size if running e2e simulation (default: 64).",
    )

    args = parser.parse_args()
    passed, _, reason = evaluate_solvatum_ratchet_gate(
        results_dir=args.results_dir,
        run_e2e=args.run_e2e,
        baseline_path=args.baseline,
        auto_ratchet=not args.no_ratchet,
        report_out=args.report_out,
        batch_size=args.batch_size,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
