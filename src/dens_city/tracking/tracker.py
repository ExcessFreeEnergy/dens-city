"""
Experiment & Benchmark Progress Tracker for dens-city.
Tracks model iterations, physical prediction accuracy, and deviation from real-world experimental reality.
"""

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RunMetrics:
    run_id: str
    timestamp: str
    git_commit: str
    species: str
    total_timesteps: int
    training_time_s: float
    throughput_sps: float
    # Physical predictions
    T_c_pred: float
    T_c_error_pct: float
    rho_l_pred: float
    rho_l_error_pct: float
    rho_v_pred: float
    hydration_layer_minima: List[float]
    rmse_rho_z: float
    rmse_pressure: float
    notes: str = ""


class ExperimentTracker:
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.runs_dir / "history.jsonl"

    def _get_git_commit(self) -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "unknown"

    def log_run(
        self,
        species: str,
        total_timesteps: int,
        training_time_s: float,
        throughput_sps: float,
        T_c_pred: float,
        rho_l_pred: float,
        rho_v_pred: float,
        hydration_layer_minima: List[float],
        rmse_rho_z: float,
        rmse_pressure: float,
        notes: str = "",
    ) -> RunMetrics:
        # Experimental ground truths (Water)
        T_c_expt = 647.1  # K
        rho_l_expt = 33.36  # nm^-3 (at 300K)

        t_c_err = ((T_c_pred - T_c_expt) / T_c_expt) * 100.0
        rho_l_err = ((rho_l_pred - rho_l_expt) / rho_l_expt) * 100.0

        run_id = f"{species}_{time.strftime('%Y%m%d_%H%M%S')}"
        record = RunMetrics(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            git_commit=self._get_git_commit(),
            species=species,
            total_timesteps=total_timesteps,
            training_time_s=training_time_s,
            throughput_sps=throughput_sps,
            T_c_pred=T_c_pred,
            T_c_error_pct=t_c_err,
            rho_l_pred=rho_l_pred,
            rho_l_error_pct=rho_l_err,
            rho_v_pred=rho_v_pred,
            hydration_layer_minima=hydration_layer_minima,
            rmse_rho_z=rmse_rho_z,
            rmse_pressure=rmse_pressure,
            notes=notes,
        )

        # Save single run file
        run_file = self.runs_dir / f"{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(asdict(record), f, indent=2)

        # Append to jsonl history
        with open(self.history_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        print(f"[Tracker] Successfully logged run '{run_id}' to {run_file}")
        return record

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        runs = []
        with open(self.history_file, "r") as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line.strip()))
        return runs

    def print_comparison_table(self):
        runs = self.load_history()
        if not runs:
            print("[Tracker] No runs recorded yet.")
            return

        print("\n" + "=" * 110)
        print("  dens-city: Multi-Run Progress & Physical Reality Benchmark History")
        print("=" * 110)
        header = f"{'Run ID':<22} | {'Commit':<8} | {'T_c (K)':<8} | {'T_c Err':<8} | {'rho_l (nm^-3)':<14} | {'rho(z) RMSE':<12} | {'SPS':<8} | {'Time (s)':<8}"
        print(header)
        print("-" * 110)
        for r in runs:
            t_c_str = f"{r['T_c_pred']:.1f}"
            t_c_err_str = f"{r['T_c_error_pct']:+.1f}%"
            rho_l_str = f"{r['rho_l_pred']:.2f}"
            rmse_str = f"{r['rmse_rho_z']:.2f}"
            sps_str = f"{r['throughput_sps']:.0f}"
            time_str = f"{r['training_time_s']:.1f}"
            print(
                f"{r['run_id']:<22} | {r['git_commit']:<8} | {t_c_str:<8} | {t_c_err_str:<8} | {rho_l_str:<14} | {rmse_str:<12} | {sps_str:<8} | {time_str:<8}"
            )
        print("=" * 110 + "\n")


if __name__ == "__main__":
    tracker = ExperimentTracker()
    tracker.print_comparison_table()
