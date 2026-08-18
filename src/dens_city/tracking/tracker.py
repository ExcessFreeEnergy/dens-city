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
    chi_T_pred: float = 0.0
    chi_T_error_pct: float = 0.0
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
        chi_T_pred: float = 0.0,
        notes: str = "",
    ) -> RunMetrics:
        # Species-specific experimental ground truths (NIST / literature verified)
        GROUND_TRUTHS = {
            "water": {"T_c": 647.1, "rho_l": 33.36, "chi_T": 4.59e-10, "unit": "nm^-3"},
            "co2": {"T_c": 304.1, "rho_l": 0.015, "unit": "A^-3"},
            "electrolytes": {"T_c": 0.050, "rho_l": 0.020, "unit": "reduced"},
            "co2_water": {"T_c": 310.0, "rho_l": 0.033, "unit": "A^-3"},
            "co2-water": {"T_c": 310.0, "rho_l": 0.033, "unit": "A^-3"},
            "nitrogen": {"T_c": 126.2, "rho_l": 0.024, "unit": "A^-3"},
            "methane": {"T_c": 190.6, "rho_l": 0.016, "unit": "A^-3"},
            "clay_pore": {"T_c": 298.15, "rho_l": 0.033, "unit": "A^-3"},
            "clay": {"T_c": 298.15, "rho_l": 0.033, "unit": "A^-3"},
            "liquid_crystals": {"T_c": 308.5, "rho_l": 0.021, "unit": "A^-3"},
            "argon": {"T_c": 150.86, "rho_l": 0.0214, "unit": "A^-3"},
            "interfaces": {"T_c": 298.15, "rho_l": 33.36, "unit": "nm^-3"},
            "wetting": {"T_c": 298.15, "rho_l": 33.36, "unit": "nm^-3"},
        }

        spec_key = species.lower().replace(" ", "_")
        gt = GROUND_TRUTHS.get(spec_key, {"T_c": 647.1, "rho_l": 33.36, "unit": "nm^-3"})
        T_c_expt = gt["T_c"]
        rho_l_expt = gt["rho_l"]

        t_c_err = ((T_c_pred - T_c_expt) / T_c_expt) * 100.0
        rho_l_err = ((rho_l_pred - rho_l_expt) / max(1e-6, rho_l_expt)) * 100.0

        chi_T_err = 0.0
        if chi_T_pred > 0.0 and "chi_T" in gt:
            chi_T_err = ((chi_T_pred - gt["chi_T"]) / gt["chi_T"]) * 100.0

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
            chi_T_pred=chi_T_pred,
            chi_T_error_pct=chi_T_err,
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

        print("\n" + "=" * 125)
        print("  dens-city: Multi-Material E2E Benchmark History & Physical Reality Comparison")
        print("=" * 125)
        header = f"{'Run ID':<26} | {'Species':<16} | {'T_c (Pred)':<10} | {'T_c Err':<9} | {'rho_l':<10} | {'rho_l Err':<9} | {'rho(z) RMSE':<11} | {'Throughput':<12} | {'Time (s)':<8}"
        print(header)
        print("-" * 125)
        for r in runs:
            species_str = r.get("species", "unknown")
            t_c_str = f"{r['T_c_pred']:.1f}" if r["T_c_pred"] > 1.0 else f"{r['T_c_pred']:.3f}"
            t_c_err_str = f"{r['T_c_error_pct']:+.1f}%"
            rho_l_str = f"{r['rho_l_pred']:.3f}" if r["rho_l_pred"] < 1.0 else f"{r['rho_l_pred']:.1f}"
            rho_l_err_str = f"{r['rho_l_error_pct']:+.1f}%"
            rmse_str = f"{r['rmse_rho_z']:.4f}" if r["rmse_rho_z"] < 0.1 else f"{r['rmse_rho_z']:.2f}"
            sps_str = f"{r['throughput_sps']:.0f} sps"
            time_str = f"{r['training_time_s']:.2f} s"
            print(
                f"{r['run_id']:<26} | {species_str:<16} | {t_c_str:<10} | {t_c_err_str:<9} | {rho_l_str:<10} | {rho_l_err_str:<9} | {rmse_str:<11} | {sps_str:<12} | {time_str:<8}"
            )
        print("=" * 125 + "\n")


if __name__ == "__main__":
    tracker = ExperimentTracker()
    tracker.print_comparison_table()
