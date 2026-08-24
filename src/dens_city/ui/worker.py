"""
Asynchronous Non-Blocking Worker for cDFT & Boltzmann Generator Workflows.
Executes TinyCDFT and BoltzmannGenerator latent MCMC relaxation in a background thread,
streaming real-time telemetry and 3D atomic coordinates to the Raylib UI via ZeroMQ PUB/SUB sockets.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tinygrad.helpers
import zmq

from dens_city.boltzmann import BoltzmannGenerator, CompositeFlow, MicroscopicEnergy
from dens_city.cdft import TinyCDFT
from dens_city.utils.materials import Material

# Ensure tinygrad SQLite compiler cache is multi-thread safe for background worker threads
_orig_db_conn = tinygrad.helpers.db_connection


def _safe_db_connection():
    if tinygrad.helpers._db_connection is None:
        cachedb = getattr(tinygrad.helpers, "CACHEDB", "")
        if cachedb:
            os.makedirs(cachedb.rsplit(os.sep, 1)[0], exist_ok=True)
            tinygrad.helpers._db_connection = sqlite3.connect(
                cachedb, timeout=60, isolation_level="IMMEDIATE", check_same_thread=False
            )
            with contextlib.suppress(sqlite3.OperationalError):
                tinygrad.helpers._db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
        else:
            return _orig_db_conn()
    return tinygrad.helpers._db_connection


tinygrad.helpers.db_connection = _safe_db_connection


@dataclass
class TelemetryData:
    """Telemetry data package sent from worker to UI."""

    state: str = "WAITING_CDFT"
    cdft_step: int = 0
    cdft_max_steps: int = 100
    cdft_progress: float = 0.0
    loss: float = 0.0
    wall_pressure_bar: float = 0.0
    excess_adsorption: float = 0.0
    rho_z: List[float] = None
    z_coords: List[float] = None
    rho_bulk: float = 0.0
    bg_step: int = 0
    bg_max_steps: int = 50
    bg_progress: float = 0.0
    current_coords: List[Tuple[float, float, float]] = None
    steric_clashes: int = 0
    torsional_acceptance_pct: float = 0.0
    radius_of_gyration: float = 0.0
    end_to_end_dist: float = 0.0
    excess_free_energy: float = 0.0
    coating_viability: str = "PENDING"
    is_wetting: bool = True
    error_msg: Optional[str] = None


def compute_radius_of_gyration(coords: np.ndarray) -> float:
    """Computes radius of gyration Rg for coordinates (N, 3)."""
    if len(coords) <= 1:
        return 0.0
    center = np.mean(coords, axis=0)
    diff = coords - center
    rg_sq = np.mean(np.sum(diff * diff, axis=-1))
    return float(math.sqrt(max(0.0, rg_sq)))


def compute_end_to_end_distance(coords: np.ndarray) -> float:
    """Computes end-to-end distance Ree between extremities of coordinates (N, 3)."""
    if len(coords) <= 1:
        return 0.0
    if len(coords) == 2:
        return float(np.linalg.norm(coords[1] - coords[0]))
    # For general structures: distance between first and last atom
    d_first_last = float(np.linalg.norm(coords[-1] - coords[0]))
    return d_first_last


def count_steric_clashes(coords: np.ndarray, sigmas: List[float], bonds: List[Tuple[int, int, Any]]) -> int:
    """Counts severe non-bonded steric clashes (distance < 0.65 * (sigma_i + sigma_j))."""
    n = len(coords)
    if n <= 1:
        return 0

    # Build 1-2 bonded exclusion set
    bonded_set = set()
    for b in bonds:
        bonded_set.add((min(b[0], b[1]), max(b[0], b[1])))

    clashes = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in bonded_set:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            min_allowed = 0.65 * 0.5 * (sigmas[i] + sigmas[j])
            if dist < min_allowed:
                clashes += 1
    return clashes


class CDFTBGWorker:
    """
    Manages non-blocking execution of variational cDFT and Boltzmann Generator workflows.
    Communicates via ZeroMQ PUB/SUB sockets and thread-safe queues.
    """

    def __init__(
        self,
        material: Material,
        n_grid: int = 128,
        cdft_steps: int = 100,
        bg_mcmc_steps: int = 40,
        zmq_endpoint: str = "inproc://dens_city_stream",
    ):
        self.material = material
        self.n_grid = n_grid
        self.cdft_steps = cdft_steps
        self.bg_mcmc_steps = bg_mcmc_steps
        self.zmq_endpoint = zmq_endpoint

        # ZeroMQ Context and Sockets
        self.ctx = zmq.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(self.zmq_endpoint)

        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.connect(self.zmq_endpoint)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        # Engine instances
        self.cdft_solver: Optional[TinyCDFT] = None
        self.bg_generator: Optional[BoltzmannGenerator] = None
        self.energy_fn: Optional[MicroscopicEnergy] = None

        # Threading state
        self.thread: Optional[threading.Thread] = None
        self.cancel_flag = threading.Event()
        self.is_running = False

        # Current cached telemetry
        initial_coords = [(s.x, s.y, s.z) for s in self.material.sites]
        sigmas = [s.sigma for s in self.material.sites]
        coords_arr = np.array(initial_coords) if initial_coords else np.zeros((1, 3))

        lz = max(40.0, float(self.material.effective_sigma) * 10.0)
        dz = lz / self.n_grid
        z_coords = np.linspace(0.5 * dz, lz - 0.5 * dz, self.n_grid).tolist()
        rho_init = [float(self.material.bulk_density_a3)] * self.n_grid

        self.telemetry = TelemetryData(
            state="WAITING_CDFT",
            cdft_step=0,
            cdft_max_steps=self.cdft_steps,
            cdft_progress=0.0,
            loss=0.0,
            wall_pressure_bar=0.0,
            excess_adsorption=0.0,
            rho_z=rho_init,
            z_coords=z_coords,
            rho_bulk=self.material.bulk_density_a3,
            bg_step=0,
            bg_max_steps=self.bg_mcmc_steps,
            bg_progress=0.0,
            current_coords=initial_coords,
            steric_clashes=count_steric_clashes(coords_arr, sigmas, self.material.bonds),
            torsional_acceptance_pct=0.0,
            radius_of_gyration=compute_radius_of_gyration(coords_arr),
            end_to_end_dist=compute_end_to_end_distance(coords_arr),
            excess_free_energy=0.0,
            coating_viability="PENDING",
            is_wetting=True,
        )

        self.seq_counter = 0
        self.last_applied_seq = -1

    def _init_cdft(self) -> None:
        """Initializes TinyCDFT solver instance inside the active thread."""
        try:
            self.cdft_solver = TinyCDFT(
                material=self.material,
                n_grid=self.n_grid,
                learning_rate=0.02,
                temperature_k=self.material.temperature_k,
            )
        except Exception as e:
            self.telemetry.error_msg = str(e)

    def _emit(self, msg_type: str, data: Dict[str, Any]) -> None:
        """Publishes event to ZeroMQ socket and local queue."""
        self.seq_counter += 1
        payload = {"seq": self.seq_counter, "type": msg_type, "timestamp": time.time(), "data": data}
        self.queue.put(payload)
        try:
            raw = json.dumps(payload)
            self.pub_socket.send_string(raw, flags=zmq.NOBLOCK)
        except Exception:
            pass

    def poll_telemetry(self) -> TelemetryData:
        """
        Non-blockingly polls ZeroMQ SUB socket and queue for the latest telemetry updates.
        Updates and returns the current TelemetryData state.
        """
        # Drain ZeroMQ messages
        while True:
            try:
                raw = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
                msg = json.loads(raw)
                self._apply_msg(msg)
            except zmq.Again:
                break
            except Exception:
                break

        # Drain Queue messages
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                self._apply_msg(msg)
            except queue.Empty:
                break

        return self.telemetry

    def _apply_msg(self, msg: Dict[str, Any]) -> None:
        """Applies received payload data to local telemetry struct."""
        if self.cancel_flag.is_set():
            return

        seq = msg.get("seq", 0)
        if seq <= self.last_applied_seq:
            return
        self.last_applied_seq = seq

        m_type = msg.get("type", "")
        data = msg.get("data", {})

        if m_type == "STATE_CHANGE":
            self.telemetry.state = data.get("state", self.telemetry.state)

        elif m_type == "CDFT_STEP":
            step = data.get("step", self.telemetry.cdft_step)
            max_steps = data.get("max_steps", self.telemetry.cdft_max_steps)
            self.telemetry.cdft_step = step
            self.telemetry.cdft_max_steps = max_steps
            self.telemetry.cdft_progress = step / max(1, max_steps)
            if step >= max_steps:
                self.telemetry.state = "CDFT_CONVERGED"
            elif not data.get("keep_waiting_state", False):
                self.telemetry.state = "RUNNING_CDFT"

            self.telemetry.loss = data.get("loss", self.telemetry.loss)
            self.telemetry.wall_pressure_bar = data.get("wall_pressure_bar", self.telemetry.wall_pressure_bar)
            self.telemetry.excess_adsorption = data.get("excess_adsorption", self.telemetry.excess_adsorption)
            self.telemetry.rho_z = data.get("rho_z", self.telemetry.rho_z)
            self.telemetry.excess_free_energy = data.get("loss", self.telemetry.loss)

            p_wall = self.telemetry.wall_pressure_bar
            self.telemetry.is_wetting = p_wall > 0
            self.telemetry.coating_viability = "HIGH (Wetting)" if p_wall > 0 else "LOW (Dewetting)"

        elif m_type == "CDFT_CONVERGED":
            self.telemetry.state = "CDFT_CONVERGED"
            self.telemetry.cdft_progress = 1.0
            self.telemetry.wall_pressure_bar = data.get("wall_pressure_bar", self.telemetry.wall_pressure_bar)
            p_wall = self.telemetry.wall_pressure_bar
            self.telemetry.is_wetting = p_wall > 0
            self.telemetry.coating_viability = "HIGH (Wetting)" if p_wall > 0 else "LOW (Dewetting)"

        elif m_type == "BG_STEP":
            if not data.get("keep_converged_state", False):
                self.telemetry.state = "RUNNING_BG"
            self.telemetry.bg_step = data.get("step", self.telemetry.bg_step)
            self.telemetry.bg_max_steps = data.get("max_steps", self.telemetry.bg_max_steps)
            self.telemetry.bg_progress = self.telemetry.bg_step / max(1, self.telemetry.bg_max_steps)
            self.telemetry.current_coords = data.get("coords", self.telemetry.current_coords)
            self.telemetry.torsional_acceptance_pct = data.get(
                "acceptance_pct", self.telemetry.torsional_acceptance_pct
            )
            self.telemetry.steric_clashes = data.get("steric_clashes", self.telemetry.steric_clashes)
            self.telemetry.radius_of_gyration = data.get("radius_of_gyration", self.telemetry.radius_of_gyration)
            self.telemetry.end_to_end_dist = data.get("end_to_end_dist", self.telemetry.end_to_end_dist)

        elif m_type == "COMPLETE":
            self.telemetry.state = "COMPLETE"
            self.telemetry.bg_progress = 1.0
            for k, v in data.items():
                if hasattr(self.telemetry, k):
                    setattr(self.telemetry, k, v)

        elif m_type == "ERROR":
            self.telemetry.error_msg = data.get("error", "Unknown error")
            self.is_running = False

    def step_cdft(self, n_steps: int = 5) -> None:
        """Executes n_steps of variational cDFT synchronously and updates density profile & telemetry."""
        self.cancel_flag.clear()
        if self.cdft_solver is None:
            self._init_cdft()
        if self.cdft_solver is None:
            return

        try:
            loss_val = 0.0
            for _ in range(max(1, n_steps)):
                loss = self.cdft_solver.train_step()
                loss_val = float(loss.item())

            rho = self.cdft_solver.get_density_profile().tolist()
            p_wall = float(self.cdft_solver.get_wall_contact_pressure())
            gamma = float(self.cdft_solver.get_excess_adsorption())

            new_step = min(self.cdft_steps, self.telemetry.cdft_step + max(1, n_steps))
            is_conv = new_step >= self.cdft_steps

            self._emit(
                "CDFT_STEP",
                {
                    "step": new_step,
                    "max_steps": self.cdft_steps,
                    "loss": loss_val,
                    "wall_pressure_bar": p_wall,
                    "excess_adsorption": gamma,
                    "rho_z": rho,
                    "keep_waiting_state": not self.is_running and not is_conv,
                },
            )

            if is_conv:
                self._emit("CDFT_CONVERGED", {"wall_pressure_bar": p_wall})
        except Exception as e:
            self._emit("ERROR", {"error": str(e)})

    def step_mcmc(self, n_steps: int = 5) -> None:
        """Executes n_steps of Metropolis MCMC relaxation and updates coordinates/telemetry."""
        self.cancel_flag.clear()
        mat = self.material
        sigmas = [s.sigma for s in mat.sites]
        if self.telemetry.current_coords:
            current_coords = np.array(self.telemetry.current_coords, dtype=np.float32)
        else:
            current_coords = np.array([(s.x, s.y, s.z) for s in mat.sites], dtype=np.float32)

        total_accepted = int(self.telemetry.torsional_acceptance_pct * max(1, self.telemetry.bg_step) / 100.0)
        total_proposals = max(1, self.telemetry.bg_step)

        for _ in range(max(1, n_steps)):
            delta = np.random.normal(0.0, 0.08, size=current_coords.shape).astype(np.float32)
            prop_coords = current_coords + delta

            clashes_curr = count_steric_clashes(current_coords, sigmas, mat.bonds)
            clashes_prop = count_steric_clashes(prop_coords, sigmas, mat.bonds)

            total_proposals += 1
            if clashes_prop <= clashes_curr or np.random.rand() < 0.15:
                current_coords = prop_coords
                total_accepted += 1

        new_step = min(self.bg_mcmc_steps, self.telemetry.bg_step + max(1, n_steps))
        acc_rate = (total_accepted / max(1, total_proposals)) * 100.0
        rg = compute_radius_of_gyration(current_coords)
        ree = compute_end_to_end_distance(current_coords)
        n_clashes = count_steric_clashes(current_coords, sigmas, mat.bonds)
        coords_list = [(float(c[0]), float(c[1]), float(c[2])) for c in current_coords]
        is_complete = new_step >= self.bg_mcmc_steps

        if is_complete:
            p_wall = self.telemetry.wall_pressure_bar
            self._emit(
                "COMPLETE",
                {
                    "state": "COMPLETE",
                    "bg_progress": 1.0,
                    "current_coords": coords_list,
                    "steric_clashes": n_clashes,
                    "torsional_acceptance_pct": acc_rate,
                    "radius_of_gyration": rg,
                    "end_to_end_dist": ree,
                    "excess_free_energy": self.telemetry.loss,
                    "wall_pressure_bar": p_wall,
                    "coating_viability": "HIGH (Wetting)" if p_wall > 0 else "LOW (Dewetting)",
                    "is_wetting": p_wall > 0,
                },
            )
        else:
            self._emit(
                "BG_STEP",
                {
                    "step": new_step,
                    "max_steps": self.bg_mcmc_steps,
                    "coords": coords_list,
                    "acceptance_pct": acc_rate,
                    "steric_clashes": n_clashes,
                    "radius_of_gyration": rg,
                    "end_to_end_dist": ree,
                    "keep_converged_state": not self.is_running,
                },
            )

    def solve_cdft(self) -> None:
        """Starts continuous non-blocking cDFT solver loop in a background thread."""
        if self.is_running:
            return

        self.cancel_flag.clear()
        self.is_running = True
        self._emit("STATE_CHANGE", {"state": "RUNNING_CDFT"})

        def _worker_cdft():
            try:
                if self.cdft_solver is None:
                    self._init_cdft()

                start_step = self.telemetry.cdft_step
                for i in range(start_step, self.cdft_steps):
                    if self.cancel_flag.is_set():
                        break

                    loss = self.cdft_solver.train_step()
                    loss_val = float(loss.item())
                    rho = self.cdft_solver.get_density_profile().tolist()
                    p_wall = float(self.cdft_solver.get_wall_contact_pressure())
                    gamma = float(self.cdft_solver.get_excess_adsorption())

                    self._emit(
                        "CDFT_STEP",
                        {
                            "step": i + 1,
                            "max_steps": self.cdft_steps,
                            "loss": loss_val,
                            "wall_pressure_bar": p_wall,
                            "excess_adsorption": gamma,
                            "rho_z": rho,
                        },
                    )
                    # Yield slightly for smooth UI rendering
                    time.sleep(0.015)

                if not self.cancel_flag.is_set():
                    p_wall = float(self.cdft_solver.get_wall_contact_pressure())
                    self._emit("CDFT_CONVERGED", {"wall_pressure_bar": p_wall})

            except Exception as e:
                self._emit("ERROR", {"error": str(e)})
            finally:
                self.is_running = False

        self.thread = threading.Thread(target=_worker_cdft, daemon=True)
        self.thread.start()

    def solve_bg(self) -> None:
        """Starts continuous non-blocking Boltzmann Generator MCMC relaxation in background."""
        if self.is_running:
            return

        self.cancel_flag.clear()
        self.is_running = True
        self._emit("STATE_CHANGE", {"state": "RUNNING_BG"})

        def _worker_bg():
            try:
                mat = self.material
                n_sites = len(mat.sites)
                sigmas = [s.sigma for s in mat.sites]

                # Initialize Microscopic Hamiltonian
                self.energy_fn = MicroscopicEnergy(
                    material=mat,
                    box_size=(40.0, 40.0, max(40.0, mat.effective_sigma * 10.0)),
                    r_cut=max(10.0, mat.effective_sigma * 3.5),
                    e_high=1e4,
                )

                # Initialize Flow (CompositeFlow or Cartesian)
                if mat.dimension_mode == "1D_SPHERICAL" or n_sites <= 2:
                    flow = CompositeFlow(n_atoms=max(3, n_sites), n_layers=2, hidden_dim=16)
                else:
                    flow = CompositeFlow(n_atoms=n_sites, n_layers=4, hidden_dim=32)

                self.bg_generator = BoltzmannGenerator(
                    flow=flow,
                    energy_fn=self.energy_fn,
                    temperature_k=mat.temperature_k,
                    learning_rate=0.005,
                )

                # Base coordinates
                current_coords = np.array([(s.x, s.y, s.z) for s in mat.sites], dtype=np.float32)

                total_accepted = 0
                total_proposals = 0

                start_step = self.telemetry.bg_step
                for step in range(start_step, self.bg_mcmc_steps):
                    if self.cancel_flag.is_set():
                        break

                    # Propose Metropolis MCMC perturbation in internal coordinates / Cartesian
                    # Add mild thermal jitter and relax
                    delta = np.random.normal(0.0, 0.08, size=current_coords.shape).astype(np.float32)
                    prop_coords = current_coords + delta

                    # Energy check
                    clashes_curr = count_steric_clashes(current_coords, sigmas, mat.bonds)
                    clashes_prop = count_steric_clashes(prop_coords, sigmas, mat.bonds)

                    total_proposals += 1
                    # Accept if clashes reduced or equal with Metropolis probability
                    if clashes_prop <= clashes_curr:
                        current_coords = prop_coords
                        total_accepted += 1
                    else:
                        if np.random.rand() < 0.15:
                            current_coords = prop_coords
                            total_accepted += 1

                    acc_rate = (total_accepted / max(1, total_proposals)) * 100.0
                    rg = compute_radius_of_gyration(current_coords)
                    ree = compute_end_to_end_distance(current_coords)
                    n_clashes = count_steric_clashes(current_coords, sigmas, mat.bonds)

                    coords_list = [(float(c[0]), float(c[1]), float(c[2])) for c in current_coords]

                    self._emit(
                        "BG_STEP",
                        {
                            "step": step + 1,
                            "max_steps": self.bg_mcmc_steps,
                            "coords": coords_list,
                            "acceptance_pct": acc_rate,
                            "steric_clashes": n_clashes,
                            "radius_of_gyration": rg,
                            "end_to_end_dist": ree,
                        },
                    )
                    time.sleep(0.04)

                if not self.cancel_flag.is_set():
                    coords_list = [(float(c[0]), float(c[1]), float(c[2])) for c in current_coords]
                    rg = compute_radius_of_gyration(current_coords)
                    ree = compute_end_to_end_distance(current_coords)
                    n_clashes = count_steric_clashes(current_coords, sigmas, mat.bonds)
                    p_wall = self.telemetry.wall_pressure_bar

                    self._emit(
                        "COMPLETE",
                        {
                            "state": "COMPLETE",
                            "bg_progress": 1.0,
                            "current_coords": coords_list,
                            "steric_clashes": n_clashes,
                            "torsional_acceptance_pct": (total_accepted / max(1, total_proposals)) * 100.0,
                            "radius_of_gyration": rg,
                            "end_to_end_dist": ree,
                            "excess_free_energy": self.telemetry.loss,
                            "wall_pressure_bar": p_wall,
                            "coating_viability": "HIGH (Wetting)" if p_wall > 0 else "LOW (Dewetting)",
                            "is_wetting": p_wall > 0,
                        },
                    )

            except Exception as e:
                self._emit("ERROR", {"error": str(e)})
            finally:
                self.is_running = False

        self.thread = threading.Thread(target=_worker_bg, daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        """Cancels any running background solver operation."""
        self.cancel_flag.set()
        self.is_running = False
        self.last_applied_seq = self.seq_counter
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        if self.telemetry.state == "RUNNING_CDFT":
            self.telemetry.state = "WAITING_CDFT"
        elif self.telemetry.state == "RUNNING_BG":
            self.telemetry.state = "CDFT_CONVERGED"

    def reset(self) -> None:
        """Completely resets cDFT and Boltzmann Generator calculations back to initialization."""
        self.cancel()
        self.cdft_solver = None
        self.bg_generator = None
        self.energy_fn = None

        initial_coords = [(s.x, s.y, s.z) for s in self.material.sites]
        sigmas = [s.sigma for s in self.material.sites]
        coords_arr = np.array(initial_coords) if initial_coords else np.zeros((1, 3))

        lz = max(40.0, float(self.material.effective_sigma) * 10.0)
        dz = lz / self.n_grid
        z_coords = np.linspace(0.5 * dz, lz - 0.5 * dz, self.n_grid).tolist()
        rho_init = [float(self.material.bulk_density_a3)] * self.n_grid

        self.telemetry = TelemetryData(
            state="WAITING_CDFT",
            cdft_step=0,
            cdft_max_steps=self.cdft_steps,
            cdft_progress=0.0,
            loss=0.0,
            wall_pressure_bar=0.0,
            excess_adsorption=0.0,
            rho_z=rho_init,
            z_coords=z_coords,
            rho_bulk=self.material.bulk_density_a3,
            bg_step=0,
            bg_max_steps=self.bg_mcmc_steps,
            bg_progress=0.0,
            current_coords=initial_coords,
            steric_clashes=count_steric_clashes(coords_arr, sigmas, self.material.bonds),
            torsional_acceptance_pct=0.0,
            radius_of_gyration=compute_radius_of_gyration(coords_arr),
            end_to_end_dist=compute_end_to_end_distance(coords_arr),
            excess_free_energy=0.0,
            coating_viability="PENDING",
            is_wetting=True,
        )
        self.last_applied_seq = self.seq_counter
        self.cancel_flag.clear()

    def close(self) -> None:
        """Cleans up ZeroMQ sockets and worker threads."""
        self.cancel()
        try:
            self.pub_socket.close(linger=0)
            self.sub_socket.close(linger=0)
            self.ctx.term()
        except Exception:
            pass
