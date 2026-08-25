"""
High-throughput batch pipeline execution logic for dens-city.
Encapsulates single-material end-to-end execution (cDFT screening -> Spatial Prior -> Boltzmann Generator -> Artifact Export)
with structured error classification and state serialization.
"""

import math
import multiprocessing
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tinygrad import nn

from dens_city.boltzmann.bijectors import Base2CartesianFlow
from dens_city.boltzmann.energy import EGNNMicroscopicEnergy, MicroscopicEnergy
from dens_city.boltzmann.generator import BoltzmannGenerator
from dens_city.boltzmann.prior import CDFTBaseDistribution
from dens_city.cdft.cdft import BatchedTinyCDFT, TinyCDFT
from dens_city.cdft.kernels import KernelBuilder
from dens_city.utils.materials import Material, MaterialLoader, MolecularBatch


class PipelineStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SUCCESS_CDFT_ONLY = "SUCCESS_CDFT_ONLY"
    SKIPPED_THERMO = "SKIPPED_THERMO"
    FAILED_TRAINING = "FAILED_TRAINING"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    FAILED_ERROR = "FAILED_ERROR"


@dataclass
class MaterialPipelineTask:
    """
    Specification of a single material processing task in the batch pipeline.
    """

    material_path_or_name: str
    out_dir: str
    temperature_k: float = 300.0
    pressure_bar: Optional[float] = 1.0
    chemical_potential_kbt: Optional[float] = None
    bulk_density_a3: Optional[float] = None
    slit_width_a: Optional[float] = None
    grid: int = 128
    cdft_steps: int = 60
    cdft_lr: float = 0.02
    bg_steps: int = 40
    bg_batch_size: int = 512
    bg_lr: float = 0.01
    bg_samples: int = 100
    bg_w_tor: float = 0.0
    bg_mcmc_steps: int = 0
    bg_mcmc_step_size: float = 0.1
    skip_bg: bool = False
    debug: bool = False
    debug_log_path: Optional[str] = None
    r_cut: Optional[float] = None
    energy_engine: str = "classical"


@dataclass
class MaterialPipelineResult:
    """
    Structured outcome and physical observables from a pipeline execution.
    """

    material_name: str
    status: str
    error_message: Optional[str] = None
    runtime_seconds: float = 0.0
    cdft_runtime_seconds: float = 0.0
    bg_runtime_seconds: float = 0.0
    num_sites: int = 0
    temperature_k: float = 0.0
    bulk_density_a3: float = 0.0
    chemical_potential_kbt: float = 0.0
    bulk_pressure_bar: float = 0.0
    wall_pressure_bar: float = 0.0
    excess_adsorption_a2: float = 0.0
    cdft_final_loss: float = 0.0
    bg_final_loss: Optional[float] = None
    solvation_free_energy_kcal_mol: Optional[float] = None
    artifact_dir: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_xyz_trajectory(
    filepath: str,
    coords: np.ndarray,
    site_names: List[str],
    energies: Optional[List[float]] = None,
    material_name: str = "",
) -> None:
    """
    Writes a multi-frame atomic trajectory in standard XYZ format.
    coords: shape (B, N, 3) where B is the number of frames and N is the number of atoms.
    """
    B, N, _ = coords.shape
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    lines = []
    for frame_idx in range(B):
        lines.append(str(N))
        e_str = (
            f" | Energy: {energies[frame_idx]:.4f} K" if (energies is not None and frame_idx < len(energies)) else ""
        )
        lines.append(f"Frame {frame_idx} | Material: {material_name}{e_str}")
        for atom_idx in range(N):
            name = site_names[atom_idx] if atom_idx < len(site_names) else "X"
            elem = "".join([c for c in name if c.isalpha()]) or name
            x, y, z = coords[frame_idx, atom_idx]
            lines.append(f"{elem:<4} {x:12.6f} {y:12.6f} {z:12.6f}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_flow_weights(filepath: str, flow: Any) -> None:
    """
    Serializes trainable weights from RealNVPFlow or CompositeFlow to an NPZ archive.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state_dict = nn.state.get_state_dict(flow)
    np_dict = {k: v.numpy() for k, v in state_dict.items()}
    np.savez(filepath, **np_dict)


def process_material_task(task: MaterialPipelineTask) -> MaterialPipelineResult:
    """
    Executes the complete single-material pipeline with strict exception handling:
    1. Thermodynamic Routing (MaterialLoader)
    2. Mean-Field Screening (TinyCDFT)
    3. Spatial Prior Handoff (CDFTBaseDistribution)
    4. Many-Body Microscopic Energy & Flow Construction (MicroscopicEnergy + CompositeFlow/RealNVP)
    5. Boltzmann Generator Training & Sampling (Reverse KL -> 3D Conformations)
    6. Artifact Export (XYZ trajectory, NPY profiles, NPZ weights, TXT summaries)
    """
    t_start = time.perf_counter()
    mat_input = task.material_path_or_name
    mat_basename = Path(mat_input).stem if os.path.exists(mat_input) or "/" in mat_input else str(mat_input)
    mat_out_dir = os.path.join(task.out_dir, mat_basename)
    os.makedirs(mat_out_dir, exist_ok=True)

    artifacts_created = []

    # 1. Thermodynamic Routing
    try:
        material = MaterialLoader.load_material(
            material_name_or_path=task.material_path_or_name,
            temperature_k=task.temperature_k,
            bulk_density_a3=task.bulk_density_a3,
            pressure_bar=task.pressure_bar,
            chemical_potential_kbt=task.chemical_potential_kbt,
        )
    except Exception as e:
        t_tot = time.perf_counter() - t_start
        err_msg = f"Thermodynamic routing failed: {str(e)}"
        status = (
            PipelineStatus.SKIPPED_THERMO
            if "spinodal" in str(e).lower() or "density" in str(e).lower()
            else PipelineStatus.FAILED_ERROR
        )
        return MaterialPipelineResult(
            material_name=mat_basename,
            status=status.value,
            error_message=err_msg,
            runtime_seconds=t_tot,
            artifact_dir=mat_out_dir,
        )

    # 2. Mean-Field Screening (cDFT)
    t_cdft_start = time.perf_counter()
    try:
        slit_w = task.slit_width_a if task.slit_width_a is not None else max(40.0, 12.0 * material.effective_sigma)
        cdft = TinyCDFT(
            material=material,
            n_grid=task.grid,
            slit_width_a=slit_w,
            temperature_k=material.temperature_k,
            bulk_density_a3=material.bulk_density_a3,
            learning_rate=task.cdft_lr,
        )
        cdft_res = cdft.solve(steps=task.cdft_steps, verbose=False)
        t_cdft = time.perf_counter() - t_cdft_start

        rho_profile = cdft.get_density_profile()
        p_wall = cdft.get_wall_contact_pressure()
        gamma_ex = cdft.get_excess_adsorption()
        cdft_loss = (
            cdft_res.get("final_loss", 0.0) if isinstance(cdft_res, dict) else (cdft_res[-1] if cdft_res else 0.0)
        )

        # Export cDFT Artifacts
        npy_path = os.path.join(mat_out_dir, "density_profile.npy")
        np.save(npy_path, rho_profile)
        artifacts_created.append(npy_path)

        csv_path = os.path.join(mat_out_dir, "density_profile.csv")
        z_grid = np.linspace(0.5 * cdft.dz_val, cdft.slit_width_a - 0.5 * cdft.dz_val, cdft.n_grid)
        np.savetxt(
            csv_path, np.column_stack([z_grid, rho_profile]), delimiter=",", header="z_angstrom,rho_a3", comments=""
        )
        artifacts_created.append(csv_path)

        summary_path = os.path.join(mat_out_dir, "cdft_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Material: {material.name}\n")
            f.write(f"Dimension Mode: {material.dimension_mode}\n")
            f.write(f"Num Sites: {material.num_sites}\n")
            f.write(f"Temperature: {material.temperature_k:.2f} K\n")
            f.write(f"Bulk Density: {material.bulk_density_a3:.6f} Å^-3\n")
            f.write(f"Bulk Pressure: {material.bulk_pressure_bar:.4f} bar\n")
            f.write(f"Chemical Potential: {material.bulk_mu:.4f} k_B T\n")
            f.write(f"Wall Contact Pressure: {p_wall:.4f} bar\n")
            f.write(f"Excess Adsorption: {gamma_ex:.6f} Å^-2\n")
            f.write(f"cDFT Solver Runtime: {t_cdft:.3f} s\n")
        artifacts_created.append(summary_path)

    except Exception as e:
        t_tot = time.perf_counter() - t_start
        err_msg = f"cDFT screening failed: {str(e)}\n{traceback.format_exc()}"
        return MaterialPipelineResult(
            material_name=mat_basename,
            status=PipelineStatus.FAILED_ERROR.value,
            error_message=err_msg,
            runtime_seconds=t_tot,
            num_sites=material.num_sites,
            temperature_k=material.temperature_k,
            bulk_density_a3=material.bulk_density_a3,
            chemical_potential_kbt=material.bulk_mu,
            bulk_pressure_bar=material.bulk_pressure_bar,
            artifact_dir=mat_out_dir,
            artifacts=artifacts_created,
        )

    # If skip_bg is set, return early with cDFT observables
    if task.skip_bg:
        t_tot = time.perf_counter() - t_start
        return MaterialPipelineResult(
            material_name=mat_basename,
            status=PipelineStatus.SUCCESS_CDFT_ONLY.value,
            runtime_seconds=t_tot,
            cdft_runtime_seconds=t_cdft,
            num_sites=material.num_sites,
            temperature_k=material.temperature_k,
            bulk_density_a3=material.bulk_density_a3,
            chemical_potential_kbt=material.bulk_mu,
            bulk_pressure_bar=material.bulk_pressure_bar,
            wall_pressure_bar=p_wall,
            excess_adsorption_a2=gamma_ex,
            cdft_final_loss=cdft_loss,
            artifact_dir=mat_out_dir,
            artifacts=artifacts_created,
        )

    # 3. Generative Handoff & Boltzmann Generator
    t_bg_start = time.perf_counter()
    try:
        n_sites = material.num_sites
        box_xy = (30.0, 30.0)
        box_size_3d = (30.0, 30.0, slit_w)

        # Microscopic Hamiltonian: Classical or EGNN MLFF
        if task.energy_engine == "egnn":
            energy_fn = EGNNMicroscopicEnergy(
                material=material,
                box_size=box_size_3d,
            )
        else:
            energy_fn = MicroscopicEnergy(
                material=material,
                box_size=box_size_3d,
                r_cut=task.r_cut,
                pad_to_128=True,
                target_n_particles=128,
            )
        n_pad_sites = energy_fn.n_particles

        # 4-Channel Base-2 Cartesian Flow (dim = 128 * 4 = 512)
        flow = Base2CartesianFlow(
            n_atoms=n_pad_sites,
            n_layers=4,
            hidden_dim=64,
        )
        flow_prior = CDFTBaseDistribution(
            rho_z=rho_profile,
            l_z=slit_w,
            box_size_xy=box_xy,
            n_particles=n_pad_sites,
        )

        # Boltzmann Generator Optimization
        generator = BoltzmannGenerator(
            flow=flow,
            energy_fn=energy_fn,
            prior=flow_prior,
            temperature_k=material.temperature_k,
            learning_rate=task.bg_lr,
            batch_size=task.bg_batch_size,
            w_torsion=task.bg_w_tor,
            dihedral_quadruplets=material.dihedral_quadruplets,
        )

        bg_losses = generator.train(
            steps=task.bg_steps,
            batch_size=task.bg_batch_size,
            verbose=False,
        )

        if not np.all(np.isfinite(bg_losses)):
            raise ValueError(f"Non-finite loss detected during Boltzmann flow training: {bg_losses}")

        bg_loss = bg_losses[-1] if bg_losses else 0.0

        # Sample Uncorrelated 3D Equilibrium Configurations in static chunks of bg_batch_size
        all_samples = []
        all_energies = []
        n_batches = math.ceil(task.bg_samples / task.bg_batch_size)
        for _ in range(n_batches):
            b_samples_pad = generator.sample(
                n_samples=task.bg_batch_size,
                return_all_pad=True,
                mcmc_steps=task.bg_mcmc_steps,
                mcmc_step_size=task.bg_mcmc_step_size,
            )
            b_energies = energy_fn.eval_energy(b_samples_pad)
            b_samples_real = (
                b_samples_pad[:, :n_sites, :] if len(b_samples_pad.shape) == 3 else b_samples_pad[:n_sites, :]
            )
            all_samples.append(b_samples_real.numpy())
            all_energies.extend(b_energies.numpy().tolist())

        samples_np = np.concatenate(all_samples, axis=0)[: task.bg_samples]
        energies = all_energies[: task.bg_samples]

        if len(samples_np.shape) == 2:
            # Reshape flat samples (B, N*3) -> (B, N, 3)
            samples_np = samples_np.reshape(task.bg_samples, n_sites, 3)

        t_bg = time.perf_counter() - t_bg_start

        # 4. Export Generative Artifacts
        site_names = [s.site_name for s in material.sites] if material.sites else [material.name]
        xyz_path = os.path.join(mat_out_dir, "trajectory.xyz")
        write_xyz_trajectory(
            filepath=xyz_path,
            coords=samples_np,
            site_names=site_names,
            energies=energies,
            material_name=material.name,
        )
        artifacts_created.append(xyz_path)

        weights_path = os.path.join(mat_out_dir, "flow_weights.npz")
        save_flow_weights(weights_path, flow)
        artifacts_created.append(weights_path)

    except Exception as e:
        t_tot = time.perf_counter() - t_start
        err_msg = f"Boltzmann Generator training failed: {str(e)}\n{traceback.format_exc()}"
        return MaterialPipelineResult(
            material_name=mat_basename,
            status=PipelineStatus.FAILED_TRAINING.value,
            error_message=err_msg,
            runtime_seconds=t_tot,
            cdft_runtime_seconds=t_cdft,
            num_sites=material.num_sites,
            temperature_k=material.temperature_k,
            bulk_density_a3=material.bulk_density_a3,
            chemical_potential_kbt=material.bulk_mu,
            bulk_pressure_bar=material.bulk_pressure_bar,
            wall_pressure_bar=p_wall,
            excess_adsorption_a2=gamma_ex,
            cdft_final_loss=cdft_loss,
            artifact_dir=mat_out_dir,
            artifacts=artifacts_created,
        )

    t_tot = time.perf_counter() - t_start
    return MaterialPipelineResult(
        material_name=mat_basename,
        status=PipelineStatus.SUCCESS.value,
        runtime_seconds=t_tot,
        cdft_runtime_seconds=t_cdft,
        bg_runtime_seconds=t_bg,
        num_sites=material.num_sites,
        temperature_k=material.temperature_k,
        bulk_density_a3=material.bulk_density_a3,
        chemical_potential_kbt=material.bulk_mu,
        bulk_pressure_bar=material.bulk_pressure_bar,
        wall_pressure_bar=p_wall,
        excess_adsorption_a2=gamma_ex,
        cdft_final_loss=cdft_loss,
        bg_final_loss=bg_loss,
        artifact_dir=mat_out_dir,
        artifacts=artifacts_created,
    )


class AsyncArtifactWriter:
    """
    Background asynchronous worker thread for non-blocking disk I/O.
    Receives (filepath, data_type, payload) items and serializes them in the background
    to prevent blocking or halting device execution.
    """

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            action, path, payload = item
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if action == "npy":
                    np.save(path, payload)
                elif action == "csv":
                    header, data = payload
                    np.savetxt(path, data, delimiter=",", header=header, comments="")
                elif action == "txt":
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(payload)
                elif action == "xyz":
                    coords, site_names, energies, mat_name = payload
                    write_xyz_trajectory(path, coords, site_names, energies, mat_name)
                elif action == "npz":
                    np.savez(path, **payload)
            except Exception:
                pass
            finally:
                self.q.task_done()

    def write_npy(self, path: str, arr: np.ndarray) -> None:
        self.q.put(("npy", path, arr))

    def write_csv(self, path: str, header: str, data: np.ndarray) -> None:
        self.q.put(("csv", path, (header, data)))

    def write_txt(self, path: str, text: str) -> None:
        self.q.put(("txt", path, text))

    def write_xyz(
        self,
        path: str,
        coords: np.ndarray,
        site_names: List[str],
        energies: Optional[List[float]] = None,
        material_name: str = "",
    ) -> None:
        self.q.put(("xyz", path, (coords, site_names, energies, material_name)))

    def write_npz(self, path: str, np_dict: Dict[str, np.ndarray]) -> None:
        self.q.put(("npz", path, np_dict))

    def flush(self) -> None:
        self.q.join()

    def close(self) -> None:
        self.q.put(None)
        self._thread.join()


def _load_single_material_worker_unpack(args: Tuple) -> Tuple[int, Optional[Material], Optional[str]]:
    """
    Unpacks task arguments and executes MaterialLoader.load_material in an isolated
    ProcessPool worker process, bypassing Python GIL for fast parallel regex and EOS solves.
    Also precomputes analytical 1D NumPy FMT planar kernels and wall potentials concurrently.
    """
    task_idx, mat_path_or_name, temp_k, density, p_bar, mu_kbt = args
    try:
        mat = MaterialLoader.load_material(
            material_name_or_path=mat_path_or_name,
            temperature_k=temp_k,
            bulk_density_a3=density,
            pressure_bar=p_bar,
            chemical_potential_kbt=mu_kbt,
        )
        n_grid = 128
        slit_w = max(40.0, 12.0 * mat.effective_sigma)
        dz = slit_w / n_grid
        fmt_dict = KernelBuilder.build_fmt_planar_kernels_np(mat.effective_sigma, dz)
        att_arr, _ = KernelBuilder.build_wca_attraction_kernel_np(mat.effective_sigma, mat.effective_epsilon_k, dz)
        v_ext_np = (
            KernelBuilder.build_slit_wall_potential_np(
                n_grid=n_grid,
                dz=dz,
                fluid_sigma=mat.effective_sigma,
                wall_sigma=3.4,
                wall_epsilon_k=50.0,
            )
            / mat.temperature_k
        )

        mat.precomputed_cdft_data = {
            "fmt_w3": fmt_dict["w3"],
            "fmt_w2": fmt_dict["w2"],
            "fmt_w1": fmt_dict["w1"],
            "fmt_w0": fmt_dict["w0"],
            "fmt_wv2": fmt_dict["wv2"],
            "fmt_wv1": fmt_dict["wv1"],
            "att_arr": att_arr,
            "v_ext_np": v_ext_np,
            "slit_w": slit_w,
            "dz": dz,
        }
        return task_idx, mat, None
    except Exception as e:
        return task_idx, None, str(e)


@dataclass
class PreparedMolecularBatch:
    """
    Encapsulates a device-ready batch where all CPU-side parsing, EOS root-finding,
    padding, and tensor allocation have already been assembled in parallel.
    """

    tasks: List[MaterialPipelineTask]
    batch_size: int
    loaded_materials: List[Material]
    task_indices: List[int]
    results_map: Dict[int, MaterialPipelineResult]
    mol_batch: Optional[MolecularBatch] = None
    batched_cdft: Optional[BatchedTinyCDFT] = None
    energy_fn: Optional[MicroscopicEnergy] = None
    t_assembly_start: float = field(default_factory=time.perf_counter)


class AsyncBatchPrefetcher:
    """
    Double-buffered background batch loader and prefetcher.
    Spawns a ProcessPoolExecutor to parse .mol2 files and solve EOS concurrently across CPU cores,
    while running a background threading.Thread that constructs device tensors and enqueues
    PreparedMolecularBatch objects into a bounded queue (maxsize=2).
    """

    def __init__(
        self,
        task_chunks: List[List[MaterialPipelineTask]],
        batch_size: int = 512,
        max_workers: Optional[int] = None,
        prefetch_depth: int = 2,
        energy_engine: str = "classical",
    ):
        self.task_chunks = task_chunks
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, os.cpu_count() or 4)
        self.energy_engine = energy_engine
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "AsyncBatchPrefetcher":
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncBatchPrefetcher")
        self._thread.start()
        return self

    def _worker_loop(self) -> None:
        try:
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_ctx) as executor:
                for chunk in self.task_chunks:
                    if self._shutdown_event.is_set():
                        break

                    prepared = self._assemble_batch(chunk, executor)
                    while not self._shutdown_event.is_set():
                        try:
                            self.queue.put(prepared, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except Exception:
            traceback.print_exc()
        finally:
            while not self._shutdown_event.is_set():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _assemble_batch(
        self,
        chunk: List[MaterialPipelineTask],
        executor: ProcessPoolExecutor,
    ) -> PreparedMolecularBatch:
        t_start = time.perf_counter()
        results_map: Dict[int, MaterialPipelineResult] = {}
        loaded_materials: List[Material] = []
        task_indices: List[int] = []

        worker_args = [
            (
                idx,
                task.material_path_or_name,
                task.temperature_k,
                task.bulk_density_a3,
                task.pressure_bar,
                task.chemical_potential_kbt,
            )
            for idx, task in enumerate(chunk)
        ]

        results = list(executor.map(_load_single_material_worker_unpack, worker_args))

        for idx, mat, err in results:
            task = chunk[idx]
            mat_input = task.material_path_or_name
            mat_basename = Path(mat_input).stem if os.path.exists(mat_input) or "/" in mat_input else str(mat_input)
            mat_out_dir = os.path.join(task.out_dir, mat_basename)
            os.makedirs(mat_out_dir, exist_ok=True)

            if mat is not None:
                loaded_materials.append(mat)
                task_indices.append(idx)
            else:
                status = (
                    PipelineStatus.SKIPPED_THERMO
                    if "spinodal" in str(err).lower() or "density" in str(err).lower()
                    else PipelineStatus.FAILED_ERROR
                )
                results_map[idx] = MaterialPipelineResult(
                    material_name=mat_basename,
                    status=status.value,
                    error_message=f"Thermodynamic routing failed: {str(err)}",
                    runtime_seconds=time.perf_counter() - t_start,
                    artifact_dir=mat_out_dir,
                )

        if not loaded_materials:
            return PreparedMolecularBatch(
                tasks=chunk,
                batch_size=self.batch_size,
                loaded_materials=[],
                task_indices=[],
                results_map=results_map,
                t_assembly_start=t_start,
            )

        return PreparedMolecularBatch(
            tasks=chunk,
            batch_size=self.batch_size,
            loaded_materials=loaded_materials,
            task_indices=task_indices,
            results_map=results_map,
            t_assembly_start=t_start,
        )

    def __iter__(self):
        return self

    def __next__(self) -> PreparedMolecularBatch:
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def close(self) -> None:
        self._shutdown_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


def execute_prepared_batch(
    prepared_batch: PreparedMolecularBatch,
    async_writer: Optional[AsyncArtifactWriter] = None,
) -> List[MaterialPipelineResult]:
    """
    Executes purely on-device operations for a pre-assembled batch:
    cDFT JIT minimization -> Boltzmann Generator JIT training -> 3D conformation sampling.
    """
    t_start = prepared_batch.t_assembly_start
    batch_tasks = prepared_batch.tasks
    loaded_materials = prepared_batch.loaded_materials
    task_indices = prepared_batch.task_indices
    results_map = dict(prepared_batch.results_map)
    batch_size = prepared_batch.batch_size

    if not loaded_materials:
        return [results_map[i] for i in range(len(batch_tasks))]

    # Instantiate device tensors on the executing thread (avoids SQLite thread-local cache issues)
    mol_batch = (
        prepared_batch.mol_batch
        if prepared_batch.mol_batch is not None
        else MolecularBatch.create_batch(
            materials=loaded_materials,
            batch_size=batch_size,
            target_n_particles=128,
        )
    )

    batched_cdft = (
        prepared_batch.batched_cdft
        if prepared_batch.batched_cdft is not None
        else BatchedTinyCDFT(
            batch=mol_batch,
            n_grid=128,
            learning_rate=batch_tasks[0].cdft_lr if batch_tasks else 0.02,
        )
    )

    engine_type = (
        batch_tasks[0].energy_engine if batch_tasks and hasattr(batch_tasks[0], "energy_engine") else "classical"
    )
    if engine_type == "egnn":
        energy_fn = (
            prepared_batch.energy_fn
            if prepared_batch.energy_fn is not None
            else EGNNMicroscopicEnergy(material=mol_batch)
        )
    else:
        energy_fn = (
            prepared_batch.energy_fn
            if prepared_batch.energy_fn is not None
            else MicroscopicEnergy(material=mol_batch, pad_to_128=True)
        )

    # 1. Batched cDFT Screening Phase
    t_c0 = time.perf_counter()
    cdft_losses = batched_cdft.solve(steps=batch_tasks[0].cdft_steps if batch_tasks else 50, verbose=False)
    t_cdft_total = time.perf_counter() - t_c0
    t_cdft_per_mat = t_cdft_total / max(1, len(loaded_materials))

    cdft_profiles = batched_cdft.get_density_profiles()
    cdft_pressures = batched_cdft.get_wall_contact_pressures()
    cdft_gammas = batched_cdft.get_excess_adsorptions()
    final_cdft_loss = cdft_losses[-1] if cdft_losses else 0.0

    for local_idx, orig_idx in enumerate(task_indices):
        mat = loaded_materials[local_idx]
        rho = cdft_profiles[local_idx]
        p_w = cdft_pressures[local_idx]
        gamma = cdft_gammas[local_idx]
        dz_val = batched_cdft.dz_vals[local_idx]
        slit_w = batched_cdft.slit_widths[local_idx]

        mat_out_dir = os.path.join(batch_tasks[orig_idx].out_dir, mat.name)
        if async_writer:
            async_writer.write_npy(os.path.join(mat_out_dir, "density_profile.npy"), rho)
            z_grid = np.linspace(0.5 * dz_val, slit_w - 0.5 * dz_val, batched_cdft.n_grid)
            async_writer.write_csv(
                os.path.join(mat_out_dir, "density_profile.csv"),
                "z_angstrom,rho_a3",
                np.column_stack([z_grid, rho]),
            )
            summary_txt = (
                f"Material: {mat.name}\nDimension Mode: {mat.dimension_mode}\n"
                f"Num Sites: {mat.num_sites}\nTemperature: {mat.temperature_k:.2f} K\n"
                f"Bulk Density: {mat.bulk_density_a3:.6f} Å^-3\nBulk Pressure: {mat.bulk_pressure_bar:.4f} bar\n"
                f"Chemical Potential: {mat.bulk_mu:.4f} k_B T\nWall Contact Pressure: {p_w:.4f} bar\n"
                f"Excess Adsorption: {gamma:.6f} Å^-2\ncDFT Solver Runtime: {t_cdft_per_mat:.3f} s\n"
            )
            async_writer.write_txt(os.path.join(mat_out_dir, "cdft_summary.txt"), summary_txt)

    # Check if skip_bg
    all_skip_bg = all(t.skip_bg for t in batch_tasks)
    if all_skip_bg:
        for local_idx, orig_idx in enumerate(task_indices):
            mat = loaded_materials[local_idx]
            task = batch_tasks[orig_idx]
            mat_out_dir = os.path.join(task.out_dir, mat.name)
            results_map[orig_idx] = MaterialPipelineResult(
                material_name=mat.name,
                status=PipelineStatus.SUCCESS_CDFT_ONLY.value,
                runtime_seconds=time.perf_counter() - t_start,
                cdft_runtime_seconds=t_cdft_per_mat,
                num_sites=mat.num_sites,
                temperature_k=mat.temperature_k,
                bulk_density_a3=mat.bulk_density_a3,
                chemical_potential_kbt=mat.bulk_mu,
                bulk_pressure_bar=mat.bulk_pressure_bar,
                wall_pressure_bar=cdft_pressures[local_idx],
                excess_adsorption_a2=cdft_gammas[local_idx],
                cdft_final_loss=final_cdft_loss,
                solvation_free_energy_kcal_mol=getattr(mat, "solvation_free_energy_kcal_mol", 0.0),
                artifact_dir=mat_out_dir,
            )
        return [results_map[i] for i in range(len(batch_tasks))]

    # 2. Batched Boltzmann Generator Phase
    t_bg_start = time.perf_counter()
    bg_steps = batch_tasks[0].bg_steps if batch_tasks else 30
    bg_samples = batch_tasks[0].bg_samples if batch_tasks else 32

    flow = Base2CartesianFlow(n_atoms=128, n_layers=4, hidden_dim=64)
    generator = BoltzmannGenerator(
        flow=flow,
        energy_fn=energy_fn,
        prior=None,
        batch_size=batch_size,
    )

    bg_losses = generator.train(steps=bg_steps, batch_size=batch_size, verbose=False)
    bg_loss = bg_losses[-1] if bg_losses else 0.0

    n_samp_batches = max(1, math.ceil(bg_samples / batch_size))
    all_batch_samples = []
    for _ in range(n_samp_batches):
        samp = generator.sample(n_samples=batch_size, return_all_pad=True)
        all_batch_samples.append(samp.numpy())

    stacked_samples = np.concatenate(all_batch_samples, axis=0)
    t_bg = time.perf_counter() - t_bg_start

    # 3. Extract Per-Material Trajectories and Dispatch Async Writes
    state_dict = nn.state.get_state_dict(flow)
    np_weights = {k: v.numpy() for k, v in state_dict.items()}

    for local_idx, orig_idx in enumerate(task_indices):
        mat = loaded_materials[local_idx]
        task = batch_tasks[orig_idx]
        mat_out_dir = os.path.join(task.out_dir, mat.name)
        site_names = [s.site_name for s in mat.sites] if mat.sites else [mat.name]

        mat_coords = stacked_samples[:bg_samples, : mat.num_sites, :]

        if async_writer:
            async_writer.write_xyz(
                path=os.path.join(mat_out_dir, "trajectory.xyz"),
                coords=mat_coords,
                site_names=site_names,
                material_name=mat.name,
            )
            async_writer.write_npz(
                path=os.path.join(mat_out_dir, "flow_weights.npz"),
                np_dict=np_weights,
            )

        results_map[orig_idx] = MaterialPipelineResult(
            material_name=mat.name,
            status=PipelineStatus.SUCCESS.value,
            runtime_seconds=time.perf_counter() - t_start,
            cdft_runtime_seconds=t_cdft_per_mat,
            bg_runtime_seconds=t_bg,
            num_sites=mat.num_sites,
            temperature_k=mat.temperature_k,
            bulk_density_a3=mat.bulk_density_a3,
            chemical_potential_kbt=mat.bulk_mu,
            bulk_pressure_bar=mat.bulk_pressure_bar,
            wall_pressure_bar=cdft_pressures[local_idx],
            excess_adsorption_a2=cdft_gammas[local_idx],
            cdft_final_loss=final_cdft_loss,
            bg_final_loss=bg_loss,
            solvation_free_energy_kcal_mol=getattr(mat, "solvation_free_energy_kcal_mol", 0.0),
            artifact_dir=mat_out_dir,
        )

    return [results_map[i] for i in range(len(batch_tasks))]


def process_batched_materials(
    batch_tasks: List[MaterialPipelineTask],
    batch_size: int = 512,
    async_writer: Optional[AsyncArtifactWriter] = None,
) -> List[MaterialPipelineResult]:
    """Convenience synchronous wrapper that prepares and executes a single batch chunk."""
    t_start = time.perf_counter()
    loaded_materials: List[Material] = []
    task_indices: List[int] = []
    results_map: Dict[int, MaterialPipelineResult] = {}

    for idx, task in enumerate(batch_tasks):
        mat_input = task.material_path_or_name
        mat_basename = Path(mat_input).stem if os.path.exists(mat_input) or "/" in mat_input else str(mat_input)
        mat_out_dir = os.path.join(task.out_dir, mat_basename)
        os.makedirs(mat_out_dir, exist_ok=True)

        try:
            mat = MaterialLoader.load_material(
                material_name_or_path=task.material_path_or_name,
                temperature_k=task.temperature_k,
                bulk_density_a3=task.bulk_density_a3,
                pressure_bar=task.pressure_bar,
                chemical_potential_kbt=task.chemical_potential_kbt,
            )
            loaded_materials.append(mat)
            task_indices.append(idx)
        except Exception as e:
            status = (
                PipelineStatus.SKIPPED_THERMO
                if "spinodal" in str(e).lower() or "density" in str(e).lower()
                else PipelineStatus.FAILED_ERROR
            )
            results_map[idx] = MaterialPipelineResult(
                material_name=mat_basename,
                status=status.value,
                error_message=f"Thermodynamic routing failed: {str(e)}",
                runtime_seconds=time.perf_counter() - t_start,
                artifact_dir=mat_out_dir,
            )

    if not loaded_materials:
        return [results_map[i] for i in range(len(batch_tasks))]

    mol_batch = MolecularBatch.create_batch(
        materials=loaded_materials,
        batch_size=batch_size,
        target_n_particles=128,
    )
    batched_cdft = BatchedTinyCDFT(
        batch=mol_batch,
        n_grid=128,
        learning_rate=batch_tasks[0].cdft_lr if batch_tasks else 0.02,
    )
    energy_fn = MicroscopicEnergy(material=mol_batch, pad_to_128=True)

    prepared = PreparedMolecularBatch(
        tasks=batch_tasks,
        batch_size=batch_size,
        loaded_materials=loaded_materials,
        task_indices=task_indices,
        results_map=results_map,
        mol_batch=mol_batch,
        batched_cdft=batched_cdft,
        energy_fn=energy_fn,
        t_assembly_start=t_start,
    )
    return execute_prepared_batch(prepared, async_writer=async_writer)
