# dens-city: Stage 1 Multi-Objective RL Molecular Swarm & High-Throughput cDFT Boltzmann Platform

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)
[![PufferLib: >=0.4.0](https://img.shields.io/badge/PufferLib-4.0-red.svg)](https://github.com/PufferAI/PufferLib)
[![PyTorch: >=2.0.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

`dens-city` is a statistical mechanics molecular design and simulation platform. It couples a **Stage 1 Multi-Objective Reinforcement Learning Molecular Swarm** (`PufferLib` 4.0 + parallel C-FFI engines) with **Stage 2 High-Throughput Variational Classical Density Functional Theory (cDFT)** and **Boltzmann Generator Normalizing Flows** in pure `tinygrad`.

---

## 1. End-to-End Molecular Discovery Pipeline

```
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                    STAGE 1: MULTI-OBJECTIVE REINFORCEMENT LEARNING SWARM                         │
 │                                (Billions -> Millions)                                            │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Target Material Specification YAMLs (tests/data/)                                                │
 │ • OLED Semiconductors  • Battery Electrolytes  • Drug Inhibitors  • Sponges  • Toughness Resins  │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ PufferLib 4.0 Molecular Swarm Policy (PyTorch)                                                   │
 │ • MolecularPortEncoder: Geometric entity encoder over 16 3D port orientation vectors             │
 │ • MolecularActionDecoder: MultiDiscrete [16, 13] heads with -1e9 invalid action masking          │
 │ • Modular Backbones: MLPBackbone & recurrent MinGRUBackbone                                      │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Vectorized Parallel C-FFI Swarm Engine (ocean/cdft_swarm/) [> 1,500 SPS]                         │
 │ • SE(3) Rigid-Body Conformer Assembly & Non-Bonded Steric Collision Detection                   │
 │ • Hyper-Fast C-cDFT Solver (Picard density relaxation, pore wetting P_wall, solvation DeltaOmega)│
 │ • Exact Microscopic Mechanics (Jacobi PMI linearity, aromatic density, FFV, sacrificial H-bonds) │
 │ • 3-Stage Curriculum Scheduler with direct C-Memory TargetSpec ctypes broadcast                  │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ PufferLib Constellation 3D Sweep Exporter & Pareto-Optimal Candidate .mol2 Output                │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                 STAGE 2: HIGH-THROUGHPUT cDFT & BOLTZMANN GENERATIVE SCREENING                   │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ AsyncBatchPrefetcher & ProcessPool (B=512)                                                       │
 │ • Multi-process CPU regex, GAFF & EOS root-finding                                               │
 │ • Pre-computed 1D NumPy FMT/WCA planar kernels                                                   │
 │ • Double-buffered prefetch queue (0.000s GPU wait)                                               │
 │ • Uniform 128-site tensor padding (B=512)                                                        │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
        ┌─────────────────────────────┐                       ┌─────────────────────────────┐
        │   Batched TinyCDFT (B=512)  │                       │  Batched Microscopic Energy │
        │  rho(z) = rho_bulk exp(psi) │                       │  Pairwise LJ 12-6 (SF)      │
        │  Rosenfeld FMT Hard-Sphere  │                       │  Coulomb Electrostatics     │
        │  WCA Attractive Dispersion  │                       │  Steele 9-3 Wall Potential  │
        │  Grouped conv2d in 1 JIT    │                       │  Noé Energy Regularization  │
        └──────────────┬──────────────┘                       └──────────────┬──────────────┘
                       │                                                     │
                       └──────────────────────────┬──────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Batched Boltzmann Generator (tinygrad)                                                           │
 │ • 4-Channel Base-2 Cartesian Flow (RealNVP Bijectors)                                            │
 │ • Invertible Z-Matrix coordinate transformation                                                  │
 │ • Reverse KL Divergence training in 1 JIT graph                                                  │
 │ • Latent MCMC equilibrium relaxation                                                             │
 └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Asynchronous Export & Single-Material 3D Raylib Visualization                                    │
 │ • Non-blocking AsyncArtifactWriter (.xyz, .npy, .npz)                                            │
 │ • High-performance 3D Raylib Interactive Visualizer (Single material inspection)                │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Microscopic Hamiltonian & Boltzmann Generator Flow

- **Shifted-Force (SF) Pairwise Energy**: Pairwise Lennard-Jones 12-6 and Coulomb electrostatics with Minimum Image Convention in $X/Y$ and exact boundary continuity at $r_{\rm cut}$.
- **Steele 9-3 Confinement**: Impenetrable confining planar walls in $Z$ with asymptotic hard boundaries.
- **Frank Noé Energy Regularization**: Soft logarithmic ceiling for overlapping atom pairs ($E \ge E_{\rm high}$) to prevent numerical gradient explosions.
- **Base-2 Cartesian Flow**: Invertible coordinate bijectors combining RealNVP affine coupling stacks with analytical Z-matrix internal coordinate transformations ($r, \theta, \phi$).

---

## 3. High-Throughput Tensor Batching & Decoupled Async Prefetching

`dens-city` implements a decoupled, double-buffered batch prefetch pipeline where target molecules are pre-parsed and stacked along Axis 0 into fixed batches of size $B=512$ (`MolecularBatch`).

- **Multi-Process CPU Assembly**: `AsyncBatchPrefetcher` uses `ProcessPoolExecutor` to parse `.mol2` files, resolve force field parameters, solve Equations of State, and precompute 1D NumPy FMT kernels concurrently across all physical CPU cores, completely bypassing the Python GIL.
- **Zero GPU Starvation**: Background double-buffering pre-assembles Batch $k+1$ in host memory while Batch $k$ is executing on the GPU, achieving **0.0000 s queue wait time** between batches.
- **Permanent JIT Cache Reuse**: Physical parameters $(\sigma, \epsilon, q, \beta, \text{mask})$ and conditioning vectors $(\sigma_{\rm eff}, \epsilon_{\rm eff}, T, \rho_{\rm bulk}, \mu)$ are passed as static device Tensors. This prevents graph recompilations across multiple runs.
- **Axis-0 Vectorization & Grouped Convolutions**: Batched cDFT and Boltzmann flow transformations evaluate up to 512 molecules simultaneously in a single forward/backward compilation pass.
- **Asynchronous Disk I/O**: `AsyncArtifactWriter` offloads file serialization (`.xyz` trajectories, `.npy` profiles, `.npz` model weights, `.csv` tables) to a background thread to prevent halting device execution.

---

## 4. Quickstart & CLI Usage

### Installation
Sync dependencies with `uv`:
```bash
uv sync
```

### Stage 1: RL Molecular Swarm Training
Train the PufferLib 4.0 policy on arbitrary material YAML specifications with 3-stage curriculum broadcasting:
```bash
# Train on Conjugated OLED Semiconductors
uv run python scripts/train_swarm.py --spec tests/data/conjugated_oled_semiconductors.yaml --num-envs 16 --total-timesteps 50000 --export-dir runs/candidates

# Train on Fluorinated Battery Electrolytes
uv run python scripts/train_swarm.py --spec tests/data/fluorinated_battery_electrolytes.yaml --num-envs 16 --total-timesteps 50000 --export-dir runs/candidates

# Train with recurrent MinGRU backbone
uv run python scripts/train_swarm.py --spec tests/data/ultra_lightweight_aliphatic_sponges.yaml --num-envs 16 --total-timesteps 50000 --recurrent
```

### Constellation Curriculum Sweeps
Run hyperparameter sweeps and serialize trial trajectories directly into the PufferLib Constellation schema:
```bash
# Launch multi-trial curriculum sweep across all material specifications
uv run python scripts/run_curriculum_sweep.py --num-trials-per-spec 3 --steps-per-trial 10000 --num-envs 8 --output-dir runs/constellation_sweeps
```

### 3D Interactive Raylib Visualizer (Single Material)
> [!NOTE]
> The interactive 3D visualizer is designed for detailed inspection of a **single material** at a time. To benchmark or screen multiple materials simultaneously, use the batch pipeline below.

Launch the real-time single-material visualizer with dynamic vdW surfaces, real-time cDFT density profile graphing, and Boltzmann Generator discovery:
```bash
# Launch interactive visualizer for a single material
uv run dens-city --interactive --materials argon
uv run dens-city --interactive --materials water
uv run dens-city --interactive --materials benzene
```

### High-Throughput Batch Pipeline (Multiple Materials)
Run the high-throughput coupled cDFT + Boltzmann Generator batch screening pipeline:
```bash
# Multi-material batch screening (default batch size 512)
uv run dens-city --materials argon water methane 5cb --batch-size 512

# Full 674-material high-throughput benchmark across FreeSolv (< 30 seconds)
uv run dens-city --materials all --benchmark

# Fast cDFT screening only (skips generative flow)
uv run dens-city --materials all --skip-bg

# Debug mode with compiler execution traces
uv run dens-city --materials argon benzene --debug
```

### 1-Line End-to-End Simulation & FreeSolv Verification
To automatically initialize test data if missing, run the complete 674-molecule end-to-end benchmark, and cross-reference thermodynamic observables against experimental and calculated FreeSolv hydration energies in a single command:
```bash
uv run python scripts/verify_e2e_against_freesolv.py --run-e2e --all
```
To verify the most recent simulation run without re-running the full benchmark:
```bash
uv run python scripts/verify_e2e_against_freesolv.py
```

---

## 5. Notes on Long-Range Forces

### Why GCMC Chokes on Long-Range Forces
Grand Canonical Monte Carlo (GCMC) simulates discrete particles through stochastic atom insertions, deletions, and displacements:
- **Reciprocal-Space Recalculation**: In Ewald summation (or Particle-Mesh Ewald), every particle insertion or deletion alters the global structure factor $\sum_j q_j e^{i \mathbf{k} \cdot \mathbf{r}_j}$. Updates across thousands of trial moves per second create a massive computational bottleneck.
- **Neutrality Violations**: Insertion of an isolated charged ion breaks electroneutrality in the box, which requires artificial background neutralizing plasma or fractional insertion schemes.
- **The Overlap Wall**: Insertion of a full molecule with Lennard-Jones cores and partial charges into a dense polar fluid (such as liquid water) suffers a $>99.9\%$ rejection rate, which demands millions of failed trial steps for a handful of accepted configurations.

### Why cDFT Solves This for Free
In cDFT, there are no particles, no trial moves, and no discrete insertions. The system contains only a continuous, smooth charge density field:

$$
\rho_q(\mathbf{r}) = \sum_i q_i \rho_i(\mathbf{r})
$$

The long-range electrostatic energy is the double integral:

$$
\mathcal{F}_{\text{coul}}[\rho] = \frac{1}{2} \iint \frac{\rho_q(\mathbf{r}) \rho_q(\mathbf{r}')}{4\pi \varepsilon_0 \varepsilon_r |\mathbf{r} - \mathbf{r}'|} \, d\mathbf{r} \, d\mathbf{r}'
$$

Instead of pairwise sums over periodic images, this integral is mathematically identical to the classical Poisson equation:

$$
\nabla^2 \phi(\mathbf{r}) = -\frac{\rho_q(\mathbf{r})}{\varepsilon_0 \varepsilon_r}
$$

### Map to tinygrad
- **1D Slit Pores**: A planar sheet of charge has a constant electric field. The 1D Green's function is $v_C(z) = -2\pi |z|$, which reduces to a 1D convolution or direct cumulative integral across the grid tensor.
- **3D Grids**: Poisson solves in a single step in Fourier space. In $k$-space, the Laplacian $\nabla^2$ becomes $-k^2$:

$$
\tilde{\phi}(\mathbf{k}) = \frac{4\pi}{\varepsilon_0 \varepsilon_r k^2} \tilde{\rho}_q(\mathbf{k})
$$

A forward 3D FFT, element-wise vector division by $k^2$, and an inverse 3D FFT solve the exact, infinite long-range field across the full periodic box in milliseconds on GPU.

---

## 6. Automated Tests & Code Quality

```bash
# Run complete test suite (133 tests)
uv run pytest tests/ -v

# Run linting and code formatting checks
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

---

## 7. Citations

- A. T. Bui, S. J. Cox, "Dielectrocapillarity for exquisite control of fluids", *arXiv:2503.09855* (2025).
- A. T. Bui, S. J. Cox, "Learning classical density functionals for ionic fluids", *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)
- A. T. Bui, S. J. Cox, "Ab initio classical density functional theory with neural functionals", *arXiv:2603.20493* (2026).
- J. Yang, R. Pan, J. Sun, J. Wu, "High-Dimensional Operator Learning for Molecular Density Functional Theory", *arXiv:2411.03698* (2024). [doi:10.48550/arxiv.2411.03698](https://doi.org/10.48550/arxiv.2411.03698)
- R. Roth, "Fundamental measure theory for hard-sphere mixtures: a review", *Journal of Physics: Condensed Matter* **22**, 063102 (2010). [doi:10.1088/0953-8984/22/6/063102](https://doi.org/10.1088/0953-8984/22/6/063102)
- F. Noé, S. Olsson, J. Köhler, H. Wu, "Boltzmann Generators – Sampling Equilibrium States of Many-Body Systems with Deep Learning", *arXiv:1812.01729* (2018). [doi:10.48550/arxiv.1812.01729](https://doi.org/10.48550/arxiv.1812.01729)

---

## 8. License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
