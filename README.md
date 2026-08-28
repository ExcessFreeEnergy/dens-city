# dens-city

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)
[![PufferLib: >=0.4.0](https://img.shields.io/badge/PufferLib-4.0-red.svg)](https://github.com/PufferAI/PufferLib)
[![PyTorch: >=2.0.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

`dens-city` is a molecular design and simulation platform coupling a **Stage 1 Multi-Objective RL Swarm** (`PufferLib` 4.0 + parallel C-FFI physics) with **Stage 2 In-Memory Streaming**, **Stage 3 Batched Variational Classical Density Functional Theory (cDFT)** & **Boltzmann Generator Normalizing Flows**, and **Stage 4 JIT-Compiled Equivariant Graph Neural Network (EGNN) Quantum Surrogate Screening** in `tinygrad`.

---

## 1. System Architecture

```
[Target Material YAMLs] (OLED, Electrolytes, Inhibitors, Sponges, Resins)
         │
         ▼
[Stage 1: Multi-Objective RL Swarm] (>1,500 SPS via PufferLib 4.0 + C-FFI)
  • Geometric 3D Port Entity Encoder & Action-Masked MultiDiscrete Decoder
  • SE(3) Rigid Assembly, Picard C-cDFT Solver, Exact Jacobi Mechanics
  • 3-Stage Curriculum Scheduler with C-memory TargetSpec ctypes broadcast
         │
         ▼
[Stage 2: In-Memory Contiguous Streaming & Zero-Copy GPU Ingestion]
  • Direct C-FFI array export without disk I/O or GAFF recalculation thrashing
         │
         ▼
[Stage 3: Batched cDFT + JIT-Compiled L-BFGS + Boltzmann Generators] (tinygrad, B=512)
  • TinyJit-Accelerated Batched L-BFGS Quasi-Newton GPU Geometry Relaxation (m=6, autograd forces)
  • Soft-Core Shifted-Force LJ + Damped Reaction-Field Coulomb Electrostatics
  • Base-2 Cartesian Flow (RealNVP Bijectors + Z-Matrix) & Latent MCMC
         │
         ▼
[Stage 4: TinyJit 7-Layer E(n)-Equivariant Graph Neural Network (EGNN)] (tinygrad, B=32)
  • DFT-Level Quantum MLFF internal energy U_egnn(x) and conservative forces F = -∇_x U
  • Single-pass GPU graph compilation for quantum force RMS stability screening
         │
         ▼
[Stage 5: Multi-Objective Pareto Frontier Ranking & Export (.mol2, .csv, .md)]
```

---

## 2. CLI Reference & Parameter Guide (`uv run dens-city`)

The `dens-city` command-line interface provides a single, unified entrypoint for all simulation, visualization, generative design, library generation, and benchmark modes.

```bash
uv run dens-city [MODE_SELECTOR] [OPTIONS...]
```

### 2.1 Execution Modes

| Mode Selector Flag | Description | Typical Use Case |
| :--- | :--- | :--- |
| *(None / Default)* | **Coupled cDFT + Boltzmann Batch Screening** | Screen `.mol2` files through cDFT thermodynamics & Boltzmann flows |
| `--interactive`, `-i` | **3D Interactive Raylib Visualizer** | Real-time orbital 3D viewer for molecular conformations |
| `--funnel`, `--run-funnel` | **5-Stage Generative Molecular Funnel** | Run end-to-end RL design $\to$ cDFT $\to$ Boltzmann $\to$ EGNN $\to$ Pareto export |
| `--benchmark-specs`, `--all-specs` | **Cross-Spec Funnel Benchmark** | Run the 5-stage funnel across all 10 material classes in `tests/data/` |
| `--train-swarm`, `--train-rl` | **Stage 1 RL Swarm Training** | Train PPO policy with curriculum learning and export `trained_policy.pt` |
| `--sweep`, `--curriculum-sweep` | **Constellation Curriculum Sweeps** | Run multi-trial hyperparameter sweep for Constellation 3D viewer |
| `--eval-swarm`, `--evaluate-specs`| **Spec Evaluation & Chemical Diagnostics** | Evaluate validity %, SA score, diversity (1-T), and unique SMILES % |
| `--generate-library`, `--gen-library` | **Combinatorial Library Generator** | High-speed C / multi-core 2D/3D combinatorial molecular generation |
| `--populate-test-data` | **Dataset & Benchmark Population** | Populate `data/test_data/` with benchmark `.mol2` files & force fields |
| `--verify-freesolv`, `--verify-e2e` | **FreeSolv Statistical Validation** | Compare predictions against FreeSolv hydration database & build report |

---

### 2.2 Parameter Options Reference

#### Material & Input Configuration
- `--materials`, `-m` : Target material names (e.g. `argon water 5cb`), path to `.mol2` files, or `'all'` (default: `argon`).
- `--data-dir`, `-d` : Directory containing `.mol2` material files (default: `data/test_data`).
- `--out-dir`, `-o` : Destination directory for structured artifacts, trajectories, summaries, and logs.
- `--spec`, `-s` : Path or keyword for material specification YAML (e.g. `tests/data/conjugated_oled_semiconductors.yaml` or `'oled'`).
- `--specs` : List of YAML specification files for curriculum sweeps.
- `--specs-dir` : Directory containing material specification YAML files (default: `tests/data`).

#### Thermodynamics & cDFT Solver Options
- `--temp`, `-t` : Reservoir temperature in Kelvin (default: `300.0` K).
- `--pressure`, `-p` : Reservoir pressure in bar (default: `1.0` bar).
- `--mu` : Chemical potential in $k_B T$ (overrides bulk pressure calculation).
- `--grid`, `-g` : Spatial grid points for cDFT 1D discretization (default: `128`).
- `--cdft-steps` : Variational cDFT optimization steps (default: `60`).
- `--cdft-lr` : cDFT solver learning rate (default: `0.02`).
- `--skip-bg` : Halt immediately after cDFT screening, skipping Boltzmann Generator training.

#### Boltzmann Generator & Geometry Relaxation Options
- `--bg-steps` : Boltzmann Generator training iterations (default: `40`).
- `--bg-lr` : Boltzmann Generator learning rate (default: `0.01`).
- `--bg-samples` : Number of 3D equilibrium conformations sampled into `.xyz` trajectory (default: `100`).
- `--bg-w-tor` : Torsional rotamer loss biasing weight (default: `0.0`).
- `--bg-mcmc-steps` : Latent Metropolis Monte Carlo relaxation steps per sample (default: `0`).
- `--bg-mcmc-step-size` : Step size for Gaussian latent perturbations (default: `0.1`).
- `--lbfgs-steps` : Batched GPU L-BFGS Quasi-Newton geometry relaxation steps (default: `50`).
- `--lbfgs-tol` : RMS force convergence threshold for L-BFGS (default: `1e-3`).

#### Quantum MLFF & EGNN Options
- `--energy-engine` : Microscopic Hamiltonian physics engine: `'classical'` (GAFF LJ + Coulomb, default) or `'egnn'` (7-layer $E(n)$-equivariant MLFF).
- `--enable-egnn` / `--no-enable-egnn` : Enable/disable Stage 4 EGNN quantum surrogate screening (default: `True`).
- `--egnn-batch-size` : GPU batch size for EGNN message-passing evaluation (default: `32`).
- `--egnn-layers` : Number of message-passing layers in the EGNN architecture (default: `7`).
- `--egnn-weights` : Optional path to pretrained EGNN weights `.npz` archive.

#### RL Swarm & Generative Funnel Options
- `--train-steps`, `--total-timesteps` : RL curriculum training timesteps (default: `5,000,000`).
- `--num-candidates` : Number of candidates to sample from trained policy via C-FFI (default: `512`).
- `--top-k` : Number of top Pareto-optimal candidates to export (default: `20`).
- `--checkpoint` : Path to existing policy `.pt` checkpoint file to skip Stage 1 training.
- `--num-envs` : Number of parallel C-FFI environment workers (default: `16`).
- `--horizon` : Rollout horizon per environment (default: `16`).
- `--learning-rate`, `--lr` : PPO learning rate (default: `3e-4`).
- `--hidden-size` : Policy latent dimension (default: `256`).
- `--recurrent` : Enable recurrent MinGRU backbone instead of MLP (default: `False`).
- `--early-stopping-lookback` : Step lookback window for EMA reward flatline detection (default: `500,000`).
- `--early-stopping-delta` : EMA reward threshold for early stopping (default: `0.01`).
- `--no-early-stopping` : Disable dynamic EMA early stopping.
- `--no-curriculum` : Disable 3-stage curriculum scheduler.
- `--no-sa-penalty` : Disable in-the-loop batch SA score penalty.
- `--sa-threshold` : SA score hinge threshold above which penalty is applied.
- `--sa-penalty-slope` : Slope multiplier for SA score excess penalty.
- `--no-dynamic-entropy` : Disable molecular-weight-scaling dynamic entropy coefficient.
- `--max-sa-score` : Maximum allowable Synthetic Accessibility (SA) Score ceiling (default: `6.0`).
- `--disable-sa-filter` : Disable Stage 5 synthesizability safety gate.
- `--checkpoint-dir` : Directory to save `trained_policy.pt` (default: `runs/checkpoints`).
- `--export-dir` : Directory to save candidate `.mol2` files (default: `runs/candidates`).

#### Combinatorial Library Generator Options
- `--target-count`, `-n` : Target number of unique molecules to generate (default: from YAML `target_molecules`).
- `--skip-3d` : Skip 3D conformer embedding (2D combinatorial generation only).
- `--skip-write` : Skip writing `.mol2` files to disk (in-memory benchmark mode).
- `--seed` : Random seed for deterministic sampling (default: `42`).

#### FreeSolv & Dataset Options
- `--all-freesolv`, `--all-data` : Populate entire FreeSolv database (642+ molecules) into `data/test_data/`.
- `--run-e2e` : Run end-to-end simulation across materials before verifying against FreeSolv.
- `--results-dir` : Directory containing `pipeline_summary.jsonl` (defaults to latest in `runs/`).
- `--database` : Path to FreeSolv `database.pickle` (default: `FreeSolv/database.pickle`).
- `--report-out` : Destination path for FreeSolv verification report (default: `data/e2e_freesolv_verification_report.md`).

#### Execution & Compiler Performance
- `--batch-size`, `-b` : Molecule batch size for parallel tensor evaluation (default: `512`).
- `--workers`, `-w` : Concurrent worker processes (default: `min(4, CPU_COUNT)`).
- `--timeout` : Maximum execution timeout per material in seconds (default: `180s`).
- `--beam` : tinygrad compiler BEAM search optimization level (default: `2`).
- `--benchmark` : Profile execution time and output comprehensive throughput table.
- `--debug` : Enable `DEBUG=2` and write per-material compiler logs to `data/logs_<timestamp>/`.

---

## 3. Practical Usage Examples

### 1. High-Throughput Batch Screening
```bash
# Screen 4 benchmark fluids with coupled cDFT + Boltzmann Generator (batch size 512)
uv run dens-city --materials argon water methane 5cb --batch-size 512

# Run full FreeSolv benchmark with BEAM=2 compiler optimization
uv run dens-city --materials all --benchmark --beam 2

# Fast cDFT screening only (skipping Boltzmann training)
uv run dens-city --materials all --skip-bg
```

### 2. 3D Interactive Raylib Visualizer
```bash
# Launch interactive 3D visualizer for argon
uv run dens-city --interactive --materials argon

# View multiple molecules (cycle using Left/Right arrow keys, orbit with mouse drag)
uv run dens-city --interactive --materials water benzene 5cb
```

### 3. 5-Stage Generative Molecular Funnel
```bash
# Run 5-stage generative funnel on OLED semiconductors (25k training steps, 512 candidates)
uv run dens-city --funnel --spec tests/data/conjugated_oled_semiconductors.yaml \
  --train-steps 25000 --num-candidates 512 --batch-size 512 --top-k 20

# Run generative funnel on battery electrolytes with keyword resolution and existing checkpoint
uv run dens-city --funnel --spec electrolytes \
  --checkpoint runs/checkpoints/trained_policy.pt --num-candidates 512 --top-k 20
```

### 4. Cross-Material Multi-Specification Funnel Benchmark
```bash
# Execute 5-stage funnel across all 10 material classes in tests/data/
uv run dens-city --benchmark-specs --train-steps 25000 --num-candidates 64 --batch-size 64
```

### 5. Stage 1 RL Swarm Training
```bash
# Train PPO policy with curriculum learning for 5M steps on 16 parallel C environments
uv run dens-city --train-swarm --spec conjugated_oled_semiconductors \
  --train-steps 5000000 --num-envs 16 --checkpoint-dir runs/checkpoints
```

### 6. Constellation Curriculum Hyperparameter Sweeps
```bash
# Run multi-trial hyperparameter sweep for Constellation 3D viewer
uv run dens-city --sweep --num-trials-per-spec 3 --steps-per-trial 10000
```

### 7. Chemical Diversity & Synthesizability Diagnostics
```bash
# Evaluate chemical validity, SA score, and Tanimoto diversity across all specs
uv run dens-city --eval-swarm --specs-dir tests/data --timesteps 10000 --num-candidates 50
```

### 8. High-Performance Combinatorial Molecular Library Generation
```bash
# Generate 50,000 unique 3D molecules with OpenMP C .mol2 export and GAFF database
uv run dens-city --generate-library --spec tests/data/conjugated_oled_semiconductors.yaml --target-count 50000

# Fast in-memory 2D combinatorial generation benchmark
uv run dens-city --generate-library --spec fluorinated_battery_electrolytes --target-count 10000 --skip-3d --skip-write
```

### 9. Test Data & FreeSolv Dataset Population
```bash
# Populate 32 core benchmark fluids in data/test_data/
uv run dens-city --populate-test-data

# Extract and populate all 642+ FreeSolv molecules
uv run dens-city --populate-test-data --all-freesolv
```

### 10. FreeSolv Statistical Verification & Validation Report
```bash
# Run full simulation and verify against FreeSolv hydration free energies
uv run dens-city --verify-freesolv --run-e2e

# Verify existing simulation results in runs/ directory
uv run dens-city --verify-freesolv --results-dir runs/batch_20260828
```

---

## 4. Automated Tests & Quality Assurance

```bash
# Run complete test suite
uv run pytest tests/ -v

# Run linting and code formatting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

---

## 5. Citations

- A. T. Bui, S. J. Cox, "Dielectrocapillarity for exquisite control of fluids", *arXiv:2503.09855* (2025).
- A. T. Bui, S. J. Cox, "Learning classical density functionals for ionic fluids", *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)
- A. T. Bui, S. J. Cox, "Ab initio classical density functional theory with neural functionals", *arXiv:2603.20493* (2026).
- J. Yang, R. Pan, J. Sun, J. Wu, "High-Dimensional Operator Learning for Molecular Density Functional Theory", *arXiv:2411.03698* (2024). [doi:10.48550/arxiv.2411.03698](https://doi.org/10.48550/arxiv.2411.03698)
- R. Roth, "Fundamental measure theory for hard-sphere mixtures: a review", *Journal of Physics: Condensed Matter* **22**, 063102 (2010). [doi:10.1088/0953-8984/22/6/063102](https://doi.org/10.1088/0953-8984/22/6/063102)
- F. Noé, S. Olsson, J. Köhler, H. Wu, "Boltzmann Generators – Sampling Equilibrium States of Many-Body Systems with Deep Learning", *arXiv:1812.01729* (2018). [doi:10.48550/arxiv.1812.01729](https://doi.org/10.48550/arxiv.1812.01729)
- V. G. Satorras, E. Hoogeboom, M. Welling, "E(n) Equivariant Graph Neural Networks", *ICML* (2021). [doi:10.48550/arXiv.2102.09844](https://doi.org/10.48550/arXiv.2102.09844)

---

## 6. License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
