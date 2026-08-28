# dens-city

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)
[![PufferLib: >=0.4.0](https://img.shields.io/badge/PufferLib-4.0-red.svg)](https://github.com/PufferAI/PufferLib)
[![PyTorch: >=2.0.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

`dens-city` is a molecular design and simulation platform coupling a **Stage 1 Multi-Objective RL Swarm** (`PufferLib` 4.0 + parallel C-FFI physics) with **Stage 2 In-Memory Streaming**, **Stage 3 Batched Variational Classical Density Functional Theory (cDFT)** & **Boltzmann Generator Normalizing Flows**, and **Stage 4 JIT-Compiled Equivariant Graph Neural Network (EGNN) Quantum Surrogate Screening** in `tinygrad`.

---

## Architecture

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

## Quickstart

### 1. Installation
```bash
uv sync
```

### 2. End-to-End Generative Molecular Funnel
```bash
# Run 5-stage generative funnel on a single material spec
uv run python scripts/run_generative_funnel.py \
  --spec tests/data/conjugated_oled_semiconductors.yaml \
  --train-steps 25000 \
  --num-candidates 512 \
  --batch-size 512 \
  --egnn-batch-size 32 \
  --top-k 20 \
  --out-dir runs/funnel_results

# Run funnel across all material classes
for spec in tests/data/*.yaml; do
  uv run python scripts/run_generative_funnel.py --spec "$spec" --train-steps 25000 --num-candidates 512 --batch-size 512 --top-k 20 --out-dir runs/funnel_results
done
```

### 3. Stage 1: RL Swarm Hyperparameter Sweeps (Constellation 3D)
```bash
# Multi-trial hyperparameter sweep for Constellation 3D viewer
uv run python scripts/run_curriculum_sweep.py --num-trials-per-spec 3 --steps-per-trial 10000
```

### 4. High-Throughput Batch Screening (FreeSolv & Database)
```bash
# High-throughput batch run (default batch size 512)
uv run dens-city --materials argon water methane 5cb --batch-size 512

# Full 674-material FreeSolv benchmark
uv run dens-city --materials all --benchmark
```

### 5. Interactive 3D Viewer (Single Material)
```bash
# Launch real-time Raylib visualizer for a single material
uv run dens-city --interactive --materials argon
```

### 6. Automated Tests & Quality
```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

---

## Citations

- A. T. Bui, S. J. Cox, "Dielectrocapillarity for exquisite control of fluids", *arXiv:2503.09855* (2025).
- A. T. Bui, S. J. Cox, "Learning classical density functionals for ionic fluids", *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)
- A. T. Bui, S. J. Cox, "Ab initio classical density functional theory with neural functionals", *arXiv:2603.20493* (2026).
- J. Yang, R. Pan, J. Sun, J. Wu, "High-Dimensional Operator Learning for Molecular Density Functional Theory", *arXiv:2411.03698* (2024). [doi:10.48550/arxiv.2411.03698](https://doi.org/10.48550/arxiv.2411.03698)
- R. Roth, "Fundamental measure theory for hard-sphere mixtures: a review", *Journal of Physics: Condensed Matter* **22**, 063102 (2010). [doi:10.1088/0953-8984/22/6/063102](https://doi.org/10.1088/0953-8984/22/6/063102)
- F. Noé, S. Olsson, J. Köhler, H. Wu, "Boltzmann Generators – Sampling Equilibrium States of Many-Body Systems with Deep Learning", *arXiv:1812.01729* (2018). [doi:10.48550/arxiv.1812.01729](https://doi.org/10.48550/arxiv.1812.01729)
- V. G. Satorras, E. Hoogeboom, M. Welling, "E(n) Equivariant Graph Neural Networks", *ICML* (2021). [doi:10.48550/arXiv.2102.09844](https://doi.org/10.48550/arXiv.2102.09844)

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

