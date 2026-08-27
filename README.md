# dens-city

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)
[![PufferLib: >=0.4.0](https://img.shields.io/badge/PufferLib-4.0-red.svg)](https://github.com/PufferAI/PufferLib)
[![PyTorch: >=2.0.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

`dens-city` is a molecular design and simulation platform coupling a **Stage 1 Multi-Objective RL Swarm** (`PufferLib` 4.0 + parallel C-FFI physics) with **Stage 2 High-Throughput Variational Classical Density Functional Theory (cDFT)** and **Boltzmann Generator Normalizing Flows** in `tinygrad`.

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
[Pareto-Optimal Candidate .mol2 Export / Constellation 3D Sweeps]
         │
         ▼
[Stage 2: Batched cDFT & Boltzmann Generators] (tinygrad, B=512)
  • Rosenfeld FMT Hard-Sphere Excess + WCA Attractive Dispersion + Poisson Electrostatics
  • Base-2 Cartesian Flow (RealNVP Bijectors + Z-Matrix) & Latent MCMC
         │
         ▼
[Single-Material Raylib 3D Viewer / Batch Artifact Export (.xyz, .npy)]
```

---

## Quickstart

### 1. Installation
```bash
uv sync
```

### 2. Stage 1: RL Molecular Swarm Training
```bash
# Train on OLED semiconductors with curriculum scheduling
uv run python scripts/train_swarm.py --spec tests/data/conjugated_oled_semiconductors.yaml --num-envs 16 --total-timesteps 50000

# Multi-trial hyperparameter sweep for Constellation 3D viewer
uv run python scripts/run_curriculum_sweep.py --num-trials-per-spec 3 --steps-per-trial 10000
```

### 3. Stage 2: High-Throughput Batch Screening
```bash
# High-throughput batch run (default batch size 512)
uv run dens-city --materials argon water methane 5cb --batch-size 512

# Full 674-material FreeSolv benchmark
uv run dens-city --materials all --benchmark
```

### 4. Interactive 3D Viewer (Single Material)
```bash
# Launch real-time Raylib visualizer for a single material
uv run dens-city --interactive --materials argon
```

### 5. Automated Tests & Quality
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

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
