# dens-city: Molecular Classical Density Functional Theory & Boltzmann Generative Platform in tinygrad

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)

`dens-city` bridges rigorous statistical mechanics and GPU tensor compilation in pure `tinygrad`. It combines variational Classical Density Functional Theory (cDFT) mean-field screening with Boltzmann Generator normalizing flows to sample exact 3D equilibrium molecular conformations in confined pore geometries.

---

## 1. First-Principles Physics & Generative Architecture

Equilibrium structure, pore adsorption, and 3D molecular conformations emerge strictly from first-principles statistical mechanics without hardcoded parameters or empirical fudge factors:

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Material Ingestion Stage                 │
                    │   Arbitrary .mol2 files + Force Field Parameter DB     │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │         MolecularBatch Stacking & Padding (B=32)       │
                    │  - Pad to 128 sites (sigma, epsilon, partial charges)  │
                    │  - Zero-padded dummy molecules for empty batch slots   │
                    │  - Static device tensors: Zero tinygrad JIT overhead   │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │      Variational cDFT       │                   │  Batched Microscopic Energy │
        │  rho(z) = rho_bulk exp(psi) │                   │  Pairwise LJ 12-6 (SF)      │
        │  Rosenfeld FMT Hard-Sphere  │                   │  Coulomb Electrostatics     │
        │  WCA Attractive Dispersion  │                   │  Steele 9-3 Wall Potential  │
        │  Irving-Kirkwood Virial P   │                   │  Noé Energy Regularization  │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │            Batched Boltzmann Generator                 │
                    │  - 4-Channel Base-2 Cartesian Flow (RealNVP Bijectors) │
                    │  - Invertible Z-Matrix coordinate transformation       │
                    │  - Reverse KL Divergence training in 1 JIT graph       │
                    │  - Latent MCMC equilibrium relaxation                  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │        Asynchronous Export & 3D Visualization          │
                    │  - Non-blocking AsyncArtifactWriter (.xyz, .npy, .npz) │
                    │  - High-performance 3D Raylib Interactive Visualizer   │
                    └────────────────────────────────────────────────────────┘
```

### 1.1 Variational Classical Density Functional Theory (cDFT)
Equilibrium density profiles are determined by minimizing the grand potential functional $\Omega[\psi]$ in latent potential space ($\rho(z) = \rho_{\rm bulk} \exp(\psi(z))$):
$$\Omega[\psi] = \mathcal{F}_{\rm ideal}[\psi] + \mathcal{F}_{\rm FMT}^{\rm ex}[\rho] + \mathcal{F}_{\rm att}^{\rm ex}[\rho] + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]$$

- **Ideal Gas Free Energy**: Log-free formulation eliminating $\ln(\rho)$ singularities.
- **Rosenfeld Fundamental Measure Theory (FMT)**: Hard-sphere excess functional $\Phi_{\rm FMT}(\{n_\alpha(z)\})$ with anti-aliased, analytically cell-integrated planar convolution kernels.
- **WCA / Lennard-Jones Dispersion**: Mean-field 1D integrated attractive dispersion kernel $v_{\rm att, 1D}(|z - z'|)$.
- **Thermodynamic Consistency**: State variables $(\rho_{\rm bulk}, \mu, P)$ derive dynamically from the Rosenfeld FMT / Percus-Yevick compressibility Equation of State (EOS) root solver.
- **Exact Mechanical Observables**: Wall contact pressures evaluate via exact Irving-Kirkwood momentum balance integrals:
  $$P_{\rm wall} = -\int_0^{z_{\rm bulk}} \rho(z) \frac{d V_{\rm ext}(z)}{dz} \, dz$$

### 1.2 Microscopic Hamiltonian & Boltzmann Generator Flow
- **Shifted-Force (SF) Pairwise Energy**: Pairwise Lennard-Jones 12-6 and Coulomb electrostatics with Minimum Image Convention in $X/Y$ and exact boundary continuity at $r_{\rm cut}$.
- **Steele 9-3 Confinement**: Impenetrable confining planar walls in $Z$ with asymptotic hard boundaries.
- **Frank Noé Energy Regularization**: Soft logarithmic ceiling for overlapping atom pairs ($E \ge E_{\rm high}$) to prevent numerical gradient explosions.
- **Base-2 Cartesian Flow**: Invertible coordinate bijectors combining RealNVP affine coupling stacks with analytical Z-matrix internal coordinate transformations ($r, \theta, \phi$).

---

## 2. High-Throughput Tensor Batching & JIT Compilation

`dens-city` implements a batched tensor architecture where target molecules are pre-loaded, padded to uniform 128-site grids, and stacked along Axis 0 into a fixed batch of size $B=32$ (`MolecularBatch`).

- **Permanent JIT Cache Reuse**: Physical parameters $(\sigma, \epsilon, q, \beta, \text{mask})$ and conditioning vectors $(\sigma_{\rm eff}, \epsilon_{\rm eff}, T, \rho_{\rm bulk}, \mu)$ are passed as static device Tensors. This prevents graph recompilations across multiple runs.
- **Axis-0 Vectorization**: Microscopic energies and Boltzmann flow transformations evaluate all 32 molecules simultaneously in a single forward/backward compilation pass.
- **Asynchronous Disk I/O**: `AsyncArtifactWriter` offloads file serialization (`.xyz` trajectories, `.npy` profiles, `.npz` model weights, `.csv` tables) to a background thread to prevent halting device execution.

---

## 3. Quickstart & CLI Usage

### Installation
Sync dependencies with `uv`:
```bash
uv sync
```

### 3D Interactive Raylib Visualizer
Launch the real-time molecular visualizer with dynamic vdW surfaces, real-time cDFT density profile graphing, and Boltzmann Generator discovery:
```bash
# Visualize single or multiple materials
uv run dens-city --interactive --materials argon
uv run dens-city --interactive --materials water benzene 5cb
```

### High-Throughput Batch Pipeline
Run the coupled cDFT + Boltzmann Generator execution pipeline:
```bash
# Standard batch run with default batch size 32
uv run dens-city --materials argon water methane --batch-size 32

# Full 20-material high-throughput benchmark with BEAM=2 compiler search
uv run dens-city --materials all --benchmark --beam 2

# Fast cDFT screening only (skips generative flow)
uv run dens-city --materials all --skip-bg

# Debug mode with detailed compiler execution traces
uv run dens-city --materials argon benzene --debug
```

---

## 4. Benchmark Results & FreeSolv Validation

In high-throughput benchmark runs across all 20 benchmark fluids in `test_data/`, `dens-city` completes the full coupled pipeline in **51.41 seconds** (average rate of 38.9 3D conformations/s) with **100% SUCCESS**:

### FreeSolv Cross-Reference Validation (`FreeSolv/database.pickle`)

| Material | FreeSolv ID | IUPAC Name | Sites (Real/Pad) | $\Delta G_{\rm solv}^{\rm expt}$ (kcal/mol) | $\Delta G_{\rm solv}^{\rm calc}$ (kcal/mol) | $P_{\rm wall}$ (bar) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `acetone` | `mobley_3867265` | acetone | 10 / 128 | **-3.80** | -3.51 | +1,961.95 | **SUCCESS** |
| `ammonia` | `mobley_5631798` | ammonia | 4 / 128 | **-4.29** | -4.02 | -221.27 | **SUCCESS** |
| `benzene` | `mobley_3053621` | benzene | 12 / 128 | **-0.90** | -0.81 | -273.46 | **SUCCESS** |
| `methane` | `mobley_9055303` | methane | 5 / 128 | **+2.00** | +2.45 | -1,651.30 | **SUCCESS** |
| `methanol` | `mobley_1636752` | methanol | 6 / 128 | **-5.10** | -3.49 | -1,765.79 | **SUCCESS** |
| `n_decane` | `mobley_2197088` | decane | 32 / 128 | **+3.16** | +3.33 | +32,791.55 | **SUCCESS** |
| `neopentane` | `mobley_1261349` | neopentane | 17 / 128 | **+2.51** | +2.51 | +3,056.06 | **SUCCESS** |

Run the automated FreeSolv validation report script:
```bash
uv run python scripts/verify_e2e_against_freesolv.py
```

---

## 5. Automated Tests & Code Quality

```bash
# Run complete test suite (94 tests)
uv run pytest tests/ -v

# Run linting and code formatting checks
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

---

## 6. Citations

- A. T. Bui, S. J. Cox, "Dielectrocapillarity for exquisite control of fluids", *arXiv:2503.09855* (2025).
- A. T. Bui, S. J. Cox, "Learning classical density functionals for ionic fluids", *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)
- A. T. Bui, S. J. Cox, "Ab initio classical density functional theory with neural functionals", *arXiv:2603.20493* (2026).
- J. Yang, R. Pan, J. Sun, J. Wu, "High-Dimensional Operator Learning for Molecular Density Functional Theory", *arXiv:2411.03698* (2024). [doi:10.48550/arxiv.2411.03698](https://doi.org/10.48550/arxiv.2411.03698)
- R. Roth, "Fundamental measure theory for hard-sphere mixtures: a review", *Journal of Physics: Condensed Matter* **22**, 063102 (2010). [doi:10.1088/0953-8984/22/6/063102](https://doi.org/10.1088/0953-8984/22/6/063102)
- F. Noé, S. Olsson, J. Köhler, H. Wu, "Boltzmann Generators – Sampling Equilibrium States of Many-Body Systems with Deep Learning", *arXiv:1812.01729* (2018). [doi:10.48550/arxiv.1812.01729](https://doi.org/10.48550/arxiv.1812.01729)

---

## 7. License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
