# AGENTS.md: Developer & Agent Reference Guide for `gcmc`

This document provides a comprehensive technical reference for the `gcmc` repository. It is designed to allow AI agents and developers to quickly understand the physical problem, theoretical framework, software architecture, file structure, test suite, and performance benchmarks without needing to re-analyze original research papers.

---

## 1. Domain Physics & Scientific Context

> [!IMPORTANT]
> **Tooling Rule**: Always use `uv` and the local virtual environment for Python environment management, package installation, and script/test execution.
> - **Environment Activation**: Run `source .venv/bin/activate` before executing python scripts, tests, and CLI commands.
> - **Commands**: `uv venv`, `uv pip install`, `uv run pytest`, `uv run ...` (or `.venv/bin/python ...`).
> - **Frameworks**: Standard dependencies include `tinygrad`, `numpy`, `scipy`, `torch`, `pytest`, `ruff`. Do not use Conda or plain system `pip`.

### 1.1 The Physical Problem: Dielectrocapillarity & Electrostriction
The `gcmc` code simulates polar, dielectric, and ionic fluids subjected to inhomogeneous electric fields and electric field gradients (EFGs), as described in the paper:
> **"Dielectrocapillarity for exquisite control of fluids"** (Anna T. Bui & Stephen J. Cox, 2025; arXiv:2503.09855).

Key physical phenomena investigated:
- **Dielectrophoretic Rise**: Neutral polar molecules experience a net dielectrophoretic force $f_{\rm DEP} \propto \nabla |E|^2$, driving the fluid towards regions of higher electric field strength.
- **Electrophoretic Rise**: Ionic fluids (restricted primitive model, RPM) experience net charge forces $f_{\rm EP} = qE$, driving anions and cations in opposite directions towards lower absolute field regions.
- **Electric-Field-Driven Critical Shift**: Sizable shifts in the liquid-vapor binodal and critical temperature $T_{\rm c}$ under microscopic EFGs.
- **Dielectrocapillarity & Controlled Capillary Condensation**: Reversible, tunable fluid uptake, hysteresis suppression, and programmable pore filling in confined nanoscale slits with applied electrostatic potentials $\phi(z)$.

---

## 2. Theoretical Framework: cDFT + LMFT + GCMC + RL Control

### 2.1 Classical Density Functional Theory (cDFT)
Equilibrium structure and thermodynamics in an open system (grand canonical ensemble) are determined by minimizing the grand potential functional:
$$\Omega([\rho, \beta\phi], T) = \mathcal{F}_{\rm intr}^{\rm id}([\rho], T) + \mathcal{F}_{\rm intr}^{\rm ex}([\rho, \beta\phi], T) + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]$$
The equilibrium density $\rho(z)$ solves the Euler-Lagrange equation:
$$\rho(z) = \frac{\zeta}{\Lambda^3} \exp\left[-\beta V_{\rm ext}(z) + \beta \mu + c^{(1)}(z; [\rho, \beta\phi], T)\right]$$
where $c^{(1)} = -\delta \beta \mathcal{F}_{\rm intr}^{\rm ex} / \delta \rho(z)$ is the one-body direct correlation functional, and $n^{(1)} = \delta \mathcal{F}_{\rm intr}^{\rm ex} / \delta (\beta\phi(z))$ is the charge density functional.

### 2.2 Short-Range Coulomb Splitting (LMFT)
Direct machine learning of non-local long-range electrostatic functionals is intractable. The Coulomb potential is partitioned into a short-ranged (SR) reference contribution $v_0(r)$ and a smooth long-range (LR) contribution $v_1(r)$:
$$\frac{1}{r} = v_0(r) + v_1(r) \equiv \frac{\text{erfc}(\kappa r)}{r} + \frac{\text{erf}(\kappa r)}{r}$$
- For water (SPC/E) and the simple dipolar Stockmayer fluid: $\kappa^{-1} = 4.5\,\text{Å}$.
- For electrolytes (RPM): $\kappa^{-1} = 5.0\,\text{Å}$.

The SR reference fluid $v_0(r)$ has strictly local correlations, making it suitable for local neural functional learning. Long-range effects are restored analytically/mean-field via local molecular field theory (LMFT):
$$c^{(1)}(z) = c_{\rm R}^{(1)}(z; [\rho, \beta\phi_{\rm R}], T) - \beta\Delta\mu$$
$$n^{(1)}(z) = n_{\rm R}^{(1)}(z; [\rho, \beta\phi_{\rm R}], T)$$
$$\phi_{\rm R}(z) = \phi(z) + \int dz' \, n(z') v_1(|z - z'|)$$

### 2.3 Role of GCMC in Data Generation
1. **Training Data Generation**: `gcmc` generates reference training data for $c_{\rm R}^{(1)}$ and $n_{\rm R}^{(1)}$ across $\sim 2000$ randomized thermodynamic and external potential configurations ($T \in [250, 500]\,\text{K}$, $\mu/k_B \in [-5000, -1000]\,\text{K}$, $\phi(z) = \frac{\phi_0}{m}\cos(2\pi mz/L_z)$).
2. **Hybrid GCMC + MD Workflow**: Dense subcritical polar fluids have low GCMC insertion/deletion acceptance rates. To sample continuous high-resolution density profiles efficiently:
   - `gcmc` runs first to determine the average equilibrium particle number $N_{\rm ave} = \langle N \rangle_{\mu, V, T}$.
   - Canonical ($NVT$) Molecular Dynamics (LAMMPS with the `pair_lj_cut_coul_GT` pair style from LMFT) is initialized with $N_{\rm ave}$ molecules to sample smooth profiles $\rho_{\rm R}(z)$ and $n_{\rm R}(z)$ on a fine grid ($\Delta z = 0.02\,\text{Å}$).

---

## 3. Dual-Engine Software Architecture (`v1` & `v2`)

`gcmc` supports two interoperable engines with optional Long-Range (LR) Ewald electrostatics:
1. **`v2` (High-Performance C++/CUDA Engine - DEFAULT)**:
   - Optimized native C++ simulation engine and batched CUDA kernels on NVIDIA GPUs (RTX 4090).
   - Short-range (SR) mode achieves **~38,000x acceleration** on GPU (over 110 Million steps/s), reducing the 2,035-condition dataset generation from $\sim 10^5$ CPU hours to **under 1 minute**.
   - Long-range (LR) mode implements full reciprocal-space Ewald summation with dynamic structure factor updates on both CPU and GPU.
   - 100% 1:1 mathematical energy and profile equivalence with `v1`.
2. **`v1` (Python Baseline Engine - REFERENCE)**:
   - Pure Python / NumPy baseline for verification and legacy regression tests.
3. **`lmft_baseline` (Embedded Zero-Dependency LMFT Reference Module)**:
   - Pure Python / NumPy standalone implementation of 1D Fourier restructuring, Stillinger-Lovett bulk thermodynamic shifts, and Picard iteration cDFT solver.

### CLI Usage:
```bash
# Default runs on v2 engine (Short-Range)
gcmc -in <input_folder>

# Explicit engine selection
gcmc -in <input_folder> --engine v2
gcmc -in <input_folder> --engine v1

# Long-Range Ewald Electrostatics (v2 engine)
gcmc -in <input_folder> --enable-long-range --ewald-alpha 0.35 --ewald-kmax 4
```

```
gcmc/
├── AGENTS.md                           # Developer & agent guide
├── pyproject.toml                      # Modern packaging configuration (uv/pip)
├── src/
│   └── gcmc/
│       ├── __init__.py                 # Root package; exports v1, v2, lmft_baseline, and cli
│       ├── main.py                     # Multi-engine CLI runner with --enable-long-range support
│       ├── lmft_baseline/              # Standalone embedded LMFT baseline solver
│       │   ├── __init__.py
│       │   └── baseline_solver.py      # 1D Fourier restructuring & Picard cDFT solver
│       ├── v1/                         # Baseline Python engine
│       │   ├── potentials.py           # Pair potentials (LJ, WCA, HS, HS+C, LJ+C with erfc)
│       │   ├── external_potentials.py  # External potentials (Slits, LJ93 walls, Cosine charges)
│       │   ├── gcmc_ff_molecule.py     # Molecular simulations (ABC dipole, H2O, CO2)
│       │   ├── gcmc_ff.py              # Atomic & ionic simulations (SingleType, RPM TwoType)
│       │   └── utils/                  # Density and c1 profile extractors
│       ├── v2/                         # High-performance C++/CUDA engine
│       │   ├── core_types.h            # Vec3, Quaternion, EwaldParams, Potentials, Molecule structs
│       │   ├── simulation_engine.h/cpp # Fast Xoroshiro128+ RNG, Ewald rho_k structure factor cache
│       │   ├── cuda_gcmc.h             # Host-device batched GCMC memory buffers & CUDA Ewald params
│       │   ├── cuda_gcmc_kernels.cu    # CUDA GPU batched GCMC kernels with shared-memory Ewald
│       │   ├── c_api.h/cpp             # C-ABI export symbols
│       │   ├── bindings.py             # Python ctypes wrapper & GCMCSimulationV2 class
│       │   ├── libgcmc_v2.so           # Compiled shared library
│       ├── ui/                         # Raylib interactive 3D/2D UI module
│       │   ├── widgets.py              # Immediate-mode scientific UI components
│       │   ├── raylib_gcmc_viewer.py   # 3D GCMC molecular viewer with orbit camera
│       │   └── raylib_cdft_viewer.py   # 2D/3D cDFT slit pore fluid manipulation & RL viewer
│       └── envs/                       # RL fluid manipulation environments
│           ├── cdft_puffer/            # PufferLib C environment for cDFT control
│           │   ├── cdft_env.h/c        # Zero-copy C ocean environment with embedded LMFT convolution
│           │   ├── cdft_env.py         # Gymnasium / PufferEnv wrapper
│           │   └── libcdft_env.so      # Compiled C environment
│           └── train_pufferl.py        # Vectorized PPO / PuffeRL training loop
└── tests/
    ├── conftest.py                     # Test runner fixtures
    ├── test_paper_baseline_lmft.py     # LMFT baseline verification against published data
    ├── test_long_range_ewald.py        # Ewald electrostatics unit & parity tests
    ├── test_engine_parity.py           # 1:1 Parity tests comparing v1 vs v2 & CUDA GPU
    ├── test_cdft_puffer_env.py         # PufferLib environment verification
    ├── test_interactive_ui.py          # Interactive UI component validation
    ├── compare_with_online_data.py     # Comparison against published dataset
    ├── v1/                             # Baseline regression tests
    └── v2/                             # Performance benchmarks
        ├── benchmark_v1_vs_v2.py       # Empirical benchmark measuring SR speedups
        └── benchmark_sr_vs_lr.py       # Empirical benchmark comparing SR vs LR Ewald
```

---

## 4. Performance Benchmarks

Measured on local workstation (NVIDIA GeForce RTX 4090 GPU, 24 GB VRAM, 16,384 CUDA cores):

| Model / System | Mode | `v1` Baseline (Python) | `v2` CPU (C++) | `v2` CUDA (RTX 4090) | Speedup (CUDA vs CPU) |
|---|---|---|---|---|---|
| **Dipole Fluid (`ABC`)** | **Short-Range (SR)** | 2,919.3 steps/s | 48,147.7 steps/s | **112,678,349 steps/s** | **2,340x (38,598x vs v1)** |
| **Dipole Fluid (`ABC`)** | **Long-Range Ewald (LR)** | N/A | 709.4 steps/s | **38,552.5 steps/s** | **54.3x** |
| **RPM Electrolyte** | **Short-Range (SR)** | 5,768.4 steps/s | 180,385.1 steps/s | **27,640,027.5 steps/s** | **153.2x (18,436x vs v1)** |
| **RPM Electrolyte** | **Long-Range Ewald (LR)** | N/A | 62,060.6 steps/s | **262,800.9 steps/s** | **4.2x** |
| **SPC/E Water (`H2O`)** | **Short-Range (SR)** | 3,337.4 steps/s | 726,251.2 steps/s | **40,578,059 steps/s** | **55.9x (12,158x vs v1)** |
| **PufferLib cDFT Env** | **Embedded LMFT** | N/A | N/A | **481,600 steps/s** | **Zero-Copy C Rollouts** |

---

## 5. Test Suite

Run full automated tests across all components:
```bash
uv run pytest tests/ -v
```
All 37 automated tests execute in ~7 seconds, validating:
- Exact 1:1 mathematical energy equivalence between `v1` and `v2`.
- Correct reciprocal space Ewald summation in C++ and CUDA GPU.
- Verified Stillinger-Lovett bulk shifts and restructuring potentials against published `OnlineData.tgz`.
- Zero-copy C PufferLib environment step and reset dynamics with embedded LMFT convolution.
