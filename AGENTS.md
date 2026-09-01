# AGENTS.md: Developer & Agent Reference Guide for `dens-city`

This document provides a comprehensive technical reference for the `dens-city` repository. It is designed to allow AI agents and developers to quickly understand the physical problem, theoretical framework, software architecture, file structure, test suite, and first-principles physics rules.

---

## 1. Domain Physics & Theoretical Framework

`dens-city` is a pure statistical mechanics Classical Density Functional Theory (cDFT) and generative molecular platform implemented in `tinygrad` and `PufferLib`.

### 1.1 Variational Classical Density Functional Theory (cDFT)
Equilibrium structure and thermodynamics in an open system (grand canonical ensemble) are determined by minimizing the grand potential functional $\Omega[\psi]$ with respect to a latent potential field $\psi(z)$, where density is parameterized as $\rho(z) = \rho_{\rm bulk} \exp(\psi(z))$:
$$\Omega[\psi] = \mathcal{F}_{\rm ideal}[\psi] + \mathcal{F}_{\rm FMT}^{\rm ex}[\rho] + \mathcal{F}_{\rm att}^{\rm ex}[\rho] + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]$$

- **Ideal Gas Free Energy (Log-Free Formulation)**:
  $$\mathcal{F}_{\rm ideal}[\psi] = k_B T \int dz \, \left[ \rho(z) \psi(z) - (\rho(z) - \rho_{\rm bulk}) \right]$$
- **Rosenfeld Fundamental Measure Theory (FMT) Hard-Sphere Excess**:
  $$\mathcal{F}_{\rm FMT}^{\rm ex}[\rho] = k_B T \int dz \, \Phi_{\rm FMT}(\{n_\alpha(z)\})$$
  $$\Phi_{\rm FMT} = -n_0 \ln(1 - n_3^*) + \frac{n_1 n_2 - \mathbf{n}_{v1} \cdot \mathbf{n}_{v2}}{1 - n_3^*} + \frac{n_2^3 - 3 n_2 |\mathbf{n}_{v2}|^2}{24\pi(1 - n_3^*)^2}$$
  where $n_3^* = \min(n_3, 1 - 10^{-5})$ prevents unphysical singularities, and weighted densities $n_\alpha(z) = (\rho * w_\alpha)(z)$ are evaluated via anti-aliased, analytically cell-integrated planar convolution kernels.
- **WCA / Lennard-Jones Attractive Dispersion Excess**:
  $$\mathcal{F}_{\rm att}^{\rm ex}[\rho] = \frac{1}{2} \int dz \int dz' \, \rho(z) v_{\rm att, 1D}(|z - z'|) \rho(z')$$
- **Exact Mechanical Observables (Irving-Kirkwood Virial Theorem)**:
  Wall contact pressure is evaluated via the exact momentum balance integral over the external potential gradient:
  $$P_{\rm wall} = -\int_0^{L_z/2} \rho(z) \frac{d V_{\rm ext}(z)}{dz} \, dz$$

---

## 2. Codebase Architecture & Subpackages

The `dens-city` codebase is modularized under `src/dens_city/`:
- **`dens_city.cdft`**: Classical Density Functional Theory engine, planar FMT convolution kernels, and variational solvers (`TinyCDFT`, `BatchedTinyCDFT`, `KernelBuilder`).
- **`dens_city.boltzmann`**: Boltzmann Generator normalizing flows (`Base2CartesianFlow`, `CompositeFlow`), microscopic Hamiltonians (`MicroscopicEnergy`), batched GPU geometry relaxation (`BatchedLBFGS`), quantum surrogate force fields (`EGNNForceField`), and latent MCMC relaxation (`BoltzmannGenerator`).
- **`dens_city.swarm`**: Multi-objective reinforcement learning swarm environment (`CDFTSwarmEnv`, `VectorizedSwarmEnv`), PPO trainer with curriculum learning (`SwarmPuffeRLTrainer`, `train_swarm_policy`), 5-stage generative funnel (`run_generative_funnel`, `run_all_specs_funnel_benchmark`), curriculum sweeps (`CurriculumSweepRunner`, `run_curriculum_sweep`), and chemistry diagnostics (`evaluate_molecule_chemistry`, `evaluate_all_swarm_specs`).
- **`dens_city.ui`**: High-performance 3D Raylib molecular visualization engine and unified CLI (`MoleculeViewer`, `run_interactive_viewer`, `main`).
- **`dens_city.utils`**: Molecular data loader, Tripos `.mol2` parser, force-field database, EOS solvers, combinatorial library generator (`generate_library`, `embed_and_export_parallel`), test dataset populator (`generate_test_data`), FreeSolv statistical validator (`verify_pipeline_against_freesolv`), and asynchronous batch prefetchers (`MaterialLoader`, `Material`, `MaterialPipelineTask`, `process_material_task`).

---

## 3. Tooling & Environment Rules

> [!IMPORTANT]
> **Tooling Rule**: Always use `uv` and the local virtual environment for Python environment management, package installation, and script/test execution.
> - **Environment Synchronization**: Run `uv sync` to sync packages from `uv.lock`.
> - **Commands**: `uv run pytest`, `uv run dens-city ...`, `uv run ruff check`, `uv run ruff format`.
> - **Lockfile Management**: Run `uv lock` to update `uv.lock` when modifying `pyproject.toml`.
> - **Frameworks**: Standard dependencies include `tinygrad`, `raylib`, `numpy`, `scipy`, `pytest`, `ruff`, `pufferlib`, `torch`, `rdkit`. Never use Conda or plain system `pip`.

> [!CAUTION]
> **Mandatory First-Principles Physics & Anti-Pattern Audit Rules**:
> 1. **Zero Hardcoded Parameters or Fudge Factors**: Never hardcode fluid properties, lookup dictionaries, aliases, or empirical constants in `src/`. All physical parameters ($\sigma_i, \epsilon_i, q_i$) must be derived strictly from arbitrary input `.mol2` files and force field parameter definitions.
> 2. **Latent Field Positivity**: Never optimize $\rho(z)$ directly. Always optimize $\psi(z)$ where $\rho(z) = \rho_{\rm bulk} \exp(\psi(z))$ to guarantee non-negativity and eliminate $\ln(\rho)$ NaN traps.
> 3. **Thermodynamic Consistency**: State variables ($\rho_{\rm bulk}, \mu, P$) must be derived dynamically from the bulk Equation of State (EOS) root solver rather than assumed constant.
> 4. **Exact Mechanical Observables**: Never use brittle spatial slices (e.g. `rho[0:15]` or `[mid-10:mid+10]`). Wall contact pressures, surface tensions, and forces must be evaluated via exact statistical mechanical integrals (e.g., Irving-Kirkwood virial tensor integral $P_{\rm wall} = -\int_0^{L/2} \rho(z) \nabla V_{\rm ext}(z) dz$).
> 5. **Exact Asymptotic Boundaries & Steric Masking**: Enforce true physical divergence ($V \to \infty$) at steric hard boundaries ($V_{\max} = 10^6\,k_B T$). Never introduce artificial spatial shifts (e.g., `max(0.2, z)`) or soft boundary clamping (`[-500, 1000]`). Use physical steric masking (`.where()`) to eliminate IEEE 754 $0 \times \infty$ NaN traps.
> 6. **Scale-Invariant Initialization & Cutoffs**: All physical cutoffs and bounding geometry must scale with the fluid's intrinsic parameters ($r_{\rm cut} = 5.0 \sigma_{\rm eff}$, $L_z = \max(40.0\,\text{Å}, 10.0 \sigma_{\rm eff})$). Initial profiles must follow exact Boltzmann asymptotics $\psi_0(z) = -\beta V_{\rm ext}(z)$.

---

## 4. CLI Usage & Verification

```bash
# Sync dependencies
uv sync

# 1. 3D Interactive Raylib Visualizer
uv run dens-city --interactive --materials argon water

# 2. Standard High-Throughput Coupled Pipeline (cDFT + Boltzmann Generator)
uv run dens-city --materials argon water methane 5cb --batch-size 512

# 3. 5-Stage Generative Molecular Funnel
uv run dens-city --funnel --spec oled --train-steps 25000 --num-candidates 512 --top-k 20

# 4. Cross-Material 5-Stage Funnel Benchmark
uv run dens-city --benchmark-specs --train-steps 25000

# 5. RL Swarm Training
uv run dens-city --train-swarm --spec oled --train-steps 5000000

# 6. Constellation Curriculum Sweeps
uv run dens-city --sweep --num-trials-per-spec 3

# 7. Multi-Specification Chemistry Diagnostics
uv run dens-city --eval-swarm --specs-dir tests/data

# 8. Combinatorial Molecular Library Generation
uv run dens-city --generate-library --spec oled --target-count 50000

# 9. Test Data Population & FreeSolv Dataset
uv run dens-city --populate-test-data --all-freesolv

# 10. FreeSolv Statistical Validation & Verification Report
uv run dens-city --verify-freesolv --run-e2e

# 11. WikiSkill Persistent Knowledge & Anti-Pattern Audit
uv run dens-city --wikiskill-status
uv run dens-city --wikiskill-audit pattern_log_free_latent_density
uv run dens-city --wikiskill-consolidate

# Run automated test suite
uv run pytest tests/ -v

# Run linting and code formatting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

---

## 5. WikiSkill Persistent Knowledge Base (Zero-Forgetting Framework)

To prevent the recurring cycle of **fixing, forgetting, and reimplementing errors** (per arXiv:2608.27454v1), all agents and developers must adhere to the three-layer knowledge workflow:
1. **Raw Layer (`.agents/wikiskill/raw/traces/`)**: Immutable test outputs and failure dumps.
2. **Persistent Wiki Layer (`.agents/wikiskill/wiki/`)**:
   - `index.md`: Catalog of known failure modes (`PROBLEM + ROOT CAUSE + FIX`).
   - `skill-impact.md`: Audit tracker of all past proposals and rejections. **Never repeat rejected approaches.**
   - `patterns/*.md`: In-depth statistical mechanics root-cause analyses and verified action rules.
3. **Skills & Rules Layer (`.agents/skills/cdft-wikiskill/` & `.agents/rules/`)**: Active, executable instructions.

