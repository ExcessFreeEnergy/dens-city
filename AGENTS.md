# AGENTS.md: Developer & Agent Reference Guide for `dens-city`

This document provides a comprehensive technical reference for the `dens-city` repository. It is designed to allow AI agents and developers to quickly understand the physical problem, theoretical framework, software architecture, file structure, test suite, and first-principles physics rules.

---

## 1. Domain Physics & Theoretical Framework

`dens-city` is a pure statistical mechanics Classical Density Functional Theory (cDFT) engine implemented in `tinygrad`.

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
- **`dens_city.cdft`**: Classical Density Functional Theory engine, planar FMT convolution kernels, and variational solvers (`TinyCDFT`, `KernelBuilder`).
- **`dens_city.boltzmann`**: Boltzmann Generator normalizing flows (`Base2CartesianFlow`, `CompositeFlow`), microscopic Hamiltonians (`MicroscopicEnergy`), spatial CDFT priors (`CDFTBaseDistribution`), and latent MCMC relaxation (`BoltzmannGenerator`).
- **`dens_city.ui`**: Interface subsystem placeholder for future interactive visualization.
- **`dens_city.materials`**: Tripos `.mol2` parser, force-field database, and Equation of State (EOS) root solvers.
- **`dens_city.pipeline`**: High-throughput multi-material batch orchestration and parallel worker pools.

---

## 3. Tooling & Environment Rules

> [!IMPORTANT]
> **Tooling Rule**: Always use `uv` and the local virtual environment for Python environment management, package installation, and script/test execution.
> - **Environment Synchronization**: Run `uv sync` to sync packages from `uv.lock`.
> - **Commands**: `uv run pytest`, `uv run python scripts/...`, `uv run ruff check`, `uv run ruff format`.
> - **Lockfile Management**: Run `uv lock` to update `uv.lock` when modifying `pyproject.toml`.
> - **Frameworks**: Standard dependencies include `tinygrad`, `numpy`, `scipy`, `pytest`, `ruff`. Never use Conda or plain system `pip`.

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

# Run single material
uv run python scripts/run_cdft.py --materials argon

# Run multi-material simulation
uv run python scripts/run_cdft.py --materials argon water methane 5cb

# Run full 20-material benchmark suite
uv run python scripts/run_cdft.py --materials all --steps 50 --no-plot

# Run automated test suite
uv run pytest tests/ -v

# Run linting and code formatting
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```
