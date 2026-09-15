# AGENTS.md: Developer & Agent Reference Guide for `dens-city`

`dens-city` is a statistical mechanics Classical Density Functional Theory (cDFT) and generative molecular platform implemented in `tinygrad` and `PufferLib`.

---

## 1. Codebase Architecture

All source code lives under `src/dens_city/`:
- **`cdft`**: Variational Classical Density Functional Theory engine, planar FMT convolution kernels, Generalized Born continuum solvation, and variational solvers (`TinyCDFT`, `BatchedTinyCDFT`, `KernelBuilder`, `GeneralizedBornSolvation`).
- **`boltzmann`**: Boltzmann Generator normalizing flows (`Base2CartesianFlow`), Hamiltonians (`MicroscopicEnergy`), geometry relaxation (`BatchedLBFGS`), and quantum surrogate force fields (`EGNNForceField`).
- **`swarm`**: Reinforcement learning molecular swarm environment (`CDFTSwarmEnv`), PPO trainer with curriculum learning (`SwarmPuffeRLTrainer`), and generative molecular funnels.
- **`ui`**: High-performance 3D Raylib molecular visualization engine and unified CLI (`main`, `MoleculeViewer`, `cli`).
- **`utils`**: Molecular data loader, Tripos `.mol2` parser, EOS solvers, benchmark datasets (`FreeSolvDataset`, `SolvatumDataset`), and validation/reporting routines (`verify_pipeline_against_dataset`).
- **`server`**: Model Context Protocol (MCP) server and FastAPI REST endpoints for autonomous agent execution.
- **`wikiskill`**: Persistent knowledge co-evolution engine, raw trace recorder, maintainer, proposer, and empirical gating harness.

---

## 2. Tooling & Environment Directives

> [!IMPORTANT]
> Always use `uv` and the local virtual environment (`.venv`). Never use system `pip` or Conda.
> - **Sync Dependencies**: `uv sync`
> - **Run Test Suite**: `uv run pytest tests/ -v`
> - **Code Quality**: `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/`
> - **CLI Execution**: `uv run dens-city [args]`

---

## 3. Persistent Knowledge Base & Anti-Pattern Audit (WikiSkill)

To eliminate the recurring cycle of **fixing, forgetting, and reimplementing errors** (per arXiv:2608.27454v1), all physical invariants, mathematical derivations, and compiler rules are maintained in the **WikiSkill persistent knowledge base**:

- **Active Skill**: [`.agents/skills/cdft-wikiskill/SKILL.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/skills/cdft-wikiskill/SKILL.md) — 22 core procedural rules for cDFT, Boltzmann flows, EGNNs, and Tinygrad.
- **Pattern Catalog**: [`.agents/wikiskill/wiki/index.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/index.md) — Single-line `[Problem | Cause | Fix]` summaries for all 25 verified patterns in `patterns/`.
- **Anti-Pattern Audit Ledger**: [`.agents/wikiskill/wiki/skill-impact.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/skill-impact.md) — Historical audit log of all past proposals and outcomes. **Never repeat previously rejected approaches.**
- **Lifecycle Automation**: Managed automatically via [`.agents/hooks.json`](file:///home/gauss/code/cdft_sim/dens-city/.agents/hooks.json):
  - `PostToolUse`: Automatically captures execution failure traces into `raw/traces/`.
  - `PreInvocation`: Intercepts user queries and proactively surfaces relevant pattern links into context.

### Essential Rules of Engagement:
1. **Zero Hardcoded Parameters**: Derive all physical parameters ($\sigma_i, \epsilon_i, q_i$) dynamically from input `.mol2` files and standard force field tables.
2. **Consult Patterns Before Modifying Solvers**: Check `wiki/index.md` before altering cDFT potentials, normalizing flows, EGNN layers, or Tinygrad JIT kernels.
3. **Audit Against Rejections**: Verify proposed approaches against `skill-impact.md` or via `uv run dens-city --wikiskill-audit <target>`.
4. **High-Throughput GPU Batch Sizing**: Maximize engine throughput by starting EGNN batch sizes at $B = 128$; decrease in powers of 2 ($64 \to 32$) only if dense pair tensors $(B \cdot s, N, N, 1)$ encounter GPU VRAM limits.
5. **Autograd Parameter Gradient Detachment**: Always clear parameter gradients (`for p in nn.state.get_parameters(self): p.grad = None`) after reverse-mode force extraction in inference loops to prevent memory graph leaks.

---

## 4. End-to-End Thermodynamic Validation Directives

`dens-city` provides direct CLI entrypoints for validating predictions against experimental thermodynamic benchmarks:

### 4.1 FreeSolv Aqueous Validation (642 molecules)
```bash
# 1. Full end-to-end simulation from scratch:
uv run dens-city --verify-freesolv --run-e2e

# 2. Fast statistical validation from existing simulation results:
uv run dens-city --verify-freesolv --results-dir runs/batch_sota_ensembled

# 3. Populate FreeSolv test dataset into data/test_data/ (if needed):
uv run dens-city --populate-test-data --all-freesolv
```

### 4.2 Solv@TUM / Solvatum Multi-Solvent Validation (5,952 pairs across 146 solvents)
```bash
# 1. Fast statistical validation from existing simulation results:
uv run dens-city --verify-solvatum

# 2. Fast statistical validation from a specific results directory:
uv run dens-city --verify-solvatum --results-dir runs/batch_solvatum_e2e

# 3. Full end-to-end multi-solvent simulation from scratch:
uv run dens-city --verify-solvatum --run-e2e

# 4. Populate Solvatum test dataset into data/test_data/ (if needed):
uv run dens-city --populate-test-data --all-solvatum

# 5. Recalibrate Universal Analytical Delta-KRR Residual Model (Cholesky LOOCV):
uv run dens-city --recalibrate-krr
```

### 4.3 Target Statistical Baselines (SOTA Thresholds)
- **FreeSolv ($N=642$)**: $\text{MAE} \le 0.184\text{ kcal/mol}$, $\text{RMSE} \le 0.302\text{ kcal/mol}$, $R^2 \ge 0.993$ (vs. GAFF baseline $1.101\text{ kcal/mol}$).
- **Solvatum ($N=5,952$)**: $\text{MAE} \le 0.123\text{ kcal/mol}$, $\text{RMSE} \le 0.258\text{ kcal/mol}$, $R^2 \ge 0.988$ (vs. literature continuum $1.809\text{ kcal/mol}$).
- **Observable Completeness**: Output rows in `pipeline_summary.jsonl` must have non-null `egnn_energy` and `egnn_force_rms`.

---

## 5. Key CLI Commands Quick Reference

```bash
# Knowledge Base & Audit
uv run dens-city --wikiskill-status
uv run dens-city --wikiskill-audit <pattern_name>
uv run dens-city --wikiskill-consolidate

# FreeSolv E2E Validation
uv run dens-city --verify-freesolv --run-e2e
uv run dens-city --verify-freesolv --results-dir runs/batch_sota_ensembled

# Solvatum Multi-Solvent Validation & KRR Recalibration
uv run dens-city --verify-solvatum
uv run dens-city --verify-solvatum --results-dir runs/batch_solvatum_e2e
uv run dens-city --recalibrate-krr

# Coupled cDFT & Boltzmann Molecular Pipeline
uv run dens-city --materials argon water methane 5cb --batch-size 512

# 3D Interactive Raylib Visualizer
uv run dens-city --interactive --materials argon water

# Generative Funnel & Swarm Training
uv run dens-city --funnel --spec oled --train-steps 25000
uv run dens-city --train-swarm --spec oled --train-steps 5000000

# Automated Verification & Quality
uv run pytest tests/ -v
uv run ruff check src/ tests/
```
