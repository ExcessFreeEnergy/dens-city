# AGENTS.md: Developer & Agent Reference Guide for `dens-city`

`dens-city` is a statistical mechanics Classical Density Functional Theory (cDFT) and generative molecular platform implemented in `tinygrad` and `PufferLib`.

---

## 1. Codebase Architecture

All source code lives under `src/dens_city/`:
- **`cdft`**: Variational Classical Density Functional Theory engine, planar FMT convolution kernels, and variational solvers (`TinyCDFT`, `BatchedTinyCDFT`, `KernelBuilder`).
- **`boltzmann`**: Boltzmann Generator normalizing flows (`Base2CartesianFlow`), Hamiltonians (`MicroscopicEnergy`), geometry relaxation (`BatchedLBFGS`), and quantum surrogate force fields (`EGNNForceField`).
- **`swarm`**: Reinforcement learning molecular swarm environment (`CDFTSwarmEnv`), PPO trainer with curriculum learning (`SwarmPuffeRLTrainer`), and generative molecular funnels.
- **`ui`**: High-performance 3D Raylib molecular visualization engine and unified CLI (`main`, `MoleculeViewer`).
- **`utils`**: Molecular data loader, Tripos `.mol2` parser, EOS solvers, and dataset population/verification routines.
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

- **Active Skill**: [`.agents/skills/cdft-wikiskill/SKILL.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/skills/cdft-wikiskill/SKILL.md) — 16 core procedural rules for cDFT, Boltzmann flows, EGNNs, and Tinygrad.
- **Pattern Catalog**: [`.agents/wikiskill/wiki/index.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/index.md) — Single-line `[Problem | Cause | Fix]` summaries for all 19 verified patterns in `patterns/`.
- **Anti-Pattern Audit Ledger**: [`.agents/wikiskill/wiki/skill-impact.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/skill-impact.md) — Historical audit log of all past proposals and outcomes. **Never repeat previously rejected approaches.**
- **Lifecycle Automation**: Managed automatically via [`.agents/hooks.json`](file:///home/gauss/code/cdft_sim/dens-city/.agents/hooks.json):
  - `PostToolUse`: Automatically captures execution failure traces into `raw/traces/`.
  - `PreInvocation`: Intercepts user queries and proactively surfaces relevant pattern links into context.

### Essential Rules of Engagement:
1. **Zero Hardcoded Parameters**: Derive all physical parameters ($\sigma_i, \epsilon_i, q_i$) dynamically from input `.mol2` files and standard force field tables.
2. **Consult Patterns Before Modifying Solvers**: Check `wiki/index.md` before altering cDFT potentials, normalizing flows, EGNN layers, or Tinygrad JIT kernels.
3. **Audit Against Rejections**: Verify proposed approaches against `skill-impact.md` or via `uv run dens-city --wikiskill-audit <target>`.

---

## 4. Key CLI Commands

```bash
# Knowledge Base & Audit
uv run dens-city --wikiskill-status
uv run dens-city --wikiskill-audit <pattern_name>
uv run dens-city --wikiskill-consolidate

# cDFT & Boltzmann Molecular Pipeline
uv run dens-city --materials argon water methane 5cb --batch-size 512

# 3D Interactive Raylib Visualizer
uv run dens-city --interactive --materials argon water

# Generative Funnel & Swarm Training
uv run dens-city --funnel --spec oled --train-steps 25000
uv run dens-city --train-swarm --spec oled --train-steps 5000000

# Automated Verification
uv run pytest tests/ -v
uv run ruff check src/ tests/
```
