# `dens-city` FastAPI & Model Context Protocol (MCP) Integration Specification

This document describes the technical architecture, REST API routes, MCP tool schemas, SQLite queue engine, and secure Zero-Trust GPU deployment for the `dens-city` server.

---

## 1. System Architecture

```
[ LLM Orchestrator (Claude / GPT-4 / Harness) ]
                       │
             (Authenticated Agentic Traffic)
                       │
            [ Cloudflare Access Tunnel ]
                       │
       ┌───────────────▼───────────────┐
       │   FastAPI Server (PID: 1001)  │
       │   - /api/v1/* REST Endpoints  │
       │   - /mcp/sse (MCP SSE Server) │
       │   - Long-Polling Dispatcher   │
       │   - Synchronous Intake Gate   │
       └───────────────┬───────────────┘
                       │ SQLite (runs/jobs/jobs.db)
       ┌───────────────▼───────────────┐
       │ Dedicated GPU Worker (spawn)  │
       │ (PID: 1002 - N=1 GPU Queue)   │
       │ - Tinygrad / PyTorch Isolated │
       │ - Progress & Log Streaming    │
       └───────────────┬───────────────┘
                       │ Persistent Disk Pools
       ┌───────────────▼───────────────┐
       │     runs/pools/{pool_id}/     │
       │  - candidate_pool             │
       │  - thermo_pool                │
       │  - pareto_result              │
       └───────────────────────────────┘
```

---

## 2. CLI Invocation

### Starting the FastAPI Server
```bash
# Start on default port 8000
uv run dens-city --serve-api --port 8000

# Or via dedicated entrypoint
uv run dens-city-server
```

### Starting the MCP Server (Stdio)
```bash
# Standard input/output transport for local agents (Claude Desktop / Cursor)
uv run dens-city --serve-mcp --transport stdio

# Or via dedicated script
uv run dens-city-mcp
```

### Disk Cleanup
```bash
# Prune intermediate pools while keeping Pareto exports
uv run dens-city --cleanup-pools --cleanup-mode keep_pareto_only
```

---

## 3. REST API Routes Reference

### Submission Endpoints (Return `HTTP 202 Accepted` + `job_id`)
- `POST /api/v1/pipeline/full`: Spawns 4-stage funnel.
- `POST /api/v1/stages/train-swarm`: Spawns Stage 1 PPO RL training.
- `POST /api/v1/stages/sample-candidates`: Spawns Stage 2 candidate sampling.
- `POST /api/v1/stages/run-cdft`: Spawns Stage 3 cDFT, Boltzmann flows & EGNN MLFF screening. Synchronously rejects invalid SMILES with `HTTP 400`.
- `POST /api/v1/stages/rank-pareto`: Spawns Stage 4 Pareto ranking & export.

### Long-Polling Observability
- `GET /api/v1/jobs/{job_id}?wait_for_completion=true&timeout_seconds=300`:
  Holds HTTP connection open until the job finishes or hits the timeout. Returns `JobStatusResponse`.
- `POST /api/v1/jobs/{job_id}/cancel`: Aborts active job.
- `GET /api/v1/jobs`: Returns last 20 jobs.

### Pre-flight & Maintenance
- `POST /api/v1/spec/validate`: Synchronous RDKit valence & spec bounds validation.
- `POST /api/v1/pools/cleanup`: Prunes intermediate scratch files.
- `GET /api/v1/pools/stats`: Returns active pool statistics and disk usage.
- `GET /healthz`: Health check probe.

---

## 4. Target Specification YAML Schema

Every new material or molecule design target submitted to `run_full_pipeline` or `train_swarm_agent` must define the 7 canonical sections:
1. `group_name` & `version`: Unique identifier and semantic version.
2. `rl_reward_targets`: `target_elasticity`, `target_tensile`, `target_toughness`, `target_lightweight`, `max_solvation_kcal`, `min_wall_pressure_bar`, `max_molecular_weight`, `min_valency`, `sa_threshold`, `sa_penalty_slope`.
3. `tensor_limits`: `max_sites` (<= 512), `allowed_atomic_numbers` (e.g. `[1, 6, 8, 14]`), `max_molecular_weight`, and optional ring/bond counts.
4. `scaffolds`: Core templates with indexed wildcard ports `[*:1]`, `[*:2]`.
5. `building_blocks`: Grouped radicals, linkers, and caps with indexed attachment ports.
6. `assembly_rules`: Reaction type, `smarts_transform` (SMIRKS), and structural limits.
7. `generation_spec`: `target_molecules`, `random_seed`, and GPU `conformer_embedding`.

### Full Commented Schema Template

```yaml
# =============================================================================
# 1. METADATA
# =============================================================================
group_name: "custom_material_class"  # [str] Snake_case material identifier
version: "1.0.0"                    # [str] Semantic versioning string
description: "High-level summary of target material and mechanical properties."

# =============================================================================
# 2. REINFORCEMENT LEARNING REWARD TARGETS (rl_reward_targets)
# Normalized target mechanical weights. Must sum to approximately 1.0.
# =============================================================================
rl_reward_targets:
  target_elasticity: 0.40           # [float: 0.0 to 1.0] Desired elastic modulus bias
  target_tensile: 0.30              # [float: 0.0 to 1.0] Desired tensile strength bias
  target_toughness: 0.20            # [float: 0.0 to 1.0] Desired fracture toughness bias
  target_lightweight: 0.10          # [float: 0.0 to 1.0] Desired density minimization bias
  max_solvation_kcal: -3.0          # [float] Maximum allowed hydration free energy (kcal/mol)
  min_wall_pressure_bar: 15.0       # [float] Target wall pressure from cDFT profile (bar)
  max_molecular_weight: 800.0       # [float] Upper molecular weight ceiling
  min_valency: 2                    # [int >= 1] Minimum required reactive/attachment valency
  sa_threshold: 4.5                 # [Optional float: 1.0 to 10.0] Synthetic accessibility threshold
  sa_penalty_slope: 2.0             # [Optional float >= 0.0] Penalty slope for SA score > threshold

# =============================================================================
# 3. TENSOR LIMITS & ATOMIC CONSTRAINTS (tensor_limits)
# Static tensor dimensions for GPU memory allocation in tinygrad.
# =============================================================================
tensor_limits:
  max_sites: 128                    # [int <= 512] Maximum number of atoms/sites per molecule
  allowed_atomic_numbers:           # [list[int]] Permitted IUPAC atomic numbers
    - 1                             # Hydrogen (H)
    - 6                             # Carbon (C)
    - 7                             # Nitrogen (N)
    - 8                             # Oxygen (O)
    - 14                            # Silicon (Si)
  max_molecular_weight: 800.0       # [float] Hard cut-off for candidate generator
  min_rotatable_bonds: 2            # [Optional int] Minimum rotatable bonds
  max_aromatic_rings: 4             # [Optional int] Maximum aromatic ring count
  min_silicon_count: 0              # [Optional int] Required heteroatom counts
  min_sp3_fraction: 0.30            # [Optional float: 0.0 to 1.0] Minimum fraction of sp3 carbons

# =============================================================================
# 4. SCAFFOLDS (scaffolds)
# Core templates with indexed wildcard attachment ports [*:1], [*:2], ...
# =============================================================================
scaffolds:
  - id: "core_template_1"
    smarts: "C(C[*:1])(C[*:2])(C[*:3])C[*:4]"
    name: "pentaerythrityl 4-arm core"
    attachment_points: 4            # [int] Must match the number of [*:N] ports

# =============================================================================
# 5. BUILDING BLOCKS (building_blocks)
# Functional groups, linkers, and terminal caps grouped by category.
# Every block MUST contain indexed attachment ports [*:1] (and [*:2] for linkers).
# =============================================================================
building_blocks:
  functional_monomers:
    - id: "dimethyl_silane_unit"
      smiles: "O[Si](C)(C)O[*:1]"
    - id: "ethylene_glycol_linker"
      smiles: "OCCO([*:1])[*:2]"
    - id: "terminal_cap"
      smiles: "C[*:1]"
    - id: "hydrogen_cap"
      smiles: "[H][*:1]"

# =============================================================================
# 6. ASSEMBLY RULES (assembly_rules)
# SMIRKS chemical reaction transforms and structural filters.
# =============================================================================
assembly_rules:
  reaction_type: "silanol_condensation"
  smarts_transform: "[Si:1][O][H].[Si:2][O][H]>>[Si:1][O][Si:2]"
  max_ring_count: 2
  min_rotatable_bonds: 4

# =============================================================================
# 7. GENERATION SPECIFICATION (generation_spec)
# Settings for 3D combinatorial enumeration and distance geometry embedding.
# =============================================================================
generation_spec:
  target_molecules: 50000           # [int] Total combinatorial enumeration budget
  random_seed: 42                   # [int] RNG seed for reproducible generation
  deterministic_enumeration: true   # [bool] Systematic (true) vs stochastic (false)
  conformer_embedding:
    algorithm: "distance_geometry_gpu"
    num_conformations_per_molecule: 10
    clash_cutoff_angstrom: 0.85
  jackhammer_mining:
    max_fragment_nodes: 32
    target_fragment_budget: 15000
```

Working reference specifications for production materials are stored in `tests/data/*.yaml`:
- `tests/data/hyperbranched_siloxane_elastomers.yaml` (Shoe Sole Midsole / Rebound Layer)
- `tests/data/sacrificial_h_bond_toughness_resins.yaml` (Shoe Sole Tread Outsole / Tear-Resistant Layer)
- `tests/data/solid_state_battery_electrolytes.yaml` (Ionic Conductors)
- `tests/data/high_strain_energetic_cages.yaml` (High-Density Polycyclic Cages)
- `tests/data/high_entropy_liquid_electrolytes.yaml` (Fluorinated & Solvating Fluids)

---

## 5. Secure Zero-Trust Deployment (Cloudflare Access Tunnel)

To allow remote LLM harnesses to trigger GPU workloads without exposing open ports:

1. Install `cloudflared`:
   ```bash
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared.deb
   ```
2. Authenticate and create tunnel:
   ```bash
   cloudflared tunnel create dens-city-gpu
   ```
3. Route tunnel to local FastAPI port:
   ```yaml
   # ~/.cloudflared/config.yml
   tunnel: <TUNNEL_UUID>
   credentials-file: /home/gauss/.cloudflared/<TUNNEL_UUID>.json
   ingress:
     - hostname: dens-city.internal.example.com
       service: http://localhost:8000
     - service: http_status:404
   ```
4. Run tunnel service:
   ```bash
   cloudflared tunnel run dens-city-gpu
   ```
5. Configure Service Token Authentication in Cloudflare Zero Trust dashboard so agent requests present `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers.
