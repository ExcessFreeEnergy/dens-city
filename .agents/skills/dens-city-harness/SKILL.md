---
name: dens-city-harness
description: Production guide for autonomous LLM harnesses to orchestrate dens-city via Model Context Protocol (MCP) and FastAPI. Details asynchronous job polling, context-window protection, stateless stage pools, synchronous chemical validation, semantic error recovery, and shoe sole multi-layer material design.
---

# `dens-city` Autonomous LLM Harness & MCP Integration Guide

`dens-city` is a statistical mechanics Classical Density Functional Theory (cDFT), generative Boltzmann normalizing flow, and $E(3)$-equivariant Graph Neural Network (EGNN) platform.

This skill equips autonomous LLM agents (e.g. Claude, GPT-4, Grok) within a broader engineering harness (e.g. CAD generative pipelines, multi-layer shoe sole design, battery manufacturing) to operate `dens-city` as an asynchronous, black-box computational microservice.

---

## 1. Core Mental Model: The Materials Architect

The LLM harness does **not** evaluate physics equations directly or handle raw 3D coordinate arrays. Instead, it acts as the **Materials Architect**:
1. It translates engineering requirements (e.g., stiffness, shock absorption, adhesion, chemical resistance) into a **Target Specification** dictionary.
2. It calls asynchronous **Model Context Protocol (MCP)** tools or REST endpoints on the GPU server.
3. It observes progress via server-side long-polling and passes lightweight **Artifact Pool IDs** between stages.
4. It consumes final exported 3D Tripos `.mol2` candidate structures and Pareto summary tables.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS LLM HARNESS / AGENT                       │
│    (Defines engineering constraints, orchestrates stages, monitors)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ MCP Tools (JSON-RPC) or REST
┌───────────────────────────────────▼────────────────────────────────────┐
│                    dens-city FASTAPI & MCP SERVER                      │
│ - Synchronous Chemical Intake Gate (RDKit / C-native valence checks)   │
│ - Server-Side Long-Polling (holds connections up to 300s)              │
│ - Semantic Error Translator (actionable recovery advice)               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ SQLite Queue (FIFO, N=1 GPU)
┌───────────────────────────────────▼────────────────────────────────────┐
│                  DEDICATED SPAWN GPU WORKER PROCESS                    │
│ Stage 1: PufferLib RL Molecular Swarm (PPO Graph Growth)               │
│ Stage 2: C-Native Vectorized Candidate Sampling (Contiguous memory)     │
│ Stage 3: Coupled cDFT Euler-Lagrange + L-BFGS + Boltzmann Flows         │
│ Stage 4: 7-Layer Invariant EGNN Quantum Force Field Screening          │
│ Stage 5: Multi-Objective Pareto Frontier Ranking & .mol2 Export        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Persistent Pool Handles
┌───────────────────────────────────▼────────────────────────────────────┐
│                runs/pools/ ARTIFACT POOL STORAGE                       │
│   candidate_pool_id ──► thermo_pool_id ──► egnn_scored_pool_id ──►    │
│                        pareto_result_uri                               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 4 Golden Rules for the LLM Harness

### Rule 1: Always Long-Poll (`wait_for_completion=True`)
- **Never** spin in a rapid polling loop. Calling `get_job_status` repeatedly when a job is `RUNNING` will flood your context window with dozens of tool calls, exhaust tokens, and cause you to forget the original design prompt.
- **Always** call `get_job_status(job_id=..., wait_for_completion=True, timeout_seconds=300)`. The server holds the request open until the job finishes or hits the timeout, compressing 30 turns into a single tool invocation.

### Rule 2: Stateless Pool Handles (Never Pass Raw Tensors)
- Never serialize molecular coordinates, matrices, or force tensors in your prompts.
- All intermediate states are stored persistently on the server and referenced via string pointer handles:
  $$\text{candidate\_pool\_id} \longrightarrow \text{thermo\_pool\_id} \longrightarrow \text{egnn\_scored\_pool\_id} \longrightarrow \text{pareto\_result\_uri}$$

### Rule 3: Validate Chemistry Before Compute
- LLMs frequently hallucinate chemically invalid structures (pentavalent carbon, unclosed rings).
- The server provides a synchronous pre-flight validator: `validate_spec(smiles_list=..., target_spec=...)`.
- Direct SMILES inputs to `run_cdft_thermo` are validated synchronously. If invalid, the server immediately rejects the call with an explanation of which atom index failed.

### Rule 4: Read `agent_action_required` on Failure
- If a job fails, the response will contain a `semantic_error` object with:
  - `error_code`
  - `summary`
  - `agent_action_required` (prescriptive guidance on what parameter to tune or what tool to call next).
- Do not blindly retry the identical call; apply the suggested remediation.

---

## 3. The 10 MCP Tools Reference

| # | Tool Name | Scope | Input Summary | Output Summary |
|---|---|---|---|---|
| 1 | `run_full_pipeline` | End-to-End | `target_spec`, `train_steps`, `num_candidates`, `top_k` | `job_id` |
| 2 | `train_swarm_agent` | Stage 1 (RL) | `target_spec`, `total_timesteps`, `curriculum`, `sa_penalty` | `job_id` $\to$ `model_weights_path` |
| 3 | `sample_candidates` | Stage 2 (Sample) | `model_weights_path`, `num_samples`, `continue_pipeline` | `job_id` $\to$ `candidate_pool_id` |
| 4 | `run_cdft_thermo` | Stage 3 (cDFT) | `candidate_pool_id` / `smiles_list`, `continue_pipeline` | `job_id` $\to$ `thermo_pool_id` |
| 5 | `run_egnn_quantum` | Stage 4 (EGNN) | `thermo_pool_id` / `candidates_dir`, `continue_pipeline` | `job_id` $\to$ `egnn_scored_pool_id` |
| 6 | `rank_pareto_frontier` | Stage 5 (Pareto) | `scored_pool_id`, `ranking_weights`, `max_sa_score` | `job_id` $\to$ `pareto_result_uri` |
| 7 | `get_job_status` | Observability | `job_id`, `wait_for_completion=True`, `timeout_seconds` | Status, progress %, logs, result |
| 8 | `cancel_job` | Control | `job_id` | `cancelled: bool` |
| 9 | `validate_spec` | Pre-flight | `smiles_list`, `target_spec` | Valid bool, offending atoms, reasons |
| 10 | `cleanup_artifacts` | Storage | `mode="keep_pareto_only"`, `older_than_hours` | Freed MB, deleted pools |

---

## 4. Pipeline Continuation Flag (`continue_pipeline`)

Tools 3, 4, and 5 support the `continue_pipeline: bool = False` argument:
- **`continue_pipeline=False` (Step-by-Step)**: Runs only that specific stage. Ideal when the LLM wants to inspect intermediate diversity, filter pools, or compare multiple models.
- **`continue_pipeline=True` (Chained Execution)**: Continues from that stage through all remaining stages to final Pareto export. For example:
  - `sample_candidates(..., continue_pipeline=True)`: Samples candidates and runs Stages 3, 4, 5 in one job.
  - `run_cdft_thermo(smiles_list=[...], continue_pipeline=True)`: Takes external molecules and screens them through Stages 3, 4, 5.

---

## 5. Full Material Specification Schema Reference

When defining new molecules or custom material layers, the LLM harness passes a `target_spec` dictionary (or YAML). To prevent rejected jobs or unphysical simulation divergence, the specification must strictly adhere to the complete 7-section schema below.

### Section Breakdown & Validation Rules

```yaml
# =============================================================================
# 1. METADATA
# =============================================================================
group_name: "custom_material_identifier"  # [Required] Unique snake_case string
version: "1.0.0"                          # [Required] Semantic version string
description: "High-level description of material purpose and mechanical/electronic targets."

# =============================================================================
# 2. REINFORCEMENT LEARNING REWARD TARGETS (rl_reward_targets)
# Normalized multi-objective reward weights and macroscopic physical limits.
# =============================================================================
rl_reward_targets:
  target_elasticity: 0.80       # [float: 0.0 to 1.0] Rewards rotatable bonds, chain flexibility, low Kuhn length
  target_tensile: 0.10          # [float: 0.0 to 1.0] Rewards rod-like PMI linearity, aromatic packing, high aspect ratio
  target_toughness: 0.10        # [float: 0.0 to 1.0] Rewards matched H-bonding donor/acceptor pairs, sacrificial bonds
  target_lightweight: 0.75      # [float: 0.0 to 1.0] Penalizes heavy atoms, rewards high fractional free volume (FFV)
  max_solvation_kcal: -3.5      # [float: kcal/mol] Solvation free energy ceiling (negative = soluble, near 0 = hydrophobic)
  min_wall_pressure_bar: 8.0    # [float: bar] Minimum cDFT pore wall contact pressure P_wall >= X (adhesion / wetting)
  max_molecular_weight: 450.0   # [float: amu] Maximum molecular mass ceiling for generated candidates
  min_valency: 2                # [int: 1 to 4] Minimum attachment ports (1 = monovalent, >=2 = polymer/network)
  sa_threshold: 4.8             # [Optional float: 1.0 to 10.0] Synthetic Accessibility penalty threshold
  sa_penalty_slope: 1.5         # [Optional float: 0.5 to 5.0] Penalty multiplier for exceeding SA threshold

# =============================================================================
# 3. GPU TENSOR LIMITS & ATOMIC CONSTRAINTS (tensor_limits)
# Governs static GPU tensor dimensions and element whitelist.
# =============================================================================
tensor_limits:
  max_sites: 128                # [int: 32, 64, 128, 180, 256] Static atom padding bound on GPU
  allowed_atomic_numbers: [1, 6, 8, 14] # [List[int]] Whitelisted elements: 1=H, 5=B, 6=C, 7=N, 8=O, 9=F, 14=Si, 15=P, 16=S, 17=Cl, 35=Br
  max_molecular_weight: 450.0   # [float] Must match rl_reward_targets.max_molecular_weight
  # Optional structural limits (enforced during filtering):
  min_aromatic_rings: 0         # [int] Minimum required aromatic ring count
  max_aromatic_rings: 2         # [int] Maximum allowed aromatic ring count
  min_rotatable_bonds: 4        # [int] Minimum rotatable single bonds
  max_rotatable_bonds: 16       # [int] Maximum rotatable single bonds
  min_fluorine_count: 0         # [int] Minimum fluorine atoms
  max_fluorine_count: 0         # [int] Maximum fluorine atoms
  min_silicon_count: 2          # [int] Minimum silicon atoms (for siloxanes)
  min_hba_count: 0              # [int] Minimum Hydrogen Bond Acceptors
  min_hbd_count: 0              # [int] Minimum Hydrogen Bond Donors
  min_sp3_fraction: 0.70        # [float: 0.0 to 1.0] Minimum fraction of sp3 hybridized carbons

# =============================================================================
# 4. SCAFFOLDS (scaffolds)
# Core templates with indexed wildcard attachment ports [*:1], [*:2], ...
# =============================================================================
scaffolds:
  - id: "core_template_1"       # [str] Unique template identifier
    smarts: "C(C[*:1])(C[*:2])(C[*:3])C[*:4]" # [str] Valid SMARTS/SMILES with wildcard attachment ports
    name: "pentaerythrityl 4-arm core"
    attachment_points: 4        # [int] Must match the number of [*:N] wildcard ports

# =============================================================================
# 5. BUILDING BLOCKS (building_blocks)
# Functional groups, linkers, and terminal caps grouped by category.
# Every block MUST contain indexed attachment ports [*:1] (and [*:2] for linkers).
# =============================================================================
building_blocks:
  functional_monomers:          # Category name (arbitrary string)
    - id: "dimethyl_silane_unit"
      smiles: "O[Si](C)(C)O[*:1]" # Valid SMILES with [*:1] attachment point
    - id: "ethylene_glycol_linker"
      smiles: "OCCO([*:1])[*:2]"  # Bifunctional linker with two attachment points
    - id: "methyl_cap"
      smiles: "C[*:1]"            # Terminal monovalent cap
    - id: "hydrogen_cap"
      smiles: "[H][*:1]"          # Terminal hydrogen cap (terminates growth)

# =============================================================================
# 6. ASSEMBLY RULES (assembly_rules)
# SMIRKS chemical reaction transforms and structural filters.
# =============================================================================
assembly_rules:
  reaction_type: "silanol_condensation"     # Human-readable reaction class
  smarts_transform: "[Si:1][O][H].[Si:2][O][H]>>[Si:1][O][Si:2]" # [str] Valid SMIRKS transform
  max_ring_count: 2                         # [int] Maximum ring count allowed
  min_rotatable_bonds: 6                    # [int] Minimum rotatable bonds
  min_h_bonds_per_molecule: 0               # [int] Minimum matched H-bonds

# =============================================================================
# 7. GENERATION SPECIFICATION (generation_spec)
# Settings for 3D combinatorial enumeration and distance geometry embedding.
# =============================================================================
generation_spec:
  target_molecules: 50000                   # [int] Total combinatorial enumeration budget
  random_seed: 42                           # [int] RNG seed for reproducible generation
  deterministic_enumeration: true           # [bool] Systematic (true) vs stochastic (false)
  conformer_embedding:
    algorithm: "distance_geometry_gpu"      # GPU distance geometry embedding engine
    num_conformations_per_molecule: 10      # Number of conformers per generated topology (10 to 50)
    clash_cutoff_angstrom: 0.85             # Minimum inter-atomic distance threshold
  jackhammer_mining:
    max_fragment_nodes: 32                  # Graph search node budget
    target_fragment_budget: 15000           # Subgraph cache size
```

---

## 6. Complete Working Specification Examples

### Example A: Hyper-Elastic Siloxane Elastomer (Midsole Layer)
*From `tests/data/hyperbranched_siloxane_elastomers.yaml`:*
```yaml
group_name: "hyperbranched_siloxane_elastomers"
version: "1.2.0"
description: "Inorganic highly-branched Si-O-Si backbones for extreme bond angles, large radius of gyration, and torsional hyper-flexibility."

rl_reward_targets:
  target_elasticity: 0.99
  target_tensile: 0.10
  target_toughness: 0.10
  target_lightweight: 0.85
  max_solvation_kcal: -2.0
  min_wall_pressure_bar: 3.0
  max_molecular_weight: 1200.0
  min_valency: 4

tensor_limits:
  max_sites: 256
  allowed_atomic_numbers: [1, 6, 8, 14]
  max_molecular_weight: 1200.0
  min_silicon_count: 5
  max_aromatic_rings: 0

scaffolds:
  - id: "d4_cyclotetrasiloxane"
    smarts: "O1[Si]([*:1])([*:2])O[Si]([*:3])([*:4])O[Si]([*:5])([*:6])O[Si]([*:7])([*:8])1"
    name: "octamethylcyclotetrasiloxane derivative"
    attachment_points: 8
  - id: "t_resin_branch"
    smarts: "[Si](O[*:1])(O[*:2])(O[*:3])[*:4]"
    name: "trifunctional silsesquioxane node"
    attachment_points: 4

building_blocks:
  silicone_monomers:
    - id: "pdms_linear_unit"
      smiles: "O[Si](C)(C)O[*:1]"
    - id: "trimethylsilyl_cap"
      smiles: "[Si](C)(C)(C)[*:1]"
    - id: "vinyl_silane_linker"
      smiles: "C=C[Si](C)(C)[*:1]"
    - id: "hydrogen_cap"
      smiles: "[H][*:1]"

assembly_rules:
  reaction_type: "silanol_condensation"
  smarts_transform: "[Si:1][O][H].[Si:2][O][H]>>[Si:1][O][Si:2]"
  max_ring_count: 2
  min_rotatable_bonds: 15

generation_spec:
  target_molecules: 50000
  random_seed: 1024
  deterministic_enumeration: true
  conformer_embedding:
    algorithm: "distance_geometry_gpu"
    num_conformations_per_molecule: 25
    clash_cutoff_angstrom: 0.90
  jackhammer_mining:
    max_fragment_nodes: 64
    target_fragment_budget: 20000
```

### Example B: Sacrificial H-Bond Toughness Resin (Adhesive Layer)
*From `tests/data/sacrificial_h_bond_toughness_resins.yaml`:*
```yaml
group_name: "sacrificial_h_bond_toughness_resins"
version: "1.0.0"
description: "Dense networks of hydrogen bond donors and acceptors designed for fracture toughness and sacrificial energy dissipation."

rl_reward_targets:
  target_elasticity: 0.20
  target_tensile: 0.20
  target_toughness: 0.95
  target_lightweight: 0.10
  max_solvation_kcal: -6.0
  min_wall_pressure_bar: 15.0
  max_molecular_weight: 550.0
  min_valency: 4

tensor_limits:
  max_sites: 128
  allowed_atomic_numbers: [1, 6, 7, 8]
  max_molecular_weight: 550.0
  min_hba_count: 4
  min_hbd_count: 4

scaffolds:
  - id: "urea_tetra_core"
    smarts: "O=C(N([*:1])[*:2])N([*:3])[*:4]"
    name: "urea 4-arm crosslinker core"
    attachment_points: 4
  - id: "pentaerythritol_ether_core"
    smarts: "C(CO[*:1])(CO[*:2])(CO[*:3])CO[*:4]"
    name: "pentaerythritol 4-arm ether core"
    attachment_points: 4

building_blocks:
  polar_hbond_units:
    - id: "hydroxyl_cap"
      smiles: "O[*:1]"
    - id: "amine_cap"
      smiles: "N[*:1]"
    - id: "urea_cap"
      smiles: "NC(=O)N[*:1]"
    - id: "amide_linker"
      smiles: "NC(=O)CCNC(=O)[*:1]"
    - id: "hydrogen_cap"
      smiles: "[H][*:1]"

assembly_rules:
  reaction_type: "condensation_and_urethane_coupling"
  smarts_transform: "[N:1][H].[C:2](=O)[O][H]>>[N:1][C:2](=O)"
  min_h_bonds_per_molecule: 6

generation_spec:
  target_molecules: 50000
  random_seed: 999
  deterministic_enumeration: true
  conformer_embedding:
    algorithm: "distance_geometry_gpu"
    num_conformations_per_molecule: 10
    clash_cutoff_angstrom: 0.85
  jackhammer_mining:
    max_fragment_nodes: 32
    target_fragment_budget: 15000
```

---

## 7. Case Study: Multi-Layer Engineered Shoe Sole

Consider an autonomous harness designing an athletic running shoe sole consisting of three distinct synthetic layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: OUTSOLE (Ground Contact)                                      │
│ Targets: High tensile strength, high wear resistance, high wall        │
│          pressure (P_wall >= 18 bar), rigid aromatic backbones         │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: MIDSOLE (Cushioning & Shock Absorption)                       │
│ Targets: Hyper-elasticity (0.95), low density, low weight (< 400 amu), │
│          flexible siloxane / aliphatic chains, high conformer entropy   │
├────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: ADHESIVE / INTERLAYER (Layer Bonding)                         │
│ Targets: High sacrificial H-bonding toughness (0.90), matched HBA/HBD  │
│          dense polar groups, high chemical valency crosslinking        │
└────────────────────────────────────────────────────────────────────────┘
```

### Agent Orchestration Flow:

#### Step 1: Synthesizing the Outsole (One-Shot Funnel)
```python
# The agent calls run_full_pipeline with high-strength constraints
outsole_spec = {
    "group_name": "shoe_outsole_high_grip",
    "rl_reward_targets": {
        "target_tensile": 0.85,
        "target_elasticity": 0.10,
        "min_wall_pressure_bar": 20.0,
        "max_molecular_weight": 750.0,
        "min_valency": 2
    }
}
resp = run_full_pipeline(target_spec=outsole_spec, train_steps=30000, num_candidates=512, top_k=10)
job_id = resp["job_id"]

# Long-poll once until completion
status = get_job_status(job_id=job_id, wait_for_completion=True, timeout_seconds=300)
outsole_mol2_dir = status["result"]["mol2_dir"]
```

#### Step 2: Synthesizing the Midsole (Exploratory Training + Sampling)
```python
# The agent trains an RL swarm on hyper-elastic targets
midsole_spec = {
    "group_name": "shoe_midsole_hyper_elastic",
    "rl_reward_targets": {
        "target_elasticity": 0.95,
        "target_lightweight": 0.80,
        "min_wall_pressure_bar": 4.0,
        "max_molecular_weight": 450.0
    }
}
train_resp = train_swarm_agent(target_spec=midsole_spec, total_timesteps=35000)
train_status = get_job_status(train_resp["job_id"], wait_for_completion=True)
model_path = train_status["result"]["model_weights_path"]

# Sample 1,024 candidates and continue directly through the pipeline
sample_resp = sample_candidates(
    model_weights_path=model_path,
    num_samples=1024,
    continue_pipeline=True,
    top_k=15
)
midsole_status = get_job_status(sample_resp["job_id"], wait_for_completion=True)
midsole_mol2_dir = midsole_status["result"]["mol2_dir"]
```

#### Step 3: Screening External Candidates for the Adhesive Layer
```python
# The agent retrieves candidate crosslinkers from an external database or prior run
candidate_smiles = [
    "O=C(NCCCCNC(=O)NCCO)NCCO",
    "c1(nc(nc(n1)NCCO)NCCO)NCCO",
    "C(CO)(CO)(CO)CO"
]

# Run synchronous intake validation
val = validate_spec(smiles_list=candidate_smiles)
if val["valid"]:
    cdft_resp = run_cdft_thermo(
        smiles_list=candidate_smiles,
        solvent_id="water",
        continue_pipeline=True,
        top_k=5
    )
    adhesive_status = get_job_status(cdft_resp["job_id"], wait_for_completion=True)
```

#### Step 4: Storage Maintenance (Pruning Scratch Pools)
```python
# After exporting top candidates for all three layers, prune heavy intermediate scratch files
cleanup_artifacts(mode="keep_pareto_only")
```

---

## 6. Interpreting Output Observables

When inspecting `funnel_summary.csv` or `pareto_result_uri`, the candidate molecules are ranked across:

| Observable | Symbol | Physical Meaning | Favorable Direction |
|---|---|---|---|
| **Wall Contact Pressure** | $P_{\rm wall}$ | Classical DFT interfacial mechanical pressure at pore boundary | Higher ($\ge 15$ bar for rigid soles) |
| **Excess Adsorption** | $\Gamma$ | Surface molecular accumulation density | Higher (indicates strong wall affinity) |
| **Boltzmann Likelihood** | $\log p_X(x)$ | Normalizing flow probability density; steric clash freedom | Higher (less negative) |
| **Conformer Variance** | $\text{Var}(U)$ | Microscopic energy variance across thermal ensemble | Lower (rigid) or moderate (elastic) |
| **EGNN Quantum Energy** | $U_{\rm EGNN}/N$ | Per-atom normalized quantum ground-state energy | Lower (thermodynamic stability) |
| **EGNN Force Residual** | $\|F_{\rm EGNN}\|_{\rm RMS}$ | Net root-mean-square quantum force norm | Lower ($< 0.05$ stationary minimum) |
| **Synthesizability** | $\text{SA}$ | RDKit Synthetic Accessibility (1 = simple, 10 = impossible) | Lower ($\le 6.0$ synthesizable) |

---

## 7. Semantic Error Recovery Matrix

If `get_job_status` returns `status="FAILED"`, inspect `semantic_error`:

| `error_code` | Root Cause | Agent Action Required |
|---|---|---|
| `CUDA_OUT_OF_MEMORY` | GPU VRAM exhausted during batch tensor operations. | Re-run with `batch_size=16` or reduce `num_candidates`. |
| `CDFT_DIVERGENCE_STERIC_CLASH` | Euler-Lagrange density profile diverged due to severe atom overlaps. | Call `sample_candidates` with `temperature=1.2` or retrain with `sa_penalty=True`. |
| `ALL_CANDIDATES_DROPPED_SA` | All generated candidates exceeded the RDKit SA synthesizability ceiling. | Increase `max_sa_score` (e.g. 7.0) or train Stage 1 with `sa_penalty=True`. |
| `ZERO_VALID_CANDIDATES_GENERATED` | No valid graphs generated within rollout budget. | Increase `max_molecular_weight` in spec or increase `total_timesteps`. |
| `CHEMICAL_INTAKE_REJECTED` | Input SMILES violated valence rules (e.g. pentavalent carbon). | Call `validate_spec` to inspect offending atom indices and fix SMILES string. |
| `ARTIFACT_POOL_NOT_FOUND` | Specified pool ID does not exist on disk. | Check `job_id` / `pool_id` string from previous stage output. |
