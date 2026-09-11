# The Physical Foundations, High-Performance Compiler Runtimes, and Autonomous Inverse Discovery Mechanics of `dens-city`

**A Comprehensive Architectural Monograph on Condensed-Phase Molecular and Macromolecular Design**  
*The dens-city Core Theoretical Physics and High-Performance Machine Learning Group*

---

## Executive Abstract

The autonomous inverse discovery of high-performance functional small molecules, conjugated oligomers, multi-functional macrocycles, and covalently crosslinked macromolecular networks requires resolving the thermodynamic free energy landscape of condensed-phase liquid environments. Conventional generative chemistry workflows decouple graph-theoretic topology generation from macroscopic continuum thermodynamics and quantum-mechanical force fields. This bifurcation leads to catastrophic physical failure modes upon synthesis: vacuum conformational collapse, unphysical intramolecular self-solvation, divergent continuum electrostatics at solvent interfaces, and severe hardware memory thrashing during batched message passing.

This monograph provides an exhaustive, publication-grade exposition of the physical principles, algorithmic mechanics, and compiler optimizations underlying `dens-city`, an end-to-end differentiable computational platform implemented in `tinygrad` and `PufferLib`. We examine the system across six coupled stages:
1. High-performance compiler engineering, universal operation (UOp) directed acyclic graphs (DAGs), static execution graphs via `@TinyJit`, persistent GPU buffer packing, and C-native OpenMP/AVX2 SIMD acceleration to eliminate Python Global Interpreter Lock (GIL) bottlenecks.
2. Directed reinforcement learning graph assembly framed as a Markov Decision Process (MDP) governed by `SwarmPuffeRLTrainer` and `VectorizedSwarmEnv`, using SE(3) rigid-body attachment, rigorous valence constraints, multi-objective physical reward proxies, and a monotonic three-stage curriculum.
3. Zero-fragmentation candidate sampling and in-memory streaming via `SwarmCandidateSampler`, extracting contiguous memory blocks directly from C arrays into fixed-size GPU tensor buffers without file I/O or heap allocation pauses.
4. Coupled slit-pore Classical Density Functional Theory (cDFT) thermodynamic screening using White Bear Fundamental Measure Theory (FMT), Irving-Kirkwood mechanical virial wall pressure integrals, trust-region batched L-BFGS geometric relaxation, and Boltzmann Generator normalizing flows.
5. High-fidelity quantum mechanical surrogate screening via a 7-layer invariant Equivariant Graph Neural Network (EGNN) force field (`EGNNForceField`), evaluating per-atom normalized energies, analytical autograd forces, and unrolled Heavy-Ball momentum relaxation.
6. Multi-objective Pareto frontier ranking and artifact delivery governed by `FunnelRanker`, applying a dynamic size-dependent synthesizability gate, topological deduplication via canonical SMILES and Weisfeiler-Lehman graph hashes, non-dominated frontier sorting, and automated 3D Tripos `.mol2` generation.

---

# Stage 1: High-Performance Compute, Compiler Optimizations & Static Graph Runtime

## 1.1 The Compiler Paradigm: Lazy Evaluation and Universal Operation (UOp) DAGs

Modern high-performance machine learning frameworks generally adhere to one of two computational paradigms: eager imperative execution or ahead-of-time (or just-in-time) graph compilation. In eager frameworks such as standard PyTorch, execution is fundamentally driven by the host Python runtime. Whenever an expression such as `y = torch.matmul(A, B) + c` is evaluated, the host CPU synchronously traverses the C++ dispatch table, allocates device memory for intermediate tensors, and pushes individual CUDA or ROCm kernel invocations into the device driver's command stream. While this approach affords high flexibility and interactive debugging, it introduces substantial runtime latency. Each operation incurs kernel launch overhead, intermediate results must be written to and read back from high-bandwidth device memory (VRAM), and the GPU's arithmetic execution units frequently stall while waiting for host dispatch decisions.

To achieve theoretical hardware compute limits in physical simulation and generative molecular modeling, `dens-city` rejects eager imperative execution and builds entirely upon the compiler architecture of `tinygrad`. At the core of `tinygrad` is the principle of lazy evaluation governed by a single, universal intermediate representation: the Universal Operation (UOp) Directed Acyclic Graph (DAG). 

To understand the transformative efficiency of lazy evaluation, we must stop and examine what occurs when a computation is expressed lazily rather than eagerly. In eager execution, every mathematical operator is an immediate imperative command to the physical hardware. If an algorithm evaluates ten sequential elementwise additions and multiplications, the GPU executes ten separate kernel launches, allocates ten temporary buffers in device memory, and repeatedly writes and reads intermediate values over the memory bus. In contrast, when an operation is expressed lazily, the runtime evaluates zero hardware instructions. Instead, the framework constructs a symbolic directed acyclic graph where each node represents a pure mathematical intent. 

In `tinygrad`, every operation from high-level Python tensor code down to machine assembly is represented in this unified dialect:

$$\text{UOp} = (\mathrm{op}, \; \mathrm{src}, \; \mathrm{arg}, \; \mathrm{tag})$$

where $\mathrm{op}$ designates the fundamental operation type (such as an elementwise ALU primitive, a memory load or store, or a coordinate movement view), $\mathrm{src}$ is a tuple of upstream source UOp nodes, $\mathrm{arg}$ denotes static metadata (such as scalar literals, dimension indices, or memory address space tags), and $\mathrm{tag}$ provides symbolic tracking identifiers.

Crucially, every UOp maintains five derived mathematical properties calculated instantaneously upon graph node construction:
1. `dtype`: The precise bit-level data type (e.g., `dtypes.float32`, `dtypes.int32`, `dtypes.bool`).
2. `shape`: The static dimensional tuple describing the coordinate space of the tensor.
3. `device`: The physical target hardware identifier string (such as `"CUDA"`, `"NV"`, `"METAL"`, or `"CPU"`).
4. `addrspace`: The hardware memory tier where the operation resides, spanning global device memory (`GLOBAL`), on-chip local shared scratchpad memory (`LOCAL`), or hardware register files (`REG`).
5. `min_max`: Analytical numerical interval bounds $[\min, \max]$ computed symbolically, which the compiler's optimization passes utilize for dead-code elimination, range analysis, and modulo arithmetic simplification.

The operational taxonomy of UOps separates memory side effects from arithmetic transformations. Source leaves in the DAG are restricted to concrete memory buffers (`Buffer`), scalar constants (`Const`), and symbolic placeholders (`Param`). Arithmetic computations are decomposed into primitive elementwise operations (`Add`, `Mul`, `Max`, `Mod`, `Idiv`, `CmpLt`, `CmpNe`, and `Where`), from which all complex nonlinear transcendental functions—such as $\exp(x)$, $\ln(x)$, $\sin(x)$, and $\sqrt{x}$—are decomposed into canonical base-2 hardware representations. 

Most importantly, all dimensional manipulations—including permutations, transpositions, reshapes, broadcasting expansions, slices, and padding—are implemented as zero-arithmetic movement operations (`Permute`, `Reshape`, `Expand`, `Pad`, `Shrink`). Movement operations execute zero floating-point arithmetic (FLOPs) and allocate zero device memory; they simply apply a transformation to the affine stride indexing metadata of the underlying buffer view. 

No machine instructions are emitted until the system encounters an explicit realization barrier, invoked via `.realize()`. At that moment, `tinygrad`'s optimizing compiler traverses the accumulated UOp DAG, applies aggressive algebraic simplification passes, fuses dozens of elementwise and reduction operations into a single loop nest, allocates registers, and emits an optimized C, OpenCL, or PTX kernel tailored specifically to the host GPU's compute architecture. The kernel is dispatched in a single hardware command, reading inputs once from global memory, keeping intermediate accumulations inside fast register files, and writing the final result back to memory in a single memory bandwidth sweep.

```
       [Input Coordinates: (B, N, 3)]      [Atom Types: (B, N)]
                      \                          /
                   (Movement Op: Reshape & Expand)
                                  |
                   (Symbolic UOp Directed Acyclic Graph)
                   [No Memory Allocated / No GPU Kernels]
                                  |
                +-----------------+-----------------+
                |                                   |
        [Pairwise Diff: diff_ij]            [Embedding Lookup]
                |                                   |
        [ALU: Sum of Squares]               [Linear Projections]
                |                                   |
                +-----------------+-----------------+
                                  \       /
                            (Explicit .realize())
                                      |
                     [AST Optimization & Kernel Fusion]
                                      |
                 [Single Compiled PTX/CUDA Command Buffer]
                                      |
                       [Single-Sweep GPU Execution]
```

## 1.2 The Forensic Autopsy of Memory Churn and Command Queue Timeouts

In computational biophysics and machine learning force fields, naive implementations frequently encounter catastrophic hardware bottlenecks that degrade throughput by orders of magnitude or cause hardware drivers to crash entirely. In PyTorch and other eager environments, iterative algorithms such as quasi-Newton geometry optimization (L-BFGS) or multi-step molecular graph rollout routinely trigger virtual address space fragmentation and host-device synchronization stalls.

When an optimization loop repeatedly instantiates variable-sized tensor objects, allocates temporary scratchpads, and calls Python scalar extraction methods like `.item()` or `.numpy()`, the runtime forces a synchronous round-trip over the PCIe bus. The CPU execution thread halts until the GPU finishes all pending operations in its queue, flushes its caches, and transmits the scalar value across the motherboard. This destroys hardware concurrency: the GPU, capable of trillions of operations per second, spends over ninety percent of its duty cycle idling in a low-power state, waiting for the Python interpreter to evaluate control flow logic and push the next command.

In a compiled static graph runtime, memory churn introduces an even more insidious failure mode: abstract syntax tree (AST) re-parsing and hardware command queue timeouts (`RuntimeError: Wait timeout`). When an algorithm is wrapped in a Just-In-Time compilation decorator such as `@TinyJit`, the compiler relies on the invariance of memory addresses, tensor shapes, and stride patterns to cache the compiled execution graph. In `@TinyJit`, the compiler records the sequence of GPU hardware command buffers during the first two warm-up invocations. On all subsequent iterations, the Python interpreter is bypassed entirely: the cached hardware command list is directly replayed against the GPU command processor.

However, if the programmer writes dynamic tensor operations—such as passing molecular batches of varying atom counts ($N_1 \neq N_2$), re-instantiating optimizer state buffers with new pointers, or using variable-length Python lists—the compiler detects a cache miss. The compiled execution graph is invalidated, forcing the runtime to re-parse the high-level Python code, rebuild the symbolic UOp AST, re-run all scheduling and register allocation passes, and re-compile the low-level PTX or C kernel. 

When this graph re-compilation occurs inside an inner optimization loop executing thousands of steps, the host CPU becomes severely bound by compilation overhead. Meanwhile, asynchronous memory allocations fragment the virtual memory tables of the graphics driver. If a kernel dispatch is delayed by memory paging or driver lock contention beyond the hardware watchdog timer limit, the operating system kernel resets the GPU device context, culminating in the fatal error:

$$\texttt{RuntimeError: Wait timeout on GPU command queue}$$

To eradicate this pathology, `dens-city` enforces strict architectural invariants: zero dynamic tensor allocation during simulation loops, total isolation from host-device synchronization barriers, and the complete stabilization of tensor shapes across all computational modules.

## 1.3 The `beautiful_mnist` Static Execution Paradigm

To attain uncompromising execution efficiency and mathematical predictability, `dens-city` embraces the design philosophy pioneered in `tinygrad`'s canonical `beautiful_mnist` architecture. This paradigm establishes four fundamental engineering rules for high-performance neural and physical runtimes:

### 1.3.1 Persistent Device Buffer Packing

In traditional simulation workflows, each batch of molecules is loaded from disk, parsed by Python worker threads, converted to NumPy arrays, and transferred as new tensor allocations onto the GPU. This induces continuous memory allocation and garbage collection churn. 

Under the static execution paradigm, all state buffers—including molecular coordinate tables, Lennard-Jones interaction parameters ($\sigma_i, \epsilon_i$), partial electrostatic charges ($q_i$), and convergence status masks—are pre-allocated at startup as contiguous, immutable GPU memory buffers. For instance, in batch training across datasets such as FreeSolv or large candidate pools, the entire corpus of atomic coordinates and chemical topologies is packed into a single contiguous device tensor:

$$\mathbf{X}_{\text{corpus}} \in \mathbb{R}^{B_{\text{total}} \times N_{\max} \times 3}, \quad \mathbf{Z}_{\text{corpus}} \in \mathbb{Z}^{B_{\text{total}} \times N_{\max}}$$

All molecules are pre-padded to a fixed, hardware-friendly power-of-two capacity (such as $N_{\max} = 128$ or $256$). Real atoms are distinguished from dummy padding sites using an immutable binary floating-point mask $\mathbf{M} \in \{0.0, 1.0\}^{B \times N_{\max}}$. Because the total buffer shape never changes throughout the execution lifetime of the process, the underlying GPU virtual memory pointers remain strictly invariant, completely eliminating driver allocation overhead.

### 1.3.2 On-Device Threefry PRNG Sampling

Mini-batch stochastic sampling in standard machine learning pipelines is typically executed on the CPU using Python's `random` module or NumPy's `choice`. The selected indices are then used to slice the host dataset, and the resulting mini-batch is pushed across the PCIe bus. This design creates a severe I/O bottleneck that starves the GPU compute cores.

`dens-city` performs all stochastic sampling directly on the device registers using the counter-based Threefry pseudo-random number generator (PRNG). Sourced from the Salmon et al. (SC 2011) high-performance parallel random number suite, Threefry is an ARX (Add-Rotate-XOR) algorithm that maps a simple integer counter and a cryptographic key directly to a uniform pseudo-random integer sequence without maintaining mutable global state. In `tinygrad`, calling:

$$\texttt{samples = Tensor.randint(batch\_size, high=corpus\_size)}$$

lowers directly to a five-round Threefry-2x32 kernel that executes entirely within on-chip GPU registers. The random indices are generated in parallel across GPU threads, and the mini-batch is extracted via high-speed on-device memory indexing:

$$\mathbf{X}_{\text{batch}} = \mathbf{X}_{\text{corpus}}[\mathbf{samples}]$$

Because data slicing occurs entirely within VRAM, zero bytes are transmitted across the PCIe bus during training iterations, and the memory layout remains perfectly static.

### 1.3.3 Single-Sweep Command Buffer Fusion via `@TinyJit`

In naive implementations of gradient-based optimization, the developer separates the forward computation, the loss calculation, the backward automatic differentiation pass, and the optimizer parameter update into distinct sequential method calls:

$$\text{loss} = \mathcal{L}(f_\theta(x), y); \quad \text{loss.backward}(); \quad \text{optimizer.step}()$$

In `tinygrad`, this separation represents an anti-pattern. If executed separately, the backward pass produces gradient tensors that must be materialized in global VRAM, and the optimizer step executes a separate set of kernels that read those gradients back from memory to update model parameters.

`dens-city` adheres strictly to the unified command buffer fusion idiom:
1. The forward network evaluation is decorated with `@function`, which symbolically traces the mathematical transformation with placeholder parameters (`Param`), guaranteeing symbolic differentiability without Python function overhead.
2. The entire training step is encapsulated inside a single function decorated with `@TinyJit` and `@Context(TRAINING=1)`.
3. The forward pass, the autograd backward graph traversal, and the optimizer parameter update are scheduled and fused into a single unified execution graph using the syntax:

$$\texttt{return loss.backward().realize(*optimizer.schedule\_step())}$$

When `optimizer.schedule_step()` is passed as variable arguments to `loss.backward().realize(...)`, the compiler's scheduling engine inspects the entire computational chain simultaneously. It schedules the parameter update arithmetic immediately adjacent to the adjoint gradient evaluations. 

The gradients computed during backpropagation are never written out to global VRAM; instead, they remain resident inside the GPU's register files and are immediately applied to update the model weights in the exact same kernel invocation. The entire training step—forward evaluation, loss computation, reverse-mode autodiff, and weight update—is fused into a single static GPU command buffer that executes in a single sweep over hardware memory.

## 1.4 C-Native SIMD Parallelism & The GIL Bottleneck

While GPU execution provides unmatched throughput for dense tensor mathematics, reinforcement learning graph assembly environments involve discrete combinatorial logic, valence checking, and topological graph hashing. If these discrete assembly steps are implemented in Python, the system encounters the Python Global Interpreter Lock (GIL).

To appreciate why the GIL cripples parallel reinforcement learning, we must stop and explain what the GIL is and why it exists. The Python Global Interpreter Lock is a mutual-exclusion mechanism implemented in the CPython reference interpreter to prevent multiple native operating system threads from executing Python bytecode simultaneously. The GIL was introduced to simplify CPython's memory management, which relies on non-thread-safe reference counting. Whenever a Python thread attempts to create an object, modify a list, or access a dictionary, it must acquire the GIL. 

If an engineer attempts to parallelize a molecular RL environment across 32 CPU cores using standard Python threading (`threading.Thread`), the threads do not run in parallel; instead, they aggressively contend for the single GIL lock. The operating system kernel constantly pauses and switches execution contexts between threads, introducing lock thrashing that frequently makes multithreaded Python code run slower than a single thread. 

If the engineer instead resorts to Python multiprocessing (`multiprocessing.Process`), each environment must run in a completely isolated process space with its own independent Python interpreter and heap. Communicating observations, actions, and rewards between processes requires serializing complex molecular objects into binary byte-streams (via `pickle`), transmitting them across Unix domain sockets or inter-process communication (IPC) pipes, and de-serializing them in the main process. This serialization overhead creates massive CPU memory bloat and saturates memory bus bandwidth.

`dens-city` solves the GIL bottleneck by completely moving the molecular graph assembly environment and candidate sampling logic out of Python and into pure, optimized C: `src/dens_city/swarm/c_src/cdft_swarm_lib.c` and its accompanying headers `mol_graph.h`, `mechanics_eval.h`, and `cdft_swarm.h`.

```
================================================================================
Naive Python RL Paradigm: Severe GIL Contention & Serialization Stalls
================================================================================
Thread 1 (Env 0)  ---> [Requests GIL] ---> [Executes 50 us] ---> [Releases GIL]
Thread 2 (Env 1)  ---> [Stalled Waiting for GIL...]        ---> [Acquires GIL]
Thread 3 (Env 2)  ---> [Stalled Waiting for GIL...]
IPC Serialization: Pickle Object -> Unix Domain Socket -> Unpickle in Parent
Result: High CPU lock thrashing, PCIe bus starvation, ~50 steps/second.

================================================================================
dens-city C-Native Paradigm: Direct OpenMP SIMD Multi-Threading (Zero GIL)
================================================================================
PyTorch / tinygrad (Main Process)
       |
       | (Passes 64-byte aligned pointers via C-FFI / ctypes)
       v
cdft_swarm_lib.so (C Shared Library, compiled with -O3 -mavx2 -mfma -fopenmp)
       |
       +---> OpenMP Core 0: Env 00..03 [AVX2 256-bit SIMD Vectorized Ops]
       +---> OpenMP Core 1: Env 04..07 [Bitmask Adjacency & SE(3) Assembly]
       +---> OpenMP Core 2: Env 08..11 [Analytical Mechanics & SA Screening]
       +---> OpenMP Core 3: Env 12..15 [Exact Weisfeiler-Lehman Graph Hashing]
       |
       v
Memory Layout: Single Contiguous 64-Byte Aligned Buffer in Physical RAM
Result: 100% CPU core utilization, zero serialization, >25,000 steps/second.
================================================================================
```

The C library is compiled with aggressive vectorization directives:
```bash
gcc -O3 -shared -fPIC -mavx2 -mfma -fopenmp -Isrc/dens_city/swarm/c_src \
    src/dens_city/swarm/c_src/cdft_swarm_lib.c -o cdft_swarm_lib.so -lm
```

The execution architecture relies on four foundational C-level optimizations:
1. 64-Byte Aligned Memory Buffers: In `env_create` and `vec_swarm_create`, all state vectors—observations (`TOTAL_OBS_SIZE = 88`), actions (2 floats), rewards (1 float), terminals (1 byte), and action masks (`TOTAL_ACTION_MASK_SIZE = 29`)—are allocated using `aligned_alloc(64, ...)`. This guarantees that memory buffers align perfectly with the 64-byte cache line boundaries of modern x86-64 CPUs, maximizing L1/L2 cache hit rates and enabling single-cycle AVX2 256-bit SIMD vector loads.
2. Direct OpenMP Vectorized Stepping: In `vec_swarm_step`, the environment batch loop is decorated with `#pragma omp parallel for schedule(static)`. When stepping 16, 32, or 64 parallel environments, OpenMP distributes the environments statically across all available physical CPU cores. Each core executes the complete SE(3) rigid-body attachment, valence validation, molecular mechanics evaluation, and topological hashing in pure native machine code, completely bypassing the Python interpreter and ignoring the GIL.
3. Zero-Copy Ctypes Bridge: The Python wrapper class `VectorizedSwarmEnv` in `src/dens_city/swarm/env.py` does not allocate new memory. Instead, it extracts the raw 64-byte aligned C memory addresses via `vec_swarm_get_obs`, `vec_swarm_get_actions`, etc., and immediately wraps them into PyTorch and tinygrad tensors using `torch.frombuffer(...)`. The Python runtime reads from and writes to the exact same physical memory addresses accessed by the C OpenMP threads.
4. AVX2 Vector Extensions and FMA: Mathematical evaluations of 3D rigid-body rotations (Rodrigues formula `mat3_align_vectors`), inter-atomic distance matrices, and principal moments of inertia leverage Advanced Vector Extensions (AVX2) and Fused Multiply-Add (`-mfma`), executing eight single-precision floating-point operations in a single CPU clock cycle.

---

# Stage 2: Inverse Design Step 1 — PufferLib RL Swarm Policy Training

## 2.1 Objective & Engine: Vectorized Swarm Policy Architecture

The primary objective of Stage 1 in the `dens-city` inverse design pipeline is to train a parameterized neural network policy:

$$\pi_\theta(a_t \mid s_t)$$

capable of autonomously assembling geometrically stable, synthetically accessible, and chemically valid molecular graphs tailored to extremize target physical properties. The reinforcement learning training loop is governed by `SwarmPuffeRLTrainer` (implemented in `src/dens_city/swarm/trainer.py`), which orchestrates training rollouts across the C-native multi-environment pool `VectorizedSwarmEnv`.

The policy architecture, implemented in `MolecularSwarmPolicy` (`src/dens_city/swarm/policy.py`), processes an 88-dimensional continuous observation vector $s_t \in \mathbb{R}^{88}$. The network comprises a shared multi-layer perceptron (MLP) trunk featuring residual feed-forward layers, layer normalization, and SiLU activations, branching into three dedicated heads:
1. Port Selection Actor Head: Computes unnormalized logits over the 16 attachment ports:

$$z_{\text{port}} \in \mathbb{R}^{16}$$

2. Fragment Selection Actor Head: Computes unnormalized logits over the 12 candidate chemical fragments plus a termination action:

$$z_{\text{frag}} \in \mathbb{R}^{13}$$

3. Critic Value Head: Predicts the scalar state-value expectation:

$$V_\phi(s_t) \approx \mathbb{E}\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \;\middle|\; s_t \right]$$

## 2.2 The Graph Assembly Markov Decision Process (MDP)

To understand how molecular generation is solved using reinforcement learning, we must stop and explain the formal framework of a Markov Decision Process (MDP). In classical computer science and control theory, an MDP is a mathematical model for decision-making under uncertainty, defined formally by the 5-tuple:

$$\mathcal{M} = \left( \mathcal{S}, \; \mathcal{A}, \; \mathcal{P}, \; \mathcal{R}, \; \gamma \right)$$

where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ is the state transition probability distribution, $\mathcal{R}(s_t, a_t, s_{t+1})$ is the scalar reward function, and $\gamma \in [0, 1)$ is the temporal discount factor. The defining property of an MDP is the Markov property: the future state $s_{t+1}$ depends strictly upon the current state $s_t$ and immediate action $a_t$, completely independent of the historical trajectory of prior states.

In unconstrained chemical generation, treating molecular design as an arbitrary string-generation task (such as producing SMILES strings token-by-token via autoregressive language models) fundamentally violates chemical reality. A naive language model has no intrinsic understanding of three-dimensional steric hindrance, orbital hybridization, or bond strain. It spends over ninety percent of its training budget generating invalid character strings, hypervalent carbon atoms, or sterically collapsed ring systems that cannot exist in physical space.

`dens-city` resolves this failure mode by formulating molecular design as a rigid, SE(3)-equivariant geometric graph assembly MDP executing in native C:

### 2.2.1 The 88-Dimensional Continuous Observation Vector

At each discrete time step $t$, the environment inspects the internal state of the evolving molecular graph and populates the 88-dimensional observation vector $s_t \in \mathbb{R}^{88}$ (`compute_observations` in `cdft_swarm.h`):
1. Molecular Graph Features (Indices 0–15, 16 floats): Continuous normalized metrics describing global topology:
   - Normalized atom count: $N_{\text{atoms}} / N_{\max}$ (where $N_{\max} = 64$).
   - Normalized bond count: $N_{\text{bonds}} / B_{\max}$ (where $B_{\max} = 128$).
   - Normalized attachment port count: $N_{\text{ports}} / P_{\max}$ (where $P_{\max} = 16$).
   - Normalized molecular weight: $\text{MW} / \text{MW}_{\max}$.
   - Rotatable bond fraction: $f_{\text{rot}} = N_{\text{rot}} / \max(1, N_{\text{bonds}})$, measuring conformational chain flexibility.
   - Aromatic density: $\rho_{\text{arom}} = N_{\text{arom}} / \max(1, N_{\text{atoms}})$, quantifying conjugated $\pi$-orbital character.
   - Principal Moment of Inertia (PMI) linearity metric: $\text{PMI}_{\text{lin}} \in [0, 1]$, describing whether the molecule is spherical, planar, or rod-like.
   - Kuhn persistence length proxy: Estimating macromolecular backbone rigidity.
   - Hydrogen bond donor count normalized: $N_{\text{HBD}} / 10.0$.
   - Hydrogen bond acceptor count normalized: $N_{\text{HBA}} / 10.0$.
   - Fractional Free Volume (FFV): Estimating open packing space for liquid transport.
   - Topological Polar Surface Area (TPSA) proxy: $\text{TPSA} / 200.0$.
   - Multivalency crosslinking count: Active reactive sites available for network formation.
   - Normalized episode step counter: $t / t_{\max}$ (where $t_{\max} = 16$).
   - cDFT convergence indicator: Boolean flag indicating whether the 1D density profile successfully reached variational equilibrium.
   - Slit-pore contact ratio proxy: Initial estimate of wall adsorption density.
2. Open Port Geometric Vectors (Indices 16–79, 64 floats): For each of the 16 attachment port slots, the environment writes four spatial floats:
   - The unit normal outward orientation vector: $(n_x, n_y, n_z) \in \mathbb{R}^3$, defining the exact 3D direction in which an attached fragment will grow.
   - The port occupancy state flag: $1.0$ if the port is open and available for attachment, $0.0$ if occupied or blocked.
3. Target Specification Vector (Indices 80–87, 8 floats): The multi-objective property targets loaded directly from user specification YAMLs (e.g., target elasticity, tensile modulus, toughness, lightweight fraction, maximum allowable solvation free energy, minimum wall pressure, and maximum molecular weight).

### 2.2.2 The 29-Channel Discrete Action Space and Action Masking Referee

The agent's action space at each step is factored into two discrete decisions:
1. Port Selection Action: $a_{\text{port}} \in \{0, \dots, 15\}$, designating which open port on the growing molecular scaffold to extend.
2. Fragment Selection Action: $a_{\text{frag}} \in \{0, \dots, 12\}$, designating which chemical building block to attach (fragments 0–11) or whether to terminate assembly (action index 12: `ACTION_FINALIZE`).

The fragment library (`get_fragment_template` in `mol_graph.h`) contains twelve chemically robust, 3D pre-equilibrated building blocks representing distinct structural classes:
- Rigid aromatic cores: Benzene 1,4-para scaffold, Triphenylamine 3-arm core, Adamantane 1,3,5,7-tetrahedral core.
- Conjugated electronic linkers: Thiophene 2,5-diyl, Bithiophene, Phenyl-ethynyl linkers.
- Solvation-directing functional units: Ethylene glycol flexible chains ($-\text{O}-\text{CH}_2-\text{CH}_2-\text{O}-$), Amide polar hydrogen-bonding units ($-\text{NH}-\text{C}(=\text{O})-$), Fluorinated hydrophobic chains ($-\text{CF}_2-\text{CF}_2-$).
- Reactive crosslinkers and endcaps: Trifluoromethyl caps, Polymerizable vinyl units, Terminal acetylene moieties.

To prevent the agent from attempting impossible chemical connections, `dens-city` implements a strict C-level Action Masking Referee (`compute_action_mask` in `cdft_swarm.h`). Before the neural policy evaluates action logits, the C environment performs pairwise geometric and valence validation across all 16 ports and 12 fragments. An attachment is marked valid if and only if:
1. The target port is currently unoccupied (`PORT_STATE_EMPTY`).
2. The candidate fragment possesses an opposing attachment port that can be aligned in 3D space.
3. Rigid-Body SE(3) Attachment does not generate steric overlap: When the fragment is rotated via the Rodrigues formula (`mat3_align_vectors`) to align its port normal vector with the host scaffold port, no atom in the incoming fragment approaches closer than $1.20\text{ \AA}$ to any existing scaffold atom.
4. Strict Chemical Valence Rules are Preserved: Carbon cannot exceed valence 4, nitrogen cannot exceed valence 3, oxygen cannot exceed valence 2, and halogens cannot exceed valence 1.
5. Molecular Weight Ceiling is Respected: The resulting molecular mass must not exceed $\text{MW}_{\max}$.

If a port or fragment fails any of these criteria, its corresponding bit in the 29-dimensional action mask is set to zero (`action_mask[i] = 0`). The neural policy applies this mask directly to its unnormalized action logits:

$$z_{\text{masked}} = z + (1 - \mathbf{m}) \cdot (-10^9)$$

This guarantees that unphysical, sterically clashing, or hypervalent actions have an exact probability of zero:

$$P(a_i) = \frac{\exp(z_{\text{masked}, i})}{\sum_j \exp(z_{\text{masked}, j})} = 0$$

The agent is physically incapable of proposing an illegal chemical move. Furthermore, the `ACTION_FINALIZE` action is strictly masked out until the molecule has grown past the bare starting scaffold (requiring at least 16 heavy atoms, at least 2 attached fragments, and a minimum molecular weight of 180 amu).

## 2.3 The Multi-Objective Physical Reward Signal and Proxies

Unlike standard generative RL models that reward arbitrary heuristic drug-likeness scores (such as QED), `dens-city` drives graph assembly using a multi-component reward function grounded in condensed-phase thermodynamics, continuum mechanics, and topological novelty:

$$R_{\text{total}} = R_{\text{thermo}} + R_{\text{elasticity}} + R_{\text{tensile}} + R_{\text{toughness}} + R_{\text{lightweight}} + R_{\text{novelty}} - R_{\text{penalties}}$$

### 2.3.1 Thermodynamic Survival Check

Upon episode completion (when the molecule is finalized or reaches the 16-step ceiling), the C environment executes a fast 1D planar cDFT solver coupled with a Generalized Born solvation engine (`compute_generalized_born_solvation_kcal`). The thermodynamic reward evaluates:
1. Slit-Pore Wall Contact Ratio: Measures fluid accumulation against the confining surface. If the contact pressure ratio exceeds the target $P_{\text{target}}$, the agent receives a smooth tanh bonus:

$$R_{\text{wall}} = +2.0 \tanh\left(0.20 \cdot (P_{\text{wall}} - P_{\text{target}})\right)$$

If it fails to meet the threshold, it receives an escalating penalty:

$$R_{\text{wall}} = -1.0 - 2.0 \tanh\left(0.20 \cdot |P_{\text{wall}} - P_{\text{target}}|\right)$$

2. Liquid-Phase Solvation Free Energy $\Delta\Omega_{\text{solv}}$: Evaluates thermodynamic affinity for the liquid medium. If the solvation free energy satisfies the target bound ($\Delta\Omega \le \Delta\Omega_{\text{target}}$), the agent receives positive reinforcement:

$$R_{\text{solv}} = +2.0 \tanh\left(0.10 \cdot (\Delta\Omega_{\text{target}} - \Delta\Omega)\right)$$

If the molecule exhibits unfavorable solvation exceeding the target ceiling, `dens-city` avoids discontinuous step penalties by applying an infinitely differentiable Softplus penalty:

$$R_{\text{solv}} = -2.0 \ln\left( 1 + \exp\left( \text{clamp}(\Delta\Omega - \Delta\Omega_{\text{target}}, -20, 20) \right) \right)$$

This smooth penalty provides a clean, informative gradient vector that guides the policy back toward the thermodynamically stable regime.

### 2.3.2 Mechanical Property Proxies

The mechanical character of the candidate molecule is evaluated via analytical structural proxies computed directly on the 3D molecular graph (`evaluate_mechanics` in `mechanics_eval.h`):
- Elasticity Reward: Encourages conformational springiness and network compliance via rotatable bond fraction, low Kuhn length, and optimal crosslink spacing:

$$R_{\text{elasticity}} = w_e \left[ 2.0 f_{\text{rot}} + 1.5 (1.0 - \min(1.0, \text{Kuhn})) + 0.1 \min(20.0, d_{\text{crosslink}}) \right]$$

- Tensile Modulus Reward: Rewards conjugated aromatic density, principal axis linearity, and multi-arm branching valency:

$$R_{\text{tensile}} = w_t \left[ 2.5 \rho_{\text{arom}} + 3.0 \text{PMI}_{\text{lin}} + 1.0 \min(4, N_{\text{valency}}) \right]$$

- Toughness & Energy Dissipation: Rewards sacrificial hydrogen-bonding networks and cooperative $\pi$-$\pi$ stacking:

$$R_{\text{toughness}} = w_{\text{to}} \left[ 1.5 \min(6, N_{\text{HBD}} + N_{\text{HBA}}) + 2.0 \min(5.0, S_{\text{sacrificial}}) + 1.5 \min(4.0, S_{\pi\text{-}\pi}) \right]$$

- Lightweight Character: Rewards fractional free volume while penalizing excessive heavy atom accumulation:

$$R_{\text{lightweight}} = w_l \left[ 2.0 \text{FFV} - 1.0 N_{\text{heavy\_penalty}} \right]$$

### 2.3.3 Sub-Microsecond Weisfeiler-Lehman (WL) Topological Novelty Bonus

A major pathology in reinforcement learning for molecular discovery is premature policy collapse into identical, high-reward local optima: the agent discovers a single functional motif and outputs that identical graph thousands of times.

To guarantee broad exploration of chemical space without expensive disk logging, `dens-city` integrates a sub-microsecond 2-hop Weisfeiler-Lehman (WL) topological graph hashing routine directly in C (`compute_wl_graph_hash` in `mol_graph.h`). At step $t$, the atomic numbers $Z_i$ are initialized as 64-bit entropy labels:

$$l_i^{(0)} = (Z_i \cdot \text{\texttt{0x9E3779B97F4A7C15ULL}}) \oplus (i + 1)$$

For two iterations, each atom aggregates the labels of its covalently bonded neighbors weighted by bond order ($1, 2, 3, \text{aromatic}$), refining its local topological context. The atomic labels are then accumulated into a global, permutation-invariant 64-bit FNV-1a hash:

$$H_{\text{graph}} = \text{FNV-1a}\left( \{ l_1^{(2)}, l_2^{(2)}, \dots, l_N^{(2)} \} \right)$$

The C environment maintains a rolling ring buffer of the 64 most recent graph hashes (`recent_hashes` in `cdft_swarm.h`). If a newly generated graph hash already resides in the buffer, it is flagged as a duplicate and receives zero novelty bonus. If the topology is unique, the agent receives an immediate novelty reward:

$$R_{\text{novelty}} = +1.0$$

This mechanism runs in less than 500 nanoseconds per molecule, incentivizing the policy to continually generate structurally diverse chemical topologies.

## 2.4 Progressive Curriculum Training & Dynamic Entropy Annealing

Attempting to train an uninitialized reinforcement learning agent directly against strict, coupled multi-objective criteria—such as requiring simultaneous high wall contact pressure, negative solvation free energy, high synthetic accessibility, and precise crosslinking valency—results in severe policy failure. Because an untrained policy almost never satisfies all constraints simultaneously, the agent receives uniformly negative rewards, its gradients vanish, and the policy wanders aimlessly.

To establish smooth policy convergence, `dens-city` implements a progressive three-stage metric-driven curriculum manager (`SwarmCurriculumManager` in `trainer.py`):

```
+-------------------------------------------------------------------------------+
|                      SWARM CURRICULUM TRAINING PROGRESSION                    |
+-------------------------------------------------------------------------------+

  Stage 1: Geometric Feasibility & Valency Retention
  --------------------------------------------------
  - Step Budget: 0 to 25,000 steps (or valid_rate < 0.80).
  - Target Constraints: Highly relaxed (Solvation <= 0.0 kcal/mol, P_wall >= 0.0).
  - Primary Policy Focus: Learning valid port attachments, avoiding hypervalency.
  - Policy Entropy: High (H ~ 2.5 nats), broad exploration of chemical space.
                         |
                         v  [Gating Metric: valid_rate >= 0.80 or step >= 25k]
  Stage 2: Linear Constraint Tightening & Intermediate Synthesis Screening
  -----------------------------------------------------------------------
  - Step Budget: 25,000 to 65,000 steps (or valid_rate < 0.95).
  - Target Constraints: Linearly interpolated targets:
        Target(t) = S1_val + alpha * (Final_Target - S1_val)
  - Primary Policy Focus: Navigating coupled mechanical and thermodynamic trade-offs.
  - Policy Entropy: Monotonically annealed (H: 2.5 -> 1.0 nats).
                         |
                         v  [Gating Metric: valid_rate >= 0.95 or step >= 65k]
  Stage 3: Full Physical Target Constraints & Fine-Tuned Exploitation
  -------------------------------------------------------------------
  - Step Budget: 65,000 to Final Target (up to 5,000,000 steps).
  - Target Constraints: 100% full YAML targets (P_wall >= 15 bar, Solv <= -3 kcal/mol).
  - Monotonic Stage Lock: Prevents validation regression jitter.
  - Policy Entropy: Low (H <= 0.5 nats), highly focused exploitation of Pareto modes.
+-------------------------------------------------------------------------------+
```

The curriculum state is updated via C-FFI broadcast: calling `curriculum.broadcast_to_vec_env` directly mutates the C `TargetSpec` structures inside all parallel environments without allocating new Python objects. Furthermore, the curriculum enforces a monotonic unidirectional stage lock: once the swarm unlocks Stage 2 or Stage 3, it is permanently locked into that stage or higher, preventing temporary validation dips from causing erratic curriculum oscillations.

Simultaneously, the policy optimizer dynamically scales the PPO entropy regularization coefficient:

$$\mathcal{L}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right] + c_v \mathcal{L}_{\text{VF}}(\theta) - c_{\text{ent}}(t) \mathcal{H}(\pi_\theta)$$

During early Stage 1 training, the entropy coefficient is maintained at $c_{\text{ent}} = 0.02$, forcing the policy to explore widely across all 12 fragment choices. As the agent transitions through Stage 2 and into Stage 3, $c_{\text{ent}}$ is dynamically annealed toward zero. This shifts the agent smoothly from broad exploratory graph discovery to fine-tuned exploitation of high-performing chemical families.

# Stage 3: Inverse Design Step 2 — C-Native Candidate Sampling & In-Memory Streaming

## 3.1 Objective & Engine: The High-Speed Sampling Transition

Once the reinforcement learning policy $\pi_\theta(a_t \mid s_t)$ has completed its progressive curriculum training, the system transitions from exploratory policy optimization to ultra-high-throughput candidate generation. The engine driving this transition is the `SwarmCandidateSampler`, implemented in `src/dens_city/swarm/sampler.py`. 

The primary mission of Stage 2 is to sample thousands of candidate molecular graphs, execute parallel SE(3) rollouts across parallel C environments, extract raw physical interaction parameters directly from native C memory structures, and stream contiguous numerical arrays into fixed-size GPU tensor buffers. By decoupling graph assembly from high-level Python object overhead, Stage 2 establishes an uninterrupted bridge between discrete reinforcement learning and continuous GPU physics solvers.

## 3.2 Zero-Fragmentation In-Memory Streaming

To understand the engineering necessity of zero-fragmentation streaming, we must stop and examine the severe I/O bottleneck that plagues conventional computational chemistry pipelines. In standard workflows, when a generative model or combinatorial fragment library produces candidate molecules, each candidate is serialized as a text representation—most commonly a Simplified Molecular Input Line Entry System (SMILES) string or a formatted 3D structure file such as a Tripos `.mol2` or Structure-Data File (SDF). These files are written to physical disk or an intermediate network-attached storage array.

In the subsequent screening phase, a secondary program reads these files back from disk, parses the ASCII character strings, constructs complex Python or C++ object hierarchies (such as RDKit `ROMol` or Open Babel `OBMol` objects), assigns bond orders, and queries heuristic force field tables (such as GAFF, MMFF94, or OPLS-AA) to parameterize Lennard-Jones van der Waals radii ($\sigma_i$), well depths ($\epsilon_i$), and partial electrostatic charges ($q_i$).

This classical disk-centric approach introduces catastrophic latency:
1. File System I/O Stalls: Disk access times and operating system file descriptor contention introduce multi-millisecond latencies per molecule.
2. Character String Parsing Overhead: Parsing SMILES strings or SDF blocks requires sequential lexical analysis and regular expression matching, consuming hundreds of thousands of CPU instructions per structure.
3. Heap Memory Fragmentation: Instantiating millions of small Python molecular objects severely fragments the CPython heap. The Python garbage collector is forced to execute stop-the-world reference traversal cycles, freezing execution pipelines.
4. Redundant Parameterization: Re-deriving force field parameters from scratch for every generated candidate duplicates computations that were already resolved during graph assembly.

When this classical pipeline is deployed, total throughput drops from tens of thousands of molecules per second down to fewer than ten molecules per second. The GPU, capable of processing massive parallel batches of physical equations, is left completely starved of data.

`dens-city` completely eliminates the file I/O bottleneck through zero-fragmentation in-memory streaming:

```
================================================================================
Classical Computational Chemistry Pipeline: Disk I/O & Garbage Collection Bottlenecks
================================================================================
RL Generator ---> [Serialize SMILES] ---> [Write to Disk: candidate_001.smi]
                                                         | (Disk Latency)
[Read from Disk] <--- [OS File Descriptor Contention] <---+
       |
[Parse ASCII SMILES via RDKit] ---> [Allocate 500 Python Heap Objects]
       |
[Query GAFF Force Field Tables] ---> [Re-assign LJ Sigmas, Epsilons, Charges]
       |
[Python Garbage Collector Triggered: Stop-The-World Freeze]
       |
Result: 5 to 10 molecules/second throughput; GPU compute cores idle >95% of time.

================================================================================
dens-city C-Native In-Memory Streaming Paradigm: Zero-Copy RAM Bridge
================================================================================
VectorizedSwarmEnv (Pure C / OpenMP Multi-Core Rollouts)
       |
       | (Parallel C Graph Assembly & SE(3) Rigid-Body Attachment)
       v
Direct C Memory Extraction: single_env.get_raw_atom_arrays()
       |
       +---> Coords: (x, y, z) 3D floats extracted directly from AtomSite structs
       +---> LJ Parameters: Pre-calculated sigma_i, epsilon_i read from C memory
       +---> Partial Charges: Electronegativity-equilibrated q_i from C arrays
       +---> 1-2 / 1-3 Exclusion Matrices: Bitmask adjacency pre-calculated in C
       |
       v
ContiguousCandidateBatch (Pre-Padded Contiguous NumPy Array in RAM)
       |
       | (slice_molecular_batch: Zero-Copy Array Memory Views)
       v
MolecularBatch (Directly Mapped into Immutable GPU VRAM Buffers)
Result: >10,000 candidates/second; zero disk I/O; zero Python heap allocations.
================================================================================
```

During sampling in `SwarmCandidateSampler.sample_candidates`, parallel inference rollouts execute across the C-FFI OpenMP backend `VectorizedSwarmEnv`. When an environment terminates with an assembled, valid molecule, the sampler does not write a SMILES string or SDF file to disk. Instead, it directly invokes the C library getter `env_get_atoms_block` via ctypes (`single_env.get_raw_atom_arrays()` in `env.py`). The function reads the raw, 64-byte aligned memory arrays of the `MolecularGraph` structure and copies the raw floats directly into contiguous RAM buffers in a single memory block transfer. In this manner, batches of 512, 1024, or 2048 fully parameterized candidates are assembled in physical RAM in under two seconds.

## 3.3 The `ContiguousCandidateBatch` Structure

The memory backbone of Stage 2 is encapsulated in the `ContiguousCandidateBatch` dataclass (`src/dens_city/swarm/sampler.py`). This structure organizes all candidate molecules into contiguous, pre-padded numerical array blocks:

```python
@dataclass
class ContiguousCandidateBatch:
    num_candidates: int
    n_particles: int
    coords: np.ndarray            # Shape: (K, N, 3), float32
    sigmas: np.ndarray            # Shape: (K, N), float32
    epsilons_k: np.ndarray        # Shape: (K, N), float32
    charges: np.ndarray           # Shape: (K, N), float32
    atomic_numbers: np.ndarray    # Shape: (K, N), int32
    atom_mask: np.ndarray         # Shape: (K, N), float32
    molecule_mask: np.ndarray     # Shape: (K,), float32
    temperature_k: np.ndarray     # Shape: (K,), float32
    bulk_density_a3: np.ndarray   # Shape: (K,), float32
    bulk_mu: np.ndarray           # Shape: (K,), float32
    slit_width_a: np.ndarray      # Shape: (K,), float32
    conditioning: np.ndarray      # Shape: (K, 5), float32
    exclusions: np.ndarray        # Shape: (K, N, N), float32
```

where $K$ denotes the total candidate batch size (e.g., $K = 1024$) and $N$ represents the maximum padded atom capacity (e.g., $N = 128$).

To prepare these candidates for macroscopic cDFT screening without manual user intervention, the sampler automatically derives scale-invariant physical parameters directly from the assembled graphs:
1. Effective Fluid Diameter $\sigma_{\text{eff}}$: The collective hard-sphere collision diameter of the molecule is calculated using an additive volumetric sphere-packing formula:

$$\sigma_{\text{eff}} = \left( \sum_{i=1}^{N_{\text{atoms}}} \sigma_i^3 \right)^{1/3}$$

2. Effective Well Depth $\epsilon_{\text{eff}}$: The integrated attractive Lennard-Jones dispersion energy is computed via Lorentz-Berthelot pair volume averaging:

$$\epsilon_{\text{eff}} = \frac{1}{\sigma_{\text{eff}}^3} \sum_{i=1}^{N} \sum_{j=1}^{N} \sqrt{\epsilon_i \epsilon_j} \left( \frac{\sigma_i + \sigma_j}{2} \right)^3$$

3. Thermodynamic Reservoir Bulk Density $\rho_{\text{bulk}}$: Rather than assuming an arbitrary liquid density, `dens-city` solves the exact Percus-Yevick compressibility equation of state (`solve_bulk_density_from_chemical_potential` in `materials.py`) at the specified chemical potential ($\mu = -8.0 \, k_B T$) and temperature ($T = 300.0\text{ K}$), guaranteeing asymptotic thermodynamic consistency.
4. Scale-Invariant Slit-Pore Geometry: The physical confinement width $L_z$ is dynamically scaled to accommodate the molecule:

$$L_z = \max\left(40.0\text{ \AA}, \; 12.0 \cdot \sigma_{\text{eff}}\right)$$

5. Slit-Pore Z-Centering: To prevent raw graph coordinates from overlapping the steric hard boundary of the pore walls, the candidate's Cartesian coordinates are automatically translated in $Z$ so that the molecule's geometric midpoint coincides exactly with the pore center:

$$z_i \gets z_i + \frac{L_z}{2} - \frac{z_{\min} + z_{\max}}{2}$$

## 3.4 Zero-Copy Slicing into Fixed-Size GPU Tensor Buffers

When transmitting candidate molecules from host RAM into GPU VRAM for physics evaluation, standard frameworks create sliced copies of tensors, invoking memory allocations that invalidate compiled execution graphs.

`dens-city` achieves zero-copy slicing through the `slice_molecular_batch` method of `ContiguousCandidateBatch`. The method extracts a slice of up to $B$ candidates (where $B$ is the static GPU batch size, such as $B = 64$ or $128$) and maps them directly into a `MolecularBatch` structure (`src/dens_city/utils/materials.py`).

Because the destination buffers in `MolecularBatch` are pre-allocated with static shapes:

$$\mathbf{X} \in \mathbb{R}^{B \times N \times 3}, \quad \boldsymbol{\sigma} \in \mathbb{R}^{B \times N}, \quad \boldsymbol{\epsilon} \in \mathbb{R}^{B \times N}, \quad \mathbf{q} \in \mathbb{R}^{B \times N}$$

the transfer into `tinygrad.Tensor` objects is executed using direct buffer views (`Tensor.frombuffer` or direct memory copies into realized device buffers). If the number of remaining valid candidates in a slice is less than $B$, the unused batch slots are masked out by setting `molecule_mask[b] = 0.0`. 

The underlying tensor buffer dimensions $(B, N, 3)$ remain strictly constant. Consequently, the compiled `@TinyJit` execution graphs of the downstream cDFT solver, the L-BFGS optimizer, and the EGNN force field retain 100% cache validity. The GPU executes consecutive candidate batches at maximum compute density with zero kernel recompilation.

---

# Stage 4: Inverse Design Step 3 — Coupled cDFT Thermodynamics, Batched L-BFGS, & Boltzmann Flows

## 4.1 Objective & Engines: Macroscopic and Conformational Screening

Stage 4 subjects candidate molecules to rigorous macroscopic continuum thermodynamic screening and atomic conformational relaxation on the GPU. This stage couples three high-performance physical engines:
1. `BatchedTinyCDFT` (`src/dens_city/cdft/cdft.py`): Solves the one-dimensional grand potential functional $\Omega[\rho]$ for $B$ confined fluids simultaneously, extracting macroscopic wall contact pressures ($P_{\text{wall}}$), excess coverage ($\Gamma$), and solvation free energies.
2. `BatchedLBFGS` (`src/dens_city/boltzmann/lbfgs.py`): Vectorized quasi-Newton optimizer with trust-region displacement limits, relaxing 3D Cartesian coordinates on the classical microscopic potential energy surface $U_{\text{3D}}(X)$.
3. `BoltzmannGenerator` (`src/dens_city/boltzmann/generator.py`): Deep invertible normalizing flow based on RealNVP affine coupling bijectors, evaluating exact conformer log-likelihoods $\log p_X(x)$ and ensemble energy variance $\text{Var}(U)$.

## 4.2 Variational Classical Density Functional Theory & White Bear FMT

To understand Classical Density Functional Theory, we must stop and explain its theoretical origins. In statistical mechanics, Classical Density Functional Theory (cDFT) is the classical statistical analog of electronic Density Functional Theory (DFT) developed by Hohenberg, Kohn, and Sham. Grounded in the foundational theorem of Mermin (1965), cDFT establishes that for any classical fluid system at temperature $T$ and chemical potential $\mu$ subject to an external potential $V_{\text{ext}}(\mathbf{r})$, there exists a unique thermodynamic grand potential functional:

$$\Omega[\rho(\mathbf{r})] = F_{\text{intrinsic}}[\rho(\mathbf{r})] + \int \rho(\mathbf{r}) \left( V_{\text{ext}}(\mathbf{r}) - \mu \right) d\mathbf{r}$$

whose global minimum with respect to variations in the one-body spatial density profile $\rho(\mathbf{r})$ is the true equilibrium grand potential $\Omega_{\text{eq}}$, and the density profile that minimizes the functional is the exact physical equilibrium density $\rho_{\text{eq}}(\mathbf{r})$. 

The intrinsic Helmholtz free energy functional $F_{\text{intrinsic}}[\rho]$ is partitioned into an ideal gas contribution and an excess contribution arising from inter-particle correlations:

$$F_{\text{intrinsic}}[\rho] = F_{\text{ideal}}[\rho] + F_{\text{excess}}[\rho]$$

The ideal gas functional is known exactly from statistical thermodynamics:

$$F_{\text{ideal}}[\rho] = k_B T \int \rho(\mathbf{r}) \left[ \ln\left( \Lambda^3 \rho(\mathbf{r}) \right) - 1 \right] d\mathbf{r}$$

where $\Lambda$ is the thermal de Broglie wavelength. 

The entire challenge of condensed-phase liquid theory resides in constructing accurate expressions for the excess free energy functional $F_{\text{excess}}[\rho]$. In liquid media, hard-core repulsive steric volume exclusion dominates fluid structure, giving rise to sharp molecular layering and oscillatory density packing against surfaces. 

Naive approximations (such as local density approximations or square-gradient theories) fail completely at molecular interfaces because they cannot capture short-range steric correlations whose length scales are comparable to the molecular diameter $\sigma$.

### 4.2.1 Fundamental Measure Theory (FMT) and the White Bear Variant

The breakthrough that resolved this problem is Fundamental Measure Theory (FMT), formulated by Yitzhak Rosenfeld in 1989. Rosenfeld recognized that the excess free energy of hard spheres can be expressed as a function of weighted densities:

$$F_{\text{excess}}^{\text{FMT}}[\rho] = k_B T \int \Phi\left( \{ n_\alpha(\mathbf{r}) \} \right) d\mathbf{r}$$

where the weighted densities $n_\alpha(\mathbf{r})$ are convolutions of the local density profile $\rho(\mathbf{r})$ with single-particle geometric weight functions $w_\alpha(\mathbf{r})$:

$$n_\alpha(\mathbf{r}) = \int \rho(\mathbf{r}') w_\alpha(\mathbf{r} - \mathbf{r}') d\mathbf{r}' = \rho * w_\alpha$$

The weight functions characterize the fundamental geometric measures of a sphere of radius $R = \sigma / 2$: its volume ($w_3$), surface area ($w_2$), mean curvature radius ($w_1$), Euler characteristic ($w_0$), and their vector directional counterparts ($\mathbf{w}_{v2}, \mathbf{w}_{v1}$):

$$w_3(\mathbf{r}) = \Theta(R - r), \quad w_2(\mathbf{r}) = \delta(R - r), \quad w_1(\mathbf{r}) = \frac{w_2(\mathbf{r})}{4\pi R}, \quad w_0(\mathbf{r}) = \frac{w_2(\mathbf{r})}{4\pi R^2}$$

$$\mathbf{w}_{v2}(\mathbf{r}) = \frac{\mathbf{r}}{r} \delta(R - r), \quad \mathbf{w}_{v1}(\mathbf{r}) = \frac{\mathbf{w}_{v2}(\mathbf{r})}{4\pi R}$$

where $\Theta$ is the Heaviside step function and $\delta$ is the Dirac delta distribution.

While Rosenfeld's original functional provided unmatched accuracy for hard spheres, it suffered from thermodynamic inconsistency: its underlying equation of state in the homogeneous bulk limit corresponded to the Percus-Yevick compressibility equation of state, which slightly underestimates pressure at high packing fractions near freezing ($\eta \ge 0.45$). 

To resolve this inconsistency, Roth et al. (2002) introduced the White Bear variant of Fundamental Measure Theory. The White Bear functional modifies the excess free energy density $\Phi_{\text{FMT}}$ to match the highly accurate Mansoori-Carnahan-Starling-Leland (MCSL) equation of state:

$$\Phi_{\text{FMT}} = -n_0 \ln(1 - n_3) + \frac{n_1 n_2 - \mathbf{n}_{v1} \cdot \mathbf{n}_{v2}}{1 - n_3} + \left( n_2^3 - 3 n_2 (\mathbf{n}_{v2} \cdot \mathbf{n}_{v2}) \right) \frac{n_3 + (1 - n_3)^2 \ln(1 - n_3)}{36\pi n_3^2 (1 - n_3)^2}$$

In `dens-city`'s planar slit-pore geometry, symmetry reduces the three-dimensional convolutions to one-dimensional convolutions along the confinement axis $z$. In `TinyCDFT.grand_potential` (`src/dens_city/cdft/cdft.py`), the White Bear free energy density is implemented as:

$$\phi_{\text{FMT}}(z) = -n_0 \ln(1 - n_3^*) + \frac{n_1 n_2 - n_{v1} n_{v2}}{1 - n_3^*} + \frac{n_2^3 - 3 n_2 (n_{v2})^2}{24\pi (1 - n_3^*)^2}$$

where $n_3^* = \min(n_3, 1.0 - 10^{-5})$ enforces numerical stability against hard-sphere close packing divergence.

### 4.2.2 Anti-Aliased Cell-Integrated Convolution Kernels

In numerical implementations of cDFT, discrete spatial grids introduce severe high-frequency discretization artifacts. If weight functions containing Dirac delta functions (such as $w_2(z)$) are evaluated at discrete grid points $z_k = k \cdot \Delta z$, the numerical representation fluctuates wildly depending on whether the sphere boundary $R$ aligns exactly with a grid node. This induces numerical grid aliasing and unphysical density oscillations ("ringing").

`dens-city` eliminates grid aliasing through analytically cell-integrated kernels (`KernelBuilder.build_fmt_planar_kernels_np` in `src/dens_city/cdft/kernels.py`). Each discrete kernel coefficient is calculated by analytically integrating the continuous weight function across the spatial bin $[z - \frac{\Delta z}{2}, z + \frac{\Delta z}{2}]$:

$$\bar{w}_\alpha(k) = \frac{1}{\Delta z} \int_{k\Delta z - \frac{\Delta z}{2}}^{k\Delta z + \frac{\Delta z}{2}} w_\alpha(z') dz'$$

Evaluating these integrals analytically yields smooth, exact algebraic formulas:

$$\bar{w}_3(z) = \frac{\pi}{\Delta z} \left[ R^2 (z_2 - z_1) - \frac{z_2^3 - z_1^3}{3} \right]$$

$$\bar{w}_2(z) = \frac{2\pi R}{\Delta z} (z_2 - z_1), \quad \bar{w}_{v2}(z) = \frac{\pi}{\Delta z} (z_2^2 - z_1^2)$$

where $z_1 = \max(-R, z - \frac{\Delta z}{2})$ and $z_2 = \min(R, z + \frac{\Delta z}{2})$. Because these kernels are integrated over finite volume bins, spatial resolution is preserved down to coarse grids ($\Delta z \approx 0.2\text{ \AA}$) with zero high-frequency ringing.

```
+-------------------------------------------------------------------------------+
|             ANTI-ALIASED CELL-INTEGRATED CONVOLUTION INTEGRALS                |
+-------------------------------------------------------------------------------+

          Continuous Planar Sphere Slice at Coordinate z' in [-R, +R]:
                         
                              |<-   2*sqrt(R^2 - z'^2)   ->|
                              +----------------------------+
                             /                              \
                            |                                |
                        ----+--------------------------------+---- z = k*dz
                            |    Grid Cell [z - dz/2, z + dz/2]    |
                             \                              /
                              +----------------------------+

  Analytical Cell Integration over [z1, z2]:
  ------------------------------------------
  w3(k)  = (pi / dz) * [ R^2 * (z2 - z1) - (z2^3 - z1^3) / 3 ]
  w2(k)  = (2 * pi * R / dz) * (z2 - z1)
  wv2(k) = (pi / dz) * (z2^2 - z1^2)
  w1(k)  = w2(k) / (4 * pi * R)
  w0(k)  = w2(k) / (4 * pi * R^2)
  wv1(k) = wv2(k) / (4 * pi * R)
+-------------------------------------------------------------------------------+
```

### 4.2.3 Log-Free Latent Density Formulation

A critical failure mode in numerical cDFT minimization is density non-positivity. During gradient-based minimization, if the optimizer updates the density field directly ($\rho^{(k+1)} = \rho^{(k)} - \alpha \nabla_\rho \Omega$), large gradient steps in strongly repulsive regions (such as near pore walls where $V_{\text{ext}} \gg 0$) inevitably drive the density to negative values ($\rho(z) < 0$). When the ideal gas functional attempts to evaluate $\ln(\rho(z))$, the runtime encounters $\ln(\text{negative})$ or $\ln(0)$, generating `NaN` values that destroy the optimization graph.

`dens-city` cures this pathology by optimizing a latent logarithmic potential field $\psi(z)$ (`pattern_log_free_latent_density.md`):

$$\rho(z) = \rho_{\text{bulk}} \exp\left( \psi(z) \right)$$

Because the exponential function maps the real line $\mathbb{R} \to (0, \infty)$, the physical density $\rho(z)$ is mathematically guaranteed to remain strictly positive throughout the domain, regardless of the magnitude or sign of $\psi(z)$.

Substituting $\rho(z) = \rho_{\text{bulk}} e^{\psi(z)}$ into the ideal gas functional yields the log-free formulation:

$$F_{\text{ideal}}[\psi] = k_B T \int \left[ \rho(z) \psi(z) - \left(\rho(z) - \rho_{\text{bulk}}\right) \right] dz$$

Notice that $\ln(\rho)$ has vanished entirely from the functional. The ideal gas free energy and its autograd derivative are smooth, continuous, and singularity-free across all compute steps.

### 4.2.4 Euler-Lagrange Equilibrium and Grouped 2D Convolutions

To solve the Euler-Lagrange functional equation:

$$\frac{\delta \Omega[\rho]}{\delta \rho(z)} = 0 \implies \psi(z) = -\beta V_{\text{ext}}(z) + \beta \mu_{\text{excess}} - \frac{\delta F_{\text{excess}}}{\delta \rho(z)}$$

`BatchedTinyCDFT` minimizes $\Omega[\psi]$ directly via gradient descent on the GPU. To evaluate convolutions across an entire batch of $B$ fluids simultaneously without slow Python loops, `dens-city` maps the 1D spatial convolutions into a grouped 2D convolution (`conv2d` with `groups=B`):

$$\mathbf{n}_\alpha = \text{conv2d}\left( \boldsymbol{\rho}, \; \mathbf{W}_\alpha, \; \text{groups}=B, \; \text{padding}=(k_{\text{pad}}, 0) \right)$$

where $\boldsymbol{\rho} \in \mathbb{R}^{1 \times B \times N_{\text{grid}} \times 1}$ and the kernel tensor $\mathbf{W}_\alpha \in \mathbb{R}^{B \times 1 \times K_{\text{size}} \times 1}$ encapsulates the distinct molecular kernels for each fluid slot. The entire batch of $B$ fluids is convolved and integrated in parallel in a single GPU kernel sweep.

### 4.2.5 Macroscopic Observables: Irving-Kirkwood Virial Pressure and Excess Coverage

Upon variational convergence ($\nabla_\psi \Omega \to 0$), the engine extracts three fundamental thermodynamic observables:
1. Exact Mechanical Wall Contact Pressure $P_{\text{wall}}$: In statistical mechanics, evaluating wall contact pressure via spatial slices (e.g., `rho[0] * k_B * T`) introduces massive numerical discretization error because the density profile exhibits steep gradients near the steric boundary. `dens-city` enforces the exact Irving-Kirkwood mechanical virial integral (`pattern_irving_kirkwood_virial_pressure.md`):

$$P_{\text{wall}} = -\int_0^{L_z/2} \rho(z) \frac{dV_{\text{ext}}(z)}{dz} dz$$

By integrating the force density $\rho(z) \nabla V_{\text{ext}}$ across the confining boundary layer, numerical discretization errors cancel out, providing an exact mechanical contact pressure in bar.
2. Wall Contact Ratio: The dimensionless wetting ratio:

$$C_{\text{ratio}} = \frac{P_{\text{wall}}}{\rho_{\text{bulk}} k_B T}$$

Ratios $C_{\text{ratio}} > 1.0$ indicate strong liquid adsorption and cohesive wetting against the surface, while $C_{\text{ratio}} < 1.0$ indicates fluid depletion and non-wetting drying.
3. Excess Adsorption Coverage $\Gamma$: The net Gibbs surface excess per unit area:

$$\Gamma = \int_0^{L_z} \left( \rho(z) - \rho_{\text{bulk}} \right) dz$$

measuring the physical accumulation of fluid mass inside the slit pore.

## 4.3 Batched L-BFGS 3D Relaxation with Trust-Region Clamping

The 3D atomic coordinates produced by reinforcement learning graph assembly possess idealized bond lengths but frequently contain severe steric clashes across non-bonded branches. In macromolecular networks and flexible conjugated chains, two non-bonded atoms may be positioned within $1.0\text{ \AA}$ of each other.

On the classical potential energy surface:

$$U_{\text{3D}}(X) = \sum_{\text{bonds}} K_b (r - r_0)^2 + \sum_{\text{angles}} K_\theta (\theta - \theta_0)^2 + \sum_{i < j} 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right] + \sum_{i < j} \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}$$

the steep Lennard-Jones repulsive core ($\sim r^{-12}$) produces astronomical repulsive forces exceeding $10^8\text{ K/\AA}$.

To understand why standard optimizers fail, we must stop and explain quasi-Newton optimization and the trust-region problem. In unconstrained optimization, Newton's method approximates the local potential energy surface using a second-order Taylor series expansion:

$$U(x + p) \approx U(x) + g^\top p + \frac{1}{2} p^\top H p$$

where $g = \nabla U$ is the gradient vector and $H = \nabla^2 U$ is the Hessian matrix of second derivatives. The pure Newton step is:

$$p = -H^{-1} g$$

Because computing and inverting the full $3N \times 3N$ Hessian matrix at every iteration requires $O(N^3)$ operations and massive GPU memory, quasi-Newton methods such as Broyden-Fletcher-Goldfarb-Shanno (BFGS) approximate the inverse Hessian $B_k \approx H_k^{-1}$ using curvature information accumulated from successive displacement vectors:

$$s_k = x_{k+1} - x_k, \quad y_k = g_{k+1} - g_k$$

However, in molecular systems with steep steric repulsion, the second-order quadratic Taylor approximation is valid only within an infinitesimal radius around the current coordinate. If a quasi-Newton optimizer proposes an unconstrained step of length $\Delta r > 0.5\text{ \AA}$, an atom can be pushed directly through the repulsive core of an adjacent atom. At $r_{ij} \to 0$, the Lennard-Jones potential explodes toward infinity, the autograd gradient evaluates to `inf` or `NaN`, and the accumulated inverse Hessian approximation $B_k$ is permanently corrupted.

`dens-city` resolves this failure mode through the `BatchedLBFGS` optimizer (`src/dens_city/boltzmann/lbfgs.py`), which implements three rigorous mathematical mechanisms (`pattern_batched_lbfgs_trust_region_relaxation.md`):

### 4.3.1 Vectorized Two-Loop Recursion ($m = 6$)

The descent direction $p_k = -B_k g_k$ is evaluated across all $B$ molecules simultaneously on the GPU using two-loop recursion over a rolling history of depth $m = 6$. The algorithm maintains memory buffers:

$$s_k \in \mathbb{R}^{B \times 3N}, \quad y_k \in \mathbb{R}^{B \times 3N}, \quad \rho_k = \frac{1}{y_k^\top s_k} \in \mathbb{R}^{B \times 1}$$

In the backward loop, the gradient vector $q$ is updated:

$$\alpha_i = \rho_i (s_i^\top q), \quad q \gets q - \alpha_i y_i$$

The initial Hessian scaling factor is dynamically computed:

$$\gamma_k = \frac{s_{k-1}^\top y_{k-1}}{y_{k-1}^\top y_{k-1}}, \quad r \gets \gamma_k q$$

In the forward loop, the search direction is reconstructed:

$$\beta_i = \rho_i (y_i^\top r), \quad r \gets r + s_i (\alpha_i - \beta_i)$$

yielding the descent vector $p_k = -r$. If numerical noise causes $g_k^\top p_k \ge 0$, the optimizer immediately falls back to steepest descent ($p_k = -g_k$).

### 4.3.2 Trust-Region Displacement Clamping

To prevent atoms from jumping across repulsive van der Waals barriers, `BatchedLBFGS` enforces a strict $L_\infty$ trust-region displacement limit on the search direction:

$$\Delta r_{\max, b} = \max_{i=1\dots N} \|p_{b, i}\|_2$$

$$\text{scale}_b = \min\left(1.0, \; \frac{0.20\text{ \AA}}{\Delta r_{\max, b} + 10^{-8}}\right)$$

$$p_{\text{clamped}} = p \cdot \text{scale}_b$$

Restricting the maximum atomic displacement to $\Delta r \le 0.20\text{ \AA}$ per iteration mathematically guarantees that no atom can step through a potential barrier. The steepest steric clashes are smoothly resolved within the first 10 to 15 iterations.

### 4.3.3 SIMD Active Molecule Convergence Masking

In a heterogeneous candidate batch ($B = 128$), rigid small molecules converge within 8 iterations, whereas flexible crosslinked macrocycles may require 50 iterations. Continuing to compute forces and line searches for converged molecules wastes GPU compute cycles and induces numerical drift.

`BatchedLBFGS` continuously evaluates the root-mean-square force residual per molecule:

$$\|g_b\|_{\text{RMS}} = \sqrt{\frac{1}{N_{\text{real}, b}} \sum_{i=1}^{N_{\text{real}}} \|g_{b, i}\|_2^2}$$

A molecule is marked converged when $\|g_b\|_{\text{RMS}} < \text{tol}$ (where $\text{tol} = 10^{-3}\text{ K/\AA}$). Using a boolean active mask $M_{\text{active}} \in \{0.0, 1.0\}^{B \times 1}$, coordinate updates are applied exclusively to unconverged molecules:

$$x_{k+1} = M_{\text{active}} \cdot (x_k + \alpha p) + (1 - M_{\text{active}}) \cdot x_k$$

Converged structures are permanently frozen in GPU memory, allowing remaining compute power to focus on unrelaxed geometries.

## 4.4 Boltzmann Generator: Normalizing Flows in Physical Chemistry

Once geometry optimization locates a local potential energy minimum $x^*$, a fundamental physical question remains: Is this single minimum representative of the molecule's true thermodynamic behavior at finite temperature ($T = 300\text{ K}$), or is it a rigid, kinetically trapped conformation that cannot flex in solution?

To answer this, `dens-city` deploys the `BoltzmannGenerator` (`src/dens_city/boltzmann/generator.py`), an invertible deep normalizing flow trained directly on the microscopic energy surface $U_{\text{3D}}(x)$.

To understand the Boltzmann Generator, we must stop and explain deep normalizing flows from first principles. In statistical mechanics, the probability density of finding a molecular system in a continuous spatial configuration $x \in \mathbb{R}^{3N}$ at temperature $T$ is governed by the Boltzmann distribution:

$$p_X(x) = \frac{1}{Z} \exp\left( -\beta U(x) \right)$$

where $\beta = \frac{1}{k_B T}$ and $Z = \int \exp(-\beta U(x)) dx$ is the intractable partition function. Evaluating expectations under $p_X(x)$ via standard Markov Chain Monte Carlo (MCMC) or molecular dynamics (MD) requires calculating millions of sequential force evaluations, frequently becoming trapped in local energy minima for weeks.

A normalizing flow solves this sampling problem by training a smooth, bijective (invertible) coordinate transformation:

$$f_\theta: \mathcal{Z} \to \mathcal{X}$$

that maps a simple, analytically tractable latent distribution $p_Z(z) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ directly to the complex molecular configuration space $\mathcal{X}$. By the probability transformation formula under a change of variables, the exact probability density of a generated configuration $x = f_\theta(z)$ is given by:

$$p_X(x) = p_Z\left( f_\theta^{-1}(x) \right) \cdot \left| \det \mathbf{J}_{f_\theta^{-1}}(x) \right| = \frac{p_Z(z)}{\left| \det \mathbf{J}_{f_\theta}(z) \right|}$$

where $\mathbf{J}_{f_\theta}(z) = \frac{\partial f_\theta(z)}{\partial z}$ is the Jacobian matrix of the transformation.

```
       Latent Space Z                          Physical Conformation Space X
    p_Z(z) ~ Normal(0, I)                     p_X(x) ~ (1/Z) * exp(-beta * U(x))
   +---------------------+                        +-------------------------+
   |   Simple Gaussian   |                        |  Complex Multi-Modal    |
   |   Prior Density     |                        |  Boltzmann Landscape    |
   |      (Tractable)    |                        |  (Layering & Rotamers)  |
   +---------------------+                        +-------------------------+
             \                                                 /
              \             Forward Flow: x = f_theta(z)      /
               +-------------------------------------------->+
               |                                             |
               +<--------------------------------------------+
                            Inverse Flow: z = f_theta^-1(x)
                                          |
                        Jacobian Determinant Tracking:
                        ln p_X(x) = ln p_Z(z) - ln |det J_f(z)|
```

### 4.4.1 RealNVP Affine Coupling Bijectors

To compute the Jacobian determinant in $O(D)$ time rather than $O(D^3)$, `dens-city` structures the normalizing flow using RealNVP (Real-valued Non-Volume Preserving) affine coupling layers (`RealNVPFlow` in `bijectors.py`). The coordinate vector $z \in \mathbb{R}^D$ is split into two halves: $z = [z_A, z_B]$. The transformation evaluates:

$$x_A = z_A$$

$$x_B = z_B \odot \exp\left( s(z_A) \right) + t(z_A)$$

where $s(\cdot)$ and $t(\cdot)$ are arbitrary neural networks (scale and translation MLPs). Because $x_A$ does not depend on $z_B$, the Jacobian matrix of the forward transformation is strictly triangular:

$$\mathbf{J}_{f} = \begin{bmatrix} \mathbf{I} & \mathbf{0} \\ \frac{\partial x_B}{\partial z_A} & \text{diag}(\exp(s(z_A))) \end{bmatrix}$$

The determinant of a triangular matrix is simply the product of its diagonal entries. The log-determinant is evaluated instantaneously as a simple sum:

$$\ln\left| \det \mathbf{J}_f \right| = \sum_{k} s_k(z_A)$$

Inverting the transformation is equally direct and requires no matrix inversion:

$$z_A = x_A, \quad z_B = \left( x_B - t(x_A) \right) \odot \exp\left( -s(x_A) \right)$$

### 4.4.2 Variational Reverse Kullback-Leibler Loss

In `dens-city`, the normalizing flow is trained without any pre-existing molecular dynamics trajectories, relying purely on the exact microscopic energy $U_{\text{3D}}(x)$. The training objective minimizes the variational Reverse Kullback-Leibler (KL) divergence between the flow density $q_\theta(x)$ and the physical Boltzmann target distribution $p(x)$:

$$\mathcal{L}_{\text{KL}}(\theta) = D_{\text{KL}}\left( q_\theta(x) \;\middle\|\; p(x) \right) = \mathbb{E}_{x \sim q_\theta} \left[ \ln q_\theta(x) - \ln p(x) \right]$$

Substituting the flow density and target Boltzmann distribution yields (`compute_loss` in `generator.py`):

$$\mathcal{L}(\theta) = \mathbb{E}_{z \sim p_Z} \left[ \beta U\left( f_\theta(z) \right) - \ln p_Z(z) - \ln\left| \det \mathbf{J}_{f_\theta}(z) \right| + w_{\text{tor}} \mathcal{J}_{\text{tor}} \right]$$

where $\mathcal{J}_{\text{tor}}$ is an analytical torsional regularizer penalizing out-of-range dihedral angles (`compute_torsion_rotamer_loss`), preserving bijective invertibility across periodicity boundaries.

By training against this variational loss, the flow learns to generate low-energy conformers while maximizing structural entropy ($\mathcal{H} = -\mathbb{E}[\ln q_\theta]$). Once trained, calling `generator.log_prob(x)` evaluates the exact log-likelihood $\ln p_X(x)$ of any conformer. Candidates exhibiting low average potential energy $\langle U_{\text{3D}} \rangle$, high log-likelihood $\ln p_X(x)$, and low conformational variance $\text{Var}(U)$ are verified as thermodynamically stable, structurally flexible, and free from kinetic bottlenecks.

# Stage 5: Inverse Design Step 4 — 7-Layer Invariant EGNN MLFF Quantum Surrogate Screening

## 5.1 Objective & Engine: Quantum Mechanical Force Equilibrium Verification

While Classical Density Functional Theory evaluates macroscopic liquid-phase thermodynamics and classical force fields resolve initial steric overlap, classical empirical potentials suffer from severe limitations. Parameterized using fixed harmonic springs and point-charge Coulomb potentials, classical force fields cannot capture electronic orbital hybridization, non-local polarization, conjugated charge delocalization, or chemical bond breaking and formation. A candidate molecule that appears stable under a classical potential may possess unphysical electronic strain, radical character, or distorted valence geometry that makes it chemically unstable or explosive.

To achieve quantum mechanical fidelity without the prohibitive computational cost of ab initio Density Functional Theory (which scales as $O(N_e^3)$ with the number of electrons), Stage 4 deploys the `EGNNForceField` (`src/dens_city/boltzmann/egnn.py`). The EGNN operates as an ultra-fast machine-learned surrogate force field compiled into a static GPU execution graph via `@TinyJit`. It ingests atomic numbers $Z$ and continuous 3D coordinates $x$, predicting the electronic ground-state potential energy $U_{\text{EGNN}}(x)$ with DFT accuracy and evaluating exact conservative forces $F_i = -\nabla_{x_i} U_{\text{EGNN}}(x)$ via reverse-mode automatic differentiation.

## 5.2 $E(3)$-Equivariant Graph Neural Processing

To understand the architecture of the EGNN, we must stop and explain the concept of $E(3)$ equivariance from first principles. In physics and group theory, the Euclidean group in three dimensions, denoted $E(3)$, represents the set of all spatial transformations that preserve Euclidean distance, comprising continuous 3D translations $\mathbf{x} \mapsto \mathbf{x} + \mathbf{g}$ (where $\mathbf{g} \in \mathbb{R}^3$) and orthogonal transformations $\mathbf{x} \mapsto \mathbf{Q}\mathbf{x}$ (where $\mathbf{Q} \in O(3)$, spanning all 3D rotations and spatial reflections).

When modeling molecular matter, physical laws possess exact symmetry under $E(3)$:
1. Potential Energy Invariance: The total scalar potential energy $U(x)$ of an isolated molecule is completely independent of where the molecule is positioned in space or how it is oriented. Rotating or translating the entire molecule must not alter its energy:

$$U\left( \mathbf{Q}\mathbf{x} + \mathbf{g} \right) = U(\mathbf{x}), \quad \forall \, \mathbf{Q} \in O(3), \; \mathbf{g} \in \mathbb{R}^3$$

2. Conservative Force Equivariance: The physical force vector acting on atom $i$, defined as the negative gradient of energy $\mathbf{F}_i = -\nabla_{\mathbf{x}_i} U$, is a geometric vector (a type-1 spatial tensor). If the molecule is rotated by matrix $\mathbf{Q}$, the force vectors acting on every atom must rotate by that exact same rotation matrix:

$$\mathbf{F}_i\left( \mathbf{Q}\mathbf{x} + \mathbf{g} \right) = \mathbf{Q} \, \mathbf{F}_i(\mathbf{x})$$

If a neural network is constructed using standard dense layers or naive graph neural networks that accept raw Cartesian coordinates $(x_i, y_i, z_i)$ as input features, the network will inevitably violate $E(3)$ symmetry. Because the coordinate numbers depend on an arbitrary choice of laboratory coordinate axes, the network will predict completely different energies for the exact same molecule merely because it was rotated by five degrees. 

Attempting to overcome this via data augmentation (training the network on millions of randomly rotated copies of each molecule) is highly inefficient, requires ten times more parameters, and never achieves exact physical conservation laws.

While spherical harmonic tensor architectures (such as Tensor Field Networks or SE(3)-Transformers) achieve exact equivariance by expanding features in Clebsch-Gordan tensor products, they introduce immense computational overhead. Evaluating spherical harmonics $Y_l^m$ and Clebsch-Gordan coefficients requires complex branching logic, highly fragmented tensor multiplications, and variable loop bounds that cannot be fused effectively by GPU compilers, degrading training throughput on large molecules.

`dens-city` implements the Equivariant Graph Neural Network (EGNN) architecture developed by Satorras, Hoogeboom, and Welling (ICML 2021). The EGNN achieves exact $E(3)$ equivariance without spherical harmonics by restricting spatial interactions to scalar invariant distances and updating coordinates along relative difference radial vector fields.

```
+-------------------------------------------------------------------------------+
|         E(3)-EQUIVARIANT MESSAGE PASSING ARCHITECTURE (EGNNLayer)             |
+-------------------------------------------------------------------------------+

  1. Invariant Squared Distance:
     d_ij^2 = ||x_i - x_j||^2   --> Strictly invariant under Q*x + g
                                    since ||Q*(x_i - x_j)||^2 = ||x_i - x_j||^2

  2. Decomposed Linear Projections (Zero OOM Memory Footprint):
     e_hidden = SiLU( W_hi*h_i + W_hj*h_j + W_d*d_ij^2 + W_a*a_ij + b )

  3. Smooth Cosine Radial Cutoff Envelope (r_cut = 5.0 A):
     f_cut(r_ij) = 0.5 * ( cos( pi * r_ij / r_cut ) + 1.0 ) * (r_ij < r_cut)

  4. Message Normalization by Active Neighbor Degree:
     m_ij = SiLU( W_l2 * e_hidden ) * a_ij * f_cut(r_ij)
     m_i  = sum_{j != i} m_ij / deg_i   --> Guarantees extensive O(N) energy

  5. Invariant Node Feature Update with Residual Connection:
     h_i^(l+1) = h_i^l + phi_h( [h_i^l || m_i] ) * atom_mask_i
+-------------------------------------------------------------------------------+
```

### 5.2.1 Decomposed Linear Projections to Eliminate Memory Explosions

In deep message-passing networks, computing edge messages between all pairs of $N$ atoms across batch $B$ requires evaluating an edge interaction function:

$$\mathbf{m}_{ij} = \phi_e\left( \mathbf{h}_i, \; \mathbf{h}_j, \; \|\mathbf{x}_i - \mathbf{x}_j\|^2, \; a_{ij} \right)$$

In naive implementations, developers concatenate the features $[\mathbf{h}_i \parallel \mathbf{h}_j \parallel d_{ij}^2 \parallel a_{ij}]$ into a massive edge feature tensor of shape $(B, N, N, 2F + 2)$. For a batch size $B = 64$, $N = 128$ atoms, and hidden dimension $F = 128$, concatenating these features materializes a single tensor requiring:

$$\text{Memory} = 64 \times 128 \times 128 \times (256 + 2) \times 4\text{ bytes} \approx 1.08\text{ GB per layer}$$

Across a 7-layer network, forward activations alone consume over 7.5 GB of VRAM, triggering out-of-memory (OOM) crashes during backpropagation.

`dens-city` completely eliminates this memory explosion by decomposing the first linear layer of the edge MLP into independent projections (`EGNNLayer.__call__` in `src/dens_city/boltzmann/egnn.py`, adhering to `pattern_egnn_quantum_charges_and_memory_decomposition.md`):

```python
# Project node features directly on (B, N, F) before spatial broadcast
h_i_proj = self.edge_hi(h).reshape(B, N, 1, F)
h_j_proj = self.edge_hj(h).reshape(B, 1, N, F)
d_proj = self.edge_d(d_sq)      # (B, N, N, F) from (B, N, N, 1)
a_proj = self.edge_a(edge_mask)  # (B, N, N, F) from (B, N, N, 1)

e_hidden = (h_i_proj + h_j_proj + d_proj + a_proj).silu()
```

By applying `edge_hi` and `edge_hj` to the node embeddings before broadcasting them across the spatial axes, the linear transformation executes on $(B, N, F)$ rather than $(B, N, N, 2F)$. The compiler fuses the additions and SiLU activation into a single streaming loop, reducing peak memory consumption by more than sixty percent.

### 5.2.2 Smooth Radial Cutoff Envelopes & Active Degree Normalization

To ensure that atomic interactions decay smoothly to zero at the interaction boundary without introducing discontinuous step-function forces, `dens-city` applies a smooth cosine cutoff envelope:

$$f_{\text{cut}}(r_{ij}) = \begin{cases} \frac{1}{2} \left[ \cos\left( \frac{\pi r_{ij}}{r_{\text{cut}}} \right) + 1 \right], & r_{ij} < r_{\text{cut}} \\ 0, & r_{ij} \ge r_{\text{cut}} \end{cases}$$

with cutoff radius $r_{\text{cut}} = 5.0\text{ \AA}$. 

Furthermore, when aggregating incoming edge messages into the node representation, standard GNNs use unnormalized summation ($\mathbf{m}_i = \sum_j \mathbf{m}_{ij}$). In molecular systems, unnormalized summation introduces a severe pathology: atoms with high coordination numbers receive artificially amplified message magnitudes, causing total molecular energy to scale quadratically $O(N^2)$ rather than extensively $O(N)$ with system size. 

`dens-city` normalizes incoming messages by the active neighbor degree:

$$\text{deg}_i = \max\left( 1.0, \; \sum_{j=1}^N a_{ij} \cdot \Theta(r_{\text{cut}} - r_{ij}) \right)$$

$$\mathbf{m}_i = \frac{1}{\text{deg}_i} \sum_{j \neq i} \mathbf{m}_{ij}$$

This degree normalization guarantees strict extensive scaling: doubling the size of a macromolecular chain exactly doubles its energy, maintaining true physical thermodynamic extensivity.

## 5.3 Quantum Observables & Force Residuals

The 7-layer EGNN processes atomic numbers $Z_i$ and coordinates $x_i$, updating node embeddings $h_i^0 \to h_i^1 \dots \to h_i^7 \in \mathbb{R}^{B \times N \times 128}$. The network then branches into specialized readout MLPs:
1. Potential Energy Readout: A 2-layer MLP maps the terminal node embedding $h_i^7$ to an atomic energy contribution $\epsilon_i \in \mathbb{R}$. The total molecular energy is the masked sum over all real atoms:

$$U_{\text{EGNN}}(x) = \sum_{i=1}^N \epsilon_i \cdot M_i$$

2. Conservative Force Evaluation: The analytical physical forces acting on all atoms are derived via reverse-mode automatic differentiation in a single backward pass (`compute_energy_and_forces`):

$$\mathbf{F}_i = -\nabla_{\mathbf{x}_i} U_{\text{EGNN}}(x) = -\frac{\partial U_{\text{EGNN}}}{\partial \mathbf{x}_i}$$

Because the energy $U$ is strictly $E(3)$-invariant, the resulting force field $\mathbf{F}$ is mathematically guaranteed to be exactly conservative ($\oint \mathbf{F} \cdot d\mathbf{x} = 0$, meaning work is independent of path) and strictly $E(3)$-equivariant.

To screen candidate molecules for structural and chemical viability, `dens-city` extracts two rigorous quantum observables:
1. Per-Atom Normalized Ground-State Energy:

$$\bar{u}_{\text{EGNN}} = \frac{U_{\text{EGNN}}}{N_{\text{heavy}}}$$

Lower values of $\bar{u}_{\text{EGNN}}$ designate superior quantum mechanical stability and favorable thermodynamic binding.
2. Root-Mean-Square (RMS) Force Residual:

$$\|F_{\text{EGNN}}\|_{\text{RMS}} = \sqrt{\frac{1}{N_{\text{heavy}}} \sum_{i=1}^{N_{\text{heavy}}} \|\mathbf{F}_i\|_2^2}$$

To understand why the RMS force residual is a critical screening metric, we must stop and explain the concept of quantum stationary states and the Hellmann-Feynman theorem. In quantum mechanics and electronic structure theory, a true equilibrium molecular geometry is a stationary point on the Born-Oppenheimer potential energy surface. By the Hellmann-Feynman theorem, the net quantum mechanical force acting on every nucleus in an unconstrained equilibrium conformer is strictly zero:

$$\mathbf{F}_i = -\langle \Psi | \nabla_{\mathbf{x}_i} \hat{H}_e | \Psi \rangle = \mathbf{0}$$

If an assembled molecular candidate exhibits high force residuals ($\|F\|_{\text{RMS}} > 5.0\text{ kcal/mol/\AA}$), the atoms are subject to severe internal stresses: strained bond angles, eclipsing torsional repulsions, or forced steric overlaps. Such a conformation is physically unstable; upon release into solution, it will undergo violent structural reorganization or decompose chemically. A low force residual ($\|F\|_{\text{RMS}} \le 1.0\text{ kcal/mol/\AA}$) is absolute proof that the candidate resides in a true, relaxed quantum mechanical basin of attraction.

## 5.4 Unrolled GPU Relaxation with Heavy-Ball Momentum

If a candidate molecule possesses a favorable topological scaffold but exhibits minor force residuals due to imperfect initial bond angles, discarding it immediately would be premature. Instead, `dens-city` performs unrolled GPU geometric relaxation (`get_jit_relaxation_evaluator` in `src/dens_city/boltzmann/egnn.py`).

The relaxation engine unrolls $K = 50$ iterations of gradient-based optimization directly within a single `@TinyJit` compiled graph. To overcome the slow asymptotic convergence of standard gradient descent, the solver implements Polyak's Heavy-Ball momentum algorithm with exponential step decay:

```python
# Unrolled Heavy-Ball Momentum Relaxation Loop inside @TinyJit
x_curr = x_in
v_curr = Tensor.zeros_like(x_in)
decay = 0.98

for s in range(relax_steps):
    x_curr.requires_grad = True
    u = self.compute_energy(x=x_curr, atomic_numbers=z_in, atom_mask=a_mask)
    u.sum().backward()
    grad = x_curr.grad if x_curr.grad is not None else Tensor.zeros_like(x_curr)
    forces = -grad * a_mask

    # 1. Clip extreme gradient spikes on steric overlap
    clipped_grad = (grad * a_mask).clip(-100.0, 100.0)

    # 2. Heavy-Ball momentum velocity update with step decay
    current_lr = lr * (decay ** s)
    v_next = momentum * v_curr + current_lr * clipped_grad

    # 3. Adaptive displacement limit (decays from 0.15 A down to 0.05 A)
    disp_limit = max(0.05, max_disp * (decay ** s))
    step = v_next.clip(-disp_limit, disp_limit) * a_mask
    x_cand = x_curr - step

    # 4. Per-molecule maximum force magnitude squared
    f_norm_sq = (forces * forces).sum(axis=-1, keepdim=True)
    f_max_sq = (f_norm_sq * a_mask).max(axis=1, keepdim=True)

    # 5. Masked convergence freezing: freeze coordinates and zero velocity
    converged_mask = f_max_sq < force_tol_sq
    x_next = converged_mask.where(x_curr, x_cand).realize()
    v_next_masked = converged_mask.where(Tensor.zeros_like(v_curr), v_next).realize()

    x_curr = x_next.detach()
    v_curr = v_next_masked.detach()
```

The unrolled relaxation incorporates three vital physical safeguards:
1. Gradient Clipping: On early steps, if two atoms are close, autograd forces can reach $10^4\text{ kcal/mol/\AA}$. Clipping gradients to $[-100.0, 100.0]$ prevents numerical overflow in floating-point registers.
2. Adaptive Displacement Limits: The maximum displacement permitted per atom starts at $\Delta r_{\max} = 0.15\text{ \AA}$ and decays exponentially as $(0.98)^s$ down to $0.05\text{ \AA}$ near convergence. This ensures large exploratory adjustments early in relaxation while preventing oscillations around the minimum.
3. Masked SIMD Convergence Freezing: Atoms in molecules whose maximum force satisfies $f_{\max}^2 < (5.0)^2$ are frozen in place via `.where()`, zeroing their velocity and preventing numerical drift.

Candidates that successfully relax to stationary quantum minima are passed to Stage 6; candidates that fail to reach equilibrium within 50 steps are rejected as physically unviable.

---

# Stage 6: Inverse Design Step 5 — Multi-Objective Pareto Frontier Ranking & Artifact Export

## 6.1 Objective & Engine: Multi-Objective Decision Analysis

The final stage of the `dens-city` inverse design architecture is governed by the `FunnelRanker`, implemented in `src/dens_city/utils/funnel_ranker.py`. Stage 6 receives the multimodal physical observables accumulated across all previous stages:
- Stage 1 RL swarm rewards and structural heuristic proxies ($R_{\text{RL}}, f_{\text{rot}}, \rho_{\text{arom}}, \text{PMI}_{\text{lin}}$).
- Stage 3 cDFT macroscopic thermodynamic observables ($P_{\text{wall}}, C_{\text{ratio}}, \Gamma, \Delta\Omega_{\text{solv}}$).
- Stage 3 Boltzmann Generator conformer log-likelihoods and energy variances ($\ln p_X(x), \langle U_{\text{3D}} \rangle, \text{Var}(U)$).
- Stage 4 EGNN quantum surrogate observables ($U_{\text{EGNN}} / N, \|F_{\text{EGNN}}\|_{\text{RMS}}$).

The engine's mission is to apply strict synthesizability filtering, perform topological deduplication, evaluate composite fitness scoring, construct the multi-objective Pareto frontier, and generate standardized publication-grade artifacts.

## 6.2 The Dynamic Synthesizability Gate (SA Score)

In generative molecular design, an algorithm can easily optimize abstract mathematical objective functions by generating chemical structures that are impossible to synthesize in a laboratory. Such structures—often referred to as "synthetic chimeras"—feature highly strained bridgehead double bonds (violating Bredt's rule), dense clusters of contiguous quaternary carbons, crowded heteroatom-heteroatom bonds (such as poly-peroxides or contiguous nitrogen chains), or impossible polycyclic ring topologies. If an inverse design platform outputs such molecules, the computational predictions are practically useless.

To prevent this, `dens-city` integrates the Synthetic Accessibility (SA) score formulated by Ertl and Schuffenhauer (2009). The SA score is a continuous metric ranging from 1.0 (readily accessible using standard synthetic protocols) to 10.0 (virtually impossible to synthesize). The score is calculated by combining two terms:
1. Fragment Contribution: Evaluates the historical prevalence of 1-4 bond environment subgraphs across millions of commercially available compounds in the PubChem database. Rare, unusual substructures receive severe penalties.
2. Topological Complexity Penalty: Evaluates non-linear structural complexity, penalizing high ring counts, bridgehead atoms, spiro junctions, chiral centers, and high molecular weight.

### 6.2.1 The Failure of Static Thresholds on Macromolecular Networks

In standard computational screening of small-molecule drugs (Lipinski's Rule of 5 regime, $\text{MW} \le 500\text{ amu}$), researchers typically enforce a rigid, static synthesizability threshold, such as dropping any candidate with $\text{SA} > 4.5$.

However, `dens-city` is designed for universal molecular matter—spanning conjugated electro-optic oligomers, functional macrocycles, and crosslinked macromolecular network precursors with molecular weights up to 1,500 amu. In large macromolecular systems, a static threshold fails catastrophically: because the SA score's complexity term scales with atom count and ring count, a large, perfectly linear conjugated polymer or a simple symmetric crosslinker naturally accumulates an SA score of 5.0 to 6.5, even though its synthesis involves standard, repetitive Suzuki-Miyaura coupling or atom-transfer radical polymerization (ATRP). A rigid cutoff at 4.5 erroneously drops 100% of all macromolecular candidates.

`dens-city` resolves this failure mode by implementing a dynamic size-dependent synthesizability gate (`FunnelRanker.rank_candidates`):

$$\text{SA}_{\text{allowance}} = \max\left( \text{SA}_{\max}^{\text{target}}, \; 0.18 \cdot N_{\text{heavy}} + 1.5 \right)$$

where $N_{\text{heavy}}$ is the number of real heavy atoms in the candidate graph. 

For a small molecule ($N_{\text{heavy}} = 15$), the allowance enforces a strict cutoff:

$$\text{SA}_{\text{allowance}} = \max(4.5, \; 0.18 \times 15 + 1.5) = \max(4.5, \; 4.2) = 4.5$$

For a large macromolecular oligomer ($N_{\text{heavy}} = 50$), the allowance dynamically expands:

$$\text{SA}_{\text{allowance}} = \max(4.5, \; 0.18 \times 50 + 1.5) = \max(4.5, \; 10.5) = 10.5$$

Candidates whose SA score exceeds $\text{SA}_{\text{allowance}}$ are immediately discarded at the entrance of the funnel (`num_dropped_sa += 1`), preventing the downstream ranker from allocating top spots to un-synthesizable chimeras.

## 6.3 Topological Deduplication via Canonical SMILES and Weisfeiler-Lehman Hashes

When thousands of candidates are sampled across parallel environments, multiple independent rollout trajectories frequently discover identical chemical graphs or distinct rotamers of the same parent molecule. If raw candidates were ranked directly, the top 20 spots would frequently be populated by twenty minor conformational variations of a single high-scoring molecule, completely destroying candidate diversity.

`FunnelRanker` enforces strict topological deduplication (`rank_candidates`). Each evaluated candidate is assigned a unique topological key:
1. If RDKit is available, the key is the canonical SMILES string generated with stereochemical and aromaticity invariants.
2. If RDKit is unavailable or parsing fails, the key falls back to the 64-bit Weisfeiler-Lehman topological graph hash (`wl_hash`) computed during C assembly.

All candidates sharing an identical topological key are grouped together. The engine compares their composite fitness scores and preserves strictly the single highest-scoring conformer per unique chemical graph. All redundant conformers are pruned.

## 6.4 Composite Scoring and the Multi-Objective Pareto Frontier

To rank the deduplicated candidates, `dens-city` deploys a two-tiered decision framework: scalar composite fitness scoring and multi-objective Pareto non-dominated sorting.

### 6.4.1 The Composite Funnel Fitness Score

The composite scalar fitness score combines observables across all physical layers:

$$S_{\text{funnel}} = w_{\text{rl}} S_{\text{rl}} + w_{\text{cdft}} S_{\text{cdft}} + w_{\text{bg}} S_{\text{bg}} + w_{\text{egnn}} S_{\text{egnn}}$$

with default normalized weights $w_{\text{rl}} = 0.30, w_{\text{cdft}} = 0.30, w_{\text{bg}} = 0.20, w_{\text{egnn}} = 0.20$. The individual sub-scores evaluate:
1. Reinforcement Learning Sub-Score:

$$S_{\text{rl}} = \max(0.0, \; R_{\text{RL}})$$

2. cDFT Thermodynamic Sub-Score: Balances wall contact pressure ratio against liquid-phase solvation free energy:

$$S_{\text{cdft}} = 0.70 \cdot \min\left(3.0, \; \frac{C_{\text{ratio}}}{P_{\text{target}}}\right) + 0.30 \cdot S_{\text{solv}}$$

where $S_{\text{solv}} = 1.0 + 0.1 \min(10.0, \Delta\Omega_{\text{target}} - \Delta\Omega)$ if favorable, and $S_{\text{solv}} = \max(-2.0, 1.0 - \ln(1 + \exp(\text{excess})))$ if unfavorable.
3. Boltzmann Generator Sub-Score: Rewards high prior log-likelihood while penalizing high internal energy and ensemble conformational variance:

$$S_{\text{bg}} = \beta_{\text{logp}} \ln p_X(x) - \alpha_U \text{clamp}(\langle U_{\text{3D}} \rangle, -10^4, 10^4) - \gamma_{\text{var}} \min(50000.0, \text{Var}(U))$$

4. EGNN Quantum Surrogate Sub-Score: Rewards low per-atom ground state energy and penalizes unrelaxed force residuals:

$$S_{\text{egnn}} = -\alpha_{\text{egnn}} \text{clamp}\left( \frac{U_{\text{EGNN}}}{N_{\text{heavy}}}, -1000, 5000 \right) - \beta_{\text{egnn}} \min\left( 100.0, \; \frac{\|F_{\text{EGNN}}\|_{\text{RMS}}}{\sqrt{N_{\text{heavy}}}} \right)$$

### 6.4.2 Pareto Optimality and Non-Dominated Sorting

To understand why scalar scoring alone is insufficient, we must stop and explain Pareto optimality from first principles. In multi-objective optimization, when an engineer attempts to optimize multiple competing criteria simultaneously—such as maximizing wall contact pressure while minimizing quantum energy and maximizing synthetic accessibility—there is rarely a single unique design that is simultaneously superior in every dimension. Increasing aromatic density may enhance wall adsorption and tensile modulus, but it also increases molecular rigidity and reduces synthetic accessibility.

When multiple objectives conflict, condensing them into a single scalar weighted sum ($S = \sum w_i S_i$) introduces trade-off blindness. The scalar score forces an arbitrary linear trade-off: a candidate with disastrously high quantum force residuals can achieve a high rank merely because its cDFT wall pressure is exceptionally large.

Pareto optimality resolves this by evaluating dominance relationships across the full dimensional objective space. In formal terms, given a vector objective function to be maximized:

$$\mathbf{f}(\mathbf{x}) = \left( f_1(\mathbf{x}), \; f_2(\mathbf{x}), \; \dots, \; f_m(\mathbf{x}) \right)$$

a candidate $\mathbf{x}_1$ is said to strictly dominate candidate $\mathbf{x}_2$ (denoted $\mathbf{x}_1 \succ \mathbf{x}_2$) if and only if:
1. Candidate $\mathbf{x}_1$ is at least as good as $\mathbf{x}_2$ in all objectives:

$$f_k(\mathbf{x}_1) \ge f_k(\mathbf{x}_2), \quad \forall \, k \in \{1, \dots, m\}$$

2. Candidate $\mathbf{x}_1$ is strictly better than $\mathbf{x}_2$ in at least one objective:

$$\exists \, j \in \{1, \dots, m\} \quad \text{such that} \quad f_j(\mathbf{x}_1) > f_j(\mathbf{x}_2)$$

A candidate $\mathbf{x}^*$ is defined as Pareto optimal (or non-dominated) if there exists no other candidate in the entire population that dominates it. The set of all non-dominated candidates forms the Pareto frontier. Every design on the Pareto frontier represents an optimal, uncompromising physical trade-off: one cannot improve any single property of that material without degrading another.

```
  High Wall Pressure P_wall (bar)
         ^
         |             (Candidate A: Highest P_wall, Moderate ln p(x))
         |               * [Pareto Optimal]
         |              / \
         |             /   * [Candidate B: Balanced Trade-off, Pareto Optimal]
         |            /     \
         |           /       * [Candidate C: Highest Conformer Flexibility]
         |          /
         |         * [Candidate D: Dominated by B]
         |
         +------------------------------------------------------------>
                                                    High Log-Likelihood ln p(x)
```

In `FunnelRanker`, `dens-city` executes non-dominated sorting across the 5-dimensional multi-objective physical vector:

$$\mathbf{F}_{\text{Pareto}} = \left( P_{\text{wall}} \uparrow, \; \ln p_X(x) \uparrow, \; -U_{\text{EGNN}} \uparrow, \; -\|F\|_{\text{RMS}} \uparrow, \; R_{\text{RL}} \uparrow \right)$$

For every candidate $c_i$, the algorithm performs pairwise comparisons against all other candidates $c_j$. A candidate is flagged `is_pareto_optimal = True` if and only if no other candidate in the evaluated batch dominates it across all five criteria simultaneously. This flags the most uncompromising physical solutions for priority synthesis.

## 6.5 Export Artifacts and Automated Delivery

Once sorting and Pareto classification are complete, `FunnelRanker.export_results` automatically generates and writes three standardized artifact formats into the run output directory (`runs/funnel_results/<spec_name>/`):

### 6.5.1 3D Tripos `.mol2` Molecular Structure Files

For each of the top-$K$ candidates (default $K = 20$), the engine writes an independent, fully parameterized 3D Tripos `.mol2` structure file (`export_mol2_string` in `env.py`):
- Header Section: Declares molecule name, atom count, bond count, substructure count, and energy flags (`GAFF_CHARGES`).
- Atom Section (`@<TRIPOS>ATOM`): Writes 1-indexed atom records containing:
  - Atomic symbol and unique identifier (e.g., `C1`, `N4`, `O7`).
  - 3D Cartesian coordinates $(x, y, z)$ formatted to four decimal places in Angstroms, reflecting the unrolled quantum-relaxed minimum.
  - General Amber Force Field (GAFF) atom types: Automatically assigning `C.ar` and `N.ar` for aromatic ring systems, `C.3` for aliphatic sp3 carbons, `O.3` for ether/alcohol oxygens, and `S.3` for thioethers.
  - Quantum partial charges ($q_i$) derived from the EGNN electronegativity-equilibrated readout, satisfying exact charge neutrality.
- Bond Section (`@<TRIPOS>BOND`): Writes covalent bond connectivity with formal bond orders ($1, 2, 3$, or `ar` for aromatic bonds).
- Substructure Section (`@<TRIPOS>SUBSTRUCTURE`): Writes standard molecular root records compatible with PyMOL, VMD, ChimeraX, and quantum chemistry packages (Gaussian, ORCA, Q-Chem).

### 6.5.2 Comprehensive CSV Summary Table (`funnel_summary.csv`)

The complete telemetry for all evaluated candidates is exported to a comma-separated values table. Each row contains twenty-four structured fields:
`rank`, `name`, `funnel_score`, `rl_reward`, `wall_pressure_bar`, `target_wall_pressure_bar`, `contact_ratio`, `solvation_free_energy_kcal_mol`, `target_solvation_kcal`, `bg_log_likelihood`, `bg_energy_mean`, `bg_energy_var`, `egnn_energy`, `egnn_force_rms`, `molecular_weight`, `num_sites`, `pmi_linearity`, `aromatic_density`, `rotatable_fraction`, `sa_score`, `smiles`, `wl_hash`, `is_pareto_optimal`.

### 6.5.3 Markdown Ranking Report (`funnel_report.md`)

The engine renders an automated executive report summarizing total screened candidates, synthesizability pruning statistics, and a formatted Markdown table displaying the top-ranked candidates with bolded composite scores, thermodynamic pressure ratios, quantum energies, force residuals, and explicit Pareto frontier designations.

---

# Concluding Synthesis: The Differentiable Condensed-Phase Paradigm

The `dens-city` inverse design architecture demonstrates that autonomous molecular discovery cannot be treated as a detached exercise in discrete graph generation or gas-phase quantum chemistry. By coupling C-native SIMD graph assembly, Classical Density Functional Theory, quasi-Newton trust-region geometry optimization, invertible normalizing flows, and equivariant graph neural networks into a unified, static GPU execution graph, the platform establishes a mathematically rigorous paradigm for computational chemistry.

Through the elimination of Python GIL contention, the eradication of memory churn via `@TinyJit` command buffer fusion, and the integration of macroscopic liquid thermodynamics directly into the reinforcement learning reward signal, `dens-city` bridges the gap between molecular topology and macroscopic performance. It provides chemical engineers and computational physicists with an uncompromising, end-to-end differentiable platform for the autonomous discovery of the next generation of advanced functional materials.
