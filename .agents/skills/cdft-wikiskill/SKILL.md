---
name: cdft-wikiskill
description: Procedural knowledge and physics invariants for the dens-city cDFT and Boltzmann generator platform. Use when modifying physics models, cDFT functional minimization, Boltzmann generators, or debugging test regressions.
---

# cDFT & Molecular Simulation Expert Guide (`cdft-wikiskill`)

This skill equips Antigravity agents with the compiled, persistent knowledge base of `dens-city` to prevent fixing, forgetting, and reimplementing known physics errors.

## Mandatory Pre-Flight Checklist Before Modifying Code
1. **Consult Known Patterns**: Inspect [`.agents/wikiskill/wiki/index.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/index.md) to check for documented failure modes relating to your task.
2. **Audit Proposal History**: Check [`.agents/wikiskill/wiki/skill-impact.md`](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/skill-impact.md) to ensure your proposed approach has not been tried and rejected in previous iterations.

## Core Physics & Software Rules

### 1. Log-Free Latent Density Optimization
- **Rule**: Never optimize \(\rho(z)\) directly.
- **Implementation**: Optimize \(\psi(z)\) with \(\rho(z) = \rho_{\rm bulk} \exp(\psi(z))\).
- **Reference**: [pattern_log_free_latent_density.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_log_free_latent_density.md).

### 2. Exact Irving-Kirkwood Virial Contact Pressure
- **Rule**: Never use spatial index slices (e.g. `rho[0:5]` or `rho[mid]`).
- **Implementation**: Evaluate \(P_{\rm wall} = -\int_0^{L_z/2} \rho(z) \frac{dV_{\rm ext}(z)}{dz} dz\).
- **Reference**: [pattern_irving_kirkwood_virial_pressure.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_irving_kirkwood_virial_pressure.md).

### 3. Anti-Aliased Rosenfeld FMT Kernels
- **Rule**: Precompute weight kernels via analytical cell-integration over grid bins \([z - dz/2, z + dz/2]\).
- **Reference**: [pattern_anti_aliased_fmt_kernels.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_anti_aliased_fmt_kernels.md).

### 4. Steric Hard Boundary Divergence & Masking
- **Rule**: Enforce \(V_{\max} = 10^6\,k_B T\) at steric walls and use `.where()` masking to eliminate \(0 \times \infty\) NaNs.
- **Reference**: [pattern_steric_boundary_divergence.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_steric_boundary_divergence.md).

### 5. Scale-Invariant Geometry & Cutoffs
- **Rule**: Scale domain length \(L_z = \max(40.0, 10.0\sigma_{\rm eff})\) and cutoff \(r_{\rm cut} = \max(15.0, 5.0\sigma_{\rm eff})\).
- **Reference**: [pattern_scale_invariant_geometry.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_scale_invariant_geometry.md).

### 6. Tinygrad JIT Realization Safety
- **Rule**: Keep tensor shapes static across `@TinyJit` calls and call `.realize()` before host CPU operations.
- **Reference**: [pattern_tinygrad_jit_graph_caching.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_tinygrad_jit_graph_caching.md).

### 7. Boltzmann Generator Dual Loss Training & Mode Collapse
- **Rule**: Combine energy training \(J_{KL} = U - H_X + H_Z\) (including log-Jacobian \(\log R_{zx}\)) with example training \(J_{ML}\) to prevent mode collapse.
- **Reference**: [pattern_boltzmann_dual_loss_training.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_boltzmann_dual_loss_training.md).

### 8. Mixed Coordinates & PCA Whitening for Molecules
- **Rule**: Never train flows directly on raw unwhitened 3D Cartesian coordinates for macromolecules. Use PCA whitening on Cartesian core and normalized internal coordinates on branches.
- **Reference**: [pattern_boltzmann_mixed_coordinates_whitening.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_boltzmann_mixed_coordinates_whitening.md).

### 9. Torsional Boundary Regularization in Dihedral Space
- **Rule**: Penalize out-of-range dihedral angles outside \([-\pi, \pi]\) with a quadratic loss \(w_{\rm tor}\) to preserve bijective invertibility \(\mathbf{z} \to \mathbf{x} \to \mathbf{z}\).
- **Reference**: [pattern_boltzmann_torsional_invertibility.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_boltzmann_torsional_invertibility.md).

### 10. Energy Log-Regularization for Steric Clashes
- **Rule**: Apply smooth logarithmic energy thresholding \(E_{\rm high}\) to prevent gradient explosions during early generative flow training.
- **Reference**: [pattern_boltzmann_energy_regularization_clipping.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_boltzmann_energy_regularization_clipping.md).

### 11. Statistical Reweighting & Free Energy Differences
- **Rule**: Reweight generated samples via \(w(\mathbf{x}) \propto \exp(-u + u_Z + \log R_{zx})\) for unbiased expectation values and compute \(\Delta A_{12} = \langle J_{KL}^{(2)}\rangle - \langle J_{KL}^{(1)}\rangle\).
- **Reference**: [pattern_boltzmann_reweighting_free_energy.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_boltzmann_reweighting_free_energy.md).

### 12. E(n) Equivariant Message Passing (EGNN)
- **Rule**: Compute edge messages from invariant squared distances \(\|\mathbf{x}_i - \mathbf{x}_j\|^2\) and update coordinates along relative difference radial fields \((\mathbf{x}_i - \mathbf{x}_j)\phi_x(\mathbf{m}_{ij})\). Never pass unconstrained raw Cartesian coordinates to edge MLPs.
- **Reference**: [pattern_egnn_equivariant_molecular_message_passing.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_egnn_equivariant_molecular_message_passing.md).

### 13. Equivariant Velocity & Soft Adjacency Inference
- **Rule**: For dynamics forecasting, update momentum via \(\mathbf{v}_i^{l+1} = \phi_v(\mathbf{h}_i^l)\mathbf{v}_i^{\rm init} + \text{accel}\) (velocities are translation-invariant differentials). Infer unknown non-bonded edges with soft sigmoid gating \(\phi_{\rm inf}(\mathbf{m}_{ij})\).
- **Reference**: [pattern_egnn_velocity_and_edge_inference.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_egnn_velocity_and_edge_inference.md).

### 14. Tinygrad Universal UOp DAG & Movement Ops
- **Rule**: Treat all tinygrad tensors as lazy UOp graphs. Leverage zero-arithmetic movement views (`Permute`, `Reshape`, `Expand`, `Pad`, `Shrink`) instead of allocating intermediate buffers or calling Python loops.
- **Reference**: [pattern_tinygrad_uop_graph_and_lazy_evaluation.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_tinygrad_uop_graph_and_lazy_evaluation.md).

### 15. Tinygrad Decomposed Tensor Primitives
- **Rule**: Compose complex tensor operations (`gemm`, `prefix_sum`, `arange`, `gather`, `scatter_add`) using tinygrad's canonical decomposed patterns to maximize kernel fusion and enable tensor-core acceleration.
- **Reference**: [pattern_tinygrad_decomposed_tensor_idioms.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_tinygrad_decomposed_tensor_idioms.md).

### 16. Idiomatic Tinygrad Training Loops & Step Fusion
- **Rule**: Follow the golden `beautiful_mnist.py` standard: wrap forward calls in `@function`, decorate training steps with `@TinyJit` and `@Context(TRAINING=1)`, sample batches on-device with `Tensor.randint` (Threefry PRNG), and fuse backward adjoints with optimizer weight updates via `loss.realize(*opt.schedule_step())`.
- **Reference**: [pattern_tinygrad_golden_idioms_beautiful_mnist.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_tinygrad_golden_idioms_beautiful_mnist.md).

### 17. Percus-Yevick Compressibility Route for FMT Consistency
- **Rule**: Never use empirical cubic equations (Peng-Robinson, Redlich-Kwong) or Carnahan-Starling for cDFT reservoir states. Derive bulk density $\rho_{\rm bulk}$ and chemical potential $\mu$ strictly via the Percus-Yevick compressibility EOS to prevent asymptotic density drift.
- **Reference**: [pattern_percus_yevick_fmt_thermodynamic_consistency.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_percus_yevick_fmt_thermodynamic_consistency.md).

### 18. Tensor-Native Generalized Born Solvation
- **Rule**: Compute dielectric aqueous hydration using $O(1)$ GPU Bondi radii gathers, Hawkins/Grycuk smooth volume descreening ($\alpha_i \ge \rho_i$ without $1/r^4$ singularity), and Still pairwise screened electrostatics.
- **Reference**: [pattern_generalized_born_implicit_solvation.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_generalized_born_implicit_solvation.md).

### 19. EGNN Memory Decomposition & Charge Neutrality
- **Rule**: In deep EGNN layers, decompose linear projections $e_{ij} = \text{SiLU}(W_{hi} h_i + W_{hj} h_j + W_d d^2 + W_a a + b)$ to avoid $(B, N, N, 258)$ OOM crashes. Normalize neighborhood messages by active degree and enforce exact formal charge conservation via mean-shift.
- **Reference**: [pattern_egnn_quantum_charges_and_memory_decomposition.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_egnn_quantum_charges_and_memory_decomposition.md).

### 20. Batched L-BFGS with Trust-Region Clamping
- **Rule**: Minimize molecular geometries on GPU using vectorized two-loop recursion ($m=6$), enforce trust-region step clamping ($\Delta r_{\max} \le 0.20\text{ \AA}$) to prevent atoms jumping across Lennard-Jones repulsive cores, and freeze converged molecules via SIMD active masking.
- **Reference**: [pattern_batched_lbfgs_trust_region_relaxation.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_batched_lbfgs_trust_region_relaxation.md).

### 21. Ensembled EGNN & Cooperative Solvation (Weinreich FML)
- **Rule**: Never evaluate hydration free energies on a single static vacuum geometry. Average predictions across thermal vibrational ensembles ($T=350\text{ K}$, $s \ge 8$) in a single vector-parallel pass $(B \cdot s, N, 3)$ without Python loops. Pool representations via multi-scale statistics (`mean`, masked `max`, `std`), provide $\ge 25.0$ kcal/mol headroom on cooperative heads to avoid $\tanh$ gradient vanishing on polyols/uracils, and use analytical KRR with $O(N^2)$ Sherman-Morrison LOOCV.
- **Reference**: [pattern_egnn_ensembled_cooperative_solvation.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_egnn_ensembled_cooperative_solvation.md).

### 22. Universal Multi-Solvent Representation & Dynamic BMCSL Cavitation
- **Rule**: Never use water's hard-sphere radius for non-aqueous fluids. Derive solvent kinetic diameters dynamically $\sigma_S = (6 V_m / \pi N_A)^{1/3}$ for BMCSL cavitation. In non-polar solvents, gate neural cooperative hydrogen-bonding readouts by Abraham H-bond capacity $\tanh((\alpha_S + \beta_S)/1.5)$ to suppress unphysical stabilization. For universal multi-solvent residual prediction, concatenate 7D normalized Abraham/physical descriptors into a 397D joint pair embedding $[\mathbf{z}_{\rm solute} \parallel 2.0\mathbf{d}_{\rm solute} \parallel 2.5\mathbf{s}_{\rm solvent}]$.
- **Reference**: [pattern_solvatum_universal_solvent_architecture.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_solvatum_universal_solvent_architecture.md).

### 23. Universal Static Padding & In-Place JIT Graph Reuse
- **Rule (The Ideal Standard)**: Zero JIT kernel recompilation overhead after Batch 0. Never allow varying molecular sizes or kernel widths to trigger JIT schedule misses. Pad molecular batch tensors to static bounds ($B=64, N_{\rm pad}=128$) and pad grouped convolution kernels to universal static lengths ($K_{\rm fmt}=41, K_{\rm att}=129$) with symmetric center-padding and zero weights outside compact physical support. Re-bind batch parameters and optimizer moments in-place via `.assign()` on persistent solver singletons rather than re-instantiating objects.
- **Reference**: [pattern_persistent_jit_graph_reuse_and_static_padding.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_persistent_jit_graph_reuse_and_static_padding.md).

### 24. Decoupled Reverse Autodiff & Conservative Force Surrogates
- **Rule (The Ideal Standard)**: In deep multi-layer equivariant networks (e.g. 7-layer EGNN), never backpropagate through an unrolled autograd tape over shared spatial distance matrices ($d_{\rm sq}$), which collapses register files into monolithic mega-reduction kernels. Decouple adjoint backpropagation layer-by-layer: $\nabla_x U = \sum_l \frac{\partial U}{\partial h_l} \frac{\partial h_l}{\partial x}$. In generative normalizing flows, train the coordinate generator via the detached conservative force surrogate work $\mathcal{L}_{\rm surr} = \mathbb{E}_z [ -\beta F_{\rm det} \cdot x_\theta(z) - \log p_z - \log |\det J| ]$, releasing forward activation graphs of the potential energy model and strictly enforcing Rule 5 (`p.grad = None`) after force evaluation.
- **Reference**: [pattern_decoupled_reverse_autodiff_and_conservative_force_surrogate.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_decoupled_reverse_autodiff_and_conservative_force_surrogate.md).

### 25. Fully Vectorized High-Throughput Batch Processing
- **Rule (The Ideal Standard)**: Zero sequential Python candidate loops during batch post-processing. Combine all molecules in a batch ($B=64$) into padded tensors $(B, N_{\rm pad}, 3)$ and $(B, N_{\rm pad})$ along Axis 0. Execute Generalized Born continuum electrostatics, EGNN force fields, and Universal Delta-KRR models in parallel, and realize all output tensors in a single unified `Tensor.realize(...)` call to collapse host-device round-trip latency to near zero. Extract individual molecular quantities using fast NumPy vector slicing.
- **Reference**: [pattern_fully_vectorized_high_throughput_batch_inference.md](file:///home/gauss/code/cdft_sim/dens-city/.agents/wikiskill/wiki/patterns/pattern_fully_vectorized_high_throughput_batch_inference.md).

## Verification Workflow
Always verify changes using the test suite:
```bash
export PATH="/home/gauss/code/cdft_sim/dens-city/.venv/bin:$PATH"
pytest tests/test_tiny_cdft.py tests/test_batched_cdft.py tests/test_wikiskill.py -v
ruff check src/ tests/
```




