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

## Verification Workflow
Always verify changes using the test suite:
```bash
export PATH="/home/gauss/code/cdft_sim/dens-city/.venv/bin:$PATH"
pytest tests/test_tiny_cdft.py tests/test_batched_cdft.py tests/test_wikiskill.py -v
ruff check src/ tests/
```

