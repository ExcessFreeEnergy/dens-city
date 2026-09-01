# Purpose & Evolution History: `cdft-wikiskill`

- **Origin**: Evolved from the WikiSkill knowledge co-evolution architecture (arXiv:2608.27454v1).
- **Motivating Problem**: Eliminate the recurring cycle of fixing, forgetting, and reimplementing errors in cDFT simulation and Boltzmann generator routines.
- **Motivating Wiki Patterns Addressed**:
  - `patterns/pattern_log_free_latent_density.md`
  - `patterns/pattern_irving_kirkwood_virial_pressure.md`
  - `patterns/pattern_anti_aliased_fmt_kernels.md`
  - `patterns/pattern_steric_boundary_divergence.md`
  - `patterns/pattern_scale_invariant_geometry.md`
  - `patterns/pattern_tinygrad_jit_graph_caching.md`
  - `patterns/pattern_tripos_mol2_forcefield_derivation.md`
  - `patterns/pattern_boltzmann_dual_loss_training.md`
  - `patterns/pattern_boltzmann_mixed_coordinates_whitening.md`
  - `patterns/pattern_boltzmann_torsional_invertibility.md`
  - `patterns/pattern_boltzmann_energy_regularization_clipping.md`
  - `patterns/pattern_boltzmann_latent_mcmc_exploration.md`
  - `patterns/pattern_boltzmann_reweighting_free_energy.md`
  - `patterns/pattern_egnn_equivariant_molecular_message_passing.md`
  - `patterns/pattern_egnn_velocity_and_edge_inference.md`
  - `patterns/pattern_tinygrad_uop_graph_and_lazy_evaluation.md`
  - `patterns/pattern_tinygrad_lowering_pipeline_and_codegen.md`
  - `patterns/pattern_tinygrad_decomposed_tensor_idioms.md`
  - `patterns/pattern_tinygrad_golden_idioms_beautiful_mnist.md`
- **Evolution History**:
  - `v1.0.0`: Initial bootstrap with 7 foundational physics and computational patterns.
  - `v1.1.0`: Added 6 Boltzmann Generator statistical mechanics and normalizing flow invariants (Noé et al., arXiv:1812.01729v2).
  - `v1.2.0`: Added 5 patterns covering E(n) Equivariant Graph Neural Networks (Satorras et al., arXiv:2102.09844v3) and Tinygrad UOp compiler specification (`data/tinyspec.tex`).
  - `v1.3.0`: Added canonical Tinygrad training loop design and step fusion patterns from `beautiful_mnist.py`.



