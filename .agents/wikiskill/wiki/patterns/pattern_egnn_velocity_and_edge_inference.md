# Pattern: EGNN Momentum Tracking, Edge Inference, and Symmetry Breaking

## Summary
- **Problem**: In molecular dynamics forecasting and quantum force-field emulation, atomic momentum/velocity must be integrated without breaking \(\mathrm{E}(3)\) equivariance; furthermore, explicit chemical bond topologies or solvent adjacency are often incomplete or unknown.
- **Root Cause**: Velocities \(\mathbf{v}_i\) transform equivariantly under rotation and reflection but are invariant under spatial translations (\(Q\mathbf{v}_i\), unaffected by \(+\mathbf{g}\)). Rigid topological graphs without node features can cause identical message passing embeddings on symmetric nodes (e.g. cycle graphs).
- **Actionable Fix**:
  1. Extend the coordinate update to track momentum via scalar velocity modulation \(\phi_v(\mathbf{h}_i^l)\mathbf{v}_i^{\rm init}\).
  2. Infer continuous edge existence via soft sigmoid message gating \(\phi_{\rm inf}(\mathbf{m}_{ij}) \in [0, 1]\).
  3. Break topological symmetry by injecting isotropic Gaussian coordinate noise \(\mathbf{x}^0 \sim \mathcal{N}(\mathbf{0}, \sigma\mathbf{I})\) while preserving exact \(\mathrm{E}(n)\) equivariance.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.egnn`

## Deep Theoretical Formulation
1. **Equivariant Velocity & Momentum Propagation**:
   - Initial velocities \(\mathbf{v}_i^{\rm init} \in \mathbb{R}^3\) transform as vectors under rotation (\(Q\mathbf{v}_i\)) but remain unchanged under coordinate shift \(\mathbf{x} \to \mathbf{x} + \mathbf{g}\).
   - In each EGCL layer:
\[\mathbf{v}_i^{l+1} = \phi_v(\mathbf{h}_i^l)\mathbf{v}_i^{\rm init} + \frac{1}{M - 1}\sum_{j \neq i} (\mathbf{x}_i^l - \mathbf{x}_j^l) \phi_x(\mathbf{m}_{ij})\]
\[\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \mathbf{v}_i^{l+1}\]
   - Proof of equivariance: \(Q\mathbf{v}_i^{l+1} = \phi_v(\mathbf{h}_i)Q\mathbf{v}_i^{\rm init} + Q\left[\frac{1}{M-1}\sum (\mathbf{x}_i - \mathbf{x}_j)\phi_x\right]\), which yields \(Q\mathbf{x}_i^{l+1} + \mathbf{g}\).

2. **Continuous Edge Inference**:
   - For long-range non-bonded dispersion or solvent shells, explicit adjacency matrices \(A_{ij}\) can be omitted or replaced by soft learned edge weights:
\[e_{ij} \approx \phi_{\rm inf}(\mathbf{m}_{ij}) = \sigma(\mathbf{W}_{\rm inf} \mathbf{m}_{ij} + b_{\rm inf}) \in [0, 1]\]
\[\mathbf{m}_i = \sum_{j \neq i} e_{ij} \mathbf{m}_{ij}\]
   - Because \(\mathbf{m}_{ij}\) is \(\mathrm{E}(n)\)-invariant, soft inferred edges preserve all spatial symmetries.

3. **Symmetry Breaking in Point Clouds & Graphs**:
   - When nodes lack distinct features (e.g. identical carbon rings), standard GNNs produce identical embeddings across symmetric nodes.
   - Initializing coordinates with Gaussian noise \(\mathbf{x}^0 \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})\) breaks structural permutation symmetry into positional representations. Because the network is \(\mathrm{E}(n)\)-equivariant to this noise, generalization is preserved.

## Verified Implementation Pattern
```python
# Velocity-aware equivariant update
def equivariant_velocity_step(
    x: Tensor,
    v_init: Tensor,
    h: Tensor,
    m_ij: Tensor,
    phi_v_mlp,
    phi_x_mlp,
) -> Tuple[Tensor, Tensor]:
    N = x.shape[0]
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N, 3)
    w_x = phi_x_mlp(m_ij)  # (N, N, 1)

    # Scale initial velocity by invariant scalar
    scale_v = phi_v_mlp(h)  # (N, 1)
    accel = (diff * w_x).mean(axis=1)  # (N, 3)

    v_next = scale_v * v_init + accel
    x_next = x + v_next
    return x_next, v_next
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Adding translational offset \(\mathbf{g}\) to velocity vectors (velocities are translation-invariant differentials).
- ❌ **Anti-Pattern**: Hard-coding fixed distance threshold cutoffs without soft edge sigmoid damping (causes step discontinuities in forces and gradients).
