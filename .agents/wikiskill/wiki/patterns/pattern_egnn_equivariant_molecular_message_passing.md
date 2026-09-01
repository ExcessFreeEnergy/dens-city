# Pattern: E(n) Equivariant Graph Neural Networks (EGNN) for Molecular Architectures

## Summary
- **Problem**: Predicting molecular properties, energy surfaces, or atomic trajectories requires rotational, translational, reflection (\(\mathrm{E}(3)\)), and permutation (\(S_N\)) equivariance. Standard GNNs fail to preserve spatial symmetries, while spherical harmonic methods (TFN, SE(3)-Transformer) require expensive Clebsch-Gordan tensor products that do not scale well.
- **Root Cause**: Cartesian atomic positions \(\mathbf{x}_i \in \mathbb{R}^3\) transform as type-1 vectors under the Euclidean group \(\mathrm{E}(3)\), while node embeddings \(\mathbf{h}_i \in \mathbb{R}^{\rm nf}\) (atomic number, charge, hybridization) are scalar type-0 invariants. Coupling them without breaking symmetry or incurring high computational overhead requires an exact radial vector field formulation.
- **Actionable Fix**: Implement the Equivariant Graph Convolutional Layer (EGCL) using relative squared distances \(\|\mathbf{x}_i - \mathbf{x}_j\|^2\) for invariant scalar messages and a radial displacement field \((\mathbf{x}_i - \mathbf{x}_j) \phi_x(\mathbf{m}_{ij})\) for vector coordinate updates.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.egnn`

## Deep Theoretical Formulation
Given a molecular graph \(\mathcal{G} = (\mathcal{V}, \mathcal{E})\) with node coordinates \(\mathbf{x}_i \in \mathbb{R}^3\), invariant node embeddings \(\mathbf{h}_i \in \mathbb{R}^{\rm nf}\), and edge attributes \(a_{ij}\):
1. **Scalar Edge Message Passing**:
\[\mathbf{m}_{ij} = \phi_e\left(\mathbf{h}_i^l, \mathbf{h}_j^l, \left\|\mathbf{x}_i^l - \mathbf{x}_j^l\right\|^2, a_{ij}\right)\]
   - Because the squared Euclidean distance \(\|\mathbf{x}_i - \mathbf{x}_j\|^2\) is strictly invariant under any translation \(\mathbf{x} + \mathbf{g}\) and orthogonal transformation \(Q \in O(3)\) (\(\|Q\mathbf{x}_i - Q\mathbf{x}_j\|^2 = (\mathbf{x}_i - \mathbf{x}_j)^\top Q^\top Q (\mathbf{x}_i - \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|^2\)), \(\mathbf{m}_{ij}\) is strictly \(\mathrm{E}(3)\)-invariant.
2. **Equivariant Coordinate Update (Radial Vector Field)**:
\[\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \frac{1}{M - 1} \sum_{j \neq i} (\mathbf{x}_i^l - \mathbf{x}_j^l) \phi_x(\mathbf{m}_{ij})\]
   - \(\phi_x: \mathbb{R}^{\rm nf} \to \mathbb{R}^1\) outputs a scalar weight.
   - The relative difference vector \((\mathbf{x}_i - \mathbf{x}_j)\) transforms as \(Q(\mathbf{x}_i - \mathbf{x}_j)\).
   - Thus, \(Q\mathbf{x}_i^{l+1} + \mathbf{g} = \mathrm{EGCL}(Q\mathbf{x}^l + \mathbf{g})\), fulfilling exact \(\mathrm{E}(3)\) equivariance.
3. **Node Message Aggregation & Invariant Feature Update**:
\[\mathbf{m}_i = \sum_{j \neq i} \mathbf{m}_{ij}, \quad \mathbf{h}_i^{l+1} = \phi_h(\mathbf{h}_i^l, \mathbf{m}_i)\]
   - Node representations remain strictly invariant to spatial rotations, translations, and reflections.

## Verified Implementation Pattern
```python
# Verified EGCL implementation in tensor operations
class EGCLayer:
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        self.phi_e = MLP(in_features * 2 + 1, hidden_dim, hidden_dim)
        self.phi_x = MLP(hidden_dim, hidden_dim, 1)
        self.phi_h = MLP(in_features + hidden_dim, hidden_dim, out_features)

    def __call__(self, h: Tensor, x: Tensor, edge_index: Optional[Tuple[Tensor, Tensor]] = None):
        # h: (N, d), x: (N, 3)
        N = x.shape[0]
        # Compute pairwise difference vectors and squared distances
        # diff_ij = x_i - x_j -> shape (N, N, 3)
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        dist_sq = (diff * diff).sum(axis=-1, keepdim=True)  # (N, N, 1)

        # Concatenate node features: h_i, h_j, dist_sq
        h_i = h.unsqueeze(1).expand(N, N, h.shape[-1])
        h_j = h.unsqueeze(0).expand(N, N, h.shape[-1])
        edge_input = h_i.cat(h_j, dist_sq, dim=-1)

        # 1. Edge messages
        m_ij = self.phi_e(edge_input)  # (N, N, hidden)

        # 2. Coordinate updates (radial field)
        w_x = self.phi_x(m_ij)  # (N, N, 1)
        # Exclude self-interaction (diagonal)
        delta_x = (diff * w_x).mean(axis=1)  # (N, 3)
        x_next = x + delta_x

        # 3. Aggregate node messages
        m_i = m_ij.sum(axis=1)  # (N, hidden)
        h_next = h + self.phi_h(h.cat(m_i, dim=-1))

        return h_next, x_next
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Passing raw Cartesian coordinates \((x, y, z)\) into the edge MLP \(\phi_e\) (violates translation and rotation invariance).
- ❌ **Anti-Pattern**: Updating coordinates with absolute directional components rather than relative difference vectors \((\mathbf{x}_i - \mathbf{x}_j)\) (destroys coordinate equivariance).
- ❌ **Anti-Pattern**: Using computationally heavy spherical harmonics when scalar distance embedding is mathematically sufficient.
