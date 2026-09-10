# Pattern: E(n) Equivariant Graph Neural Networks (EGNN) for Molecular Architectures

## Summary
- **Problem**:
  1. Predicting molecular properties, energy surfaces, or atomic trajectories requires rotational, translational, reflection (\(\mathrm{E}(3)\)), and permutation (\(S_N\)) equivariance. Standard GNNs fail to preserve spatial symmetries, while spherical harmonic methods (TFN, SE(3)-Transformer) require expensive Clebsch-Gordan tensor products that do not scale well.
  2. Evaluating thermodynamic ensemble properties (e.g. hydration free energy \(\Delta G_{\rm solv}\)) on a single static vacuum geometry introduces severe prediction variance (often \(>5\text{ kcal/mol}\) across rotamers of the same molecule) because free energy is an integral over configurational phase space \(\Gamma(T)\), not a single-point electronic property.
- **Root Cause**:
  1. Cartesian atomic positions \(\mathbf{x}_i \in \mathbb{R}^3\) transform as type-1 vectors under \(\mathrm{E}(3)\), while node embeddings \(\mathbf{h}_i \in \mathbb{R}^{\rm nf}\) (atomic number, charge, solvent features) are scalar type-0 invariants. Coupling them requires an exact radial vector field formulation.
  2. Single-conformation QML neglects conformational entropy and thermal fluctuations. Furthermore, naive conformer looping in Python destroys GPU memory coalescing and breaks `@TinyJit` static graph compilation.
- **Actionable Fix**:
  1. Implement the Equivariant Graph Convolutional Layer (EGCL) using relative squared distances \(\|\mathbf{x}_i - \mathbf{x}_j\|^2\) for invariant scalar messages and a radial displacement field \((\mathbf{x}_i - \mathbf{x}_j) \phi_x(\mathbf{m}_{ij})\) for vector coordinate updates.
  2. Implement vector-parallel multi-conformation ensembling per Weinreich et al. (2021) Eq. (4):
     \[\langle \mathbf{X} \rangle(T) \approx \frac{1}{s} \sum_{k=1}^s \mathbf{X}_k\]
     by reshaping input batches \((B, s, N, 3) \to (B \cdot s, N, 3)\), executing all 7 EGNN layers in a single hardware pass, and reducing across conformers.
  3. Aggregate node representations into molecular descriptors via multi-scale invariant pooling (`mean`, masked `max`, `std` over real atoms), producing an invariant 384-dimensional vector \(\mathbf{z}_{\rm mol} \in \mathbb{R}^{384}\).
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.egnn`, `dens_city.boltzmann.train_charges`

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

4. **Conformational Ensembling Across Phase Space \(\Gamma(T)\) (Weinreich FML Principle)**:
   - In statistical thermodynamics, the hydration free energy is an integral over the phase space of configurations accessible at temperature \(T\):
     \[\Delta G_{\rm solv}(T) = -k_B T \ln \frac{\int_{\Gamma(T)} e^{-\beta (U_{\rm solute} + U_{\rm solvent} + U_{\rm int})} d\Gamma}{\int_{\Gamma(T)} e^{-\beta U_{\rm solute}} d\Gamma}\]
   - Basing predictions on a single arbitrary vacuum conformation \(\mathbf{x}_0\) introduces large spurious errors. As shown in Weinreich et al. (2021) Fig. 4, predictions across 976 conformers of \(\text{C}_7\text{H}_{10}\text{O}_2\) fluctuate by up to \(5.3\text{ kcal/mol}\).
   - Computing the Boltzmann ensemble average over \(s\) configurations sampled from thermal vibrational basins (\(T=350\text{ K}\)) or normalizing flow trajectories:
     \[\langle \mathbf{h}_i \rangle = \frac{1}{s} \sum_{k=1}^s \mathbf{h}_i(\mathbf{x}_k), \quad \langle q_i \rangle = \frac{1}{s} \sum_{k=1}^s q_i(\mathbf{x}_k)\]
     guarantees convergence of the molecular representation (relative deviation \(<4\%\) at \(s=10\)) and eliminates single-geometry variance.

5. **Multi-Scale Invariant Graph Pooling**:
   - To aggregate local node states \(\mathbf{h}_i \in \mathbb{R}^{B \times N \times 128}\) into a global molecular descriptor without losing extreme activation sites:
     \[\text{mean}_k = \frac{1}{N_{\rm real}} \sum_{i=1}^{N} \mathbf{h}_{i,k} \cdot m_i\]
     \[\text{max}_k = \max_{i=1\dots N} \left( \mathbf{h}_{i,k} \cdot m_i - (1 - m_i) \cdot 10^4 \right)\]
     \[\text{std}_k = \sqrt{\frac{1}{N_{\rm real}} \sum_{i=1}^N (\mathbf{h}_{i,k} - \text{mean}_k)^2 \cdot m_i + 10^{-6}}\]
     \[\mathbf{z}_{\rm mol} = [\text{mean} \parallel \text{max} \parallel \text{std}] \in \mathbb{R}^{B \times 384}\]
   - The penalization \(-(1 - m_i) \cdot 10^4\) strictly isolates real atoms from padded zero slots, ensuring that dummy padding does not corrupt the maximum reduction.

## Verified Implementation Pattern
```python
# Multi-conformation vector-parallel ensembling in Tinygrad
def compute_ensembled_solvation_readouts(
    self,
    x_ensemble: Tensor,  # (B, s, N, 3) or (s, N, 3)
    atomic_numbers: Tensor,  # (B, N)
    atom_mask: Tensor,  # (B, N, 1)
    total_charge: Tensor,  # (B, 1, 1)
    base_charges: Tensor,  # (B, N)
    solvent_features: Optional[Tensor] = None,  # (B, N, 4)
    detach_trunk: bool = True,
    dielectric_constant: float = 78.4,
    return_global: bool = False,
) -> Tuple[Tensor, ...]:
    # Flatten batch and conformer axes: (B * s, N, 3)
    if len(x_ensemble.shape) == 3:
        s_conf, n_particles, _ = x_ensemble.shape
        b_size = 1
        x_flat = x_ensemble
    else:
        b_size, s_conf, n_particles, _ = x_ensemble.shape
        x_flat = x_ensemble.reshape(b_size * s_conf, n_particles, 3)

    # Expand masks and baseline descriptors across conformer dimension
    z_exp = atomic_numbers.unsqueeze(1).expand(b_size, s_conf, n_particles).reshape(b_size * s_conf, n_particles)
    m_exp = atom_mask.unsqueeze(1).expand(b_size, s_conf, n_particles, 1).reshape(b_size * s_conf, n_particles, 1)
    tq_exp = total_charge.unsqueeze(1).expand(b_size, s_conf, 1, 1).reshape(b_size * s_conf, 1, 1)
    bq_exp = base_charges.unsqueeze(1).expand(b_size, s_conf, n_particles).reshape(b_size * s_conf, n_particles)

    # Execute all 7 EGNN layers in a single vector-parallel pass (no Python loops)
    q_pred_flat, delta_vdw_flat, _, delta_g_coop_flat = self.compute_solvation_readouts(
        x=x_flat,
        atomic_numbers=z_exp,
        atom_mask=m_exp,
        total_charge=tq_exp,
        base_charges=bq_exp,
        solvent_features=sf_exp,
        detach_trunk=detach_trunk,
        return_global=True,
    )

    # Generalized Born solvation over all conformations simultaneously
    gb_flat = gb_solver.compute_solvation_free_energy(
        x=x_flat, charges=q_pred_flat, atomic_numbers=z_exp, atom_mask=m_exp,
        dielectric_constant=dielectric_constant
    )

    # Ensemble expectation values: <q_i>, <ΔG_vdw>, <ΔG_GB>, <ΔG_coop>
    q_mean = q_pred_flat.reshape(b_size, s_conf, n_particles).mean(axis=1)
    vdw_mean = delta_vdw_flat.reshape(b_size, s_conf).mean(axis=1)
    gb_mean = gb_flat.reshape(b_size, s_conf).mean(axis=1)
    coop_mean = delta_g_coop_flat.reshape(b_size, s_conf).mean(axis=1)

    return q_mean, vdw_mean, gb_mean, coop_mean
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Passing raw Cartesian coordinates \((x, y, z)\) into the edge MLP \(\phi_e\) (violates translation and rotation invariance).
- ❌ **Anti-Pattern**: Updating coordinates with absolute directional components rather than relative difference vectors \((\mathbf{x}_i - \mathbf{x}_j)\) (destroys coordinate equivariance).
- ❌ **Anti-Pattern**: Evaluating free energies on a single static vacuum geometry (Weinreich pitfall: causes multi-kcal/mol spurious variance across stereoisomers).
- ❌ **Anti-Pattern**: Looping over conformers \(k=1\dots s\) in Python rather than flattening \((B, s, N, 3) \to (B \cdot s, N, 3)\) (destroys GPU kernel fusion and breaks `@TinyJit`).
- ❌ **Anti-Pattern**: Using naive unmasked max-pooling over padded dummy atoms (pollutes molecular representation with zeros from dummy padding; always subtract \((1 - m) \cdot 10^4\)).

