# Pattern: Statistical Reweighting & Reaction-Coordinate-Free Free Energy Differences

## Summary
- **Problem**: Computing free energy differences between disconnected metastable states (e.g. folded vs. unfolded conformations) requires months of brute-force MD simulations or complex reaction coordinate tuning.
- **Root Cause**: Standard enhanced sampling methods (Umbrella Sampling, Metadynamics) require a continuous physical order parameter connecting the states; without it, phase space overlap between disconnected states is zero.
- **Actionable Fix**: Train independent Boltzmann Generators for each disconnected state to link them to a common Gaussian reference latent state. The free energy difference is directly computed from the average latent KL divergence values:
\[\Delta A_{12} = \langle J_{KL}^{(2)} \rangle - \langle J_{KL}^{(1)} \rangle\]
and individual sample observables are computed via exact statistical reweighting:
\[w_X(\mathbf{x}) = \frac{\mu_X(\mathbf{x})}{q_X(\mathbf{x})} \propto \exp\left( -u(F_{zx}(\mathbf{z})) + u_Z(\mathbf{z}) + \log R_{zx}(\mathbf{z}) \right)\]
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.generator`

## Deep Root Cause Analysis
For any generated sample \(\mathbf{x} = F_{zx}(\mathbf{z})\), the generative probability density \(q_X(\mathbf{x})\) is known exactly via the change-of-variables theorem:
\[q_X(\mathbf{x}) = \mu_Z(F_{xz}(\mathbf{x})) \cdot R_{xz}(\mathbf{x})\]
Because \(q_X(\mathbf{x})\) is known analytically, every generated configuration \(\mathbf{x}\) can be assigned an exact statistical importance weight:
\[w_X(\mathbf{x}) = \frac{e^{-u(\mathbf{x})}}{q_X(\mathbf{x})}\]
Unbiased equilibrium expectation values for any physical observable \(O(\mathbf{x})\) are computed via:
\[\langle O \rangle = \frac{\sum_{i=1}^N w_X(\mathbf{x}_i) O(\mathbf{x}_i)}{\sum_{i=1}^N w_X(\mathbf{x}_i)}\]
Furthermore, because \(J_{KL} = U - H_X + H_Z\) represents the exact free energy difference \(\Delta A\) between the standard Gaussian reference state and the molecular state, training two independent flows on states 1 and 2 allows direct evaluation of \(\Delta A_{12} = \langle J_{KL}^{(2)} \rangle - \langle J_{KL}^{(1)} \rangle\) using orders of magnitude fewer energy evaluations than Umbrella Sampling.

## Verified Solution & Action Rules
1. Assign statistical weights \(w_i = \exp\left(-u(\mathbf{x}_i) + u_Z(\mathbf{z}_i) + \log R_{zx}(\mathbf{z}_i)\right)\) to all generated samples.
2. Filter out low-weight outlier bins (bins with aggregate weight \(< 0.01\) samples) when constructing free energy histograms \(-k_B T \ln p(R)\).
3. Track the convergence of \(\langle J_{KL}\rangle\) over training batches to compute equilibrium free energy differences between disconnected conformational basins.

```python
# Verified Implementation Pattern
def compute_importance_weights(
    x_gen: Tensor,
    z_latent: Tensor,
    log_det_zx: Tensor,
    energy_fn,
) -> Tensor:
    u_phys = energy_fn(x_gen)
    u_prior = 0.5 * (z_latent * z_latent).sum(axis=-1)
    # log weight = -u_phys + u_prior + log_det_zx
    log_w = -u_phys + u_prior + log_det_zx
    # Stabilize by subtracting maximum log weight
    log_w_stabilized = log_w - log_w.max()
    weights = log_w_stabilized.exp()
    return weights / weights.sum()
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Treating raw generated configurations as unbiased equilibrium samples without importance reweighting.
- ❌ **Anti-Pattern**: Computing free energy histograms without subtracting \(\max(\log w)\), leading to floating point underflow.
- ❌ **Anti-Pattern**: Forcing artificial physical reaction coordinates when independent reference-state coupling is sufficient.
