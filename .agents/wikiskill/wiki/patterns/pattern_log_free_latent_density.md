# Pattern: Log-Free Latent Density Field Parameterization

## Summary
- **Problem**: Optimization producing NaNs or negative densities during variational free energy functional minimization.
- **Root Cause**: Direct unconstrained optimization of density \(\rho(z)\) steps into negative values, causing \(\ln(\rho)\) singularities and gradient divergence.
- **Actionable Fix**: Parameterize density via \(\rho(z) = \rho_{\rm bulk} \exp(\psi(z))\) and optimize the unconstrained latent potential field \(\psi(z)\).
- **Related Skills / Modules**: `cdft-physics`, `dens_city.cdft`

## Deep Root Cause Analysis
In variational Classical Density Functional Theory (cDFT), the equilibrium density profile minimizes the grand potential functional:
\[\Omega[\rho] = \mathcal{F}_{\rm id}[\rho] + \mathcal{F}_{\rm ex}[\rho] + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]\]
The ideal gas free energy term contains \(\int dz \, \rho(z) (\ln(\rho(z)/\rho_{\rm bulk}) - 1)\).
If an optimizer updates \(\rho(z)\) directly, any trial step where \(\rho(z) \le 0\) produces an immediate `NaN` or unphysical negative density. Standard heuristics like `np.clip(rho, 1e-12, None)` distort the analytical functional derivative and stall gradient-based convergence (L-BFGS / Adam).

## Verified Solution & Action Rules
1. Define the latent potential field \(\psi(z) \in \mathbb{R}\).
2. Set density as \(\rho(z) = \rho_{\rm bulk} \exp(\psi(z))\).
3. Express the ideal free energy in the log-free formulation:
\[\mathcal{F}_{\rm ideal}[\psi] = k_B T \int dz \, \left[ \rho(z) \psi(z) - (\rho(z) - \rho_{\rm bulk}) \right]\]
4. Compute functional derivatives with respect to \(\psi(z)\):
\[\frac{\delta \Omega}{\delta \psi(z)} = \rho(z) \left( k_B T \psi(z) + V_{\rm ext}(z) - \mu + \frac{\delta \mathcal{F}_{\rm ex}}{\delta \rho(z)} \right)\]

```python
# Verified Implementation Pattern
def compute_density(psi: Tensor, rho_bulk: float) -> Tensor:
    return rho_bulk * psi.exp()

def ideal_free_energy(psi: Tensor, rho: Tensor, rho_bulk: float, dz: float, kbt: float) -> Tensor:
    # Log-free ideal gas free energy functional
    integrand = rho * psi - (rho - rho_bulk)
    return (integrand * dz).sum() * kbt
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Optimizing `rho` directly in gradient descent or L-BFGS.
- ❌ **Anti-Pattern**: Using `rho.clip(1e-15, None)` or `np.maximum(rho, 0.0)` to mask non-positivity.
- ❌ **Anti-Pattern**: Evaluating `rho * rho.log()` directly without the exponential latent transformation.
