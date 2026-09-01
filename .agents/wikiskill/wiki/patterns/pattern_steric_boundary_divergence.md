# Pattern: Steric Hard Boundary Divergence & Physical Masking

## Summary
- **Problem**: Fluid density penetrates solid walls, or optimizer produces IEEE 754 \(0 \times \infty = \text{NaN}\) floating point exceptions at hard wall boundaries.
- **Root Cause**: Using artificial spatial offsets (e.g. `max(0.2, z)`) or soft boundary clamping (`[-500, 1000]`) fails to enforce true physical steric exclusion and allows unphysical fluid leakage into the substrate.
- **Actionable Fix**: Enforce true physical potential divergence (\(V_{\max} = 10^6\,k_B T\)) at steric exclusion regions and use physical boolean tensor masking (`.where()`) to prevent \(0 \times \infty\) NaN traps.
- **Related Skills / Modules**: `cdft-physics`, `dens_city.cdft`

## Deep Root Cause Analysis
At atomic or planar boundaries where fluid atoms cannot physically enter (\(z < \sigma_{\rm wall}/2\)), the repulsive potential must diverge to infinity.
If a developer implements soft clamping such as `np.clip(V_ext, -500, 1000)`, the resulting Boltzmann factor \(\exp(-1000) \approx 0\) might seem small, but gradient updates to the grand potential functional can still accumulate non-zero density in the wall, violating the hard-core exclusion.
Conversely, setting \(V_{\rm ext} = \infty\) produces IEEE 754 `inf`. When computing the external energy \(\int dz \, \rho(z) V_{\rm ext}(z)\), the product \(\rho(z) \times \infty\) evaluates to \(0 \times \infty = \text{NaN}\).

## Verified Solution & Action Rules
1. Define a large, finite physical divergence barrier: \(V_{\max} = 10^6\,k_B T\).
2. Set \(V_{\rm ext}(z) = V_{\max}\) inside the sterically forbidden core.
3. Apply explicit steric masking: where \(V_{\rm ext} \ge 10^5\), force \(\rho(z) = 0\) and mask energy integrands to exactly 0 using `.where()` operations:

```python
# Verified Implementation Pattern
STERIC_VMAX = 1.0e6  # k_B T

def build_external_potential(z: Tensor, wall_sigma: float) -> Tensor:
    is_core = z < (0.5 * wall_sigma)
    v_ext = compute_9_3_wall_potential(z)
    return is_core.where(Tensor([STERIC_VMAX]), v_ext)

def apply_steric_mask(rho: Tensor, v_ext: Tensor) -> Tensor:
    # Explicit zeroing of density in excluded core to prevent 0 * inf NaNs
    is_excluded = v_ext >= (0.1 * STERIC_VMAX)
    return is_excluded.where(Tensor([0.0]), rho)
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using artificial coordinate offsets like `z_safe = np.maximum(0.2, z)`.
- ❌ **Anti-Pattern**: Soft potential clamping like `v_ext = np.clip(v_ext, -100, 500)`.
- ❌ **Anti-Pattern**: Allowing \(\rho(z) > 0\) inside steric exclusion zones.
