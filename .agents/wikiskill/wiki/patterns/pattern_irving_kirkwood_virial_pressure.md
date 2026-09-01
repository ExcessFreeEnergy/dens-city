# Pattern: Exact Irving-Kirkwood Virial Contact Pressure Integral

## Summary
- **Problem**: Wall contact pressure diverges from exact mechanical equilibrium or depends erratically on grid resolution.
- **Root Cause**: Using arbitrary spatial index slices (e.g., `rho[0:5]` or `rho[mid]`) to evaluate wall contact pressure breaks momentum conservation under continuous external potentials.
- **Actionable Fix**: Compute wall contact pressure using the exact statistical mechanical Irving-Kirkwood momentum balance integral over the external potential gradient.
- **Related Skills / Modules**: `cdft-physics`, `dens_city.cdft`

## Deep Root Cause Analysis
For a hard planar wall at \(z = 0\), the contact theorem states \(P_{\rm wall} = k_B T \rho(0^+)\). However, for realistic, continuous external wall potentials \(V_{\rm ext}(z)\) (such as 9-3 or 10-4-3 Lennard-Jones walls), there is no single "contact plane". The fluid density is smooth and vanishes as \(z \to 0\).
Estimating contact pressure by indexing `rho[0]` or taking an ad-hoc average `rho[:10].mean()` yields values that fluctuate wildly with grid resolution \(dz\) and cutoff positions.

The exact mechanical balance equation (derived from the divergence of the Irving-Kirkwood pressure tensor \(\nabla \cdot \mathbf{P} + \rho \nabla V_{\rm ext} = 0\)) integrates across the half-box:
\[P_{\rm wall} = -\int_0^{L_z/2} \rho(z) \frac{d V_{\rm ext}(z)}{dz} \, dz\]
At mechanical equilibrium, this integral matches the bulk thermodynamic pressure \(P_{\rm bulk}\) regardless of grid discretization.

## Verified Solution & Action Rules
1. Never evaluate wall pressure via spatial slice indexing.
2. Differentiate \(V_{\rm ext}(z)\) analytically or via conservative central differences.
3. Integrate the momentum balance force across the wall interface:

```python
# Verified Implementation Pattern
def compute_wall_contact_pressure(rho: Tensor, dV_ext_dz: Tensor, dz: float, mid_idx: int) -> float:
    # Half-box integral from wall (z=0) to midplane (z=L_z/2)
    force_integrand = rho[:mid_idx] * dV_ext_dz[:mid_idx]
    P_wall = -(force_integrand.sum() * dz).item()
    return float(P_wall)
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using `rho[0] * k_B * T` or `rho[1] * k_B * T` for continuous walls.
- ❌ **Anti-Pattern**: Using hardcoded spatial slices like `rho[0:15].mean()`.
- ❌ **Anti-Pattern**: Calculating contact pressure at an arbitrary spatial cutoff index.
