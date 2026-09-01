# Pattern: Anti-Aliased Cell-Integrated Rosenfeld FMT Planar Convolution Kernels

## Summary
- **Problem**: Oscillations, energy conservation errors, or grid aliasing in Rosenfeld Fundamental Measure Theory (FMT) weighted densities.
- **Root Cause**: Evaluating hard-sphere weight functions \(w_\alpha(z)\) as discrete point samples on a coarse 1D grid introduces step discontinuities and aliasing artifacts near sphere boundaries.
- **Actionable Fix**: Precompute convolution kernels via exact analytical cell-integration over each spatial grid bin \([z - dz/2, z + dz/2]\).
- **Related Skills / Modules**: `cdft-physics`, `dens_city.cdft.kernels`

## Deep Root Cause Analysis
Rosenfeld FMT requires 6 weighted densities \(n_\alpha(z) = \int dz' \rho(z') w_\alpha(z - z')\):
- Scalar kernels: \(w_3(r), w_2(r), w_1(r), w_0(r)\)
- Vector kernels: \(\mathbf{w}_{v2}(r), \mathbf{w}_{v1}(r)\)

For planar geometry perpendicular to the \(z\)-axis, the 3D hard-sphere delta and theta functions integrate to 1D planar weight kernels with compact support \(|z| \le R\):
- \(w_3(z) = \pi (R^2 - z^2)\)
- \(w_2(z) = 2\pi R\)
- \(\mathbf{w}_{v2}(z) = 2\pi z \, \hat{\mathbf{z}}\)

Evaluating these piecewise functions directly at discrete points \(z_k = k \cdot dz\) causes severe discretization error when \(R\) is not an exact integer multiple of \(dz\). The boundary point \(|z| \approx R\) is improperly weighted, breaking the dimensional consistency relation \(\frac{\partial w_3}{\partial R} = w_2\).

## Verified Solution & Action Rules
1. Integrate each weight function analytically across the bin \([z - dz/2, z + dz/2]\):
\[\bar{w}_\alpha(z_k) = \frac{1}{dz} \int_{z_k - dz/2}^{z_k + dz/2} w_\alpha(z') \, dz'\]
2. Ensure the discrete sum of the volume kernel satisfies \(\sum_k \bar{w}_3(z_k) dz = \frac{4}{3}\pi R^3\) to machine precision.
3. Use FFT or symmetric 1D direct convolution with anti-aliased kernels.

```python
# Verified Implementation Pattern
def build_anti_aliased_fmt_kernels(R: float, dz: float) -> Dict[str, np.ndarray]:
    K = int(np.ceil(R / dz))
    z = np.arange(-K, K + 1) * dz
    z_low = np.maximum(-R, z - 0.5 * dz)
    z_high = np.minimum(R, z + 0.5 * dz)
    valid = z_high > z_low

    # w3 cell integral: \int pi (R^2 - z^2) dz = pi [R^2 z - z^3/3]
    def w3_primitive(z_val):
        return np.pi * (R**2 * z_val - (z_val**3) / 3.0)

    w3_cell = np.zeros_like(z)
    w3_cell[valid] = (w3_primitive(z_high[valid]) - w3_primitive(z_low[valid])) / dz
    return {"w3": w3_cell, "R": R, "dz": dz}
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Evaluating `w3 = np.pi * (R**2 - z**2) * (np.abs(z) <= R)` pointwise on grid.
- ❌ **Anti-Pattern**: Omitting vector kernels \(\mathbf{n}_{v2}\) in inhomogeneous planar geometries.
