# Pattern: Scale-Invariant Simulation Geometry & Interaction Cutoffs

## Summary
- **Problem**: Simulation box truncation errors, artificial confinement effects, or unphysical finite-size finite-cutoff errors when switching between small molecules (e.g. Argon \(\sigma \approx 3.4\,\text{Å}\)) and large mesogens (e.g. 5CB \(\sigma \approx 18.0\,\text{Å}\)).
- **Root Cause**: Hardcoding fixed grid lengths in Angstroms (e.g. \(L_z = 30.0\,\text{Å}\)) or fixed cutoffs (\(r_{\rm cut} = 12.0\,\text{Å}\)) breaks when applied to fluids of varying molecular size.
- **Actionable Fix**: Dynamically scale all spatial domain dimensions, grid points, and interaction cutoffs from the fluid's intrinsic effective diameter \(\sigma_{\rm eff}\).
- **Related Skills / Modules**: `cdft-physics`, `dens_city.cdft`, `dens_city.utils.materials`

## Deep Root Cause Analysis
For a small atom like Argon (\(\sigma \approx 3.405\,\text{Å}\)), a box length of \(40\,\text{Å}\) spans \(\approx 11.7\,\sigma\), allowing density oscillations to decay cleanly to bulk \(\rho_{\rm bulk}\) at the midplane.
However, for large conjugated liquid crystals like 5CB (\(\sigma_{\rm eff} \approx 16\,\text{Å}\)), a fixed \(40\,\text{Å}\) box spans only \(2.5\,\sigma\). The wall perturbation from the left wall overlaps with the perturbation from the right wall, destroying bulk boundary conditions and corrupting the Irving-Kirkwood contact pressure.

## Verified Solution & Action Rules
1. Compute the fluid's effective hard-sphere diameter \(\sigma_{\rm eff}\) dynamically from the input `.mol2` geometry or Barker-Henderson temperature-dependent diameter.
2. Set the 1D spatial domain length:
\[L_z = \max\left(40.0\,\text{Å}, \, 10.0 \cdot \sigma_{\rm eff}\right)\]
3. Set the attractive dispersion interaction cutoff:
\[r_{\rm cut} = \max\left(15.0\,\text{Å}, \, 5.0 \cdot \sigma_{\rm eff}\right)\]
4. Set the spatial grid step \(dz \le \min\left(0.05\,\text{Å}, \, 0.01 \cdot \sigma_{\rm eff}\right)\).

```python
# Verified Implementation Pattern
def compute_scale_invariant_domain(sigma_eff: float, dz_target: float = 0.05) -> Tuple[float, int, float]:
    L_z = max(40.0, 10.0 * sigma_eff)
    r_cut = max(15.0, 5.0 * sigma_eff)
    num_grid = int(np.ceil(L_z / dz_target))
    # Ensure even number of grid points for symmetric midpoint
    if num_grid % 2 != 0:
        num_grid += 1
    actual_dz = L_z / num_grid
    return L_z, num_grid, actual_dz
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Hardcoding `L_z = 30.0` or `L_z = 50.0` as constants.
- ❌ **Anti-Pattern**: Using a fixed `r_cut = 10.0` regardless of molecular dimensions.
- ❌ **Anti-Pattern**: Assuming the box midpoint is bulk without verifying \(L_z \ge 10\sigma_{\rm eff}\).
