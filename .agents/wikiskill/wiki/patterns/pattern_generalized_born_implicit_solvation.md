# Pattern: Tensor-Native Generalized Born Implicit Solvation Engine

## Summary
- **Problem**: Evaluating polar hydration free energies in water for high-throughput molecular ensembles ($B=512$) is computationally intractable with explicit solvent boxes ($10^5$ atoms) and suffers from host-device synchronization stalls when using CPU-based Poisson-Boltzmann grid solvers.
- **Root Cause**: Naive Generalized Born (GB) implementations rely on CPU-side atom radius lookups and numerical integration of atomic volumes with $1/r^4$ descreening divergences when atoms overlap closely ($r_{ij} \to 0$).
- **Actionable Fix**: Implement an end-to-end differentiable GPU-resident Generalized Born solver:
  1. $O(1)$ GPU Bondi radii gather from a static device-resident radius tensor ($Z \in [0, 118]$).
  2. Hawkins/Grycuk smooth volume descreening with bounded saturation:
     \[\alpha_i(\mathbf{x}) = \rho_i \left( 1.0 + \min\left(1.5, \, \sum_{j \ne i} \frac{0.12 \sigma_j^3}{r_{ij}^3 + \rho_i^3} \right) \right)\]
     guaranteeing effective Born radii satisfy $\alpha_i \ge \rho_i$ without negative descreening or division-by-zero singularities.
  3. Still pairwise analytical solvation free energy:
     \[\Delta G_{\rm GB} = -\frac{1}{2}\left( \frac{1}{\varepsilon_{\rm in}} - \frac{1}{\varepsilon_{\rm out}} \right) \cdot \frac{e^2}{4\pi\varepsilon_0} \sum_{i, j} \frac{q_i q_j}{\sqrt{r_{ij}^2 + \alpha_i \alpha_j \exp\left(-\frac{r_{ij}^2}{4\alpha_i\alpha_j}\right)}}\]
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.cdft.generalized_born`

## Deep Theoretical Formulation
In aqueous solution, the electrostatic contribution to solvation free energy decomposes into self-polarization and pairwise cross-polarization screening:
1. **Bondi Radius Initialization**: Intrinsic van der Waals radii $\rho_i$ are assigned via atomic numbers $Z_i$ ($H=1.20\text{ \AA}, C=1.70\text{ \AA}, N=1.55\text{ \AA}, O=1.52\text{ \AA}$).
2. **Hawkins-Cramer-Truhlar / Grycuk Volume Descreening**:
   - The effective Born radius $\alpha_i$ represents the effective distance of atom $i$ from the dielectric continuum boundary:
   - When atom $j$ displaces high-dielectric water around atom $i$, the effective self-energy drops.
   - Using $r_{ij}^3 + \rho_i^3$ in the denominator smoothly floors the interaction at zero separation, eliminating infinite forces.
   - Bounding the sum with $\min(1.5, \dots)$ prevents numerical runaway on high-coordination cluster cores.
3. **Still Polarization Integral**:
   - The Still function $f_{\rm GB}(r_{ij}, \alpha_i, \alpha_j) = \sqrt{r_{ij}^2 + \alpha_i \alpha_j \exp\left(-\frac{r_{ij}^2}{4\alpha_i\alpha_j}\right)}$ smoothly interpolates between:
     - Short distance ($r_{ij} \to 0$): $f_{\rm GB} \to \sqrt{\alpha_i \alpha_j}$ (Born ion self-energy).
     - Long distance ($r_{ij} \gg \alpha$): $f_{\rm GB} \to r_{ij}$ (standard Coulomb screening).

## Verified Implementation Pattern
```python
class GeneralizedBornSolvation:

    def __init__(
        self,
        dielectric_out: float = 78.4,
        dielectric_in: float = 1.0,
        radius_offset: float = 0.09,
    ):
        self.factor = -0.5 * (1.0 / dielectric_in - 1.0 / dielectric_out) * 332.0637  # kcal*A/(mol*e^2)
        self.offset = radius_offset

    def compute_born_radii(self, coords: Tensor, radii: Tensor, mask: Tensor) -> Tensor:
        # coords: (B, N, 3), radii: (B, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        r2 = (diff * diff).sum(axis=-1)
        r = (r2 + 1e-8).sqrt()

        rho = radii.unsqueeze(-1)
        sigma_j = radii.unsqueeze(1)

        # Hawkins descreening term: 0.12 * sigma_j^3 / (r_ij^3 + rho_i^3)
        denom = (r * r2) + (rho * rho * rho)
        term = 0.12 * (sigma_j**3) / denom

        # Mask diagonal and dummy atoms
        term = term * mask.unsqueeze(1) * mask.unsqueeze(2)
        term = term * (1.0 - Tensor.eye(coords.shape[1])).reshape(1, coords.shape[1], coords.shape[1])

        # Effective Born radii
        descreen = term.sum(axis=-1).clip(0.0, 1.5)
        alpha = radii * (1.0 + descreen)
        return alpha * mask

    def compute_energy(self, coords: Tensor, charges: Tensor, alpha: Tensor, mask: Tensor) -> Tensor:
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        r2 = (diff * diff).sum(axis=-1)

        alpha_i = alpha.unsqueeze(2)
        alpha_j = alpha.unsqueeze(1)
        alpha_prod = alpha_i * alpha_j

        # Still smoothing function
        f_gb = (r2 + alpha_prod * (-r2 / (4.0 * alpha_prod + 1e-8)).exp()).sqrt()

        # Pairwise screened interaction
        q_prod = charges.unsqueeze(2) * charges.unsqueeze(1)
        energy_matrix = (q_prod / (f_gb + 1e-8)) * mask.unsqueeze(1) * mask.unsqueeze(2)
        return self.factor * energy_matrix.sum(axis=(-1, -2))
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Performing CPU-side host-device synchronization to lookup per-atom Bondi radii during high-throughput flow training.
- ❌ **Anti-Pattern**: Using raw $1/r_{ij}^4$ descreening formulas without denominator flooring ($r^3 + \rho^3$), leading to `NaN` gradients upon steric overlaps.
