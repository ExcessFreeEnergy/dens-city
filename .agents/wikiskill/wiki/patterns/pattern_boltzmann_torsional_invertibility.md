# Pattern: Dihedral Angle Torsional Regularization & Invertibility in Normalizing Flows

## Summary
- **Problem**: Invertibility violations during the transformation cycle \(\mathbf{z} \to \mathbf{x} \to \mathbf{z}\) in internal coordinate normalizing flows, or unphysical dihedral angle wrapping artifacts.
- **Root Cause**: Dihedral angles are periodically defined on the circle \(\mathbb{S}^1 \cong [-\pi, \pi]\). When a continuous normalizing flow samples unconstrained real numbers \(\phi \notin [-\pi, \pi]\), periodic wrapping reconstructs the correct 3D Cartesian coordinates but breaks the mathematical bijection with the latent space \(\mathbf{z}\).
- **Actionable Fix**: Incorporate a quadratic torsional penalty loss (\(w_{\rm tor}\)) during training that penalizes angles generated outside \([-\pi, \pi]\), and discard any rare sample that violates numerical bijection.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.bijectors`

## Deep Root Cause Analysis
In statistical mechanics, internal coordinate transformations map 3D atomic positions \(\mathbf{x}\) into bond lengths \(d \in \mathbb{R}^+\), bond angles \(\alpha \in [0, \pi]\), and dihedral angles \(\phi \in [-\pi, \pi]\).
When mapping from latent space \(\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\) back to internal coordinates, the neural network may predict a dihedral value \(\phi = 3.5\,\text{rad} > \pi\).
- In Cartesian placement, \(\phi = 3.5\) wraps to \(\phi - 2\pi = -2.783\,\text{rad}\), producing a valid 3D molecular structure.
- However, when mapping this structure back via \(\mathbf{x} \to \mathbf{z}'\), the inverse coordinate converter extracts \(\phi = -2.783\), leading to \(\mathbf{z}' \neq \mathbf{z}\).
- This breaks the change-of-variables formula:
\[q_X(\mathbf{x}) = \mu_Z(F_{xz}(\mathbf{x})) \cdot |\det J_{xz}|\]
and invalidates statistical reweighting.

## Verified Solution & Action Rules
1. Define a smooth quadratic boundary loss for all dihedral angles:
\[L_{\rm tor}(\boldsymbol{\phi}) = \sum_i \left( \max(0, |\phi_i| - \pi) \right)^2\]
2. Add this loss with weight \(w_{\rm tor} \ge 1.0\) to the overall objective during energy-based training.
3. Validate cycle consistency in unit tests: verify \(\| F_{xz}(F_{zx}(\mathbf{z})) - \mathbf{z} \|_\infty < 10^{-4}\) for all accepted samples.

```python
# Verified Implementation Pattern
def compute_torsional_boundary_loss(dihedrals: Tensor, margin: float = 3.14159265) -> Tensor:
    # Penalize angles outside [-pi, pi]
    excess = (dihedrals.abs() - margin).relu()
    return (excess * excess).sum(axis=-1).mean()
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Applying hard modulo \((\phi \pmod{2\pi})\) inside differentiable computation graphs without boundary loss (causes discontinuous gradient spikes at \(\pm \pi\)).
- ❌ **Anti-Pattern**: Ignoring round-trip invertibility checks \(\mathbf{z} \to \mathbf{x} \to \mathbf{z}\) in test suites.
