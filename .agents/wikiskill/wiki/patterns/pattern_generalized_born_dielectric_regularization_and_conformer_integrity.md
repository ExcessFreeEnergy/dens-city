# Pattern: Generalized Born Dielectric Regularization & Rigid-Rotor Ensemble Integrity

## Summary
- **The Ideal / Objective Goal**: Zero unphysical dielectric singularities (\Delta G_{\\rm GB} \ll -50\\text{ kcal/mol} on neutral molecules) and zero conformer hijacking in thermal Boltzmann ensembles. Generated conformational ensembles preserve 100% of covalent bond lengths and valence angles to machine precision (|d - d_0| < 10^{-4}\\text{ \\AA}). Solute charges in Generalized Born continuum solvers scale with pure q^2 electrostatics, and effective Born radii reflect true physical volume descreening (\\alpha_i \\ge \\rho_i up to 30\\text{ \\AA}) across arbitrary solvent dielectrics (1 \\le \\epsilon_r \\le 200).
- **Problem**: Extreme solvation energy prediction outliers (e.g. errors of -60 to -150\\text{ kcal/mol} on phosphates, esters, and high-dielectric solvents like formamide) caused by distorted conformers with stretched covalent bonds, artificial centroid contraction, or collapsed Born radii hijacking the Boltzmann ensemble.
- **Root Cause**:
  1. In neutral molecules (\\sum q_i = 0), Generalized Born self-energy \\Delta G_{\\rm self} \\approx -150\\text{ kcal/mol} and pairwise Still screening \\Delta G_{\\rm pair} \\approx +140\\text{ kcal/mol} cancel by >90% on physical ground-state geometries (r \\approx 1.5\\text{ \\AA}). When coordinates have stretched bonds (r > 2.5\\text{ \\AA}) or centroid contraction, pairwise cancellation collapses while self-energy remains large and negative.
  2. Capping intramolecular internal energy penalties at small values (e.g. +50\\text{ kcal/mol}) creates an inverse Boltzmann trap: the artificial Born gain (\\Delta G_{\\rm GB} \\approx -140\\text{ kcal/mol}) overpowers the internal penalty, assigning 100% of the ensemble weight (w_k = 1.0) to the broken conformer.
  3. Artificially clamping volume descreening integrals (\\alpha_i \\le 2.5 \\rho_i) prevents buried atoms in high-dielectric media (\\epsilon_r > 80) from expanding physically.
  4. Squashing solute charges in continuum solvers breaks the physical q^2 scaling required for divalent and trivalent ions.
- **Actionable Fix**:
  1. **Bond Length Invariance via Rigid-Rotor Dihedral Rotations**: Conformer diversity must be generated exclusively by rigid-rotor rotations around rotatable single bonds using Rodrigues rotation formula. Any downstream atom collinear with the bond axis must be bent around an orthogonal axis to preserve tetrahedral geometry. Every covalent bond length must satisfy |d - d_0| = 0 to float precision.
  2. **Severe Intramolecular Strain Gating**: Assign prohibitive penalties (E_{\\rm int} \\ge 10^4\\text{ kcal/mol}) to any conformer with bond stretch >0.35\\text{ \\AA} or clash <0.85\\text{ \\AA}. Reject invalid conformers (E_{\\rm int} \\ge 500\\text{ kcal/mol}) with w_k = 0.0 in the Boltzmann weighting, and clamp solution-phase Born modulation to \\Delta(\\Delta G_{\\rm GB}) \\in [-15, +15]\\text{ kcal/mol}.
  3. **Uncapped Hawkins/Grycuk Descreening**: Allow effective radii to descreen physically up to 30.0\\text{ \\AA} with \\alpha_i \\ge \\rho_i.
  4. **Unconstrained Physical Charge Scaling**: Preserve pure q^2 electrostatics in continuum solvers.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.cdft.generalized_born`, `dens_city.boltzmann.egnn`, `dens_city.utils.materials`

---

## The Ideal Architectural Standard

In a rigorous statistical mechanics and molecular ML engine, a thermal conformational ensemble must never corrupt the covalent topology of the molecule. The ideal pipeline satisfies:

$$\\max_{k, (i, j) \\in \\text{bonds}} |d_k(i, j) - d_0(i, j)| < 10^{-4}\\text{ \\AA}$$
$$w_k(E_{\\rm int} \\ge 500\\text{ kcal/mol}) = 0.0$$

```
+-----------------------------------------------------------------------------------+
|                     IDEAL CONFORMATIONAL ENSEMBLE PIPELINE                        |
+-----------------------------------------------------------------------------------+
|  1. Input Geometry: Ground-state Cartesian coordinates x_0 and chemical bonds     |
|  2. Rigid-Rotor Rotamer Generation:                                               |
|     - Identify rotatable single bonds not in rings                                |
|     - Pivot subtree exclusively around atom a2: x_rot = R(u, theta) (x - x_a2)    |
|     - If collinear (u x r = 0), bend around orthogonal axis to tetrahedral angle  |
|     => Bond Length Invariance: 100% of covalent bonds preserved exactly           |
|  3. Physical Conformer Validation:                                                |
|     - Reject raw flow samples with stretched bonds (|d - d0| > 0.35 A)            |
|     - Compute harmonic bond strain and non-bonded steric clash penalties          |
|     - Assign E_int = 10000 kcal/mol to broken/clashing conformers                 |
|  4. Clamped Boltzmann Reweighting:                                                |
|     - Mask invalid conformers: valid_mask = (E_int < 500 kcal/mol)                |
|     - Bound Born modulation: delta_GB = clip(GB_k - GB_0, -15.0, +15.0)           |
|     - E_eff = E_int + delta_GB; fallback to ground state if w_sum == 0            |
|  5. Continuum Solvation Physics:                                                  |
|     - Grycuk volume descreening: alpha_i = rho_i * (1 + sum_j descreen_j)         |
|     - Unconstrained q^2 scaling for physical monoatomic and polyatomic ions       |
+-----------------------------------------------------------------------------------+
```

### Rigid-Rotor Dihedral Rotation Invariant
For any bond $(a_1, a_2)$ with downstream subtree $S(a_2)$:
$$\\mathbf{x}_i^{\\prime} = \\mathbf{x}_{a_2} + \\mathbf{R}(\\hat{\\mathbf{u}}, \\theta) (\\mathbf{x}_i - \\mathbf{x}_{a_2}) \\quad \\forall i \\in S(a_2)$$
Because $\\mathbf{R}$ is an orthogonal matrix and $\\mathbf{x}_{a_2}$ is the stationary pivot, the distance between $a_1$ and $a_2$ is identical, and all internal distances within $S(a_2)$ are invariant:
$$\\|\\mathbf{x}_i^{\\prime} - \\mathbf{x}_j^{\\prime}\\| = \\|\\mathbf{R} (\\mathbf{x}_i - \\mathbf{x}_j)\\| = \\|\\mathbf{x}_i - \\mathbf{x}_j\\| \\quad \\forall i, j \\in S(a_2)$$
$$\\|\\mathbf{x}_{a_2}^{\\prime} - \\mathbf{x}_{a_1}^{\\prime}\\| = \\|\\mathbf{x}_{a_2} - \\mathbf{x}_{a_1}\\|$$

### Multipole Cancellation Invariant in Generalized Born
For a neutral molecule ($\\sum_i q_i = 0$), the total Born free energy is:
$$\\Delta G_{\\rm GB} = -\\frac{1}{2}\\left(1 - \\frac{1}{\\epsilon_r}\\right) \\left[ \\sum_i \\frac{q_i^2}{\\alpha_i} + \\sum_{i \\ne j} \\frac{q_i q_j}{f_{\\rm GB}(r_{ij})} \\right]$$
At bonded distances ($r_{ij} \\approx \\alpha_i \\approx \\alpha_j$), $f_{\\rm GB}(r_{ij}) \\approx \\alpha_i$, so:
$$\\sum_i \\frac{q_i^2}{\\alpha_i} + \\sum_{i \\ne j} \\frac{q_i q_j}{f_{\\rm GB}(r_{ij})} \\approx \\frac{1}{\\bar{\\alpha}} \\left(\\sum_i q_i\\right)^2 = 0$$
Preserving physical bond lengths is the fundamental prerequisite that guarantees multipole cancellation and prevents artificial Born dielectric explosions.
