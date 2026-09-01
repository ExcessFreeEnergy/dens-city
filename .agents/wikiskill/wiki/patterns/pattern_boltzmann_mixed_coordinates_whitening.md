# Pattern: Mixed Coordinate Representation & PCA Whitening for Macromolecules

## Summary
- **Problem**: Training Boltzmann Generators directly in 3D Cartesian coordinates fails on macromolecules and polymers, generating unphysical structures with broken covalent bonds and distorted bond angles.
- **Root Cause**: Cartesian coordinates couple stiff high-frequency vibrational degrees of freedom (bond lengths \(\sim 1.5\,\text{Å}\) with force constants \(k \sim 1000\,\text{kcal/mol}\cdot\text{Å}^2\)) with soft low-frequency conformational changes (torsions, rotations), creating ill-conditioned loss landscapes.
- **Actionable Fix**: Split coordinates into Cartesian backbone atoms and Internal Coordinate (IC) side-chain/branch atoms: whiten Cartesian coordinates via PCA (discarding 6 global translation/rotation modes) and normalize internal coordinates (bonds, angles, dihedrals) to zero mean and unit variance.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.bijectors`

## Deep Root Cause Analysis
For a molecule with \(N\) atoms in 3D space:
1. Pure Cartesian generators require the neural network to rediscover the precise rigid geometry of chemical bonds and angles across \(3N\) correlated dimensions. Even a \(0.05\,\text{Å}\) bond stretch can incur an energy penalty of hundreds of \(k_B T\).
2. The **Mixed Coordinate Transformation (\(M\)-layer)** solves this by separating degrees of freedom:
   - **Cartesian Set \(\mathbf{x}^C\)**: Heavy backbone atoms (e.g. C, N, S in protein or conjugated core). Whiten via PCA:
     \[\mathbf{z}^C = \mathbf{\Lambda}^{-1/2} \mathbf{R}^\top \mathbf{x}^C\]
     where \(\mathbf{R}\) are eigenvectors and \(\mathbf{\Lambda}\) are variances of the covariance matrix \(\mathbf{X}^\top \mathbf{X}\). The 6 zero eigenvalues corresponding to global translation and rotation are removed.
   - **Internal Coordinate Set \(\mathbf{x}^I\)**: Branch/side-chain atoms parameterized relative to 3 parent atoms by distance \(d_{ij}\), valence angle \(\alpha_{ijk}\), and dihedral angle \(\phi_{ijkl}\). Each IC is normalized:
     \[\bar{q} = \frac{q - \mu_q}{\sigma_q}\]
3. After this whitening and normalization transformation, the input data distribution is already nearly standard Gaussian, dramatically accelerating flow convergence and preserving valid chemistry.

## Verified Solution & Action Rules
1. Never train flow bijectors directly on raw unwhitened 3D Cartesian coordinates for flexible molecules.
2. Build an invertible coordinate transformation pipeline:
   - Superimpose configurations to reference structure to remove translation and rotation.
   - Whiten Cartesian coordinates with PCA.
   - Transform branching atoms to internal coordinates and normalize by empirical means and standard deviations.

```python
# Verified Implementation Pattern
class PCAWhiteningLayer:
    def __init__(self, ref_coords: np.ndarray, threshold: float = 1e-6):
        cov = np.cov(ref_coords.T)
        evals, evecs = np.linalg.eigh(cov)
        # Keep non-zero degrees of freedom (drop 6 translation/rotation modes)
        valid = evals > threshold
        self.R = evecs[:, valid]
        self.inv_sqrt_lambda = 1.0 / np.sqrt(evals[valid])
        self.sqrt_lambda = np.sqrt(evals[valid])
        self.log_det_xz = -0.5 * np.sum(np.log(evals[valid]))
        self.log_det_zx = 0.5 * np.sum(np.log(evals[valid]))

    def forward_xz(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.R) * self.inv_sqrt_lambda

    def forward_zx(self, z: np.ndarray) -> np.ndarray:
        return (z * self.sqrt_lambda) @ self.R.T
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Passing raw unnormalized Cartesian coordinate matrices directly to RealNVP / normalizing flows.
- ❌ **Anti-Pattern**: Leaving global rigid body translation and rotation degrees of freedom unconstrained.
- ❌ **Anti-Pattern**: Re-estimating PCA whitening components dynamically per batch rather than fixing them on reference training datasets.
