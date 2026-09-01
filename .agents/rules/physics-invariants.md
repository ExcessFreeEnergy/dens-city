# Physics Invariants & Anti-Pattern Rules for `dens-city`

When editing any code under `src/dens_city/` or `tests/`:

1. **Zero Hardcoded Parameters**: Derive all physical parameters (\(\sigma_i, \epsilon_i, q_i\)) strictly from input `.mol2` files and force field definitions.
2. **Latent Field Positivity**: Optimize \(\psi(z)\) where \(\rho(z) = \rho_{\rm bulk} \exp(\psi(z))\). Never optimize \(\rho(z)\) directly.
3. **Exact Mechanical Observables**: Evaluate wall contact pressure via \(P_{\rm wall} = -\int_0^{L/2} \rho(z) \nabla V_{\rm ext}(z) dz\). Never use spatial slicing.
4. **Exact Asymptotic Boundaries**: Enforce true physical divergence (\(V_{\max} = 10^6\,k_B T\)) at steric walls and use `.where()` masking to eliminate \(0 \times \infty\) NaNs.
5. **Scale-Invariant Initialization**: All spatial grids and cutoffs must scale with fluid diameter: \(L_z = \max(40.0, 10.0\sigma_{\rm eff})\), \(r_{\rm cut} = \max(15.0, 5.0\sigma_{\rm eff})\).
6. **Consult Wiki First**: Check `.agents/wikiskill/wiki/index.md` and `skill-impact.md` before implementing numerical routines.
