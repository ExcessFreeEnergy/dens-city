# Pattern: Percus-Yevick Equation of State Consistency with Rosenfeld FMT

## Summary
- **Problem**: In Classical Density Functional Theory (cDFT), the spatial fluid density profile drifts away from true physical equilibrium, or bulk pressure does not match reservoir conditions during variational free energy functional minimization.
- **Root Cause**: Inconsistent thermodynamic routes. The Rosenfeld Fundamental Measure Theory (FMT) excess hard-sphere functional is mathematically consistent with the **Percus-Yevick (PY) compressibility Equation of State (EOS)**. Using empirical cubic equations (e.g. Peng-Robinson) or Carnahan-Starling to solve for the bulk reservoir state ($\rho_{\rm bulk}, \mu$) creates a thermodynamic mismatch against the FMT functional, causing the variational optimizer to alter the asymptotic density far from walls.
- **Actionable Fix**: Derive bulk reservoir density $\rho_{\rm bulk}$ and chemical potential $\mu$ strictly using the Percus-Yevick compressibility route:
\[Z_{\rm PY}(\eta) = \frac{P_{\rm bulk}}{\rho_{\rm bulk} k_B T} = \frac{1 + \eta + \eta^2}{(1 - \eta)^3}\]
where $\eta = \frac{\pi}{6} \rho_{\rm bulk} \sigma_{\rm eff}^3$ is the hard-sphere packing fraction.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.utils.materials`, `dens_city.cdft.tiny_cdft`

## Deep Theoretical Formulation
The grand potential functional minimized in `dens-city` is:
\[\Omega[\psi] = \mathcal{F}_{\rm ideal}[\psi] + \mathcal{F}_{\rm FMT}^{\rm ex}[\rho] + \mathcal{F}_{\rm att}^{\rm ex}[\rho] + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]\]
In the bulk limit far from walls ($V_{\rm ext} \to 0, \rho(z) \to \rho_{\rm bulk}, \psi(z) \to 0$):
1. The bulk excess free energy density from Rosenfeld FMT corresponds to:
\[\Phi_{\rm FMT}(\rho_{\rm bulk}) = \frac{k_B T}{\pi \sigma^3} \left[ -\frac{6\eta}{\sigma} \ln(1 - \eta) + \frac{18\eta^2}{(1 - \eta) \sigma} + \frac{18\eta^3}{(1 - \eta)^2 \sigma} \right]\]
2. The pressure derived via the thermodynamic relation $P = -\left(\frac{\partial F}{\partial V}\right)_{T, N} = \rho \mu - f$ recovers the exact Percus-Yevick compressibility pressure:
\[P_{\rm bulk}^{\rm PY} = \rho_{\rm bulk} k_B T \frac{1 + \eta + \eta^2}{(1 - \eta)^3}\]
3. If $\mu$ or $\rho_{\rm bulk}$ is supplied from a mismatched EOS (such as Carnahan-Starling $Z_{\rm CS} = \frac{1+\eta+\eta^2-\eta^3}{(1-\eta)^3}$ or Peng-Robinson), $\left.\frac{\delta \Omega}{\delta \rho}\right|_{\rho_{\rm bulk}} \ne 0$. The variational solver is forced to continuously adjust the asymptotic bulk density to satisfy its internal FMT minimum, introducing unphysical density gradients.

## Verified Implementation Pattern
```python
def solve_eos_bulk_density_py(temp_k: float, pressure_bar: float, sigma_eff: float) -> float:
    """Solves Percus-Yevick compressibility EOS for exact thermodynamic consistency with FMT."""
    kb = 1.380649e-23
    p_pa = pressure_bar * 1e5
    kt = kb * temp_k
    sigma_m = sigma_eff * 1e-10

    # Dimensionless pressure P* = P * sigma^3 / (k_B * T)
    p_star = p_pa * (sigma_m**3) / kt

    # Solve Z_PY(eta) * eta * (6/pi) = P*
    # (1 + eta + eta^2) / (1 - eta)^3 * eta * (6 / pi) = P*
    def objective(eta):
        z_py = (1.0 + eta + eta**2) / ((1.0 - eta) ** 3)
        return (6.0 / np.pi) * eta * z_py - p_star

    # Bisection on physical packing fraction range [0.001, 0.49]
    eta_sol = scipy.optimize.brentq(objective, 1e-4, 0.49)
    # rho = 6 * eta / (pi * sigma^3) in particles/Angstrom^3
    rho_bulk_a3 = (6.0 * eta_sol) / (np.pi * (sigma_eff**3))
    return float(rho_bulk_a3)
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using empirical cubic EOS (Peng-Robinson, Redlich-Kwong) to set boundary chemical potentials for Rosenfeld FMT solvers.
- ❌ **Anti-Pattern**: Setting chemical potential $\mu$ as a hardcoded constant across temperatures instead of computing $\mu_{\rm bulk} = k_B T \ln(\rho_{\rm bulk} \Lambda^3) + \mu_{\rm ex}^{\rm PY}$.
