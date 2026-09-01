# Pattern: Energy Log-Regularization & Steric Clash Clipping in Flow Training

## Summary
- **Problem**: Gradient explosions (`inf` or `NaN` gradients) during the initial stages of energy-based (\(J_{KL}\)) training in Boltzmann Generators.
- **Root Cause**: Early untrained normalizing flows generate random 3D atomic configurations where non-bonded atom pairs overlap closely (\(r_{ij} < 0.5\,\text{Å}\)). The Lennard-Jones repulsive term \((\sigma/r_{ij})^{12}\) evaluates to \(10^{10} \dots 10^{20}\), producing astronomical loss values and blowing up optimizer momentum buffers.
- **Actionable Fix**: Apply a smooth, monotonic log-regularization threshold \(E_{\rm high}\) to the microscopic energy during backpropagation, and gradually relax \(E_{\rm high}\) downward as the generator learns valid physical geometries.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.energy`

## Deep Root Cause Analysis
During training by energy, the objective evaluates the potential energy \(U(F_{zx}(\mathbf{z}))\).
In condensed matter or multi-particle systems, placing \(N\) particles simultaneously from an un-converged flow results in at least one near-zero interatomic distance \(r_{ij} \to 0\).
- If \(U\) is passed directly to the backward pass, gradients \(\nabla_{\mathbf{x}} U \sim -12 \frac{\epsilon}{\sigma} (\frac{\sigma}{r_{ij}})^{13}\) exceed floating point range and produce `NaN` weights.
- Simply clipping gradients (\(\text{clip\_grad\_norm}\)) prevents numerical crash but produces random direction updates that fail to resolve the specific overlapping atom pair.
- The solution introduced in Noé et al. is smooth energy log-regularization:
\[E_{\rm reg}(E) = \begin{cases} E & E < E_{\rm high} \\ E_{\rm high} + \log(E - E_{\rm high} + 1) & E_{\rm high} \le E < E_{\max} \\ E_{\rm high} + \log(E_{\max} - E_{\rm high} + 1) & E \ge E_{\max} \end{cases}\]
This maintains the exact gradient direction \(\nabla E\) while scaling down the gradient magnitude by \(\frac{1}{E - E_{\rm high} + 1}\).

## Verified Solution & Action Rules
1. Implement the three-tier smooth logarithmic energy regularization.
2. Initialize \(E_{\rm high} = 10^{10}\,k_BT\) during initial flow stages and anneal it down to \(10^4 \dots 10^3\,k_BT\) as the fraction of low-energy samples approaches \(>95\%\).

```python
# Verified Implementation Pattern
def regularize_energy(energy: Tensor, e_high: float = 1e4, e_max: float = 1e20) -> Tensor:
    # 1. Linear regime: E < E_high
    # 2. Log regime: E_high <= E < E_max
    # 3. Plateau: E >= E_max
    diff = (energy - e_high).relu()
    log_reg = e_high + (diff + 1.0).log()
    return (energy < e_high).where(energy, log_reg)
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using raw un-regularized Lennard-Jones \((\sigma/r)^{12}\) potentials in generative flow loss functions.
- ❌ **Anti-Pattern**: Using discontinuous hard clipping like `min(E, 1000)` which zeroes out gradients precisely when atom pairs need repulsive separation forces.
