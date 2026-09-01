# Pattern: Adaptive Latent-Space MCMC Exploration Across Barrier Heights

## Summary
- **Problem**: Molecular Dynamics (MD) or standard physical-space Monte Carlo simulations get trapped in local metastable energy wells for millions of steps, failing to sample rare conformational transitions (e.g. open/closed dimer, protein folding).
- **Root Cause**: High physical energy barriers (\(> 10\dots 20\,k_B T\)) between metastable states require exponentially long simulation times (\(\tau \sim e^{\Delta F / k_BT}\)) in Cartesian space.
- **Actionable Fix**: Perform Metropolis Monte Carlo in the latent space of the Boltzmann Generator, proposing large non-local structural rearrangements in a single step with exact acceptance probability:
\[p_{\rm acc} = \min\left(1, \, \exp(-\Delta E)\right), \quad \Delta E = u(F_{zx}(\mathbf{z}')) - u(\mathbf{x}) - \log R_{zx}(\mathbf{z}') + \log R_{xz}(\mathbf{x})\]
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.generator`

## Deep Root Cause Analysis
In physical space \(\mathbf{x}\), moving between two conformational states (e.g., open and closed dimer) requires coordinated reorganization of all surrounding solvent particles. Small incremental MD steps cannot traverse the intermediate high-energy state.
In the latent space \(\mathbf{z}\) of a trained Boltzmann Generator, the complex energy landscape is repacked such that distinct metastable states are mapped into neighboring regions of a standard Gaussian \(\mathcal{N}(\mathbf{0}, \mathbf{I})\).
Proposing a step \(\mathbf{z}' = \mathbf{z} + s \boldsymbol{\eta}\) (with step size \(s \sim 0.1 \dots 1.0\)) translates in physical space to a global concerted multi-atom rearrangement that completely bypasses the physical pathway barrier.

## Verified Solution & Action Rules
1. Propose latent step: \(\mathbf{z}' = F_{xz}(\mathbf{x}) + s \mathcal{N}(\mathbf{0}, \mathbf{I})\).
2. Compute new configuration: \(\mathbf{x}' = F_{zx}(\mathbf{z}')\).
3. Compute the generalized Metropolis-Hastings energy difference including Jacobian volume corrections:
\[\Delta E = u(\mathbf{x}') - u(\mathbf{x}) - \log R_{zx}(\mathbf{z}') + \log R_{xz}(\mathbf{x})\]
4. Accept with probability \(\alpha = \min(1, e^{-\Delta E})\). If accepted, replace \(\mathbf{x} \gets \mathbf{x}'\).
5. Retain past accepted samples in a replay buffer to continuously retrain and refine the flow.

```python
# Verified Implementation Pattern
def latent_metropolis_step(flow, x_current: Tensor, energy_fn, step_size: float = 0.2) -> Tuple[Tensor, bool]:
    z_cur, log_det_xz = flow.forward_xz(x_current)
    z_prop = z_cur + step_size * Tensor.randn(*z_cur.shape)
    x_prop, log_det_zx = flow.forward_zx(z_prop)

    u_cur = energy_fn(x_current)
    u_prop = energy_fn(x_prop)

    delta_E = u_prop - u_cur - log_det_zx + (-log_det_xz)
    p_acc = (-delta_E).exp().clip(0.0, 1.0)
    accepted = float(Tensor.rand(1).item()) < float(p_acc.item())
    return (x_prop if accepted else x_current), accepted
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Omitting the Jacobian determinant ratio \(\frac{R_{xz}(\mathbf{x})}{R_{zx}(\mathbf{z}')}\) in the Metropolis acceptance criterion (destroys detailed balance and leads to biased equilibrium distributions).
- ❌ **Anti-Pattern**: Using rigid step sizes \(s\) without monitoring the acceptance rate (optimal acceptance \(\approx 20\% - 50\%\)).
