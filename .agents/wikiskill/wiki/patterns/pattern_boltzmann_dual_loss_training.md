# Pattern: Dual Loss Training & Mode Collapse Prevention in Boltzmann Generators

## Summary
- **Problem**: Invertible normalizing flows trained purely on energy collapse to a single delta-peak energy minimum (mode collapse), or when trained purely on maximum likelihood fail to sample new low-energy states outside the training set.
- **Root Cause**: Training by energy alone (\(J_{KL}\)) focuses on the global minimum if entropy is insufficiently weighted; training by example alone (\(J_{ML}\)) is bounded by existing data and cannot discover unseen metastable states.
- **Actionable Fix**: Train Boltzmann Generators using the combined dual loss function \(J = w_{ML} J_{ML} + w_{KL} J_{KL} + w_{RC} J_{RC}\), where \(J_{KL} = U - H_X + H_Z\) maximizes configurational entropy \(H_X\) via the log-Jacobian determinant while minimizing potential energy.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.generator`

## Deep Root Cause Analysis
In generative molecular modeling, the target equilibrium distribution is the Boltzmann distribution:
\[\mu_X(\mathbf{x}) = Z_X^{-1} e^{-u(\mathbf{x})}, \quad u(\mathbf{x}) = \frac{U(\mathbf{x})}{k_B T}\]
When training an invertible neural network \(F_{zx}: \mathbf{z} \to \mathbf{x}\) from a Gaussian prior \(\mu_Z(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})\):
1. **Latent Reverse KL Loss (Training by Energy)**:
\[J_{KL} = \mathbb{E}_{\mathbf{z} \sim \mu_Z} \left[ u(F_{zx}(\mathbf{z})) - \log R_{zx}(\mathbf{z}) \right] = U - H_X + H_Z\]
   - The first term \(\mathbb{E}[u]\) pulls samples toward low-energy states.
   - The second term \(-\mathbb{E}[\log R_{zx}]\) rewards volume expansion in configuration space, preventing all latent samples from collapsing into a single zero-variance point.
2. **Maximum Likelihood Loss (Training by Example)**:
\[J_{ML} = \mathbb{E}_{\mathbf{x} \sim \rho_X} \left[ \frac{1}{2} \| F_{xz}(\mathbf{x}) \|^2 - \log R_{xz}(\mathbf{x}) \right]\]
   - Guides the flow into relevant metastable basins during early iterations.

In high-dimensional molecular systems, pure \(J_{KL}\) from random initialization gets trapped in local minimum basins or produces catastrophic steric overlaps. Staged scheduling—starting with high \(w_{ML}\) and transitioning to dominant \(w_{KL}\)—is essential for stable convergence.

## Verified Solution & Action Rules
1. Implement staged loss weighting:
   - **Stage 1 (Warmup)**: \(w_{ML} = 1.0, w_{KL} = 10^{-6} \dots 10^{-3}\) to fit initial conformers.
   - **Stage 2 (Energy Refinement)**: Gradually ramp \(w_{KL} \to 1.0\) while decreasing \(w_{ML}\).
2. Ensure the log-Jacobian determinant \(\log R_{zx}\) is computed analytically from invertible coupling layers.

```python
# Verified Implementation Pattern
def compute_boltzmann_generator_loss(
    flow,
    batch_x: Tensor,
    prior_z: Tensor,
    energy_fn,
    w_ml: float = 1.0,
    w_kl: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    # 1. Training by Example (ML Loss)
    z_pred, log_det_xz = flow.forward_xz(batch_x)
    loss_ml = 0.5 * (z_pred * z_pred).sum(axis=-1).mean() - log_det_xz.mean()

    # 2. Training by Energy (KL Loss)
    x_gen, log_det_zx = flow.forward_zx(prior_z)
    u_gen = energy_fn(x_gen)
    loss_kl = u_gen.mean() - log_det_zx.mean()

    total_loss = w_ml * loss_ml + w_kl * loss_kl
    return total_loss, {"loss_ml": loss_ml.item(), "loss_kl": loss_kl.item()}
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Training solely on \(J_{KL}\) from scratch without ML pre-training on valid conformers.
- ❌ **Anti-Pattern**: Omitting the Jacobian determinant \(\log R_{zx}\) in \(J_{KL}\), which eliminates entropy maximization and guarantees mode collapse.
- ❌ **Anti-Pattern**: Using static \(w_{ML}=0\) in the first 100 epochs on complex polymers.
