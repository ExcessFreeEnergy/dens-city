# Pattern: Decoupled Layer-by-Layer Reverse Autodiff & Conservative Force Surrogates

## Summary
- **The Ideal / Objective Goal**: Zero backward mega-reduction kernel fusing. Multi-layer equivariant networks evaluate conservative atomic forces $F = -\nabla_x U$ via decoupled layer-by-layer reverse adjoint backpropagation, keeping live memory footprints minimal ($O(B \cdot N)$ instead of $O(L \cdot B \cdot N^2)$). Normalizing flows optimize coordinate distributions via detached conservative force surrogate work, completely releasing potential energy forward activation memory graphs.
- **Problem**: When a deep neural network (e.g. 7-layer EGNN) computes potential energy $U(x)$, all layers share the coordinate leaf tensor $x$ via distance matrices $d_{ij}^2$. Calling `u.backward()` forces Tinygrad's compiler to fuse all 7 layers into a monolithic backward reduction kernel (e.g. 35 buffers, 4.6 GB DRAM, 372 ms execution), causing register spilling to off-chip DRAM. In generative flows, backpropagating $U(f_\theta(z))$ through the flow tape holds both flow and force-field graphs in memory simultaneously, causing CUDA Out-of-Memory (OOM) crashes.
- **Root Cause**: Monolithic autograd graphs across multi-layer equivariant networks couple forward intermediate activations across all layers. In normalizing flows, differentiating composite objective $U(f_\theta(z))$ directly via end-to-end autodiff is unnecessary because conservative physical forces $F(x) = -\nabla_x U(x)$ satisfy the exact chain rule:
$$\nabla_\theta U(f_\theta(z)) = \left. \frac{\partial U}{\partial x} \right|_{x=f_\theta(z)} \frac{\partial f_\theta(z)}{\partial \theta} = - F(x) \cdot \frac{\partial f_\theta(z)}{\partial \theta} = \nabla_\theta \left[ - F(x) \cdot f_\theta(z) \right]$$
- **Actionable Fix**:
  1. Compute forces layer-by-layer in reverse topological order: evaluate layer $l$ gradients $\nabla_{x_l} U$ and pass adjoints backward without accumulating multi-layer graph paths.
  2. In generative flows, evaluate $F(x)$ on detached coordinates $x = f_\theta(z).\text{detach}()$, and backpropagate the surrogate work loss $- F_{\rm det} \cdot f_\theta(z)$.
  3. Enforce **Rule 5**: Always reset parameter gradients (`for p in nn.state.get_parameters(model): p.grad = None`) immediately after force extraction in inference/sampling loops.
- **Related Skills / Modules**: `tinygrad-autograd`, `dens_city.boltzmann`, `dens_city.boltzmann.egnn`

---

## The Ideal Architectural Standard

### 1. Decoupled Layer-by-Layer Adjoint Backpropagation
For an $L$-layer network where each layer produces node representations $h_l = \text{Layer}_l(h_{l-1}, x)$, intermediate representations $h_l$ are detached and evaluated sequentially backward:

```
+-----------------------------------------------------------------------------+
|                     IDEAL LAYER-BY-LAYER REVERSE AUTODIFF                   |
+-----------------------------------------------------------------------------+
|  Forward Pass:                                                              |
|    h_0 = Embedding(Z)                                                       |
|    For l = 1..L: h_l = Layer_l(h_{l-1}, d_sq(x)).realize()                 |
|    U_total = Readout(h_L)                                                   |
|                                                                             |
|  Backward Pass:                                                             |
|    g_L = ∇_{h_L} U_total                                                    |
|    F_total = 0                                                              |
|    For l = L down to 1:                                                     |
|      [g_{l-1}, f_l] = ∇_{h_{l-1}, x} (g_l · Layer_l(h_{l-1}, d_sq(x)))      |
|      F_total += f_l                                                         |
|      g_l = g_{l-1}                                                          |
|    Forces F = - F_total                                                     |
+-----------------------------------------------------------------------------+
```

```python
# The Ideal: Decoupled Adjoint Backpropagation
h_final = h_list[-1].detach()
h_final.requires_grad = True
u_total = self.readout(h_final)
[g] = u_total.sum().gradient(h_final)

f_total = Tensor.zeros(B, N, 3)
for layer_idx in reversed(range(self.num_layers)):
    layer = self.layers[layer_idx]
    h_in = h_list[layer_idx].detach()
    h_in.requires_grad = True
    x_l = x_prep.detach()
    x_l.requires_grad = True

    h_out = layer(h_in, x_l)
    step_loss = (g * h_out).sum()
    [g_prev, f_l] = step_loss.gradient(h_in, x_l)
    Tensor.realize(g_prev, f_l)
    g = g_prev
    f_total = (f_total + f_l).realize()

forces = -f_total * atom_mask
for p in nn.state.get_parameters(self):
    p.grad = None
```

### 2. Conservative Force Surrogate Flow Training
Instead of maintaining the autograd tape through the microscopic Hamiltonian $U(x)$ into normalizing flow parameters $\theta$:

```python
# The Ideal: Detached Force Surrogate Step Loss
x, log_pz, log_det, j_tor = self._forward_transform(z)

# Evaluate exact forces on detached coordinates
x_eval = x.detach()
u_val, forces = self.energy_fn.compute_forces(x_eval)
forces_det = forces.detach()

# Conservative force work: - F · x(θ)
work = (forces_det * x).sum(axis=(-1, -2))
loss_surrogate = - self.beta * work - log_pz - log_det

# Loss value for tracking matches true microscopic energy
loss_exact = self.beta * u_val.detach() - log_pz - log_det

step_loss = loss_surrogate - loss_surrogate.detach() + loss_exact
step_loss.backward()

# Rule 5 Invariant: Detach parameter gradients on force field
for p in nn.state.get_parameters(self.energy_fn):
    p.grad = None
```

---

## Anti-Patterns to Contrast Against (What Bad Code Looks Like)

- ❌ **Anti-Pattern (Monolithic Shared-Leaf Autograd)**: Calling `u.sum().backward()` directly on an unrolled 7-layer EGNN where coordinate tensor `x` was defined at the input. This fuses all 7 layers into a 35-buffer mega-reduction kernel that spills registers to DRAM.
- ❌ **Anti-Pattern (Coupled Flow-Hamiltonian Tape)**: Running `loss = beta * energy_fn(flow(z)) - log_pz - log_det` followed by `loss.backward()` when `energy_fn` contains deep neural network layers. This retains full forward activations for both networks simultaneously, doubling VRAM usage.
- ❌ **Anti-Pattern (Leaking Parameter Gradients)**: Omitting `p.grad = None` after force calculation. Retaining parameter gradient buffers causes Tinygrad's autograd graph to leak across iterations.
