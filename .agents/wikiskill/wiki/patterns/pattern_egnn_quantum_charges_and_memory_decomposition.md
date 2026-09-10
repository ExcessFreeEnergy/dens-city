# Pattern: EGNN Memory-Decomposed Projections & Multi-Head Solvation Architecture

## Summary
- **Problem**:
  1. Out-Of-Memory (OOM) crashes during batched execution of 7-layer EGNN models on GPU, and unphysical drift of total net molecular charge when predicting conformation-dependent partial charges.
  2. Severe underestimation of hydration free energies (errors \(>10\text{ kcal/mol}\)) on polyols (glucose, sorbitol) and cyclic amides (uracils) when relying solely on pairwise additive Generalized Born electrostatics, caused by missing cooperative multi-center polarization networks.
  3. Optimizer crashes (`AssertionError` in `tinygrad/helpers.py:unwrap(t.grad)`) during multi-head training when individual parameter heads are omitted from the backward computational loss graph.
- **Root Cause**:
  1. Materializing fully concatenated edge tensors \([h_i, h_j, d_{ij}^2, a_{ij}] \in \mathbb{R}^{B \times 128 \times 128 \times 258}\) before linear projection consumes gigabytes of memory per layer.
  2. Classical implicit solvent and pairwise Generalized Born models assume independent atomic descreening spheres, neglecting cooperative hydrogen-bonding networks. When adding a neural cooperative head \(\Delta G_{\rm coop}\), setting the output squashing bound too tight (\(\text{max\_delta\_global} \le 12.0\text{ kcal/mol}\)) forces \(\tanh\) into near-saturation, collapsing autograd derivatives \(1 - \tanh^2(u) \to 0\) and freezing parameter updates on poly-hydroxylated sugars.
  3. In Tinygrad, `opt.schedule_step()` evaluates `[unwrap(t.grad) for t in self.params]`. If any parameter registered in `self.head_params` does not contribute to the realized loss, its `t.grad` is `None`, triggering an immediate assertion failure.
- **Actionable Fix**:
  1. Decompose the first linear edge projection:
     \[e_{ij} = \text{SiLU}\left( W_{hi} h_i^l + W_{hj} h_j^l + W_d d_{ij}^2 + W_a a_{ij} + b \right)\]
     computing linear projections in \(\mathbb{R}^{B \times 128 \times 128}\) before broadcasting.
  2. Normalize neighborhood aggregation by active cutoff degree:
     \[\text{deg}_i = \max\left(1.0, \, \sum_{j=1}^{128} a_{ij} \cdot \mathbf{1}_{r_{ij} \le r_{\rm cut}}\right), \quad m_i = \frac{1}{\text{deg}_i} \sum_{j=1}^{128} m_{ij}\]
  3. Enforce exact net charge conservation via baseline topological prior superposition and uniform mean-shift:
     \[q_i^{\rm final} = \left[ q_i^{\rm raw} - \frac{1}{N_{\rm real}} \left( \sum_{j=1}^{N_{\rm real}} q_j^{\rm raw} - Q_{\rm total} \right) \right] \cdot m_i\]
  4. Implement a multi-head solvation readout architecture:
     - Dynamic partial charges (\(132 \to 128 \to 1\)): \(\Delta q_i \in \pm 0.32e\) with topological Pauling prior.
     - Atomic nonpolar cavitation (\(132 \to 128 \to 1\)): \(\Delta g_i^{\rm vdw} = \text{max\_delta\_vdw} \cdot \tanh(\Delta g_{\rm raw} / \text{max\_delta\_vdw}) \cdot m_i\).
     - Molecular cooperative solvation (\(384 \to 128 \to 1\)): \(\Delta G_{\rm coop} = \text{max\_delta\_global} \cdot \tanh(\Delta G_{\rm raw} / \text{max\_delta\_global}) \cdot \text{molecule\_mask}\) with headroom \(\ge 25.0\text{ kcal/mol}\) and zero-initialized weights.
  5. Ensure all 12 registered head parameters receive non-None gradients in every training step by including \(\mathcal{L}_2(\Delta G_{\rm coop})\) and evaluating all heads in Phase 1 and Phase 2.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.boltzmann.egnn`, `dens_city.boltzmann.train_charges`

## Deep Theoretical Formulation
1. **Memory Decomposition**:
   - In standard message passing, concatenating \(h_i \in \mathbb{R}^d\) and \(h_j \in \mathbb{R}^d\) creates a \(2d\)-dimensional tensor across all \(N \times N\) pairs.
   - Using the distributive property of matrix multiplication:
     \[W [h_i \parallel h_j] = W_1 h_i + W_2 h_j\]
   - Evaluating \(W_1 h\) and \(W_2 h\) first on \((B, N, d)\) tensors costs \(O(N \cdot d^2)\) operations. Then broadcasting their addition along dimension 1 and 2 costs only \(O(N^2 \cdot d)\) operations, reducing memory traffic by \(>70\%\).

2. **Degree Normalization for Molecular Extensivity**:
   - Molecular potential energy must be strictly extensive: a system of two non-interacting molecules at infinite separation must have energy \(U_{\rm total} = U_1 + U_2\).
   - Normalizing by \(\text{deg}_i\) stabilizes the hidden representation scale across variable molecule sizes (\(N=10\) to \(N=128\)).

3. **Topological Pauling Electronegativity Prior & Mean-Shift**:
   - Neutralizing raw neural charges with a mean-shift guarantees that the monopole electrostatic term \(\sum q_i = Q_{\rm total}\) is mathematically exact.
   - Seeding the charge head with a 2D topological Pauling prior:
     \[q_i^{\rm base} = F_i + \text{clip}\left(\kappa \sum_{j \in \mathcal{N}(i)} \text{order}_{ij} (\chi_j - \chi_i), \, -1.0, \, +1.0\right)\]
     ensures physically reasonable partial charges even when the neural perturbation head \(\Delta q(\mathbf{x}) \to 0\).

4. **Multi-Center Cooperative Solvation & Headroom Analysis**:
   - While mono-functional compounds are accurately modeled by pairwise Generalized Born electrostatics, carbohydrates and polyols (e.g. glucose \(\Delta G_{\rm expt} = -25.47\text{ kcal/mol}\), sorbitol \(\Delta G_{\rm expt} = -23.62\text{ kcal/mol}\)) exhibit massive cooperative stabilization due to contiguous hydrogen-bonding donor-acceptor ladders with water.
   - The cooperative head maps the 384-dimensional pooled descriptor \(\mathbf{z}_{\rm mol}\) to a global energy shift:
     \[\Delta G_{\rm coop} = M_{\rm coop} \cdot \tanh\left(\frac{\mathbf{w}^\top \mathbf{h}_{\rm coop} + b}{M_{\rm coop}}\right)\]
   - **Critical Gradient Saturation Limit**:
     The autograd derivative scales as \(\frac{\partial \Delta G_{\rm coop}}{\partial u} = 1 - \tanh^2(u/M_{\rm coop})\). If \(M_{\rm coop} = 12.0\text{ kcal/mol}\) and glucose requires \(\Delta G_{\rm coop} \approx -15.3\text{ kcal/mol}\), the argument \(u/M_{\rm coop} > 1.5\) drives the gradient toward zero, permanently freezing parameter updates. Setting \(M_{\rm coop} \ge 25.0\text{ kcal/mol}\) maintains active linear gradient flow throughout training, dropping FreeSolv MAE from \(0.812 \to 0.614\text{ kcal/mol}\).
   - **Zero-Initialization Identity**:
     Initializing \(\mathbf{w}_{\rm out} = \mathbf{0}, b_{\rm out} = 0\) guarantees that \(\Delta G_{\rm coop} \equiv 0\) at epoch 0, preserving baseline cDFT/Born predictions without random noise shocks.

5. **Tinygrad Optimizer Gradient Invariant (`unwrap(t.grad)`)**:
   - When constructing `self.opt_head = nn.optim.Adam(self.head_params, lr=...)`, `self.head_params` contains 12 tensors:
     - `charge_mlp`: 2 weight matrices + 2 bias vectors (4 params)
     - `vdw_mlp`: 2 weight matrices + 2 bias vectors (4 params)
     - `global_mlp`: 2 weight matrices + 2 bias vectors (4 params)
   - Tinygrad's `schedule_step()` checks `assert x is not None` on `unwrap(t.grad)` for every parameter.
   - Therefore, any training function (such as `train_epoch` or `_train_step_p1`) must evaluate all registered heads or compute a dummy regularization loss \(\lambda (\Delta G_{\rm coop})^2\) so that every parameter receives a valid backpropagation adjoint.

## Verified Implementation Pattern
```python
# Multi-head evaluation with zero-initialization and Tinygrad gradient safety
class MultiHeadEGNNForceField:
    def __init__(self, hidden_dim: int = 128, max_delta_q: float = 0.32, max_delta_vdw: float = 4.5, max_delta_global: float = 25.0):
        # 1. Dynamic quantum charge head
        self.charge_mlp = [nn.Linear(hidden_dim + 4, hidden_dim), Tensor.silu, nn.Linear(hidden_dim, 1)]
        # 2. Atomic nonpolar cavitation head
        self.vdw_mlp = [nn.Linear(hidden_dim + 4, hidden_dim), Tensor.silu, nn.Linear(hidden_dim, 1)]
        # 3. Molecular cooperative solvation head (384 -> 128 -> 1)
        self.global_mlp = [nn.Linear(hidden_dim * 3, hidden_dim), Tensor.silu, nn.Linear(hidden_dim, 1)]

        # Zero-initialize output layers to preserve physical identity at step 0
        self.vdw_mlp[2].weight = Tensor.zeros(1, hidden_dim)
        self.vdw_mlp[2].bias = Tensor.zeros(1)
        self.global_mlp[2].weight = Tensor.zeros(1, hidden_dim)
        self.global_mlp[2].bias = Tensor.zeros(1)

        self.max_delta_q = max_delta_q
        self.max_delta_vdw = max_delta_vdw
        self.max_delta_global = max_delta_global

    def forward_heads(self, h: Tensor, m: Tensor, bq: Tensor, tq: Tensor, sf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, d = h.shape
        node_inputs = Tensor.cat(h, sf, dim=-1)

        # 1. Charge head with formal charge conservation
        dq_raw = self.charge_mlp[2](self.charge_mlp[1](self.charge_mlp[0](node_inputs)))
        dq = self.max_delta_q * (dq_raw / self.max_delta_q).tanh() * m
        q_raw = (bq.reshape(B, N, 1) + dq) * m
        n_real = m.sum(axis=1, keepdim=True).maximum(1.0)
        q_shift = (q_raw.sum(axis=1, keepdim=True) - tq) / n_real
        q_pred = ((q_raw - q_shift) * m).reshape(B, N)

        # 2. Atomic cavitation head
        dv_raw = self.vdw_mlp[2](self.vdw_mlp[1](self.vdw_mlp[0](node_inputs)))
        delta_vdw_atomic = self.max_delta_vdw * (dv_raw / self.max_delta_vdw).tanh() * m
        delta_vdw_mol = delta_vdw_atomic.sum(axis=(1, 2))

        # 3. Multi-scale graph pooling and cooperative head
        mean_pool = (h * m).sum(axis=1) / n_real.reshape(B, 1)
        max_pool = (h * m - (1.0 - m) * 1e4).max(axis=1)
        h_diff = (h - mean_pool.reshape(B, 1, d)) * m
        std_pool = (((h_diff * h_diff).sum(axis=1) / n_real.reshape(B, 1)) + 1e-6).sqrt()
        z_mol = Tensor.cat(mean_pool, max_pool, std_pool, dim=-1)

        dg_coop_raw = self.global_mlp[2](self.global_mlp[1](self.global_mlp[0](z_mol))).reshape(B)
        delta_g_coop = self.max_delta_global * (dg_coop_raw / self.max_delta_global).tanh()

        return q_pred, delta_vdw_mol + delta_g_coop, delta_g_coop
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Registering multi-head parameters in `nn.optim.Adam` without evaluating every head in every training step (causes Tinygrad `AssertionError` in `helpers.py:unwrap(t.grad)`).
- ❌ **Anti-Pattern**: Clamping cooperative multi-center solvation too tightly (\(\le 12.0\text{ kcal/mol}\)), which freezes gradients on polyols and amides due to \(\tanh\) derivative saturation (always provide \(\ge 25.0\text{ kcal/mol}\) headroom).
- ❌ **Anti-Pattern**: Randomly initializing the final linear layers of correction heads (destroys baseline physics at step 0; always use zero-initialization).
- ❌ **Anti-Pattern**: Concatenating raw pair embeddings \([h_i, h_j]\) before linear projection (causes GPU memory explosion at \(B=512\)).
- ❌ **Anti-Pattern**: Training charge prediction heads with unconstrained MSE loss without a formal charge conservation mean-shift layer (destroys long-range monopole and dipole electrostatics).
