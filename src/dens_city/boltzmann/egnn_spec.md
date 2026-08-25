# Specification: EGNN Machine Learned Force Field (`egnn_spec.md`)

This document provides a comprehensive technical specification of the **$E(n)$-Equivariant Graph Neural Network (EGNN) Machine Learned Force Field (MLFF)** in `dens-city`. It details the physical foundation, mathematical formulation, memory-optimized decomposed kernel projections, pipeline integration, and protocols for training compatible neural potential models.

---

## 1. Domain Foundations & Physical Principles

### 1.1 Moving Beyond Static Classical Force Fields
Traditional molecular force fields (e.g., GAFF, OPLS, CHARMM) approximate interatomic interactions using fixed partial charges $q_i$, static Lennard-Jones parameters $(\sigma_i, \epsilon_i)$, and empirical Lorentz-Berthelot mixing rules:

$$
U_{\rm classical}(\mathbf{x}) = \sum_{i < j} 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right] + \sum_{i < j} \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}
$$

While computationally lightweight, fixed-charge models fail to capture:
- **Electronic Polarizability**: Dipole enhancements induced by local dielectric environments (e.g., polyols, sugars, ionic clusters).
- **Many-Body Charge Transfer**: Conformation-dependent charge reorganization across conjugated bonds.
- **Quantum Chemical Accuracy**: Density Functional Theory (DFT) level potential energy surfaces.

### 1.2 $E(n)$ Symmetry & Conservative Force Fields
For an isolated molecular system in 3D Euclidean space ($n=3$), the potential energy $U(\mathbf{x})$ must obey strict translational and rotational invariance:

$$
U(R \mathbf{x} + \mathbf{t}) = U(\mathbf{x}) \quad \forall R \in \text{SO}(3), \, \mathbf{t} \in \mathbb{R}^3
$$

Conservative interatomic forces $\mathbf{F}_i = -\nabla_{\mathbf{r}_i} U(\mathbf{x})$ are derived via reverse-mode automatic differentiation and rotate equivariantly:

$$
\mathbf{F}_i(R \mathbf{x} + \mathbf{t}) = R \mathbf{F}_i(\mathbf{x})
$$

The EGNN architecture guarantees exact $E(n)$ invariance for scalar energy $U$ and exact equivariance for forces $\mathbf{F}$ by constructing all message-passing interactions exclusively from invariant pairwise squared Euclidean distances $d_{ij}^2 = \|\mathbf{r}_i - \mathbf{r}_j\|^2$ and atomic numbers $Z_i$.

---

## 2. Mathematical Formulation & Architecture

The `EGNNForceField` consists of an initial embedding layer, $L=7$ sequential message-passing layers, a node-wise energy readout MLP, and an autograd force evaluator.

```
                              Atomic Numbers Z in R^(B x 128)
                                             │
                                             ▼
                             Node Embedding: h^0 in R^(B x 128 x 128)
                                             │
    ┌────────────────────────────────────────┴────────────────────────────────────────┐
    │                                                                                 │
    ▼                                                                                 ▼
Layer 1 ──► Layer 2 ──► Layer 3 ──► Layer 4 ──► Layer 5 ──► Layer 6 ──► Layer 7: h^7 in R^(B x 128 x 128)
Inside Each Layer:                                                                    │
1. Distances: d_ij^2 = ||x_i - x_j||^2 in R^(B x 128 x 128 x 1)                       │
2. Decomposed Edge Projection: e_ij = SiLU(W_hi h_i + W_hj h_j + W_d d_ij^2 + W_a a_ij)│
3. Edge Message: m_ij = SiLU(W_2 e_ij + b_2) * a_ij                                   │
4. Neighbor Aggregation: m_i = sum_j m_ij in R^(B x 128 x 128)                        │
5. Node Update: h_i^(l+1) = (h_i^l + phi_h([h_i^l, m_i])) * atom_mask_i               │
                                                                                      │
                                             ┌────────────────────────────────────────┘
                                             ▼
                             Atomic Energies: eps_i = MLP_node(h_i^7) * atom_mask_i
                                             │
                                             ▼
                             Total Potential Energy: U_total = sum_i eps_i * mol_mask
                                             │
                                             ▼
                             Conservative Forces: F = -nabla_x U_total
```

### 2.1 Inputs & Ingestion
- **Batch Shape**: Fixed uniform dimensions $(B, N)$ where $N = 128$ particles, padded with dummy zeros for batch uniformity.
- **Atomic Numbers**: $Z \in \mathbb{R}^{B \times 128}$, mapped to atomic numbers ($H=1, C=6, N=7, O=8, \dots$, dummy $= 0$).
- **Coordinates**: $\mathbf{x} \in \mathbb{R}^{B \times 128 \times 3}$.
- **Masks**:
  - Atom mask: $m_i \in \mathbb{R}^{B \times 128 \times 1}$ ($1.0$ for real atoms, $0.0$ for dummy slots).
  - Edge mask: $a_{ij} = m_i \cdot m_j \cdot (1 - \delta_{ij}) \in \mathbb{R}^{B \times 128 \times 128 \times 1}$.
  - Molecule mask: $M_b \in \mathbb{R}^{B}$ ($1.0$ for valid batch items, $0.0$ for dummy slots).

### 2.2 Layer-0 Node Embedding
Atomic numbers are one-hot encoded and projected through a trainable linear layer into initial hidden representations $h^0 \in \mathbb{R}^{B \times 128 \times 128}$:

$$
h_i^0 = \text{Linear}_{128 \to 128}\left( \text{OneHot}(Z_i) \right) \cdot m_i
$$

### 2.3 Layer-Wise Message Passing ($l = 0, \dots, 6$)

#### Step A: Pairwise Relative Squared Distances
$$
d_{ij}^2 = \sum_{c=0}^2 (x_{i, c} - x_{j, c})^2 \in \mathbb{R}^{B \times 128 \times 128 \times 1}
$$

#### Step B: Memory-Optimized Decomposed Edge Projection ($\phi_e$)
To prevent memory exhaustion from materializing $(B, 128, 128, 258)$ edge tensors, the first linear transformation is mathematically decomposed:

$$
e_{ij} = \text{SiLU}\left( W_{hi} h_i^l + W_{hj} h_j^l + W_d d_{ij}^2 + W_a a_{ij} + b \right) \in \mathbb{R}^{B \times 128 \times 128 \times 128}
$$

$$
m_{ij} = \text{SiLU}\left( W_2 e_{ij} + b_2 \right) \cdot a_{ij}
$$

#### Step C: Neighborhood Aggregation
$$
m_i = \sum_{j=1}^{128} m_{ij} \in \mathbb{R}^{B \times 128 \times 128}
$$

#### Step D: Node State Update ($\phi_h$) with Residual Connection
$$
\phi_h([h_i^l, m_i]) = W_{h2} \cdot \text{SiLU}\left( W_{h1} [h_i^l, m_i] + b_{h1} \right) + b_{h2}
$$

$$
h_i^{l+1} = \left( h_i^l + \phi_h([h_i^l, m_i]) \right) \cdot m_i
$$

*Note: Coordinate updates are explicitly omitted to maintain strict $E(n)$ invariance of the scalar output.*

### 2.4 Readout & Conservative Force Generation
The final layer node embeddings $h^7 \in \mathbb{R}^{B \times 128 \times 128}$ are mapped to atomic energy contributions $\epsilon_i$:

$$
\epsilon_i = \left[ W_{\rm out2} \cdot \text{SiLU}\left( W_{\rm out1} h_i^7 + b_{\rm out1} \right) + b_{\rm out2} \right] \cdot m_i \in \mathbb{R}^{B \times 128 \times 1}
$$

$$
U_{\rm total}(\mathbf{x}) = \left( \sum_{i=1}^{128} \epsilon_i \right) \cdot M_b \in \mathbb{R}^B
$$

$$
\mathbf{F}_i = -\frac{\partial U_{\rm total}}{\partial \mathbf{r}_i} \in \mathbb{R}^{B \times 128 \times 3}
$$

---

## 3. Hardware Scaling & Compute Tiering

| Feature | Classical Route (`--energy-engine classical`) | EGNN MLFF Route (`--energy-engine egnn`) |
| :--- | :--- | :--- |
| **Physical Model** | GAFF Lennard-Jones + Coulomb (PBC) | 7-Layer Invariant EGNN Potential |
| **Parameter Source** | Semi-empirical lookup (`forcefield_parameters.json`) | DFT / Quantum Trained Weights |
| **Default Batch Size** | $B = 512$ molecules | $B = 32$ molecules (auto-throttled) |
| **Sampling Throughput** | $2,200+$ conformations/second | $20-100$ conformations/second |
| **Polarizability** | Fixed point charges | Dynamic many-body contextual polarization |
| **Primary Use Case** | Large combinatorial library screening | High-accuracy polar/charged edge-cases (polyols, ions) |

---

## 4. How to Train a Compatible Model for `dens-city`

To train a custom EGNN model on reference quantum datasets (such as QM9, ANI-1x, MD17, or SPICE) that can be directly ingested by `dens-city`:

### 4.1 Target Loss Function
Train against reference total potential energies $U_{\rm DFT}$ and conservative forces $\mathbf{F}_{\rm DFT}$:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{b=1}^B \left[ \left| U_{\rm pred}^{(b)} - U_{\rm DFT}^{(b)} \right|^2 + \frac{\lambda_F}{3 N_{\rm real}} \sum_{i=1}^{N_{\rm real}} \left\| \mathbf{F}_{\rm pred, i}^{(b)} - \mathbf{F}_{\rm DFT, i}^{(b)} \right\|^2 \right]
$$

where $\lambda_F \approx 10.0$ to $100.0$ balances force fitting against energy fitting.

### 4.2 Minimal Training Script Example (`train_egnn.py`)

```python
import numpy as np
from tinygrad import Tensor, nn, dtypes
from dens_city.boltzmann.egnn import EGNNForceField

# 1. Instantiate Model with Power-of-2 Dimensions
model = EGNNForceField(num_layers=7, hidden_dim=128, max_atomic_number=128, n_particles=128)
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-3)

# 2. Training Loop on DFT Mini-Batches
def train_step(coords_np, z_np, atom_mask_np, energy_dft_np, forces_dft_np):
    Tensor.training = True
    opt.zero_grad()

    x = Tensor(coords_np, requires_grad=True)
    z = Tensor(z_np, dtype=dtypes.float32)
    mask = Tensor(atom_mask_np, dtype=dtypes.float32)
    u_ref = Tensor(energy_dft_np, dtype=dtypes.float32)
    f_ref = Tensor(forces_dft_np, dtype=dtypes.float32)

    # Forward Energy
    u_pred = model.compute_energy(x, z, mask)
    loss_energy = ((u_pred - u_ref) ** 2).mean()

    # Autograd Forces
    loss_energy.backward()
    f_pred = -x.grad * mask
    loss_force = ((f_pred - f_ref) ** 2).sum() / mask.sum()

    total_loss = loss_energy + 10.0 * loss_force
    total_loss.backward()
    opt.step()
    return float(total_loss.item())

# 3. Save Checkpoint to Standard NPZ Archive
def export_weights(model, filepath="egnn_weights.npz"):
    state_dict = nn.state.get_state_dict(model)
    np_dict = {k: v.numpy() for k, v in state_dict.items()}
    np.savez(filepath, **np_dict)
    print(f"Exported EGNN weights to {filepath}")
```

### 4.3 Checkpoint State Dictionary Format
Exported `.npz` weight archives must match the following naming convention:
- `embedding.weight`: `(128, 128)`
- `embedding.bias`: `(128,)`
- `layers.{0..6}.edge_hi.weight`: `(128, 128)`
- `layers.{0..6}.edge_hj.weight`: `(128, 128)`
- `layers.{0..6}.edge_d.weight`: `(128, 1)`
- `layers.{0..6}.edge_a.weight`: `(128, 1)`
- `layers.{0..6}.edge_a.bias`: `(128,)`
- `layers.{0..6}.edge_l2.weight`: `(128, 128)`
- `layers.{0..6}.edge_l2.bias`: `(128,)`
- `layers.{0..6}.node_mlp.0.weight`: `(128, 256)`
- `layers.{0..6}.node_mlp.0.bias`: `(128,)`
- `layers.{0..6}.node_mlp.2.weight`: `(128, 128)`
- `layers.{0..6}.node_mlp.2.bias`: `(128,)`
- `readout_mlp.0.weight`: `(128, 128)`
- `readout_mlp.0.bias`: `(128,)`
- `readout_mlp.2.weight`: `(1, 128)`
- `readout_mlp.2.bias`: `(1,)`

---

## 5. CLI Usage & Verification

```bash
# Run Classical Route (Default, high throughput B=512)
uv run dens-city --materials water argon --energy-engine classical

# Run EGNN Route (Auto-throttled to B=32)
uv run dens-city --materials water argon benzene 5cb --energy-engine egnn --benchmark

# Run test suite
uv run pytest tests/test_egnn_forcefield.py tests/test_engine_routing.py -v
```
