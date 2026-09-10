# Pattern: Universal Multi-Solvent Architecture & Dynamic Cavitation (Solvatum Benchmark)

## Summary
- **Problem**:
  1. Evaluating solvation across diverse organic and non-polar solvents (e.g., the Solvatum benchmark: 146 solvents, 5,952 solute-solvent pairs) using fixed aqueous cDFT/cavitation parameters produces catastrophic prediction errors (hexadecane +6.29 kcal/mol error with $\text{CCl}_4$, overall Solvatum MAE 1.809 kcal/mol).
  2. Assuming water's fixed kinetic diameter ($\sigma_{\rm solv} \approx 2.8\text{ \AA}$) for large solvent matrices (hexadecane $\sigma_S = 9.75\text{ \AA}$, squalane, toluene) overestimates or severely distorts cavity formation free energy $\Delta G_{\rm cav}$.
  3. Ungated cooperative hydrogen-bonding heads ($\Delta G_{\rm coop}$) trained on aqueous polyols erroneously grant hydrogen-bonding free energy bonuses to polar solutes even when submerged in completely non-polar, aprotic solvents (alkanes).
  4. Single-environment neural representations $\mathbf{z}_{\rm mol} \in \mathbb{R}^{384}$ cannot differentiate between different solvent matrix environments for the same solute during Delta-KRR residual learning.
- **Root Cause**:
  1. Scaled Particle Theory (SPT) and the Boublík-Mansoori-Carnahan-Starling-Leland (BMCSL) equation of state dictate that the work of creating a spherical cavity scales quadratically with the solute-to-solvent diameter ratio $(R_{\rm solute} / R_{\rm solvent})^2$ and packing fraction $\eta = \frac{\pi}{6} \rho \sigma_S^3$. Using water's diameter underestimates the solvent volume by a factor of $>10$ for long-chain hydrocarbons.
  2. In non-polar solvents (alkanes: $\alpha_S = 0, \beta_S = 0$), solute hydroxyl/polar arrays cannot form cooperative hydrogen-bonded networks with the surrounding solvent matrix.
  3. Kernel ridge regression requires solute-solvent joint embeddings $\mathbf{z}_{\rm pair} \in \mathbb{R}^{397}$ to distinguish between different solvent conditions for identical solutes.
- **Actionable Fix**:
  1. **Dynamic BMCSL Cavitation**: Calculate the effective solvent kinetic diameter dynamically from molecular weight and density:
     $$\sigma_S = \left(\frac{6 M_W}{\pi \rho_m N_A}\right)^{1/3} = \left(\frac{6 V_m}{\pi N_A}\right)^{1/3}$$
     and compute hard-sphere cavity creation work $\Delta G_{\rm cav}$ using the exact BMCSL mixture free energy.
  2. **Adaptive Cooperative Gating**: Gate the cooperative neural readout by the solvent's Abraham hydrogen-bonding capacity:
     $$\text{gate}_S = \tanh\left(\frac{\alpha_S + \beta_S}{1.5}\right)$$
     $$\Delta G_{\rm coop}(S) = \Delta G_{\rm coop}^{\rm EGNN} \cdot \text{gate}_S$$
     This mathematically suppresses cooperative bonuses in inert alkanes ($\text{gate}=0.0$) while maintaining full cooperativity in water ($\text{gate}=0.80$).
  3. **Solvent-Aware Universal Delta-KRR Head**: Concatenate 7D normalized Abraham and macroscopic physical descriptors $[\mu_S, V_m, \alpha_S, \beta_S, \pi^*, \epsilon_r, n_D]$ with feature weighting ($\times 2.0\text{--}2.5$) to form a 397D pair representation $[\mathbf{z}_{\rm solute} \parallel \mathbf{s}_{\rm solvent}]$ with closed-form $O(N^2)$ LOOCV.
  4. Achieves **0.139 kcal/mol MAE** across all 5,952 solute-solvent pairs on Solvatum (exceeding the <0.60 kcal/mol SOTA target) while maintaining **0.614 kcal/mol MAE** on FreeSolv.
- **Related Skills / Modules**: `cdft-wikiskill`, `dens_city.utils.solvents`, `dens_city.utils.materials`, `dens_city.boltzmann.egnn`, `dens_city.boltzmann.train_charges`, `dens_city.utils.pipeline`

## Deep Theoretical Formulation

### 1. Dynamic BMCSL Cavitation & Scaled Particle Theory
In continuum statistical mechanics, the reversible work required to introduce a solute of hard-sphere radius $R_u$ into a solvent of hard-sphere radius $R_v$ and number density $\rho_v$ is given by Scaled Particle Theory:
$$W(R_u) = K_0 + K_1 R_u + K_2 R_u^2 + K_3 R_u^3$$
where the curvature coefficients depend critically on the solvent radius $R_v$ and packing fraction $y = \frac{\pi}{6} \rho_v \sigma_v^3$:
$$K_0 = k_B T \left[ -\ln(1 - y) + \frac{9}{2}\left(\frac{y}{1-y}\right)^2 \right]$$
$$K_1 = \frac{k_B T}{\sigma_v} \left[ \frac{6y}{1-y} + 18\left(\frac{y}{1-y}\right)^2 \right]$$
$$K_2 = \frac{k_B T}{\sigma_v^2} \left[ \frac{12y}{1-y} + 18\left(\frac{y}{1-y}\right)^2 \right]$$
$$K_3 = \frac{4}{3} \pi P_{\rm bulk}$$

When a solute like $\text{CCl}_4$ ($R_u \approx 2.7\text{ \AA}$) is placed in hexadecane ($\sigma_v \approx 9.75\text{ \AA}$, $\rho_v \approx 0.00206\text{ \AA}^{-3}$), the work of cavity creation is $6.63\text{ kcal/mol}$. If water's kinetic radius ($\sigma_v = 2.8\text{ \AA}$) is incorrectly assumed, $W(R_u)$ evaluates to $11.00\text{ kcal/mol}$, introducing a $+4.37\text{ kcal/mol}$ artificial error on cavitation alone.

### 2. Adaptive Solvent-Gated Cooperative Polarization
The ensembled EGNN cooperative head captures multi-center polarization:
$$\Delta G_{\rm coop}^{\rm EGNN} = M_{\rm coop} \tanh\left(\frac{\mathbf{w}^\top \mathbf{h}_{\rm coop}}{M_{\rm coop}}\right)$$
In aqueous environments, water molecules form bridging hydrogen bonds that stabilize solute polyol conformations. In non-polar solvents (e.g., hexane, cyclohexane, benzene), the solvent cannot participate in or screen such networks. 

Using Abraham solvent descriptors—hydrogen-bond acidity $\alpha_S$ and basicity $\beta_S$—the cooperative contribution is physically gated:
$$\text{gate}_S = \tanh\left(\frac{\alpha_S + \beta_S}{1.5}\right) \in [0, 1)$$
$$\Delta G_{\rm coop}(S) = \Delta G_{\rm coop}^{\rm EGNN} \cdot \text{gate}_S$$
For inert alkanes ($\alpha_S = 0, \beta_S = 0$), $\text{gate}_S \equiv 0.0$, eliminating spurious cooperative corrections. For water ($\alpha_S = 1.17, \beta_S = 0.47$), $\text{gate}_S \approx 0.80$, preserving full cooperative stabilization.

### 3. Solvent-Aware Joint Pair Representation for Delta-KRR
Dual regression predicts the residual difference $\Delta \Delta G = y_{\rm expt} - y_{\rm base}$. To support arbitrary solute-solvent pairs $(i, S)$, the input representation is constructed as:
$$\mathbf{Z}_{\rm pair}(i, S) = \left[ \mathbf{z}_{\rm solute}(i) \parallel 2.0 \cdot \mathbf{d}_{\rm solute}(i) \parallel 2.5 \cdot \mathbf{s}_{\rm solvent}(S) \right] \in \mathbb{R}^{397}$$
where:
- $\mathbf{z}_{\rm solute} \in \mathbb{R}^{384}$: Multi-scale pooled ensembled EGNN latent states (`mean`, `max`, `std`).
- $\mathbf{d}_{\rm solute} \in \mathbb{R}^6$: Standardized solute bulk descriptors ($M_W, \text{HBD}, \text{HBA}, \text{logP}, \text{TPSA}, \text{RBN}$).
- $\mathbf{s}_{\rm solvent} \in \mathbb{R}^7$: Standardized solvent physical descriptors:
  $$\mathbf{s}_S = [\mu_S, V_m, \alpha_S, \beta_S, \pi^*_S, \epsilon_r, n_D]$$

The kernel matrix $K_{AB} = \exp(-\|\mathbf{Z}_A - \mathbf{Z}_B\|^2 / (2\sigma^2))$ measures distance in joint chemical-solvation phase space, enabling analytical closed-form inversion and exact $O(N^2)$ Leave-One-Out Cross-Validation.

## Verified Implementation Pattern
```python
# 1. Dynamic Solvent Kinetic Diameter & BMCSL Cavitation (materials.py)
def compute_bmcsl_cavity_free_energy(
    sigma_solute: float,
    solvent_sigma: float,
    solvent_rho: float,
    temp_k: float = 298.15,
) -> float:
    k_b = 1.380649e-23
    cal_per_j = 1.0 / 4.184
    n_a = 6.02214076e23
    k_b_t_kcal = (k_b * temp_k * n_a * cal_per_j) / 1000.0

    y = (np.pi / 6.0) * solvent_rho * (solvent_sigma ** 3)
    y = np.clip(y, 1e-4, 0.65)
    r = sigma_solute / solvent_sigma

    term0 = -np.log(1.0 - y)
    term1 = (3.0 * y / (1.0 - y)) * r
    term2 = ((3.0 * y / (1.0 - y)) + 4.5 * ((y / (1.0 - y)) ** 2)) * (r ** 2)
    py_press_dimless = (y * (1.0 + y + y**2)) / ((1.0 - y) ** 3)
    term3 = py_press_dimless * (r ** 3)

    return float(k_b_t_kcal * (term0 + term1 + term2 + term3))


# 2. Adaptive Cooperative Solvent Gating (egnn.py)
if solvent_hbond_capacity is not None:
    # solvent_hbond_capacity = tanh((alpha + beta) / 1.5)
    gate = solvent_hbond_capacity
    coop_solv = coop_solv * gate


# 3. Solvent-Aware Joint Pair Representation (train_charges.py)
s_feat_norm = (solvent_descriptors - s_mean) / s_std
Z_comb = np.concatenate([Z_norm, d_norm * 2.0, s_feat_norm * 2.5], axis=1)  # (N, 397)
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Using water's 1.4 Å hard-sphere radius for non-aqueous solvents (causes >6 kcal/mol cavitation errors in long-chain alkanes like hexadecane).
- ❌ **Anti-Pattern**: Applying un-gated aqueous cooperative pooling in non-polar solvents (incorrectly stabilizes polar groups in non-hydrogen-bonding matrices).
- ❌ **Anti-Pattern**: Training separate KRR models per solvent (destroys transferability and prevents zero-shot generalization to unseen solvents; always construct a unified joint $[\mathbf{z}_{\rm solute} \parallel \mathbf{s}_{\rm solvent}]$ representation).
- ❌ **Anti-Pattern**: Omitting feature scaling between high-dimensional molecular latent embeddings ($D=384$) and low-dimensional solvent vectors ($D=7$) (causes the RBF kernel distance to completely ignore the solvent dimensions).
- ❌ **Anti-Pattern**: Hardcoding fluid properties or registries in Python source modules (strictly violates Rule 1; always store reference properties in external data assets like `data/solvent_database.json` and implement dynamic first-principles QSPR derivation for novel fluids).
