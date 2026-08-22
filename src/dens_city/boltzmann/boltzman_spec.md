# Specification: Boltzmann Generator & Normalizing Flow Engine (`boltzman_spec.md`)

This document specifies the architecture, statistical mechanics, deep generative normalizing flows, base-2 dyadic optimization, and computational flow for the `boltzmann` subpackage within `dens-city`.

---

## 1. Domain Overview & Theoretical Bridge

### 1.1 The Generative Bridge: Mean-Field cDFT to Atomistic Sampling
While 1D/3D Classical Density Functional Theory computes the continuous single-body equilibrium density profile $\rho(\mathbf{r})$, atomistic molecular dynamics and thermodynamic validation require sampling uncorrelated many-body configurations $\mathbf{x} = (\mathbf{r}_1, \dots, \mathbf{r}_N)$ from the Boltzmann distribution:
$$p(\mathbf{x}) = \frac{1}{Z} \exp(-\beta U(\mathbf{x}))$$
where $U(\mathbf{x}) = U_{\rm pair}(\mathbf{x}) + U_{\rm wall}(\mathbf{x})$ is the exact microscopic Hamiltonian, and $\beta = 1 / (k_B T)$.

### 1.2 Variational Reverse Kullback-Leibler (KL) Training
The Boltzmann Generator uses an invertible normalizing flow $f_\theta: \mathcal{Z} \to \mathcal{X}$ mapping a base latent distribution $p_z(\mathbf{z})$ to Cartesian molecular coordinates $\mathbf{x} = f_\theta(\mathbf{z})$.
The parameters $\theta$ are optimized by minimizing the reverse KL divergence between the generated density $q_\theta(\mathbf{x})$ and the unnormalized Boltzmann target $e^{-\beta U(\mathbf{x})}$:
$$\mathcal{L}(\theta) = D_{\rm KL}(q_\theta \parallel p) = \mathbb{E}_{\mathbf{z} \sim p_z} \left[ \beta U(f_\theta(\mathbf{z})) - \log p_z(\mathbf{z}) - \log \left| \det J_{f_\theta}(\mathbf{z}) \right| \right]$$
where $\log \left| \det J_{f_\theta}(\mathbf{z}) \right|$ is the exact analytical Jacobian log-determinant accumulated across flow layers.

---

## 2. Computational Flow Architecture

```mermaid
flowchart TD
    subgraph P["1. prior.py (Spatial Prior)"]
        Rho["Input cDFT Profile ρ_cDFT(z)"] --> CDF["Inverse-CDF 1D Distribution"]
        CDF --> Origins["Sampled Origin Pool (B, 3)"]
    end

    subgraph E["2. energy.py (Microscopic Hamiltonian)"]
        Mol["Material Sites: N_real Atoms"] --> Pad["Base-2 Pad: N_pad = 2^k ∈ {1..64}"]
        Pad --> Ham["U(x) = Shifted-Force LJ + Coulomb + Steele (9-3) Slit Walls"]
    end

    subgraph B["3. bijectors.py (4-Channel Base-2 Flow)"]
        Latent["Latent Noise z ~ N(0, I) [dim = N_pad * 4 = 2^k]"]
        Flow["Base2CartesianFlow: RealNVP Stack of Dyadic Affine Couplings"]
        Latent --> Flow
        Flow --> Coords["Generated Cartesian Coordinates x (B, N_pad, 3) + log|det J|"]
    end

    subgraph G["4. generator.py (Variational Training & Sampling)"]
        subgraph LOOP["JIT Training Step (@TinyJit)"]
            Loss["Reverse-KL Loss: L = β * U(x) - log p_z(z) - log|det J|"]
            Grad["Autograd: ∇_θ L -> Optimizer Step (Adam / Muon)"]
            Loss --> Grad
        end
        
        Sample["generator.sample() -> Slice Real Atoms -> Export trajectory.xyz"]
    end

    %% Data Flow Connections
    Origins --> Flow
    Coords --> Ham
    Ham --> Loss
    Origins --> Loss
    Flow --> Loss
    Grad --> Flow
    Flow --> Sample
```

---

## 3. Module Specifications & Interfaces

### 3.1 `prior.py` — cDFT Base Distribution Prior
- **`CDFTBaseDistribution`**:
  - `__init__(rho_z, l_z, box_size_xy=(30.0, 30.0), n_particles=1, n_grid=None)`: Constructs 1D cumulative distribution function (CDF) from $\rho_{\rm cDFT}(z)$.
  - `sample(n_samples=1, as_4channel=False)`: Generates configurations with uniform transverse $(x, y) \in [0, L_x] \times [0, L_y]$ and piecewise-linear inverse-CDF interpolation along $z$.
  - `log_prob(pos)`: Vectorized tinygrad tensor evaluation of exact base prior log probability:
    $$\log p_0(\mathbf{r}_1 \dots \mathbf{r}_N) = \sum_{i=1}^N \left[ -\ln(L_x L_y) + \ln(\rho_{\rm cDFT}(z_i)) - \ln\left(\int \rho_{\rm cDFT}(z) dz\right) \right]$$

### 3.2 `energy.py` — Shifted-Force Microscopic Hamiltonian & Energy Regularization
- **`regularize_energy(energy, e_high=1e4, e_max=1e20)`**:
  - Continuous logarithmic regularization $E_{\rm reg} = E_{\rm high} + \log(E - E_{\rm high} + 1)$ for $E \ge E_{\rm high}$.
  - Autograd-safe: hard-bounds excess to $0.0$ before `.log()` evaluation to prevent NaN gradient poisoning.
- **`MicroscopicEnergy`**:
  - `__init__(material, box_size, r_cut=None, pad_to_power_of_2=True, wall_type="stele93", e_high=1e4, e_max=1e20)`: Configures pairwise parameters $\sigma_{ij}, \epsilon_{ij}, q_{ij}$, minimum image periodic boundary conditions in $XY$, confining wall parameters, and Noé regularization thresholds.
  - `compute_pair_energy(pos, shift=True)`: Pairwise shifted-force Lennard-Jones (12-6) + Coulomb electrostatic energy with periodic distance wrap. Supports both $(B, N, 3)$ and $(B, N, 4)$ inputs.
  - `compute_wall_energy(pos)`: External Steele (9-3) or hard wall confining potential energy in $Z$.
  - `regularize_energy(energy, e_high=None, e_max=None)`: Regularizes high-energy configurations using instance thresholds.
  - `eval_energy`: JIT-compiled function wrapper (`TinyJit(self.__call__)`).

### 3.3 `bijectors.py` — Normalizing Flows, Invertible Bijectors & Torsional Kinematics
- **`AffineCouplingLayer`**:
  - Dyadic partitioning ($\text{dim}_a = \text{dim} // 2, \text{dim}_b = \text{dim} // 2$) with power-of-2 hidden dimensions.
  - Exact analytical forward, inverse, and Jacobian log-determinant $\sum s(x_A)$.
- **`RealNVPFlow`**:
  - Stacked sequence of alternating affine coupling layers.
- **`Base2CartesianFlow`**:
  - 4-channel base-2 coordinate flow $(B, N_{\rm pad}, 4)$ with flat dimension $\text{dim} = N_{\rm pad} \times 4 = 2^k \in \{4, 8, 16, 32, 64, 128, 256\}$.
  - 100% dyadic factorable matrix multiplications; eliminates serial Python unrolling loops.
- **`CompositeFlow` & `ZMatrixBijector`**:
  - Differentiable mapping between internal coordinates (bonds, angles, torsions) and 3D Cartesian coordinates via the Natural Extension Reference Frame (NeRF) algorithm.
- **`compute_cartesian_dihedrals(coords, quadruplets)` & `compute_torsion_rotamer_loss(phi, periodicity=3)`**:
  - Vectorized 3D dihedral extraction for explicit `.mol2` bond graph quadruplets $(a, b, c, d)$.
  - Autograd-safe: applies regularized safe norms (`.maximum(1e-8).sqrt()`) on normal vectors to eliminate division-by-zero singularities on collinear $180^\circ$ bond angles.
  - Evaluates 3-fold Fourier rotamer potential $J_{\rm tor}(\phi) = \frac{1}{BK}\sum (1 + \cos(3\phi))$ with global minima at trans ($\pm 180^\circ$) and gauche ($\pm 60^\circ$).

### 3.4 `generator.py` — Variational Boltzmann Generator with Torsional Biasing
- **`BoltzmannGenerator`**:
  - `__init__(flow, energy_fn, prior=None, temperature_k=300.0, learning_rate=0.01, batch_size=64, w_torsion=0.0, dihedral_quadruplets=None)`: Initializes optimizer, device origin pool, realized temperature/torsion buffers, and bond graph quadruplets.
  - `compute_loss(z, origin=None)`: Evaluates Reverse-KL loss $\mathcal{L}(\theta) = \beta U(f_\theta(z)) - \log p_z(z) - \log |\det J| + w_{\rm tor} J_{\rm tor}$.
    - **Dual-Path Execution**:
      - `CompositeFlow`: Directly slices internal coordinate torsions $\phi$ with zero cross products and zero division-by-zero risk.
      - `Base2CartesianFlow`: Evaluates safe regularized Cartesian dihedrals for explicit `.mol2` bond graph quadruplets.
  - `train(steps=100, batch_size=64, verbose=False)`: JIT-compiled optimization loop over flow parameters using `Adam` or `Muon`.
  - `sample(n_samples=1, return_all_pad=False, mcmc_steps=0, mcmc_step_size=0.1)`: Draws equilibrium configurations and automatically slices dummy padding sites to return real molecular atoms $(B, N_{\rm real}, 3)$. Dispatches to `sample_relaxed` when `mcmc_steps > 0`.
  - `log_prob(x)`: Evaluates exact generated density $\log q_\theta(\mathbf{x}) = \log p_z(f^{-1}(\mathbf{x})) + \log |\det J_{f^{-1}}(\mathbf{x})|$.

### 3.5 Latent Space MCMC Relaxation Engine (`generator.sample_relaxed`)
- **Motivation & Principle**:
  While normalizing flows generate one-shot configurations rapidly, high-dimensional target distributions inevitably contain rare steric overlap outliers. Rather than running costly gradient descent in 3D configuration space, MCMC is performed directly in the Gaussian latent space $\mathcal{Z}$.
- **Target Latent Distribution & Metropolis Criterion**:
  The exact Boltzmann equilibrium distribution in latent space is $p_Z(\mathbf{z}) \propto \exp(-E_{\rm eff}(\mathbf{z}))$, where:
  $$E_{\rm eff}(\mathbf{z}) = \beta U(f_\theta(\mathbf{z})) - \log p_0(\mathbf{z}) - \log \left| \det J_{f_\theta}(\mathbf{z}) \right|$$
  Perturbations $\mathbf{z}' = \mathbf{z} + s \, \boldsymbol{\eta}$ ($\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$) are accepted with exact Metropolis probability:
  $$p_{\rm acc} = \min(1, \exp(-\Delta E)) = \exp\left(\min(0, -(E_{\rm eff}(\mathbf{z}') - E_{\rm eff}(\mathbf{z})))\right)$$
- **Numerical Protections & Dyadic Architecture**:
  - **IEEE 754 Exponent Capping**: Uses `(-delta_e).minimum(0.0).exp()` to eliminate `inf` overflow on massive energy relief steps ($\Delta E \ll 0$).
  - **Dynamic Stochasticity**: Preserves unjitted outer Python Markov loops to avoid freezing pseudo-random number generator buffers.
  - **Base-2 Batched Tensor Parallelism**: Vectorized across all $B$ configurations simultaneously via `.where()` conditional selection.

---

## 4. Base-2 Dyadic Optimization & Power-of-2 Architecture

To maximize GPU hardware efficiency, minimize kernel compilation overhead, and ensure seamless `@TinyJit` execution in tinygrad, the engine strictly enforces base-2 dyadic optimization across data pipelines, energy evaluators, and normalizing flows:

- **7 Power-of-2 Buckets**: All 20 materials collapse into $N_{\rm pad} \in \{1, 2, 4, 8, 16, 32, 64\}$, shrinking kernel compilation graphs to just 7 structural topologies. Arbitrary molecular topologies share identical static tensor shapes, eliminating combinatorial JIT recompilations and graph cache thrashing.
- **GPU Base-2 Matrix Alignment**: Eliminates warp loop peeling and non-aligned strides in pairwise matrix tensor reductions. Tensor operations align with warp boundaries (32/64 threads) and SIMD register lanes, achieving maximum memory bandwidth coalescing and ALU utilization.
- **Exact Observable Extraction**: Padded dummy sites have zero interactions ($\epsilon_i = 0, q_i = 0, \text{wall}_i = 0$) and are masked out via upper-triangular and atom reduction masks (`self.is_real_atom`). Generated configurations are automatically sliced back to $N_{\rm real}$ physical atoms on export (`generator.sample()`), preserving exact statistical mechanical observables.

