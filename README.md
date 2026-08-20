# dens-city: Pure First-Principles Classical Density Functional Theory (cDFT) in tinygrad

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)

`dens-city` is a pure statistical mechanics Classical Density Functional Theory (cDFT) engine built from scratch in `tinygrad`.

---

## 1. First-Principles Physics Architecture

All physical properties emerge strictly from variational minimization of the Grand Potential functional $\Omega[\rho]$:
$$\Omega[\rho] = \mathcal{F}_{\rm ideal}[\rho] + \mathcal{F}_{\rm FMT}^{\rm ex}[\rho] + \mathcal{F}_{\rm att}^{\rm ex}[\rho] + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]$$

- **Zero Hardcoded Parameters**: Steric diameter $\sigma_{\rm eff} = (\sum \sigma_i^3)^{1/3}$ and cohesive well depth $\epsilon_{\rm eff} = \sum \epsilon_i$ derive directly from arbitrary input `.mol2` files and force field definitions.
- **Thermodynamic Consistency**: Bulk state variables $(\rho_{\rm bulk}, \mu, P)$ derive dynamically from the Rosenfeld FMT / Percus-Yevick compressibility Equation of State (EOS) root solver.
- **Irving-Kirkwood Virial Observables**: Wall contact pressures and surface forces evaluate via exact momentum balance integrals over the interaction domain:
  $$P_{\rm wall} = -\int_0^{z_{\rm bulk}} \rho(z) \frac{d V_{\rm ext}(z)}{dz} \, dz$$
- **Anti-Aliased Weight Functions**: Analytical continuous cell integrals over Rosenfeld Fundamental Measure Theory (FMT) 1D planar weight kernels ($w_3, w_2, w_1, w_0, \mathbf{w}_{v2}, \mathbf{w}_{v1}$) and WCA attractive dispersion kernels.

---

## 2. Why GCMC Chokes on Long-Range Forces

Grand Canonical Monte Carlo (GCMC) simulates discrete particles through stochastic atom insertions, deletions, and displacements:
- **Reciprocal-Space Recalculation**: In Ewald summation (or Particle-Mesh Ewald), every particle insertion or deletion alters the global structure factor $\sum_j q_j e^{i \mathbf{k} \cdot \mathbf{r}_j}$. Updates across thousands of trial moves per second create a massive computational bottleneck.
- **Neutrality Violations**: Insertion of an isolated charged ion breaks electroneutrality in the box, which requires artificial background neutralizing plasma or fractional insertion schemes.
- **The Overlap Wall**: Insertion of a full molecule with Lennard-Jones cores and partial charges into a dense polar fluid (such as liquid water) suffers a $>99.9\%$ rejection rate, which demands millions of failed trial steps for a handful of accepted configurations.

---

## 3. Why cDFT Solves This for Free

In cDFT, there are no particles, no trial moves, and no discrete insertions. The system contains only a continuous, smooth charge density field:
$$\rho_q(\mathbf{r}) = \sum_i q_i \rho_i(\mathbf{r})$$

The long-range electrostatic energy is the double integral:
$$\mathcal{F}_{\text{coul}}[\rho] = \frac{1}{2} \iint \frac{\rho_q(\mathbf{r}) \rho_q(\mathbf{r}')}{4\pi \varepsilon_0 \varepsilon_r |\mathbf{r} - \mathbf{r}'|} \, d\mathbf{r} \, d\mathbf{r}'$$

Instead of pairwise sums over periodic images, this integral is mathematically identical to the classical Poisson equation:
$$\nabla^2 \phi(\mathbf{r}) = -\frac{\rho_q(\mathbf{r})}{\varepsilon_0 \varepsilon_r}$$

### Map to tinygrad
- **1D Slit Pores**: A planar sheet of charge has a constant electric field. The 1D Green's function is $v_C(z) = -2\pi |z|$, which reduces to a 1D convolution or direct cumulative integral across the grid tensor.
- **3D Grids**: Poisson solves in a single step in Fourier space. In $k$-space, the Laplacian $\nabla^2$ becomes $-k^2$:
  $$\tilde{\phi}(\mathbf{k}) = \frac{4\pi}{\varepsilon_0 \varepsilon_r k^2} \tilde{\rho}_q(\mathbf{k})$$
  A forward 3D FFT, element-wise vector division by $k^2$, and an inverse 3D FFT solve the exact, infinite long-range field across the full periodic box in milliseconds on GPU.

> [!NOTE]
> **Molecular Flexibility**: The current engine solves rigid molecules (water, benzene, methane) efficiently. For large, flexible drug molecules (such as ibuprofen or long-chain polymers), conformational degrees of freedom demand rotational averages over bond-angle conformations, which remains an active research frontier.

---

## 4. Quickstart & CLI Usage

Activate the virtual environment:
```bash
source .venv/bin/activate

# Single material simulation
python scripts/run_cdft.py --materials argon

# Multi-material sweep
python scripts/run_cdft.py --materials argon water methane 5cb

# Run all 20 benchmark fluids with JIT acceleration
python scripts/run_cdft.py --materials all --steps 50

# Specify thermodynamic pressure for Equation of State (EOS) root solve
python scripts/run_cdft.py --materials benzene --pressure 1.01325
```

---

## 5. Automated Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## 6. Citations

- A. T. Bui, S. J. Cox, "Dielectrocapillarity for exquisite control of fluids", *arXiv:2503.09855* (2025).
- A. T. Bui, S. J. Cox, "Learning classical density functionals for ionic fluids", *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)
- A. T. Bui, S. J. Cox, "Ab initio classical density functional theory with neural functionals", *arXiv:2603.20493* (2026).
- J. Yang, R. Pan, J. Sun, J. Wu, "High-Dimensional Operator Learning for Molecular Density Functional Theory", *arXiv:2411.03698* (2024). [doi:10.48550/arxiv.2411.03698](https://doi.org/10.48550/arxiv.2411.03698)
- R. Roth, "Fundamental measure theory for hard-sphere mixtures: a review", *Journal of Physics: Condensed Matter* **22**, 063102 (2010). [doi:10.1088/0953-8984/22/6/063102](https://doi.org/10.1088/0953-8984/22/6/063102)

---

## 7. License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
