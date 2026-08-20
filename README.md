# dens-city: Pure First-Principles Classical Density Functional Theory (cDFT) in tinygrad

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![tinygrad: >=0.13.0](https://img.shields.io/badge/tinygrad-0.13.0+-orange.svg)](https://github.com/tinygrad/tinygrad)

`dens-city` is a pure statistical mechanics Classical Density Functional Theory (cDFT) simulation platform built from scratch in `tinygrad`.

---

## 1. First-Principles Physics Architecture

All physical properties emerge strictly from variational minimization of the Grand Potential functional $\Omega[\rho]$:
$$\Omega[\rho] = \mathcal{F}_{\rm ideal}[\rho] + \mathcal{F}_{\rm FMT}^{\rm ex}[\rho] + \mathcal{F}_{\rm att}^{\rm ex}[\rho] + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]$$

- **Zero Hardcoded Parameters**: Every molecular fluid's steric diameter $\sigma_{\rm eff} = (\sum \sigma_i^3)^{1/3}$ and cohesive well depth $\epsilon_{\rm eff} = \sum \epsilon_i$ are derived directly from arbitrary input `.mol2` files and force field definitions.
- **Thermodynamic Consistency**: Bulk state variables $(\rho_{\rm bulk}, \mu, P)$ are derived dynamically from the Carnahan-Starling + Mean-Field Equation of State (EOS) root solver.
- **Irving-Kirkwood Virial Observables**: Wall contact pressures and surface forces are evaluated via exact statistical mechanical momentum balance integrals:
  $$P_{\rm wall} = -\int_0^{L_z/2} \rho(z) \frac{d V_{\rm ext}(z)}{dz} \, dz$$
- **Anti-Aliased Weight Functions**: Analytical continuous cell integrals over Rosenfeld Fundamental Measure Theory (FMT) 1D planar weight kernels ($w_3, w_2, w_1, w_0, \mathbf{w}_{v2}, \mathbf{w}_{v1}$) and WCA attractive dispersion kernels.

---

## 2. Numerical Implementation & Steric Masking (The $0 \times \infty$ NaN Trap)

### The Problem:
At impenetrable steric boundaries ($z \le \sigma_{\rm wall}/2$), the physical external potential diverges to infinity ($V_{\rm ext} \to \infty$). In the Boltzmann distribution $\rho = \rho_{\rm bulk} \exp(-\beta V_{\rm ext})$, the fluid density is strictly zero ($\rho = 0.0$).

In IEEE 754 floating-point arithmetic on GPUs:
- $0.0 \times \infty = \text{NaN}$ (in the external potential loss term $f_{\rm ext} = \int \rho(z) V_{\rm ext}(z) dz$)
- $0.0 \times \ln(0.0) = \text{NaN}$ (in the ideal gas entropy loss term $f_{\rm ideal} = \int \rho \ln(\rho/\rho_b) dz$)

The moment `NaN` enters the loss function, autograd backpropagation destroys all optimizer tensors.

### The First-Principles Solution:
1. **Massive Finite Brick-Wall Barrier**: Hard boundaries are set to $V_{\max} = 10^6\,k_B T$, which evaluates $\exp(-10^6) \equiv 0.0$ in float32 without floating-point overflow.
2. **Physical Steric Masking**: Using tinygrad's `.where()` operator, energy evaluations are masked out inside the hard core where density is analytically zero:
   ```python
   # Fluid-accessible domain mask
   self.steric_mask = (self.v_ext < 50.0).contiguous()
   self.v_ext_masked = self.steric_mask.where(self.v_ext, 0.0).contiguous()

   # Masked external potential energy
   f_ext = (rho * self.v_ext_masked).sum() * self.dz

   # Masked ideal gas free energy
   rho_safe = rho.maximum(1e-15)
   f_id_density = rho * (rho_safe / self.bulk_density).log() - (rho - self.bulk_density)
   f_ideal = self.steric_mask.where(f_id_density, 0.0).sum() * self.dz
   ```
3. **Boltzmann Asymptotic Initialization**:
   $\psi_0(z) = -\beta V_{\rm ext}(z)$, guaranteeing zero fluid density penetration inside steric walls from step 0.

---

## 3. Quickstart & CLI Usage

Always activate the local virtual environment with `uv`:
```bash
source .venv/bin/activate

# Run single material
python scripts/run_cdft.py --materials argon

# Run multi-material sweep
python scripts/run_cdft.py --materials argon water methane 5cb

# Run all 20 benchmark fluids with JIT acceleration
python scripts/run_cdft.py --materials all --steps 50

# Specify thermodynamic pressure for Equation of State (EOS) root solving
python scripts/run_cdft.py --materials benzene --pressure 1.01325
```

---

## 4. Automated Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## 5. License
GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
