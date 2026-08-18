# dens-city Compute Steps & Theoretical Pipeline

This document details the 20 fundamental compute steps, physical formulas, statistical mechanics operations, and corresponding source implementations in `dens-city`.

---

## Complete Compute Steps Overview

| Step | Formula / Operation | Implemented In | Description |
|---|---|---|---|
| **1. Grand Potential** | $\Omega[\rho] = \mathcal{F}^{\rm id} + \mathcal{F}^{\rm ex} + \int dz \, \rho(z)(V^{\rm ext} - \mu)$ | `src/dens_city/solver/thermo_integration.py` | Grand canonical potential functional minimization |
| **2. Euler–Lagrange** | $c^{(1)}(z) = \ln(\zeta^{-1}\Lambda^3\rho) + \beta(V^{\rm ext} - \mu)$ | `src/dens_city/envs/dens_city_env.c` | Exact Euler-Lagrange self-consistency relation |
| **3. Short-Range LMFT** | $c^{(1)}(z) = c_{\rm R}^{(1)} - \beta\Delta\mu^{\rm SL} + \beta\phi^{\rm R}$ | `src/dens_city/envs/dens_city_env.c` | Local Molecular Field Theory Coulomb splitting |
| **4. 1D Restructuring** | $\phi^{\rm R}(z) = \phi(z) + \frac{1}{L_z} \sum_{k \neq 0} \frac{4\pi}{k^2} \tilde{n}(k) e^{ikz} e^{-k^2/4\kappa^2}$ | `src/dens_city/envs/dens_city_env.c` | 1D Fourier electrostatic restructuring potential |
| **5. Stillinger–Lovett** | $\Delta\mu^{\rm SL} = \frac{1}{2\beta\rho_b\kappa^{-3}\sqrt{\pi}^3}\left(\frac{\epsilon-1}{\epsilon}\right) - \frac{2\rho_b^2}{3\kappa^{-3}\sqrt{\pi}}$ | `src/dens_city/core/engine.cpp` | Bulk thermodynamic Stillinger-Lovett chemical potential shift |
| **6. 3D Long-Range Ewald** | $U_{\rm recip} = \frac{1}{2V} \sum_{\mathbf{k} \neq 0} \frac{4\pi}{k^2} e^{-k^2/4\alpha^2} \|\tilde{\rho}(\mathbf{k})\|^2 - \frac{\alpha}{\sqrt{\pi}}\sum_i q_i^2$ | `src/dens_city/core/cuda_kernels.cu` | Reciprocal-space 3D Ewald sum with GPU shared memory |
| **7. Fundamental Measure Theory** | $\Phi_{\text{hs}} = -n_0 \ln(1-n_3) + \frac{n_1 n_2 - \mathbf{n}_1 \cdot \mathbf{n}_2}{1-n_3} + \frac{n_2^3 - 3n_2 \mathbf{n}_2^2}{24\pi(1-n_3)^2}$ | `src/dens_city/solver/fmt.py` | White Bear mark II tensor hard-sphere FMT functional |
| **8. Barker-Henderson Diameter** | $d(T) = \int_0^{r_{\rm min}} \left[1 - \exp\left(-\frac{u_0(r)}{k_B T}\right)\right] dr$ | `src/dens_city/solver/dispersion.py` | Temperature-dependent effective hard-core diameter |
| **9. Slab Attractive Dispersion** | $\bar{u}_{\rm att}(\|z\|) = 2\pi \int_{\|z\|}^{r_{\rm cut}} r \, u_{\rm att}(r) \, dr$ | `src/dens_city/solver/dispersion.py` | 1D slab-integrated attractive dispersion potential |
| **10. COLN Operator** | $c_1(x, \theta, \phi) = \sum_{l,m} c_{ml}(x, \bar{\rho}) Y_{ml}(\theta, \phi) \cdot [1 + \hat{\rho}(\theta, \phi)]$ | `src/dens_city/models/coln.py` | Convoluted Operator Learning on $S^2$ spherical harmonics |
| **11. Nematic Order $S$** | $S_{\rm order}(z) = \frac{1}{\bar{\rho}(z)} \int d\Omega \, \rho(z, \theta, \phi) \left(\frac{3\cos^2\theta - 1}{2}\right)$ | `src/dens_city/pipelines/co2/supercritical.py` | Second-rank orientational order parameter profile |
| **12. Buckingham Exp-6** | $u(r) = \frac{q_i q_j}{4\pi\epsilon_0 r}{\rm erf}\left(\frac{r}{\sqrt{2}\sigma_{ij}}\right) + A_{ij}e^{-B_{ij}r} - \frac{C_{ij}}{r^6}$ | `src/dens_city/core/cuda_kernels.cu` | Born-Mayer Buckingham exp-6 soft-repulsion pair style |
| **13. Hyper-DFT** | $\rho_{\rm H}(z) = \rho_{\rm H}^{(1)}(z; [\rho_{\rm O}], T)$ | `src/dens_city/envs/train.py` | Sub-molecular conditional stoichiometric density mapper |
| **14. Line Integration** | $\mathcal{F}^{\rm ex} = -k_B T \int_0^1 d\lambda \int dz \, c^{(1)}(z; [\lambda\rho], T) \rho(z)$ | `src/dens_city/solver/thermo_integration.py` | Path thermodynamic line integration for free energy |
| **15. Bulk Pressure EOS** | $P(\rho_b, T) = k_B T \rho_b(1 - c^{(1)}) - \frac{\mathcal{F}^{\rm ex}}{V}$ | `src/dens_city/solver/thermo_integration.py` | Exact virial-free energy equation of state |
| **16. Structure Factor $S(k)$** | $S(k) = \frac{1}{1 - \rho_b \hat{c}^{(2)}(k)}$ where $c^{(2)} = -\frac{\delta c^{(1)}}{\delta \rho}$ | `src/dens_city/solver/correlation.py` | Static structure factor via Fourier OZ inversion |
| **17. Effective Pressure** | $\tilde{P}(H) = P + \Pi(H) = -\int dz \, \rho(z) \frac{dV_{\rm wall}}{dz}$ | `src/dens_city/pipelines/water/confinement.py` | Contact-value theorem & disjoining pressure in slits |
| **18. Fisher–Widom Line** | Crossover of correlation decay: $\alpha_0 = \tilde{\alpha}_0$ | `src/dens_city/pipelines/co2/supercritical.py` | Monotonic to oscillatory asymptotic correlation crossover |
| **19. Widom Lines** | Maxima of correlation length $\xi$ and compressibility $\chi_T$ | `src/dens_city/pipelines/co2/supercritical.py` | Supercritical pseudo-boiling crossover ridges |
| **20. Binodal Solver** | Picard iteration with Anderson acceleration | `src/dens_city/solver/picard_solver.py` | Fast convergent liquid-vapor coexistence solver |
