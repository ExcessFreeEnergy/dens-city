# dens-city Test Data & Molecular Structures

This directory contains canonical Tripos `.mol2` molecular structure files and complete force field parameter lookup tables for the 20 fundamental benchmark fluid systems and materials simulated in `dens-city`.

---

## 1. Credit & Citations

### 1.1 FreeSolv (Mobley Lab)
Small organic molecule models, GAFF parameterizations, Amber `.prmtop`/`.frcmod`, GROMACS `.top`, and LAMMPS `.lmp` files in this directory are sourced from the **FreeSolv Database** (v0.52):
- **Repository**: [https://github.com/MobleyLab/FreeSolv](https://github.com/MobleyLab/FreeSolv)
- **Primary Publication**:
  > Mobley, D. L., & Guthrie, J. P. (2014). *FreeSolv: a database of experimental and calculated hydration free energies, with input files.* **Journal of Computer-Aided Molecular Design**, 28(7), 711–720. [doi:10.1007/s10822-014-9747-x](https://doi.org/10.1007/s10822-014-9747-x)
- **Calculated Hydration Reference**:
  > Matos, G. D. R., Kyu, D. Y., Loeffler, H. H., Chodera, J. D., Shirts, M. R., & Mobley, D. L. (2017). *Approaches for Calculating Solvation Free Energies and Enthalpies That Reconcile with Experiment.* **Journal of Chemical & Engineering Data**, 62(5), 1559–1569. [doi:10.1021/acs.jced.7b00104](https://doi.org/10.1021/acs.jced.7b00104)

### 1.2 Fluid & Molecular Force Field Literature
The inorganic, ionic, polymeric, and coarse-grained models follow standard statistical mechanics literature:
- **Water (SPC/E)**: Berendsen et al., *J. Phys. Chem.* 91, 6269 (1987).
- **Carbon Dioxide (TraPPE)**: Potoff & Siepmann, *AIChE J.* 47, 1676 (2001).
- **Nitrogen (TraPPE)**: Potoff & Siepmann, *AIChE J.* 47, 1676 (2001).
- **Argon (LJ 12-6)**: Michels et al., *Physica* 15, 627 (1949).
- **Restricted Primitive Model (RPM NaCl / CaCl2)**: Stillinger & Lovett, *J. Chem. Phys.* 49, 1991 (1968).
- **Liquid Crystals (5CB)**: Maier-Saupe mean-field & GAFF biphenyl mesogens.
- **Surfactants (SDS)**: Shelley et al., *J. Phys. Chem. B* 105, 4464 (2001).

---

## 2. Molecular Structure Files (.mol2)

| # | System / Material | File | Source / Origin | Description |
|---|---|---|---|---|
| 1 | **Water** | [`water.mol2`](./water.mol2) | Canonical SPC/E | Rigid 3-site planar geometry ($r_{\rm OH} = 1.0\,\text{Å}$, $\theta = 109.47^\circ$, $q_{\rm O} = -0.8476\,e$, $q_{\rm H} = +0.4238\,e$) |
| 2 | **Nitrogen** | [`nitrogen.mol2`](./nitrogen.mol2) | TraPPE | Linear rigid diatomic ($r_{\rm NN} = 1.10\,\text{Å}$) |
| 3 | **Methane** | [`methane.mol2`](./methane.mol2) | FreeSolv (`mobley_9055303`) | GAFF all-atom tetrahedral $CH_4$ |
| 4 | **Carbon dioxide** | [`carbon_dioxide.mol2`](./carbon_dioxide.mol2) | TraPPE | Linear rigid triatomic ($r_{\rm CO} = 1.16\,\text{Å}$, $q_{\rm C} = +0.70\,e$, $q_{\rm O} = -0.35\,e$) |
| 5 | **Argon** | [`argon.mol2`](./argon.mol2) | Noble Gas LJ | Monoatomic sphere ($\sigma = 3.405\,\text{Å}$, $\epsilon/k_B = 119.8\,\text{K}$) |
| 6 | **Sodium chloride** | [`sodium_chloride.mol2`](./sodium_chloride.mol2) | 1:1 RPM | Contact ion pair ($\text{Na}^+ / \text{Cl}^-$, $r = 2.36\,\text{Å}$) |
| 7 | **Calcium chloride** | [`calcium_chloride.mol2`](./calcium_chloride.mol2) | 2:1 RPM | Triatomic electrolyte ($\text{Ca}^{2+} / 2\,\text{Cl}^-$, $r = 2.70\,\text{Å}$) |
| 8 | **n-Decane** | [`n_decane.mol2`](./n_decane.mol2) | FreeSolv (`mobley_2197088`) | GAFF all-atom linear alkane $C_{10}H_{22}$ |
| 9 | **Neopentane** | [`neopentane.mol2`](./neopentane.mol2) | FreeSolv (`mobley_1261349`) | GAFF all-atom spherical branched alkane $C(CH_3)_4$ |
| 10 | **Polyethylene** | [`polyethylene.mol2`](./polyethylene.mol2) | All-trans Oligomer | Eicosane $C_{20}H_{42}$ zigzag chain with GAFF partial charges |
| 11 | **Methanol** | [`methanol.mol2`](./methanol.mol2) | FreeSolv (`mobley_1636752`) | GAFF all-atom $CH_3OH$ |
| 12 | **Ammonia** | [`ammonia.mol2`](./ammonia.mol2) | FreeSolv (`mobley_5631798`) | GAFF all-atom pyramidal $NH_3$ |
| 13 | **Hydrogen fluoride** | [`hydrogen_fluoride.mol2`](./hydrogen_fluoride.mol2) | Associating 1D | Diatomic polar HF ($r_{\rm HF} = 0.917\,\text{Å}$, $q = \pm 0.45\,e$) |
| 14 | **Benzene** | [`benzene.mol2`](./benzene.mol2) | FreeSolv (`mobley_3053621`) | GAFF all-atom aromatic $C_6H_6$ ring |
| 15 | **5CB** | [`5cb.mol2`](./5cb.mol2) | Nematic LC Mesogen | 4-Cyano-4'-pentylbiphenyl with cyano group, biphenyl core, and pentyl tail |
| 16 | **Sodium dodecyl sulfate** | [`sodium_dodecyl_sulfate.mol2`](./sodium_dodecyl_sulfate.mol2) | Surfactant Model | Anionic headgroup ($-\text{OSO}_3^-$) + $\text{Na}^+$ counterion + $C_{12}$ lipophilic tail |
| 17 | **Sulfur hexafluoride** | [`sulfur_hexafluoride.mol2`](./sulfur_hexafluoride.mol2) | Octahedral Fluid | Octahedral $SF_6$ ($r_{\rm SF} = 1.56\,\text{Å}$, $q_{\rm S} = +1.50\,e$, $q_{\rm F} = -0.25\,e$) |
| 18 | **Acetone** | [`acetone.mol2`](./acetone.mol2) | FreeSolv (`mobley_3867265`) | GAFF all-atom polar solvent $(CH_3)_2CO$ |
| 19 | **Colloidal hard sphere** | [`colloidal_hard_sphere.mol2`](./colloidal_hard_sphere.mol2) | Asakura-Oosawa | Coarse-grained excluded-volume sphere ($D = 15.0\,\text{Å}$) |
| 20 | **Hydrogen** | [`hydrogen.mol2`](./hydrogen.mol2) | Quantum / Diatomic | Molecular $H_2$ ($r_{\rm HH} = 0.7414\,\text{Å}$) |

---

## 3. Force Field Parameter Files & LJ Lookups

Because standard Tripos `.mol2` files record the atom types (column 6 in `@<TRIPOS>ATOM`) rather than hardcoded pairwise $\sigma$ and $\epsilon$ values, this directory packages complete force field parameter files:

### 3.1 Parameter Lookup Tables
- **[`gaff.dat`](./gaff.dat)**: Standard AMBER format parameter file with `MASS` and `NONBON` sections mapping each GAFF atom type to $R_{\min}/2$ and $\epsilon$.
  - Conversion relation: $\sigma = 2 \cdot (R_{\min}/2) / 2^{1/6} \approx 1.7818 \cdot (R_{\min}/2)$.
- **[`forcefield_parameters.json`](./forcefield_parameters.json)**: Machine-readable JSON dictionary mapping every atom type directly to $\sigma$ (in Å and nm), $\epsilon$ (in kcal/mol, kJ/mol, and Kelvin), atomic number, mass, and descriptions.
- **[`forcefield_parameters.csv`](./forcefield_parameters.csv)**: Tabular CSV of the nonbonded Lennard-Jones parameter database.

### 3.2 Simulation Engine Topology & Parameter Packages
The subdirectories contain engine-ready topology and parameter sets:
- **[`amber/`](./amber/)**: Amber topology files (`.prmtop`), modification parameter files (`.frcmod`), and coordinate files (`.inpcrd`) for all FreeSolv benchmark fluids.
- **[`gromacs/`](./gromacs/)**: GROMACS topologies (`.top`) with explicit `[ atomtypes ]` sigma/epsilon directives and structure files (`.gro`).
- **[`lammps/`](./lammps/)**: LAMMPS input scripts (`.input`) and data files (`.lmp`) with `Masses`, `Bond Coeffs`, `Angle Coeffs`, and `Pair Coeffs`.

---

## 4. Directory Management & Git Tracking

Unlike the larger raw cache in `data/`, the `test_data/` directory is **tracked in Git** to provide self-contained, reproducible unit and integration tests across all benchmark pipelines.
