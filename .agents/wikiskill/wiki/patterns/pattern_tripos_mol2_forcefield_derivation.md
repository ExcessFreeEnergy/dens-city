# Pattern: Strict First-Principles Derivation of Molecular & Force Field Parameters

## Summary
- **Problem**: Inconsistencies across fluids or unphysical simulation results when introducing new molecules or testing datasets.
- **Root Cause**: Hardcoding fluid properties (\(\sigma, \epsilon, q\)), lookup aliases, or empirical constants in code rather than parsing them dynamically from input `.mol2` files and standard force field parameter definitions (GAFF, OPLS, TraPPE).
- **Actionable Fix**: Derive all microscopic and statistical mechanical parameters strictly from input `.mol2` atomic geometries, Sybyl atom types, and force field tables.
- **Related Skills / Modules**: `cdft-physics`, `dens_city.utils.materials`

## Deep Root Cause Analysis
In high-throughput screening and generative pipelines, input molecules range from simple Lennard-Jones fluids (Argon, Krypton) to complex polycyclic aromatic hydrocarbons (conjugated OLEDs) and flexible electrolytes.
If a module relies on dictionary lookups like `if mol_name == 'water': sigma = 3.16`, it silently fails when evaluating novel generated compounds or arbitrary FreeSolv molecules, either falling back to bogus defaults or crashing.

## Verified Solution & Action Rules
1. Parse 3D coordinates, partial charges \(q_i\), and Sybyl atom types directly from the Tripos `.mol2` format.
2. Map atom types to Lennard-Jones parameters (\(\sigma_i, \epsilon_i\)) via standard force field databases.
3. Compute molecular volume, effective spherical diameter \(\sigma_{\rm eff}\), and dipole moments dynamically from atomic coordinates.
4. Solve the bulk Equation of State (EOS) numerically at runtime to find consistent state variables \((\rho_{\rm bulk}, \mu, P)\).

```python
# Verified Implementation Pattern
def load_material_from_mol2(mol2_path: str, temperature: float) -> Material:
    atoms, bonds, charges, types = parse_tripos_mol2(mol2_path)
    sigma_i, epsilon_i = match_forcefield_parameters(types)
    sigma_eff = compute_effective_molecular_diameter(atoms, sigma_i)
    rho_bulk, mu = solve_bulk_eos(temperature, sigma_eff, epsilon_i)
    return Material(
        name=Path(mol2_path).stem,
        temperature=temperature,
        sigma_eff=sigma_eff,
        rho_bulk=rho_bulk,
        chemical_potential=mu,
    )
```

## Anti-Patterns to Avoid
- ❌ **Anti-Pattern**: Hardcoding fluid-specific `if/elif` branches or fudge factors in physics routines.
- ❌ **Anti-Pattern**: Assuming a fixed bulk density \(\rho_{\rm bulk} = 0.84\) without solving the EOS for the specified temperature and potential.
