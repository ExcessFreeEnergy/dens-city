"""
dens-city Physical Pipelines:
1. Water (SCAN, RPBE-D3, TIP4P, Nanoconfinement, Binodal, Hyper-DFT)
2. CO2 (TraPPE, Buckingham Exp-6 Gaussian, Fisher-Widom, Widom lines, COLN 3D Orientational Order)
3. Electrolytes (RPM 1:1, 2:1 Multivalent Overcharging & Charge Inversion, Differential Capacitance)
4. CO2/H2O Binary Mixture (Solvation Free Energy, Mutual Solubility, Competitive Pore Filling)
5. Nitrogen (Linear Diatomic N2 Flue Gas Separation, Adsorption Selectivity)
6. Methane (CH4 TraPPE United-Atom Shale Organic Confinement, Binodal, Enhanced Gas Recovery)
7. Montmorillonite Clay (Aluminosilicate Slit Pores, Hydration & Osmotic Swelling Pressures)
8. Liquid Crystals & Patchy Particles (Nematic Director Fields, Isotropic-Nematic Binodal)
9. Argon (Pure Lennard-Jones Baseline, FMT + Barker-Henderson/WCA MCA Coexistence)
10. Interfaces & Wetting (Hydrophobic/Hydrophilic Planar Wetting, Capillary Drying, Lum-Chandler-Weeks Crossover)
11. Helium-4 (^4He Quantum NQE Fluid, Feynman-Hibbs Effective Potential, Non-Freezing Zero-Point Stability)
12. Room-Temperature Ionic Liquids ([BMIM][PF6] Steric Crowding, Camel-Shaped Capacitance, Alternating Overscreening)
13. Flexible Macromolecules (Polyethylene N>100 Chains, Wertheim TPT1 Connectivity, Wall Entropic Depletion)
14. Liquid Metals (Liquid Gallium/Mercury Jellium Coupling, Friedel Density Oscillations, Ultra-High Surface Tension)
15. Azeotropes (Water-Ethanol Binary Mixture, Non-Ideal VLE, Minimum-Boiling Azeotrope at 95.6 wt% Ethanol)
16. Amphiphilic Surfactants (SDS Self-Assembly, Critical Micelle Concentration CMC=8.2 mM, Aggregation N=62)
17. Associating 1D Fluids (Hydrogen Fluoride HF (HF)_6 Cyclic Hexamers, Vapor Compressibility Z < 0.5)
18. Colloidal Depletion (Asakura-Oosawa Entropic Attraction, Pure Entropic Demixing Without Energetic Potential)
19. Supercooled Glasses (Kob-Andersen 80/20 Non-Additive LJ, Split Second Peak in g_AA(r), Avoided Crystallization)
20. Steric Shielding (Sulfur Hexafluoride SF6 Octahedral Fluorine Cage, Giant Excluded Volume, Triple Point T_t=222.35 K)
21. Quantum Water (SCAN, RPBE-D3 Ab Initio Multiscale Benchmark)
22. Quantum CO2 (PBE-D3, BLYP-D3 Supercritical Crossover Benchmark)
"""

from dens_city.pipelines import (
    argon,
    associating_1d,
    azeotropes,
    clay_pore,
    co2,
    co2_water,
    colloids,
    electrolytes,
    fluorinated,
    glasses,
    interfaces,
    ionic_liquids,
    liquid_crystals,
    liquid_metals,
    methane,
    nitrogen,
    polymers,
    quantum,
    quantum_co2,
    quantum_water,
    surfactants,
    water,
)

__all__ = [
    "argon",
    "associating_1d",
    "azeotropes",
    "clay_pore",
    "co2",
    "co2_water",
    "colloids",
    "electrolytes",
    "fluorinated",
    "glasses",
    "interfaces",
    "ionic_liquids",
    "liquid_crystals",
    "liquid_metals",
    "methane",
    "nitrogen",
    "polymers",
    "quantum",
    "quantum_co2",
    "quantum_water",
    "surfactants",
    "water",
]
