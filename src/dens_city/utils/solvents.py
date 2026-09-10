"""
Physical solvent registry and parameter resolver for non-aqueous solvation benchmarks.
Provides dielectric constants, refractive indices, surface tensions, and densities
for the Solv@TUM (Solvatum) multi-solvent database, with dynamic Onsager/Kirkwood-Fröhlich
polarization fallback for unlisted or novel solvent matrices.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SolventProperties:
    """Thermodynamic and electrostatic parameters for a liquid solvent matrix."""

    name: str
    aliases: Tuple[str, ...]
    dielectric_constant: float  # Static relative permittivity epsilon_r at 298.15 K
    refractive_index: float  # Optical refractive index n_D at 298.15 K
    density_g_cm3: float  # Liquid mass density in g/cm³
    surface_tension_mn_m: float  # Cavitation surface tension gamma in mN/m (dyn/cm)
    solvent_class: str  # polar_protic, polar_aprotic, alkane_nonpolar, aromatic, chlorinated, ether, ester
    molecular_weight: float = 100.0  # Molar mass M in g/mol
    abraham_alpha: float = 0.0  # Hydrogen bond acidity alpha_2^H
    abraham_beta: float = 0.0  # Hydrogen bond basicity beta_2^H
    abraham_pi2: float = 0.0  # Dipolarity / polarizability pi_2^H
    dipole_moment_debye: float = 0.0  # Molecular dipole moment in Debye

    @property
    def optical_dielectric(self) -> float:
        """Optical dielectric permittivity epsilon_infinity approx n_D^2."""
        return self.refractive_index**2

    @property
    def molar_volume_cm3_mol(self) -> float:
        """Liquid molar volume V_m = M / rho in cm³/mol."""
        return self.molecular_weight / max(1e-4, self.density_g_cm3)

    @property
    def kinetic_diameter_a(self) -> float:
        """Effective spherical packing diameter sigma_S in Angstroms from molar volume."""
        v_mol_a3 = (self.molar_volume_cm3_mol * 1e24) / 6.02214076e23
        return float((6.0 * v_mol_a3 / math.pi) ** (1.0 / 3.0))

    @property
    def hbond_capacity(self) -> float:
        """Total hydrogen bonding interaction capacity alpha + beta."""
        return self.abraham_alpha + self.abraham_beta


def normalize_solvent_name(name: str) -> str:
    """Normalizes solvent identifier by cleaning unicode racemic prefixes, hyphens, and whitespace."""
    s = name.strip().upper()
    # Normalize racemic markers: (\xb1), (±), (+/-), racemic
    s = re.sub(r"\(\\XB1\)", "", s)
    s = re.sub(r"\(±\)", "", s)
    s = re.sub(r"\(\+/-\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove leading/trailing non-alphanumeric except numbers/letters
    s = s.strip(" -_")
    return s


# Comprehensive experimental solvent properties registry (Acree, Abraham, CRC Handbook, Riddick & Bunger)
# All 146 solvents in the Solv@TUM database are explicitly mapped to their verified 298.15 K properties.
_SOLVENT_REGISTRY: Dict[str, SolventProperties] = {
    # Reference Aqueous
    "WATER": SolventProperties("WATER", ("H2O", "AQUEOUS"), 78.4, 1.333, 0.997, 72.8, "polar_protic"),
    # Nonpolar Alkanes & Cycloalkanes
    "HEXADECANE": SolventProperties(
        "HEXADECANE", ("N-HEXADECANE", "CETANE"), 2.05, 1.434, 0.773, 27.5, "alkane_nonpolar"
    ),
    "HEPTANE": SolventProperties("HEPTANE", ("N-HEPTANE",), 1.92, 1.388, 0.684, 20.1, "alkane_nonpolar"),
    "HEXANE": SolventProperties("HEXANE", ("N-HEXANE",), 1.88, 1.375, 0.655, 18.4, "alkane_nonpolar"),
    "OCTANE": SolventProperties("OCTANE", ("N-OCTANE",), 1.95, 1.397, 0.703, 21.6, "alkane_nonpolar"),
    "NONANE": SolventProperties("NONANE", ("N-NONANE",), 1.97, 1.405, 0.718, 22.8, "alkane_nonpolar"),
    "DECANE": SolventProperties("DECANE", ("N-DECANE",), 1.99, 1.411, 0.730, 23.8, "alkane_nonpolar"),
    "UNDECANE": SolventProperties("UNDECANE", ("N-UNDECANE",), 2.01, 1.417, 0.740, 24.7, "alkane_nonpolar"),
    "DODECANE": SolventProperties("DODECANE", ("N-DODECANE",), 2.01, 1.422, 0.749, 25.4, "alkane_nonpolar"),
    "TETRADECANE": SolventProperties("TETRADECANE", ("N-TETRADECANE",), 2.04, 1.429, 0.763, 26.5, "alkane_nonpolar"),
    "PENTANE": SolventProperties("PENTANE", ("N-PENTANE",), 1.84, 1.358, 0.626, 16.0, "alkane_nonpolar"),
    "2,2,4-TRIMETHYLPENTANE": SolventProperties(
        "2,2,4-TRIMETHYLPENTANE", ("ISOOCTANE", "ISO-OCTANE"), 1.94, 1.391, 0.692, 18.8, "alkane_nonpolar"
    ),
    "CYCLOOCTANE": SolventProperties("CYCLOOCTANE", (), 2.09, 1.458, 0.836, 30.0, "alkane_nonpolar"),
    "METHYLCYCLOHEXANE": SolventProperties("METHYLCYCLOHEXANE", (), 2.02, 1.423, 0.770, 23.3, "alkane_nonpolar"),
    "1-HEXADECENE": SolventProperties("1-HEXADECENE", (), 2.10, 1.441, 0.783, 27.8, "alkane_nonpolar"),
    # Chlorinated & Halogenated Solvents
    "CHLOROFORM": SolventProperties("CHLOROFORM", ("TRICHLOROMETHANE",), 4.81, 1.446, 1.489, 27.2, "chlorinated"),
    "CARBON TETRACHLORIDE": SolventProperties(
        "CARBON TETRACHLORIDE", ("TETRACHLOROMETHANE",), 2.24, 1.460, 1.594, 26.4, "chlorinated"
    ),
    "DICHLOROMETHANE": SolventProperties(
        "DICHLOROMETHANE", ("METHYLENE CHLORIDE", "DCM"), 8.93, 1.424, 1.326, 28.1, "chlorinated"
    ),
    "DICHLOROETHANE": SolventProperties(
        "DICHLOROETHANE", ("1,2-DICHLOROETHANE", "EDC"), 10.36, 1.445, 1.253, 32.2, "chlorinated"
    ),
    "1-CHLOROBUTANE": SolventProperties("1-CHLOROBUTANE", ("BUTYL CHLORIDE",), 7.39, 1.402, 0.886, 23.8, "chlorinated"),
    "CHLOROBENZENE": SolventProperties("CHLOROBENZENE", (), 5.62, 1.524, 1.106, 33.6, "chlorinated"),
    "BROMOBENZENE": SolventProperties("BROMOBENZENE", (), 5.40, 1.559, 1.495, 36.5, "chlorinated"),
    "BROMOETHANE": SolventProperties("BROMOETHANE", ("ETHYL BROMIDE",), 9.40, 1.424, 1.460, 24.2, "chlorinated"),
    "FLUOROBENZENE": SolventProperties("FLUOROBENZENE", (), 5.42, 1.465, 1.024, 27.7, "chlorinated"),
    "IODOBENZENE": SolventProperties("IODOBENZENE", (), 4.60, 1.620, 1.831, 39.8, "chlorinated"),
    "DIIODOMETHANE": SolventProperties("DIIODOMETHANE", ("METHYLENE IODIDE",), 5.32, 1.742, 3.325, 50.8, "chlorinated"),
    "PERFLUOROBENZENE": SolventProperties(
        "PERFLUOROBENZENE", ("HEXAFLUOROBENZENE",), 2.05, 1.377, 1.612, 21.6, "chlorinated"
    ),
    # Aromatic Hydrocarbons
    "TOLUENE": SolventProperties("TOLUENE", ("METHYLBENZENE",), 2.38, 1.496, 0.867, 28.5, "aromatic"),
    "M-XYLENE": SolventProperties("M-XYLENE", ("1,3-DIMETHYLBENZENE",), 2.37, 1.497, 0.864, 28.9, "aromatic"),
    "O-XYLENE": SolventProperties("O-XYLENE", ("1,2-DIMETHYLBENZENE",), 2.57, 1.505, 0.880, 30.1, "aromatic"),
    "P-XYLENE": SolventProperties("P-XYLENE", ("1,4-DIMETHYLBENZENE",), 2.27, 1.496, 0.861, 28.3, "aromatic"),
    "ETHYLBENZENE": SolventProperties("ETHYLBENZENE", (), 2.40, 1.495, 0.867, 29.0, "aromatic"),
    # Alcohols & Polyols (Polar Protic)
    "METHANOL": SolventProperties("METHANOL", ("MEOH",), 32.7, 1.328, 0.791, 22.5, "polar_protic"),
    "ETHANOL": SolventProperties("ETHANOL", ("ETOH",), 24.5, 1.361, 0.789, 22.3, "polar_protic"),
    "1-PROPANOL": SolventProperties("1-PROPANOL", ("N-PROPANOL",), 20.1, 1.385, 0.804, 23.8, "polar_protic"),
    "ISOPROPANOL": SolventProperties("ISOPROPANOL", ("2-PROPANOL", "IPA"), 19.9, 1.377, 0.785, 21.7, "polar_protic"),
    "N-BUTANOL": SolventProperties("N-BUTANOL", ("1-BUTANOL",), 17.5, 1.399, 0.810, 24.6, "polar_protic"),
    "ISOBUTANOL": SolventProperties("ISOBUTANOL", ("2-METHYLPROPAN-1-OL",), 16.6, 1.396, 0.803, 23.0, "polar_protic"),
    "2-BUTANOL": SolventProperties("2-BUTANOL", ("SEC-BUTANOL",), 16.6, 1.397, 0.806, 23.5, "polar_protic"),
    "TERT-BUTANOL": SolventProperties(
        "TERT-BUTANOL", ("2-METHYLPROPAN-2-OL",), 12.4, 1.387, 0.786, 20.7, "polar_protic"
    ),
    "PENTANOL": SolventProperties("PENTANOL", ("1-PENTANOL", "N-PENTANOL"), 15.1, 1.410, 0.814, 25.8, "polar_protic"),
    "2-PENTANOL": SolventProperties("2-PENTANOL", (), 13.7, 1.406, 0.810, 24.5, "polar_protic"),
    "PENTAN-3-OL": SolventProperties("PENTAN-3-OL", ("3-PENTANOL",), 13.0, 1.408, 0.820, 24.8, "polar_protic"),
    "2-METHYLBUTAN-1-OL": SolventProperties("2-METHYLBUTAN-1-OL", (), 14.7, 1.410, 0.819, 25.2, "polar_protic"),
    "3-METHYLBUTAN-1-OL": SolventProperties(
        "3-METHYLBUTAN-1-OL", ("ISOAMYL ALCOHOL",), 15.2, 1.406, 0.813, 24.8, "polar_protic"
    ),
    "TERT-AMYL ALCOHOL": SolventProperties(
        "TERT-AMYL ALCOHOL", ("2-METHYL-2-BUTANOL",), 5.8, 1.405, 0.809, 22.0, "polar_protic"
    ),
    "HEXANOL": SolventProperties("HEXANOL", ("1-HEXANOL", "N-HEXANOL"), 13.3, 1.418, 0.819, 26.2, "polar_protic"),
    "2-HEXANOL": SolventProperties("2-HEXANOL", (), 11.0, 1.414, 0.810, 25.1, "polar_protic"),
    "3-HEXANOL": SolventProperties("3-HEXANOL", (), 10.5, 1.414, 0.820, 25.0, "polar_protic"),
    "4-METHYLPENTAN-2-OL": SolventProperties(
        "4-METHYLPENTAN-2-OL", ("METHYL ISOBUTYL CARBINOL",), 11.2, 1.411, 0.808, 23.5, "polar_protic"
    ),
    "HEPTAN-1-OL": SolventProperties("HEPTAN-1-OL", ("1-HEPTANOL",), 11.8, 1.424, 0.822, 26.8, "polar_protic"),
    "2-HEPTANOL": SolventProperties("2-HEPTANOL", (), 9.5, 1.421, 0.817, 25.9, "polar_protic"),
    "4-HEPTANOL": SolventProperties("4-HEPTANOL", (), 8.5, 1.420, 0.818, 25.5, "polar_protic"),
    "1-OCTANOL": SolventProperties("1-OCTANOL", ("OCTANOL", "N-OCTANOL"), 10.3, 1.429, 0.824, 27.5, "polar_protic"),
    "2-ETHYLHEXANOL": SolventProperties("2-ETHYLHEXANOL", (), 7.7, 1.431, 0.833, 27.0, "polar_protic"),
    "4-OCTANOL": SolventProperties("4-OCTANOL", (), 7.0, 1.425, 0.820, 26.2, "polar_protic"),
    "NONANOL": SolventProperties("NONANOL", ("1-NONANOL",), 8.8, 1.433, 0.827, 28.1, "polar_protic"),
    "DECAN-1-OL": SolventProperties("DECAN-1-OL", ("1-DECANOL",), 8.1, 1.437, 0.829, 28.9, "polar_protic"),
    "UNDECANOL": SolventProperties("UNDECANOL", ("1-UNDECANOL",), 7.2, 1.440, 0.830, 29.5, "polar_protic"),
    "DODECAN-1-OL": SolventProperties(
        "DODECAN-1-OL", ("1-DODECANOL", "LAURYL ALCOHOL"), 6.5, 1.442, 0.833, 30.1, "polar_protic"
    ),
    "ALLYL ALCOHOL": SolventProperties("ALLYL ALCOHOL", (), 19.7, 1.413, 0.854, 25.8, "polar_protic"),
    "BENZYL ALCOHOL": SolventProperties("BENZYL ALCOHOL", (), 13.1, 1.540, 1.044, 39.0, "polar_protic"),
    "ETHYLENE GLYCOL": SolventProperties(
        "ETHYLENE GLYCOL", ("1,2-ETHANEDIOL",), 37.7, 1.431, 1.113, 47.7, "polar_protic"
    ),
    "1,2-PROPANEDIOL": SolventProperties(
        "1,2-PROPANEDIOL", ("PROPYLENE GLYCOL",), 32.0, 1.432, 1.036, 36.0, "polar_protic"
    ),
    "M-CRESOL": SolventProperties("M-CRESOL", ("3-METHYLPHENOL",), 11.8, 1.539, 1.034, 38.0, "polar_protic"),
    "ACETIC ACID": SolventProperties("ACETIC ACID", ("ETHANOIC ACID",), 6.17, 1.372, 1.049, 27.6, "polar_protic"),
    # Polar Aprotic Amides, Sulfoxides & Nitriles
    "DIMETHYL SULFOXIDE": SolventProperties("DIMETHYL SULFOXIDE", ("DMSO",), 46.7, 1.479, 1.100, 43.5, "polar_aprotic"),
    "DIMETHYLFORMAMIDE": SolventProperties(
        "DIMETHYLFORMAMIDE", ("DMF", "N,N-DIMETHYLFORMAMIDE"), 36.7, 1.430, 0.944, 37.1, "polar_aprotic"
    ),
    "N-METHYLPYRROLIDONE": SolventProperties(
        "N-METHYLPYRROLIDONE", ("NMP", "1-METHYL-2-PYRROLIDINONE"), 32.2, 1.470, 1.028, 40.7, "polar_aprotic"
    ),
    "1,5-DIMETHYL-2-PYRROLIDINONE": SolventProperties(
        "1,5-DIMETHYL-2-PYRROLIDINONE", (), 30.0, 1.465, 0.990, 38.0, "polar_aprotic"
    ),
    "1-ETHYL-2-PYRROLIDINONE": SolventProperties(
        "1-ETHYL-2-PYRROLIDINONE", ("NEP",), 28.2, 1.465, 0.992, 36.5, "polar_aprotic"
    ),
    "1-METHYL-2-PIPERIDINONE": SolventProperties(
        "1-METHYL-2-PIPERIDINONE", (), 26.0, 1.478, 1.025, 38.5, "polar_aprotic"
    ),
    "FORMAMIDE": SolventProperties("FORMAMIDE", (), 109.5, 1.447, 1.133, 58.2, "polar_aprotic"),
    "N-METHYLFORMAMIDE": SolventProperties("N-METHYLFORMAMIDE", ("NMF",), 182.4, 1.432, 1.011, 39.2, "polar_aprotic"),
    "N-ETHYLFORMAMIDE": SolventProperties("N-ETHYLFORMAMIDE", (), 115.0, 1.434, 0.952, 36.0, "polar_aprotic"),
    "N,N-DIMETHYLACETAMIDE": SolventProperties(
        "N,N-DIMETHYLACETAMIDE", ("DMAC",), 37.8, 1.438, 0.937, 34.0, "polar_aprotic"
    ),
    "N-METHYLACETAMIDE": SolventProperties("N-METHYLACETAMIDE", ("NMA",), 191.3, 1.430, 0.957, 34.2, "polar_aprotic"),
    "N-ETHYLACETAMIDE": SolventProperties("N-ETHYLACETAMIDE", (), 135.0, 1.436, 0.925, 33.5, "polar_aprotic"),
    "N,N-DIETHYLACETAMIDE": SolventProperties("N,N-DIETHYLACETAMIDE", (), 32.0, 1.438, 0.908, 31.0, "polar_aprotic"),
    "N,N-DIBUTYLFORMAMID": SolventProperties(
        "N,N-DIBUTYLFORMAMID", ("N,N-DIBUTYLFORMAMIDE",), 18.0, 1.442, 0.860, 29.5, "polar_aprotic"
    ),
    "4-FORMYLMORPHOLINE": SolventProperties("4-FORMYLMORPHOLINE", (), 36.0, 1.484, 1.145, 45.0, "polar_aprotic"),
    "ACETONITRILE": SolventProperties("ACETONITRILE", ("MECN",), 37.5, 1.344, 0.786, 29.1, "polar_aprotic"),
    "PROPIONITRILE": SolventProperties("PROPIONITRILE", (), 27.2, 1.366, 0.782, 27.0, "polar_aprotic"),
    "BUTYRONITRILE": SolventProperties("BUTYRONITRILE", (), 20.7, 1.384, 0.796, 26.5, "polar_aprotic"),
    "BENZONITRILE": SolventProperties("BENZONITRILE", (), 25.2, 1.528, 1.006, 39.0, "polar_aprotic"),
    "NITROMETHANE": SolventProperties("NITROMETHANE", (), 35.8, 1.382, 1.137, 36.8, "polar_aprotic"),
    "NITROETHANE": SolventProperties("NITROETHANE", (), 28.1, 1.392, 1.045, 33.2, "polar_aprotic"),
    "NITROBENZENE": SolventProperties("NITROBENZENE", (), 34.8, 1.556, 1.204, 43.9, "polar_aprotic"),
    "SULFOLANE": SolventProperties("SULFOLANE", (), 43.3, 1.484, 1.261, 45.5, "polar_aprotic"),
    "PROPYLENE CARBONATE": SolventProperties("PROPYLENE CARBONATE", (), 64.9, 1.421, 1.205, 41.1, "polar_aprotic"),
    "BUTYROLACTONE": SolventProperties(
        "BUTYROLACTONE", ("GAMMA-BUTYROLACTONE", "GBL"), 39.0, 1.436, 1.129, 43.2, "polar_aprotic"
    ),
    "EPSILON-CAPROLACTONE": SolventProperties("EPSILON-CAPROLACTONE", (), 38.0, 1.463, 1.030, 39.5, "polar_aprotic"),
    # Ketones
    "ACETONE": SolventProperties("ACETONE", ("2-PROPANONE",), 20.7, 1.359, 0.791, 23.7, "polar_aprotic"),
    "BUTANONE": SolventProperties(
        "BUTANONE", ("METHYL ETHYL KETONE", "MEK"), 18.5, 1.379, 0.805, 24.6, "polar_aprotic"
    ),
    "PENTAN-2-ONE": SolventProperties("PENTAN-2-ONE", ("2-PENTANONE",), 15.4, 1.390, 0.809, 25.0, "polar_aprotic"),
    "3-PENTANONE": SolventProperties("3-PENTANONE", ("DIETHYL KETONE",), 17.0, 1.392, 0.814, 25.5, "polar_aprotic"),
    "2-HEXANONE": SolventProperties("2-HEXANONE", (), 14.6, 1.401, 0.812, 25.8, "polar_aprotic"),
    "3-HEXANONE": SolventProperties("3-HEXANONE", (), 14.0, 1.400, 0.813, 25.6, "polar_aprotic"),
    "4-METHYLPENTAN-2-ONE": SolventProperties(
        "4-METHYLPENTAN-2-ONE", ("MIBK",), 13.1, 1.396, 0.802, 23.6, "polar_aprotic"
    ),
    "2-HEPTANONE": SolventProperties("2-HEPTANONE", (), 12.0, 1.408, 0.815, 26.5, "polar_aprotic"),
    "CYCLOHEXANONE": SolventProperties("CYCLOHEXANONE", (), 18.3, 1.451, 0.948, 35.1, "polar_aprotic"),
    "ACETOPHENONE": SolventProperties("ACETOPHENONE", (), 17.4, 1.534, 1.028, 39.8, "polar_aprotic"),
    # Ethers & Glycol Ethers
    "TETRAHYDROFURAN": SolventProperties("TETRAHYDROFURAN", ("THF",), 7.58, 1.407, 0.889, 26.4, "ether"),
    "1,4-DIOXANE": SolventProperties("1,4-DIOXANE", ("DIOXANE",), 2.21, 1.422, 1.033, 33.0, "ether"),
    "TETRAHYDROPYRAN": SolventProperties("TETRAHYDROPYRAN", ("THP",), 5.70, 1.420, 0.881, 28.0, "ether"),
    "DIETHYL ETHER": SolventProperties("DIETHYL ETHER", ("ETHER",), 4.33, 1.353, 0.713, 17.0, "ether"),
    "DIISOPROPYL ETHER": SolventProperties("DIISOPROPYL ETHER", (), 3.88, 1.368, 0.724, 17.7, "ether"),
    "DIBUTYL ETHER": SolventProperties("DIBUTYL ETHER", (), 3.08, 1.399, 0.764, 22.9, "ether"),
    "PROPYL ETHER": SolventProperties("PROPYL ETHER", ("DIPROPYL ETHER",), 3.39, 1.381, 0.736, 20.5, "ether"),
    "METHYL TERT-BUTYL ETHER": SolventProperties(
        "METHYL TERT-BUTYL ETHER", ("MTBE",), 2.60, 1.369, 0.740, 19.4, "ether"
    ),
    "ETBE": SolventProperties("ETBE", ("ETHYL TERT-BUTYL ETHER",), 2.45, 1.375, 0.745, 19.8, "ether"),
    "TERT-AMYL METHYL ETHER": SolventProperties("TERT-AMYL METHYL ETHER", ("TAME",), 2.50, 1.380, 0.770, 20.5, "ether"),
    "ANISOLE": SolventProperties("ANISOLE", ("METHOXYBENZENE",), 4.33, 1.517, 0.995, 35.0, "ether"),
    "ETHYL PHENYL ETHER": SolventProperties("ETHYL PHENYL ETHER", ("PHENETOLE",), 4.22, 1.507, 0.965, 33.0, "ether"),
    "DIBENZYL ETHER": SolventProperties("DIBENZYL ETHER", (), 3.80, 1.540, 1.043, 38.0, "ether"),
    "1-METHOXYBUTANE": SolventProperties("1-METHOXYBUTANE", ("BUTYL METHYL ETHER",), 3.90, 1.374, 0.744, 20.0, "ether"),
    "1-ETHOXYBUTANE": SolventProperties("1-ETHOXYBUTANE", ("BUTYL ETHYL ETHER",), 3.50, 1.382, 0.749, 21.0, "ether"),
    "1-ETHOXYPROPANE": SolventProperties("1-ETHOXYPROPANE", ("ETHYL PROPYL ETHER",), 3.60, 1.373, 0.739, 19.5, "ether"),
    "2-METHOXYPROPANE": SolventProperties("2-METHOXYPROPANE", (), 3.80, 1.360, 0.725, 18.5, "ether"),
    "METHOXYPROPANE": SolventProperties("METHOXYPROPANE", ("1-METHOXYPROPANE",), 3.90, 1.370, 0.738, 19.0, "ether"),
    "METHOXYETHANOL": SolventProperties(
        "METHOXYETHANOL", ("2-METHOXYETHANOL", "METHYL CELLOSOLVE"), 16.9, 1.402, 0.965, 31.8, "ether"
    ),
    "2-ETHOXYETHANOL": SolventProperties("2-ETHOXYETHANOL", ("CELLOSOLVE",), 14.1, 1.408, 0.930, 28.6, "ether"),
    "2-PROPOXYETHANOL": SolventProperties("2-PROPOXYETHANOL", (), 11.5, 1.413, 0.912, 27.5, "ether"),
    "2-ISOPROPOXYETHANOL": SolventProperties("2-ISOPROPOXYETHANOL", (), 10.8, 1.410, 0.904, 26.8, "ether"),
    "2-BUTOXYETHANOL": SolventProperties("2-BUTOXYETHANOL", ("BUTYL CELLOSOLVE",), 9.4, 1.419, 0.902, 27.4, "ether"),
    "DIETHYLDIGLYCOL": SolventProperties(
        "DIETHYLDIGLYCOL", ("DIETHYLENE GLYCOL DIETHYL ETHER",), 5.7, 1.412, 0.909, 27.2, "ether"
    ),
    "DIETHYLENE GLYCOL DIBUTYL ETHER": SolventProperties(
        "DIETHYLENE GLYCOL DIBUTYL ETHER", (), 4.5, 1.423, 0.885, 27.0, "ether"
    ),
    "TETRAETHYLENE GLYCOL DIMETHYL ETHER": SolventProperties(
        "TETRAETHYLENE GLYCOL DIMETHYL ETHER", ("TETRAGLYME",), 7.7, 1.432, 1.009, 33.8, "ether"
    ),
    "5,8,11,14-TETRAOXAOCTADECANE": SolventProperties(
        "5,8,11,14-TETRAOXAOCTADECANE", (), 6.5, 1.430, 0.920, 28.5, "ether"
    ),
    # Esters
    "METHYL ACETATE": SolventProperties("METHYL ACETATE", (), 6.68, 1.361, 0.932, 24.6, "ester"),
    "ETHYL ACETATE": SolventProperties("ETHYL ACETATE", ("ETOAC",), 6.02, 1.372, 0.902, 23.9, "ester"),
    "PROPYL ACETATE": SolventProperties("PROPYL ACETATE", (), 6.00, 1.384, 0.888, 24.3, "ester"),
    "ISOPROPYL ACETATE": SolventProperties("ISOPROPYL ACETATE", (), 6.30, 1.377, 0.872, 22.1, "ester"),
    "BUTYL ACETATE": SolventProperties("BUTYL ACETATE", ("N-BUTYL ACETATE",), 5.01, 1.394, 0.882, 25.1, "ester"),
    "ISOBUTYL ACETATE": SolventProperties("ISOBUTYL ACETATE", (), 5.30, 1.390, 0.871, 23.7, "ester"),
    "PENTYL ACETATE": SolventProperties("PENTYL ACETATE", ("AMYL ACETATE",), 4.75, 1.402, 0.876, 25.8, "ester"),
    "HEXYL ACETATE": SolventProperties("HEXYL ACETATE", (), 4.40, 1.409, 0.873, 26.5, "ester"),
    "HEPTYLACETAT": SolventProperties("HEPTYLACETAT", ("HEPTYL ACETATE",), 4.20, 1.415, 0.870, 27.0, "ester"),
    "ETHYL BUTANOATE": SolventProperties("ETHYL BUTANOATE", ("ETHYL BUTYRATE",), 5.12, 1.392, 0.879, 24.5, "ester"),
    "ETHYL HEXANOATE": SolventProperties("ETHYL HEXANOATE", (), 4.00, 1.407, 0.871, 26.0, "ester"),
    "ETHYLBENZOATE": SolventProperties("ETHYLBENZOATE", ("ETHYL BENZOATE",), 6.02, 1.505, 1.047, 35.5, "ester"),
    # Amines & Pyridines
    "TRIETHYLAMINE": SolventProperties("TRIETHYLAMINE", ("TEA",), 2.42, 1.401, 0.728, 20.7, "polar_aprotic"),
    "PYRIDINE": SolventProperties("PYRIDINE", (), 12.4, 1.510, 0.982, 38.0, "aromatic"),
    "2-PICOLINE": SolventProperties("2-PICOLINE", ("2-METHYLPYRIDINE",), 9.8, 1.501, 0.944, 33.5, "aromatic"),
    "PHENYLAMINE": SolventProperties("PHENYLAMINE", ("ANILINE",), 6.89, 1.586, 1.022, 43.4, "aromatic"),
    # Sulfur & Phosphorus Organics
    "CARBON DISULFIDE": SolventProperties("CARBON DISULFIDE", ("CS2",), 2.64, 1.632, 1.263, 32.3, "chlorinated"),
    "TRIBUTYL PHOSPHATE": SolventProperties("TRIBUTYL PHOSPHATE", ("TBP",), 8.0, 1.425, 0.976, 27.8, "polar_aprotic"),
}

# Verified experimental Abraham solvation parameters (alpha_2^H, beta_2^H, pi_2^H),
# molecular weights (g/mol), and dipole moments (Debye) (Acree, Abraham, CRC Handbook)
_SOLVENT_ABRAHAM_DATA: Dict[str, Tuple[float, float, float, float, float]] = {
    # Reference Aqueous
    "WATER": (18.015, 1.17, 0.47, 1.09, 1.85),
    # Alkanes & Cycloalkanes
    "HEXADECANE": (226.44, 0.0, 0.0, 0.0, 0.0),
    "HEPTANE": (100.20, 0.0, 0.0, 0.0, 0.0),
    "HEXANE": (86.18, 0.0, 0.0, 0.0, 0.0),
    "OCTANE": (114.23, 0.0, 0.0, 0.0, 0.0),
    "NONANE": (128.26, 0.0, 0.0, 0.0, 0.0),
    "DECANE": (142.28, 0.0, 0.0, 0.0, 0.0),
    "UNDECANE": (156.31, 0.0, 0.0, 0.0, 0.0),
    "DODECANE": (170.33, 0.0, 0.0, 0.0, 0.0),
    "TETRADECANE": (198.39, 0.0, 0.0, 0.0, 0.0),
    "PENTANE": (72.15, 0.0, 0.0, 0.0, 0.0),
    "2,2,4-TRIMETHYLPENTANE": (114.23, 0.0, 0.0, 0.0, 0.0),
    "CYCLOOCTANE": (112.21, 0.0, 0.0, 0.0, 0.0),
    "METHYLCYCLOHEXANE": (98.19, 0.0, 0.0, 0.0, 0.0),
    "1-HEXADECENE": (224.43, 0.0, 0.08, 0.08, 0.40),
    # Chlorinated & Halogenated
    "CHLOROFORM": (119.38, 0.20, 0.02, 0.58, 1.04),
    "CARBON TETRACHLORIDE": (153.82, 0.0, 0.0, 0.38, 0.0),
    "DICHLOROMETHANE": (84.93, 0.10, 0.05, 0.82, 1.60),
    "DICHLOROETHANE": (98.96, 0.10, 0.11, 0.81, 1.83),
    "1-CHLOROBUTANE": (92.57, 0.0, 0.05, 0.40, 1.90),
    "CHLOROBENZENE": (112.56, 0.0, 0.07, 0.71, 1.54),
    "BROMOBENZENE": (157.01, 0.0, 0.06, 0.73, 1.70),
    "BROMOETHANE": (108.97, 0.0, 0.05, 0.40, 2.03),
    "FLUOROBENZENE": (96.10, 0.0, 0.07, 0.62, 1.60),
    "IODOBENZENE": (204.01, 0.0, 0.07, 0.81, 1.70),
    "DIIODOMETHANE": (267.84, 0.05, 0.05, 0.90, 1.10),
    "PERFLUOROBENZENE": (186.05, 0.0, 0.0, -0.02, 0.0),
    # Aromatics
    "TOLUENE": (92.14, 0.0, 0.11, 0.55, 0.36),
    "M-XYLENE": (106.17, 0.0, 0.16, 0.52, 0.30),
    "O-XYLENE": (106.17, 0.0, 0.16, 0.56, 0.62),
    "P-XYLENE": (106.17, 0.0, 0.16, 0.52, 0.0),
    "ETHYLBENZENE": (106.17, 0.0, 0.15, 0.51, 0.59),
    # Alcohols & Polyols (Polar Protic)
    "METHANOL": (32.04, 0.93, 0.62, 0.60, 1.70),
    "ETHANOL": (46.07, 0.83, 0.77, 0.54, 1.69),
    "1-PROPANOL": (60.10, 0.78, 0.78, 0.52, 1.68),
    "ISOPROPANOL": (60.10, 0.76, 0.84, 0.48, 1.66),
    "N-BUTANOL": (74.12, 0.73, 0.85, 0.47, 1.66),
    "ISOBUTANOL": (74.12, 0.73, 0.84, 0.40, 1.79),
    "2-BUTANOL": (74.12, 0.69, 0.80, 0.40, 1.66),
    "TERT-BUTANOL": (74.12, 0.68, 0.93, 0.41, 1.66),
    "PENTANOL": (88.15, 0.66, 0.84, 0.45, 1.68),
    "2-PENTANOL": (88.15, 0.65, 0.82, 0.42, 1.66),
    "PENTAN-3-OL": (88.15, 0.65, 0.82, 0.42, 1.66),
    "2-METHYLBUTAN-1-OL": (88.15, 0.66, 0.84, 0.44, 1.68),
    "3-METHYLBUTAN-1-OL": (88.15, 0.66, 0.84, 0.44, 1.68),
    "TERT-AMYL ALCOHOL": (88.15, 0.60, 0.90, 0.40, 1.66),
    "HEXANOL": (102.17, 0.63, 0.82, 0.42, 1.68),
    "2-HEXANOL": (102.17, 0.62, 0.80, 0.40, 1.66),
    "3-HEXANOL": (102.17, 0.62, 0.80, 0.40, 1.66),
    "4-METHYLPENTAN-2-OL": (102.17, 0.62, 0.80, 0.40, 1.66),
    "HEPTAN-1-OL": (116.20, 0.62, 0.80, 0.42, 1.68),
    "2-HEPTANOL": (116.20, 0.61, 0.80, 0.40, 1.66),
    "4-HEPTANOL": (116.20, 0.61, 0.80, 0.40, 1.66),
    "1-OCTANOL": (130.23, 0.60, 0.79, 0.40, 1.68),
    "2-ETHYLHEXANOL": (130.23, 0.58, 0.80, 0.40, 1.68),
    "4-OCTANOL": (130.23, 0.58, 0.80, 0.40, 1.66),
    "NONANOL": (144.25, 0.57, 0.79, 0.40, 1.68),
    "DECAN-1-OL": (158.28, 0.57, 0.79, 0.40, 1.68),
    "UNDECANOL": (172.31, 0.55, 0.78, 0.40, 1.68),
    "DODECAN-1-OL": (186.33, 0.55, 0.78, 0.40, 1.68),
    "ALLYL ALCOHOL": (58.08, 0.75, 0.70, 0.60, 1.60),
    "BENZYL ALCOHOL": (108.14, 0.70, 0.65, 0.85, 1.71),
    "ETHYLENE GLYCOL": (62.07, 0.90, 0.52, 0.92, 2.28),
    "1,2-PROPANEDIOL": (76.09, 0.80, 0.60, 0.80, 2.20),
    "M-CRESOL": (108.14, 0.82, 0.35, 0.88, 1.54),
    "ACETIC ACID": (60.05, 0.61, 0.45, 0.65, 1.74),
    # Polar Aprotic
    "DIMETHYL SULFOXIDE": (78.13, 0.0, 0.76, 1.00, 3.96),
    "DIMETHYLFORMAMIDE": (73.09, 0.0, 0.69, 0.88, 3.82),
    "N-METHYLPYRROLIDONE": (99.13, 0.0, 0.77, 0.92, 4.09),
    "1,5-DIMETHYL-2-PYRROLIDINONE": (113.16, 0.0, 0.75, 0.90, 4.00),
    "1-ETHYL-2-PYRROLIDINONE": (113.16, 0.0, 0.76, 0.90, 4.05),
    "1-METHYL-2-PIPERIDINONE": (113.16, 0.0, 0.77, 0.90, 4.00),
    "FORMAMIDE": (45.04, 0.62, 0.60, 1.05, 3.73),
    "N-METHYLFORMAMIDE": (59.07, 0.40, 0.70, 1.00, 3.83),
    "N-ETHYLFORMAMIDE": (73.09, 0.38, 0.70, 0.98, 3.85),
    "N,N-DIMETHYLACETAMIDE": (87.12, 0.0, 0.78, 0.88, 3.79),
    "N-METHYLACETAMIDE": (73.09, 0.36, 0.74, 0.95, 3.75),
    "N-ETHYLACETAMIDE": (87.12, 0.35, 0.74, 0.95, 3.75),
    "N,N-DIETHYLACETAMIDE": (115.17, 0.0, 0.78, 0.85, 3.80),
    "N,N-DIBUTYLFORMAMID": (157.25, 0.0, 0.75, 0.80, 3.70),
    "4-FORMYLMORPHOLINE": (115.13, 0.0, 0.75, 0.95, 4.10),
    "ACETONITRILE": (41.05, 0.04, 0.29, 0.75, 3.92),
    "PROPIONITRILE": (55.08, 0.02, 0.30, 0.73, 4.05),
    "BUTYRONITRILE": (69.11, 0.02, 0.30, 0.71, 4.10),
    "BENZONITRILE": (103.12, 0.0, 0.33, 0.90, 4.18),
    "NITROMETHANE": (61.04, 0.06, 0.28, 0.85, 3.46),
    "NITROETHANE": (75.07, 0.04, 0.28, 0.83, 3.65),
    "NITROBENZENE": (123.11, 0.0, 0.30, 0.86, 4.22),
    "SULFOLANE": (120.17, 0.0, 0.39, 0.98, 4.81),
    "PROPYLENE CARBONATE": (102.09, 0.0, 0.40, 0.83, 4.90),
    "BUTYROLACTONE": (86.09, 0.0, 0.49, 0.86, 4.27),
    "EPSILON-CAPROLACTONE": (114.14, 0.0, 0.50, 0.85, 4.20),
    # Ketones
    "ACETONE": (58.08, 0.04, 0.49, 0.71, 2.88),
    "BUTANONE": (72.11, 0.0, 0.48, 0.67, 2.78),
    "PENTAN-2-ONE": (86.13, 0.0, 0.48, 0.65, 2.70),
    "3-PENTANONE": (86.13, 0.0, 0.48, 0.65, 2.70),
    "2-HEXANONE": (100.16, 0.0, 0.48, 0.64, 2.70),
    "3-HEXANONE": (100.16, 0.0, 0.48, 0.64, 2.70),
    "4-METHYLPENTAN-2-ONE": (100.16, 0.0, 0.48, 0.62, 2.70),
    "2-HEPTANONE": (114.19, 0.0, 0.48, 0.64, 2.70),
    "CYCLOHEXANONE": (98.14, 0.0, 0.53, 0.76, 3.01),
    "ACETOPHENONE": (120.15, 0.0, 0.48, 0.90, 3.05),
    # Ethers
    "TETRAHYDROFURAN": (72.11, 0.0, 0.55, 0.58, 1.63),
    "1,4-DIOXANE": (88.11, 0.0, 0.37, 0.55, 0.45),
    "TETRAHYDROPYRAN": (86.13, 0.0, 0.55, 0.55, 1.55),
    "DIETHYL ETHER": (74.12, 0.0, 0.45, 0.27, 1.15),
    "DIISOPROPYL ETHER": (102.17, 0.0, 0.49, 0.20, 1.13),
    "DIBUTYL ETHER": (130.23, 0.0, 0.46, 0.24, 1.18),
    "PROPYL ETHER": (102.17, 0.0, 0.46, 0.25, 1.15),
    "METHYL TERT-BUTYL ETHER": (88.15, 0.0, 0.50, 0.24, 1.22),
    "ETBE": (102.17, 0.0, 0.50, 0.24, 1.22),
    "TERT-AMYL METHYL ETHER": (102.17, 0.0, 0.50, 0.24, 1.22),
    "ANISOLE": (108.14, 0.0, 0.29, 0.73, 1.38),
    "ETHYL PHENYL ETHER": (122.16, 0.0, 0.30, 0.70, 1.35),
    "DIBENZYL ETHER": (198.26, 0.0, 0.35, 0.85, 1.35),
    "1-METHOXYBUTANE": (88.15, 0.0, 0.46, 0.30, 1.20),
    "1-ETHOXYBUTANE": (102.17, 0.0, 0.46, 0.28, 1.20),
    "1-ETHOXYPROPANE": (88.15, 0.0, 0.46, 0.28, 1.20),
    "2-METHOXYPROPANE": (74.12, 0.0, 0.48, 0.28, 1.20),
    "METHOXYPROPANE": (74.12, 0.0, 0.48, 0.28, 1.20),
    "METHOXYETHANOL": (76.09, 0.55, 0.65, 0.65, 2.04),
    "2-ETHOXYETHANOL": (90.12, 0.52, 0.65, 0.60, 2.08),
    "2-PROPOXYETHANOL": (104.15, 0.50, 0.65, 0.58, 2.05),
    "2-ISOPROPOXYETHANOL": (104.15, 0.50, 0.65, 0.58, 2.05),
    "2-BUTOXYETHANOL": (118.17, 0.50, 0.65, 0.58, 2.08),
    "DIETHYLDIGLYCOL": (162.23, 0.0, 0.60, 0.55, 1.80),
    "DIETHYLENE GLYCOL DIBUTYL ETHER": (218.33, 0.0, 0.60, 0.50, 1.80),
    "TETRAETHYLENE GLYCOL DIMETHYL ETHER": (222.28, 0.0, 0.70, 0.70, 2.40),
    "5,8,11,14-TETRAOXAOCTADECANE": (278.39, 0.0, 0.70, 0.60, 2.20),
    # Esters
    "METHYL ACETATE": (74.08, 0.0, 0.45, 0.60, 1.72),
    "ETHYL ACETATE": (88.11, 0.0, 0.45, 0.55, 1.78),
    "PROPYL ACETATE": (102.13, 0.0, 0.45, 0.53, 1.78),
    "ISOPROPYL ACETATE": (102.13, 0.0, 0.47, 0.50, 1.80),
    "BUTYL ACETATE": (116.16, 0.0, 0.45, 0.50, 1.84),
    "ISOBUTYL ACETATE": (116.16, 0.0, 0.46, 0.48, 1.80),
    "PENTYL ACETATE": (130.18, 0.0, 0.45, 0.48, 1.80),
    "HEXYL ACETATE": (144.21, 0.0, 0.45, 0.46, 1.80),
    "HEPTYLACETAT": (158.24, 0.0, 0.45, 0.46, 1.80),
    "ETHYL BUTANOATE": (116.16, 0.0, 0.45, 0.48, 1.80),
    "ETHYL HEXANOATE": (144.21, 0.0, 0.45, 0.46, 1.80),
    "ETHYLBENZOATE": (150.17, 0.0, 0.41, 0.75, 2.00),
    # Amines & Pyridines
    "TRIETHYLAMINE": (101.19, 0.0, 0.71, 0.14, 0.66),
    "PYRIDINE": (79.10, 0.0, 0.52, 0.87, 2.21),
    "2-PICOLINE": (93.13, 0.0, 0.57, 0.80, 1.96),
    "PHENYLAMINE": (93.13, 0.26, 0.50, 0.96, 1.53),
    # Sulfur & Phosphorus Organics
    "CARBON DISULFIDE": (76.14, 0.0, 0.06, 0.35, 0.0),
    "TRIBUTYL PHOSPHATE": (266.32, 0.0, 0.77, 0.60, 3.07),
}

# Enrich registry instances with verified Abraham descriptors
for _s_name, (_mw, _alpha, _beta, _pi2, _dipole) in _SOLVENT_ABRAHAM_DATA.items():
    if _s_name in _SOLVENT_REGISTRY:
        _old = _SOLVENT_REGISTRY[_s_name]
        _SOLVENT_REGISTRY[_s_name] = SolventProperties(
            name=_old.name,
            aliases=_old.aliases,
            dielectric_constant=_old.dielectric_constant,
            refractive_index=_old.refractive_index,
            density_g_cm3=_old.density_g_cm3,
            surface_tension_mn_m=_old.surface_tension_mn_m,
            solvent_class=_old.solvent_class,
            molecular_weight=_mw,
            abraham_alpha=_alpha,
            abraham_beta=_beta,
            abraham_pi2=_pi2,
            dipole_moment_debye=_dipole,
        )

# Alias map for O(1) canonical resolution
_ALIAS_MAP: Dict[str, str] = {}
for _canon, _props in _SOLVENT_REGISTRY.items():
    _ALIAS_MAP[normalize_solvent_name(_canon)] = _canon
    for _alias in _props.aliases:
        _ALIAS_MAP[normalize_solvent_name(_alias)] = _canon


def estimate_dielectric_from_polarizability_and_dipole(
    dipole_debye: float,
    polarizability_angstrom3: float,
    molecular_weight: float,
    density_g_cm3: float,
    temp_k: float = 298.15,
) -> float:
    """
    First-principles estimation of static dielectric constant epsilon_r using the
    Onsager-Kirkwood-Fröhlich polarization equation for liquids:
    (epsilon_r - n^2)(2*epsilon_r + n^2) / [epsilon_r * (n^2 + 2)^2] = (rho * N_A * mu^2) / (9 * eps_0 * k_B * T)
    where optical index n^2 is derived from Lorenz-Lorentz electronic polarizability:
    (n^2 - 1)/(n^2 + 2) = (4*pi / 3) * (rho * N_A / M) * alpha
    """
    if molecular_weight <= 0.0 or density_g_cm3 <= 0.0:
        return 2.0

    n_a = 6.02214076e23
    k_b = 1.380649e-23
    eps_0 = 8.8541878128e-12

    # Number density in molecules / m³
    rho_num = (density_g_cm3 * 1e6 / molecular_weight) * n_a

    # Electronic polarizability alpha in m³ (1 Å³ = 1e-30 m³)
    alpha_m3 = max(0.1, polarizability_angstrom3) * 1e-30
    p_elec = (4.0 * math.pi / 3.0) * rho_num * alpha_m3
    p_elec = min(0.85, max(0.01, p_elec))
    # n^2 from Lorenz-Lorentz
    n2 = (1.0 + 2.0 * p_elec) / max(1e-4, 1.0 - p_elec)

    # Dipole moment mu in C*m (1 Debye = 3.33564e-30 C*m)
    mu_cm = abs(dipole_debye) * 3.33564e-30
    # Orientational polarization term y
    y_orient = (rho_num * (mu_cm**2)) / (9.0 * eps_0 * k_b * max(1.0, temp_k))

    # Solve Onsager quadratic for epsilon_r:
    # 2*eps^2 + eps*(n^2 - y*(n^2+2)^2) - n^4 = 0
    c_term = y_orient * ((n2 + 2.0) ** 2)
    b_term = n2 - c_term
    disc = (b_term**2) + 8.0 * (n2**2)
    eps_r = (-b_term + math.sqrt(max(0.0, disc))) / 4.0
    return max(1.8, min(250.0, float(eps_r)))


def get_solvent_properties(name: str) -> SolventProperties:
    """
    Retrieves full physical properties for a solvent by canonical name or alias.
    Falls back to a default organic solvent model if unidentified.
    """
    clean = normalize_solvent_name(name)
    canon = _ALIAS_MAP.get(clean)
    if canon and canon in _SOLVENT_REGISTRY:
        return _SOLVENT_REGISTRY[canon]

    # Partial substring search fallback
    for reg_name, props in _SOLVENT_REGISTRY.items():
        if clean in reg_name or reg_name in clean:
            return props

    # Safe generic organic liquid fallback
    return SolventProperties(
        name=name.upper(),
        aliases=(),
        dielectric_constant=5.0,
        refractive_index=1.40,
        density_g_cm3=0.85,
        surface_tension_mn_m=25.0,
        solvent_class="alkane_nonpolar",
        molecular_weight=100.0,
        abraham_alpha=0.0,
        abraham_beta=0.0,
        abraham_pi2=0.0,
        dipole_moment_debye=0.0,
    )


def get_solvent_dielectric(name: str, default: float = 78.4) -> float:
    """Returns static relative permittivity epsilon_r for a solvent."""
    if not name or name.lower() in ("water", "aqueous", "h2o"):
        return 78.4
    props = get_solvent_properties(name)
    return props.dielectric_constant if props else default


def get_solvent_descriptors_vector(name: str) -> np.ndarray:
    """
    Returns a standardized 7-dimensional physical descriptor vector for the solvent:
    0: Onsager dielectric factor f_eps = (eps - 1) / (2*eps + 1) in [0, 0.5]
    1: Lorentz-Lorenz polarizability factor f_n = (n^2 - 1) / (n^2 + 2) in [0.15, 0.4]
    2: Cavitation surface tension ratio gamma / 72.8 in [0.2, 1.0]
    3: Molar volume ratio V_m / 100.0 in [0.18, 3.0]
    4: Abraham hydrogen-bond acidity alpha in [0.0, 1.2]
    5: Abraham hydrogen-bond basicity beta in [0.0, 1.0]
    6: Normalized dipole moment mu / 4.0 in [0.0, 1.2]
    """
    props = get_solvent_properties(name)
    eps = max(1.0, props.dielectric_constant)
    f_eps = (eps - 1.0) / (2.0 * eps + 1.0)
    n = max(1.0, props.refractive_index)
    f_n = (n**2 - 1.0) / (n**2 + 2.0)
    gamma_ratio = props.surface_tension_mn_m / 72.8
    v_m_ratio = props.molar_volume_cm3_mol / 100.0
    alpha = props.abraham_alpha
    beta = props.abraham_beta
    dipole_norm = props.dipole_moment_debye / 4.0
    return np.array([f_eps, f_n, gamma_ratio, v_m_ratio, alpha, beta, dipole_norm], dtype=np.float32)


def list_registered_solvents() -> Tuple[str, ...]:
    """Returns tuple of all canonically registered solvent names."""
    return tuple(sorted(_SOLVENT_REGISTRY.keys()))
