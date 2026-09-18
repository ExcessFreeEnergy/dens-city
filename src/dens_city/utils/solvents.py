"""
Generic, Extensible Physical Solvent Parameter Architecture.
Provides dielectric constants, optical refractive indices, surface tensions,
molar volumes, and Abraham parameters for liquid solvent matrices.
Eliminates hardcoded dictionary registries by loading from external data assets
and employing dynamic first-principles (Onsager-Kirkwood-Fröhlich / Lorentz-Lorenz)
derivation for novel or unlisted fluids.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SolventProperties:
    """Thermodynamic, electrostatic, and structural parameters for a liquid solvent matrix."""

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
    s = str(name).strip().upper()
    # Normalize racemic markers: (\xb1), (±), (+/-), racemic
    s = s.replace("(\\XB1)", "").replace("(±)", "").replace("(+/-)", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" -_")
    return s


def estimate_dielectric_from_polarizability_and_dipole(
    dipole_debye: float,
    polarizability_angstrom3: float,
    molecular_weight: float,
    density_g_cm3: float,
    temp_k: float = 298.15,
    kirkwood_g: float = 1.0,
) -> float:
    """
    First-principles estimation of static dielectric constant epsilon_r using the
    Onsager-Kirkwood-Fröhlich polarization equation for liquids:
    (epsilon_r - n^2)(2*epsilon_r + n^2) / [epsilon_r * (n^2 + 2)^2] = (rho * N_A * g_K * mu^2) / (9 * eps_0 * k_B * T)
    where optical index n^2 is derived from Lorenz-Lorentz electronic polarizability:
    (n^2 - 1)/(n^2 + 2) = (4*pi / 3) * (rho * N_A / M) * alpha
    and g_K is the Kirkwood dipole correlation factor (g_K = 1 for non-associated fluids, > 1 for H-bonding networks).
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
    # Orientational polarization term y with Kirkwood dipole correlation factor g_K
    gk_val = max(1.0, float(kirkwood_g))
    y_orient = (rho_num * gk_val * (mu_cm**2)) / (9.0 * eps_0 * k_b * max(1.0, temp_k))

    # Solve Onsager quadratic for epsilon_r:
    # (eps - n^2)(2*eps + n^2) = y * eps * (n^2 + 2)^2
    # 2*eps^2 - eps*[n^2 + y*(n^2+2)^2] - n^4 = 0
    c_term = y_orient * ((n2 + 2.0) ** 2)
    b_prime = n2 + c_term
    disc = (b_prime**2) + 8.0 * (n2**2)
    eps_r = (b_prime + math.sqrt(max(0.0, disc))) / 4.0
    return max(1.8, min(250.0, float(eps_r)))


class SolventDatabase:
    """
    Generic, extensible physical solvent repository.
    Loads definitions dynamically from external JSON/YAML assets, preventing hardcoded tables in Python code.
    Supports runtime registration and extensible search paths.
    """

    _default_instance: Optional[SolventDatabase] = None

    def __init__(self, data_path: Optional[Path | str] = None):
        self._solvents: Dict[str, SolventProperties] = {}
        self._alias_map: Dict[str, str] = {}
        self._loaded_paths: List[Path] = []

        # Resolve data path
        candidate_paths = self._find_candidate_paths(data_path)
        for p in candidate_paths:
            if p.exists() and p.is_file():
                self.load_from_file(p)
                break

    @staticmethod
    def _find_candidate_paths(explicit_path: Optional[Path | str] = None) -> List[Path]:
        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        env_path = os.environ.get("SOLVENT_DATABASE_PATH")
        if env_path:
            candidates.append(Path(env_path))

        pkg_data = Path(__file__).resolve().parent.parent / "data" / "solvent_database.json"
        repo_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                pkg_data,
                Path.cwd() / "data" / "solvent_database.json",
                repo_root / "data" / "solvent_database.json",
                Path.cwd() / "data" / "test_data" / "solvent_database.json",
                repo_root / "data" / "test_data" / "solvent_database.json",
            ]
        )
        return candidates

    def load_from_file(self, path: Path | str) -> int:
        """Loads solvent definitions from an external JSON file."""
        p = Path(path)
        if not p.exists():
            return 0
        content = p.read_text(encoding="utf-8")
        data: Dict[str, Any] = json.loads(content)
        count = 0
        for name, item in data.items():
            aliases = tuple(item.get("aliases", []))
            props = SolventProperties(
                name=item.get("name", name).upper(),
                aliases=aliases,
                dielectric_constant=float(item.get("dielectric_constant", 2.0)),
                refractive_index=float(item.get("refractive_index", 1.40)),
                density_g_cm3=float(item.get("density_g_cm3", 0.85)),
                surface_tension_mn_m=float(item.get("surface_tension_mn_m", 25.0)),
                solvent_class=str(item.get("solvent_class", "alkane_nonpolar")),
                molecular_weight=float(item.get("molecular_weight", 100.0)),
                abraham_alpha=float(item.get("abraham_alpha", 0.0)),
                abraham_beta=float(item.get("abraham_beta", 0.0)),
                abraham_pi2=float(item.get("abraham_pi2", 0.0)),
                dipole_moment_debye=float(item.get("dipole_moment_debye", 0.0)),
            )
            self.register_solvent(props)
            count += 1
        self._loaded_paths.append(p)
        return count

    def register_solvent(self, props: SolventProperties) -> None:
        """Registers a solvent property object and indexes its canonical name and aliases."""
        canon = normalize_solvent_name(props.name)
        self._solvents[canon] = props
        self._alias_map[canon] = canon
        for alias in props.aliases:
            self._alias_map[normalize_solvent_name(alias)] = canon

    def get(self, name: str) -> Optional[SolventProperties]:
        """Retrieves SolventProperties by exact canonical name or registered alias."""
        clean = normalize_solvent_name(name)
        canon = self._alias_map.get(clean)
        if canon and canon in self._solvents:
            return self._solvents[canon]

        return None

    def list_solvents(self) -> Tuple[str, ...]:
        """Returns sorted tuple of registered canonical solvent names."""
        return tuple(sorted(self._solvents.keys()))

    @classmethod
    def get_default(cls) -> SolventDatabase:
        """Returns singleton default SolventDatabase instance."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance


@functools.lru_cache(maxsize=1024)
def derive_solvent_properties_from_structure(
    name: str,
    molecular_weight: Optional[float] = None,
    density_g_cm3: Optional[float] = None,
    dipole_moment_debye: Optional[float] = None,
    polarizability_a3: Optional[float] = None,
) -> SolventProperties:
    """
    First-principles and QSPR dynamic derivation of solvent parameters for arbitrary/unlisted fluids.
    Calculates dielectric constant via Onsager-Kirkwood-Fröhlich, refractive index via Lorentz-Lorenz
    and RDKit Molar Refractivity, and Abraham hydrogen-bonding factors via molecular graph descriptors.
    """
    clean = normalize_solvent_name(name)

    mol = None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski

        # Attempt parsing SMILES or InChI
        mol = Chem.MolFromSmiles(name)
        if mol is None:
            mol = Chem.MolFromInchi(name)
    except Exception:
        mol = None

    if mol is not None:
        mw = float(molecular_weight if molecular_weight is not None else Descriptors.ExactMolWt(mol))
        mr = float(Crippen.MolMR(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        tpsa = float(Descriptors.TPSA(mol))
        mol_h = Chem.AddHs(mol)

        # Accurate donor/acceptor counts from chemical graph (capturing protic solvents like water/polyols)
        hbd_atoms = sum(
            sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "H")
            for a in mol_h.GetAtoms()
            if a.GetSymbol() in ("O", "N", "S")
        )
        hba_atoms = sum(1 for a in mol_h.GetAtoms() if a.GetSymbol() in ("O", "N") and a.GetFormalCharge() <= 0)
        hbd_val = max(hbd, hbd_atoms)
        hba_val = max(hba, hba_atoms)

        is_aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
        is_hal = any(a.GetAtomicNum() in (9, 17, 35, 53) for a in mol.GetAtoms())
        has_ether = (
            any(
                a.GetAtomicNum() == 8 and a.GetDegree() == 2 and not any(b.GetIsAromatic() for b in a.GetNeighbors())
                for a in mol.GetAtoms()
            )
            and hbd_val == 0
        )

        if hbd_val > 0:
            s_class = "polar_protic"
            default_rho = 1.00
        elif is_hal:
            s_class = "chlorinated"
            default_rho = 1.30
        elif is_aromatic:
            s_class = "aromatic"
            default_rho = 0.88
        elif tpsa > 15.0:
            s_class = "polar_aprotic"
            default_rho = 0.95
        elif has_ether:
            s_class = "ether"
            default_rho = 0.75
        else:
            s_class = "alkane_nonpolar"
            default_rho = 0.75

        rho = float(density_g_cm3 if density_g_cm3 is not None else default_rho)

        # Electronic polarizability from Molar Refractivity:
        # MR = (4*pi / 3) * N_A * alpha => alpha (A^3) = (3 / (4*pi*N_A)) * MR approx 0.39643 * MR
        alpha_pol = float(polarizability_a3 if polarizability_a3 is not None else (0.39643 * mr))

        # Optical index from Lorenz-Lorentz:
        # (n^2 - 1) / (n^2 + 2) = (MR * rho) / mw
        fn = min(0.45, max(0.05, (mr * rho) / max(1.0, mw)))
        n_d = math.sqrt((1.0 + 2.0 * fn) / max(1e-4, 1.0 - fn))

        # Dipole moment estimation
        if dipole_moment_debye is not None:
            mu = float(dipole_moment_debye)
        else:
            mu = 0.0
            try:
                res = AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=10)
                if res == 0:
                    conf = mol_h.GetConformer()
                    dip = np.zeros(3)
                    mp = AllChem.MMFFGetMoleculeProperties(mol_h)
                    if mp is not None:
                        charges = [mp.GetMMFFPartialCharge(i) for i in range(mol_h.GetNumAtoms())]
                    else:
                        AllChem.ComputeGasteigerCharges(mol_h)
                        charges = [
                            float(a.GetProp("_GasteigerCharge"))
                            if a.HasProp("_GasteigerCharge")
                            and a.GetProp("_GasteigerCharge") not in ("nan", "-nan", "inf", "-inf")
                            else 0.0
                            for a in mol_h.GetAtoms()
                        ]
                    for i, atom in enumerate(mol_h.GetAtoms()):
                        pos = np.array(conf.GetAtomPosition(i))
                        dip += charges[i] * pos
                    mu = float(np.linalg.norm(dip) * 4.8032)
                else:
                    mu = float(max(0.0, 0.05 * tpsa))
            except Exception:
                mu = float(max(0.0, 0.05 * tpsa))

        alpha_h = float(min(1.5, 0.35 * hbd_val))
        beta_h = float(min(1.5, 0.25 * hba_val + 0.005 * tpsa))
        pi2 = float(min(1.5, 0.20 * mu + 0.005 * tpsa))

        # First-principles Kirkwood dipole correlation factor g_K for hydrogen-bonding networks:
        # Constructive dipole alignment in polar protic networks enhances effective dipole moment
        hbond_factor = alpha_h * beta_h
        gk = 1.0 + 2.2 * hbond_factor + 0.4 * alpha_h if hbond_factor > 0.05 else (1.0 + 0.3 * alpha_h)

        # Dielectric constant via Onsager-Kirkwood-Frohlich
        eps_r = estimate_dielectric_from_polarizability_and_dipole(
            dipole_debye=mu,
            polarizability_angstrom3=alpha_pol,
            molecular_weight=mw,
            density_g_cm3=rho,
            kirkwood_g=gk,
        )

        # First-principles cohesive surface tension based on Stefan-Eötvös cohesive density:
        vm = mw / max(0.1, rho)
        vm_area = max(1.0, vm ** (2.0 / 3.0))
        gamma = float(18.0 + 10.0 * pi2 + 950.0 * hbond_factor / vm_area)
    else:
        # Fallback for unparseable solvent labels (derive class from chemical name keywords)
        clean_lower = clean.lower()
        if any(k in clean_lower for k in ("ether", "ox", "furan", "thf", "glyme", "dioxane")):
            s_class = "ether"
            default_rho = 0.75
            default_gamma = 24.0
        elif any(k in clean_lower for k in ("chloro", "bromo", "fluoro", "iodo", "halo")):
            s_class = "chlorinated"
            default_rho = 1.30
            default_gamma = 27.0
        elif any(k in clean_lower for k in ("benzene", "tolu", "xyl", "aromat")):
            s_class = "aromatic"
            default_rho = 0.88
            default_gamma = 28.5
        elif any(k in clean_lower for k in ("ol", "water", "acid", "amine")):
            s_class = "polar_protic"
            default_rho = 1.00
            default_gamma = 30.0
        elif any(k in clean_lower for k in ("nitrile", "sulfoxide", "formamide", "ketone", "one", "ester", "acetate")):
            s_class = "polar_aprotic"
            default_rho = 0.95
            default_gamma = 26.0
        else:
            s_class = "alkane_nonpolar"
            default_rho = 0.75
            default_gamma = 22.0

        mw = molecular_weight if molecular_weight is not None else 100.0
        rho = density_g_cm3 if density_g_cm3 is not None else default_rho
        if dipole_moment_debye is not None:
            mu = float(dipole_moment_debye)
        else:
            default_mu_map = {
                "ether": 1.2,
                "chlorinated": 1.5,
                "polar_protic": 1.7,
                "polar_aprotic": 2.8,
                "aromatic": 0.3,
                "alkane_nonpolar": 0.0,
            }
            mu = default_mu_map.get(s_class, 0.0)
        alpha_pol = polarizability_a3 if polarizability_a3 is not None else (mw / 10.0)
        alpha_h, beta_h, pi2 = 0.0, 0.0, 0.0
        n_d = 1.40
        gamma = default_gamma
        eps_r = estimate_dielectric_from_polarizability_and_dipole(
            dipole_debye=mu,
            polarizability_angstrom3=alpha_pol,
            molecular_weight=mw,
            density_g_cm3=rho,
        )

    props = SolventProperties(
        name=clean,
        aliases=(),
        dielectric_constant=float(round(eps_r, 2)),
        refractive_index=float(round(n_d, 4)),
        density_g_cm3=float(rho),
        surface_tension_mn_m=float(gamma),
        solvent_class=s_class,
        molecular_weight=float(mw),
        abraham_alpha=float(alpha_h),
        abraham_beta=float(beta_h),
        abraham_pi2=float(pi2),
        dipole_moment_debye=float(mu),
    )

    # Dynamically register into database for future O(1) queries
    SolventDatabase.get_default().register_solvent(props)
    return props


VACUUM_PROPERTIES = SolventProperties(
    name="vacuum",
    aliases=("vacuum", "gas", "vapor", "none", ""),
    dielectric_constant=1.0,
    refractive_index=1.0,
    density_g_cm3=0.0,
    surface_tension_mn_m=0.0,
    solvent_class="vacuum",
    molecular_weight=0.0,
    abraham_alpha=0.0,
    abraham_beta=0.0,
    abraham_pi2=0.0,
    dipole_moment_debye=0.0,
)


@functools.lru_cache(maxsize=1024)
def get_solvent_properties(name: str) -> SolventProperties:
    """
    Retrieves full physical properties for a solvent by canonical name or alias.
    Falls back to first-principles QSPR derivation if not in the external database.
    """
    if not name or str(name).strip().lower() in ("vacuum", "gas", "vapor", "none", ""):
        return VACUUM_PROPERTIES

    db = SolventDatabase.get_default()
    props = db.get(name)
    if props is not None:
        return props

    # Dynamic first-principles derivation fallback
    return derive_solvent_properties_from_structure(name)


@functools.lru_cache(maxsize=1024)
def get_solvent_dielectric(
    name: str,
    default: Optional[float] = None,
    temp_k: Optional[float] = None,
) -> float:
    """
    Returns relative permittivity epsilon_r for a solvent, with dynamic property lookup
    and optional temperature adjustment. Raises KeyError if solvent is unregistered and no default is provided.
    """
    props = None
    if not name or str(name).strip().lower() in ("vacuum", "gas", "vapor", "none", ""):
        props = VACUUM_PROPERTIES
    else:
        db = SolventDatabase.get_default()
        props = db.get(name)

    if props is None:
        if default is not None:
            return float(default)
        raise KeyError(f"Solvent '{name}' not found in SolventDatabase and no default dielectric provided.")

    eps_298 = float(props.dielectric_constant)
    if temp_k is not None and temp_k > 0:
        # Temperature adjustment: eps(T) = eps_0 * exp(-b * (T - 298.15))
        temp_diff = temp_k - 298.15
        b_coef = 0.0046 if props.solvent_class == "polar_protic" else 0.0020
        return float(max(1.0, eps_298 * math.exp(-b_coef * temp_diff)))
    return eps_298


def get_solvent_descriptors_vector(name: str) -> np.ndarray:
    """
    Returns a standardized 7-dimensional physical descriptor vector for the solvent:
    0: Onsager dielectric factor f_eps = (eps - 1) / (2*eps + 1) in [0, 0.5]
    1: Lorentz-Lorenz polarizability factor f_n = (n^2 - 1) / (n^2 + 2) in [0.15, 0.4]
    2: Continuous cavitation surface tension scale gamma / (gamma + 25.0) in [0, 1)
    3: Log-scaled molar volume ln(1 + V_m / 50.0)
    4: Abraham hydrogen-bond acidity alpha in [0.0, 1.5]
    5: Abraham hydrogen-bond basicity beta in [0.0, 1.5]
    6: Bounded dipole moment tanh(mu / 3.0) in [0, 1)
    """
    props = get_solvent_properties(name)
    eps = max(1.0, props.dielectric_constant)
    f_eps = (eps - 1.0) / (2.0 * eps + 1.0)
    n = max(1.0, props.refractive_index)
    f_n = (n**2 - 1.0) / (n**2 + 2.0)
    gamma_scale = props.surface_tension_mn_m / max(1.0, props.surface_tension_mn_m + 25.0)
    v_m_scale = math.log1p(max(0.0, props.molar_volume_cm3_mol) / 50.0)
    alpha = props.abraham_alpha
    beta = props.abraham_beta
    dipole_norm = math.tanh(max(0.0, props.dipole_moment_debye) / 3.0)
    return np.array([f_eps, f_n, gamma_scale, v_m_scale, alpha, beta, dipole_norm], dtype=np.float32)


def list_registered_solvents() -> Tuple[str, ...]:
    """Returns tuple of all canonically registered solvent names."""
    return SolventDatabase.get_default().list_solvents()
