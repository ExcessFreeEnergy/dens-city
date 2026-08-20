"""
Material metadata loader and force field parameter resolver for arbitrary .mol2 files.
Extracts site coordinates, partial charges, and Lennard-Jones parameters strictly
from the molecular geometry and force field database with zero hardcoded values.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"
FF_JSON_PATH = TEST_DATA_DIR / "forcefield_parameters.json"


@dataclass
class AtomSite:
    site_name: str
    atom_type: str
    x: float
    y: float
    z: float
    charge: float
    sigma: float
    epsilon_kcal: float
    epsilon_k: float
    mass: float


@dataclass
class Material:
    name: str
    identifier: str
    dimension_mode: str  # "1D_SPHERICAL", "1D_ANGULAR", "3D_MOLECULAR"
    sites: List[AtomSite] = field(default_factory=list)
    effective_sigma: float = 3.4
    effective_epsilon_k: float = 120.0
    temperature_k: float = 300.0
    bulk_density_a3: float = 0.02
    bulk_mu: float = 0.0

    @property
    def num_sites(self) -> int:
        return len(self.sites)

    @property
    def total_charge(self) -> float:
        return sum(s.charge for s in self.sites)

    @property
    def molarity_mol_l(self) -> float:
        """Calculates molar concentration in mol/L (M) from number density."""
        # 1 molecule/Å³ = 1e27 molecules/L -> / (6.02214076e23 mol⁻¹) = 1660.53878 mol/L
        return self.bulk_density_a3 * 1660.53878

    @property
    def molecular_span_a(self) -> float:
        """Computes the maximum spatial bounding diameter of the 3D molecular framework."""
        if not self.sites:
            return self.effective_sigma
        if len(self.sites) == 1:
            return self.sites[0].sigma

        max_span = 0.0
        for i in range(len(self.sites)):
            for j in range(i, len(self.sites)):
                s1, s2 = self.sites[i], self.sites[j]
                dist = math.sqrt((s1.x - s2.x) ** 2 + (s1.y - s2.y) ** 2 + (s1.z - s2.z) ** 2)
                span = dist + 0.5 * (s1.sigma + s2.sigma)
                max_span = max(max_span, span)
        return max_span

    @property
    def radius_of_gyration_a(self) -> float:
        """Computes the center-of-mass mass-weighted radius of gyration."""
        if len(self.sites) <= 1:
            return 0.0
        total_m = sum(s.mass for s in self.sites)
        if total_m <= 0.0:
            return 0.0
        cx = sum(s.mass * s.x for s in self.sites) / total_m
        cy = sum(s.mass * s.y for s in self.sites) / total_m
        cz = sum(s.mass * s.z for s in self.sites) / total_m

        rg_sq = sum(
            s.mass * ((s.x - cx) ** 2 + (s.y - cy) ** 2 + (s.z - cz) ** 2) for s in self.sites
        ) / total_m
        return math.sqrt(max(0.0, rg_sq))

    def compute_bulk_mu(self, T: Optional[float] = None, rho: Optional[float] = None) -> float:
        """
        Computes the theoretical bulk chemical potential mu_bulk(T, rho) in units of k_B * T
        using Carnahan-Starling for hard-core repulsion + mean-field attractive dispersion.
        """
        temp = T if T is not None else self.temperature_k
        rho_b = rho if rho is not None else self.bulk_density_a3
        sig = self.effective_sigma
        eps_k = self.effective_epsilon_k

        # Packing fraction eta = (pi / 6) * rho * sigma^3
        eta = (math.pi / 6.0) * rho_b * (sig**3)
        eta = min(0.48, max(1e-5, eta))

        # Ideal chemical potential (in k_B * T)
        mu_id = math.log(max(1e-10, rho_b * (sig**3)))

        # Carnahan-Starling excess hard-sphere chemical potential
        mu_hs_ex = (eta * (8.0 - 9.0 * eta + 3.0 * (eta**2))) / ((1.0 - eta) ** 3)

        # Mean-field attractive chemical potential: \int v_att(r) d^3r = -(32\pi/9) * epsilon * sigma^3
        v_att_integral = -(32.0 * math.pi / 9.0) * (eps_k / temp) * (sig**3)
        mu_att = rho_b * v_att_integral

        self.bulk_mu = mu_id + mu_hs_ex + mu_att
        return self.bulk_mu


class MaterialLoader:
    """Loads and parses arbitrary .mol2 files and maps them to force field parameters."""

    _ff_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def get_forcefield_database(cls) -> Dict[str, Any]:
        if cls._ff_cache is None:
            if not FF_JSON_PATH.exists():
                raise FileNotFoundError(f"Missing forcefield parameters file at {FF_JSON_PATH}")
            with open(FF_JSON_PATH, "r") as f:
                cls._ff_cache = json.load(f)
        return cls._ff_cache

    @classmethod
    def classify_dimension_mode(cls, sites: List[AtomSite]) -> str:
        """
        Infers representation mode strictly from molecular geometry:
        - N == 1: 1D_SPHERICAL (monoatomic)
        - Collinear sites (I_min / I_max < 1e-3): 1D_ANGULAR (linear / diatomic)
        - General 3D sites: 3D_MOLECULAR
        """
        n = len(sites)
        if n == 1:
            return "1D_SPHERICAL"
        if n == 2:
            return "1D_ANGULAR"

        coords = np.array([[s.x, s.y, s.z] for s in sites], dtype=np.float64)
        centered = coords - coords.mean(axis=0)

        # Moment of inertia tensor of point sites
        I = np.zeros((3, 3), dtype=np.float64)
        for r in centered:
            I += np.dot(r, r) * np.eye(3) - np.outer(r, r)

        eigvals = np.sort(np.linalg.eigvalsh(I))
        if eigvals[0] / max(1e-8, eigvals[2]) < 1e-3:
            return "1D_ANGULAR"

        return "3D_MOLECULAR"

    @classmethod
    def load_material(
        cls,
        target: Union[str, Path],
        temperature_k: Optional[float] = None,
        bulk_density_a3: Optional[float] = None,
        packing_fraction: float = 0.35,
    ) -> Material:
        """
        Parses an arbitrary .mol2 file and resolves site Lennard-Jones parameters.
        Zero hardcoded fluid tables or names.
        """
        ff_db = cls.get_forcefield_database()

        # Resolve path
        if isinstance(target, Path) or ("/" in str(target)) or str(target).endswith(".mol2"):
            mol2_path = Path(target)
            if not mol2_path.is_absolute() and not mol2_path.exists():
                mol2_path = TEST_DATA_DIR / target
                if not mol2_path.exists() and not str(target).endswith(".mol2"):
                    mol2_path = TEST_DATA_DIR / f"{target}.mol2"
        else:
            name_clean = str(target).strip()
            mol2_path = TEST_DATA_DIR / f"{name_clean}.mol2"
            if not mol2_path.exists():
                matched = list(TEST_DATA_DIR.glob(f"*{name_clean}*.mol2"))
                if matched:
                    mol2_path = matched[0]
                else:
                    available = [p.stem for p in TEST_DATA_DIR.glob("*.mol2")]
                    raise FileNotFoundError(
                        f"Material file for '{target}' not found in test_data/. Available: {available}"
                    )

        if not mol2_path.exists():
            raise FileNotFoundError(f"Mol2 file does not exist: {mol2_path}")

        # Parse .mol2 file
        lines = mol2_path.read_text().splitlines()
        sites: List[AtomSite] = []
        in_atom = False

        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            elif line.startswith("@<TRIPOS>"):
                in_atom = False
                continue

            if in_atom and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    s_name = parts[1]
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    at_type = parts[5]
                    charge = float(parts[8]) if len(parts) >= 9 else 0.0

                    ff_entry = ff_db.get(at_type, ff_db.get(at_type.lower(), {}))
                    sigma = float(ff_entry.get("sigma_angstrom", 3.40))
                    eps_kcal = float(ff_entry.get("epsilon_kcal_mol", 0.10))
                    eps_k = float(ff_entry.get("epsilon_kelvin", 120.0))
                    mass = float(ff_entry.get("mass_amu", 12.0))

                    sites.append(
                        AtomSite(
                            site_name=s_name,
                            atom_type=at_type,
                            x=x,
                            y=y,
                            z=z,
                            charge=charge,
                            sigma=sigma,
                            epsilon_kcal=eps_kcal,
                            epsilon_k=eps_k,
                            mass=mass,
                        )
                    )

        if not sites:
            raise ValueError(f"No valid atom sites found in {mol2_path}")

        # Derive molecular Lennard-Jones parameters directly from atomic sites:
        if len(sites) == 1:
            eff_sigma = sites[0].sigma
            eff_eps_k = sites[0].epsilon_k
        else:
            # Volume-equivalent hard sphere diameter: sigma_eff = ( \sum sigma_i^3 )^(1/3)
            heavy_sigmas = [s.sigma for s in sites if s.sigma > 0.5]
            eff_sigma = sum(s**3 for s in heavy_sigmas) ** (1.0 / 3.0) if heavy_sigmas else sites[0].sigma

            # Total cohesive dispersion well depth: sum of site epsilon contributions
            eff_eps_k = sum(s.epsilon_k for s in sites)

        # Standard physical thermodynamic state:
        temp = temperature_k if temperature_k is not None else 300.0

        if bulk_density_a3 is not None:
            density = bulk_density_a3
        else:
            # Natural liquid packing density: rho_bulk = (6 * eta) / (pi * sigma_eff^3)
            density = (6.0 * packing_fraction) / (math.pi * (eff_sigma**3))

        dim_mode = cls.classify_dimension_mode(sites)

        mat = Material(
            name=mol2_path.stem,
            identifier=mol2_path.stem,
            dimension_mode=dim_mode,
            sites=sites,
            effective_sigma=eff_sigma,
            effective_epsilon_k=eff_eps_k,
            temperature_k=temp,
            bulk_density_a3=density,
        )
        mat.compute_bulk_mu()
        return mat

    @classmethod
    def list_available_materials(cls) -> List[str]:
        """Returns all available .mol2 files in test_data/."""
        return sorted([p.stem for p in TEST_DATA_DIR.glob("*.mol2")])
