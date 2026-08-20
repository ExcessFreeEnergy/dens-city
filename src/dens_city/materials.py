"""
Material metadata loader, force field resolver, and representation classifier.
Ingests molecular models from test_data/ and sets up physical thermodynamic parameters.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    effective_sigma: float = 3.405
    effective_epsilon_k: float = 119.8
    temperature_k: float = 300.0
    bulk_density_a3: float = 0.02  # molecules / Angstrom^3
    bulk_mu: float = 0.0

    @property
    def num_sites(self) -> int:
        return len(self.sites)

    @property
    def total_charge(self) -> float:
        return sum(s.charge for s in self.sites)

    def compute_bulk_mu(self, T: Optional[float] = None, rho: Optional[float] = None) -> float:
        """
        Computes bulk chemical potential mu_bulk(T, rho) in units of k_B * T
        using Carnahan-Starling for hard-core repulsion + mean-field attractive dispersion.
        """
        temp = T if T is not None else self.temperature_k
        rho_b = rho if rho is not None else self.bulk_density_a3
        sig = self.effective_sigma
        eps_k = self.effective_epsilon_k

        # Packing fraction
        eta = (math.pi / 6.0) * rho_b * (sig**3)
        eta = min(0.48, max(1e-5, eta))

        # Ideal chemical potential (in k_B * T)
        mu_id = math.log(max(1e-10, rho_b * (sig**3)))

        # Carnahan-Starling excess hard-sphere chemical potential
        mu_hs_ex = (eta * (8.0 - 9.0 * eta + 3.0 * (eta**2))) / ((1.0 - eta) ** 3)

        # Mean-field attractive chemical potential: \int v_att(r) d^3r = -(32\pi/9) * epsilon * sigma^3
        # In units of k_B * T:
        v_att_integral = -(32.0 * math.pi / 9.0) * (eps_k / temp) * (sig**3)
        mu_att = rho_b * v_att_integral

        self.bulk_mu = mu_id + mu_hs_ex + mu_att
        return self.bulk_mu


class MaterialLoader:
    """Loads and standardizes molecular force field datasets for cDFT simulations."""

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
    def classify_dimension_mode(cls, name: str, sites: List[AtomSite]) -> str:
        """Infers optimal cDFT representation mode based on molecular geometry and symmetry."""
        name_lower = name.lower()
        if len(sites) == 1:
            return "1D_SPHERICAL"

        # Linear diatomics / triatomics / nematics -> 1D + Angular
        if name_lower in ["nitrogen", "carbon_dioxide", "hydrogen_fluoride", "hydrogen", "5cb"]:
            return "1D_ANGULAR"

        # Spherical / high-symmetry polyatomics -> 1D Spherical effective representation
        if name_lower in ["methane", "neopentane", "sulfur_hexafluoride", "sodium_chloride", "calcium_chloride", "colloidal_hard_sphere"]:
            return "1D_SPHERICAL"

        # Arbitrary 3D molecular solutes
        return "3D_MOLECULAR"

    @classmethod
    def load_material(
        cls,
        name: str,
        temperature_k: Optional[float] = None,
        bulk_density_a3: Optional[float] = None,
    ) -> Material:
        """Loads a material from test_data/ by name or identifier."""
        ff_db = cls.get_forcefield_database()
        name_clean = name.lower().replace("-", "_").replace(" ", "_")

        # Map common aliases
        alias_map = {
            "water_spce": "water",
            "co2": "carbon_dioxide",
            "n2": "nitrogen",
            "hf": "hydrogen_fluoride",
            "nacl": "sodium_chloride",
            "cacl2": "calcium_chloride",
            "sf6": "sulfur_hexafluoride",
            "decane": "n_decane",
            "colloid": "colloidal_hard_sphere",
            "h2": "hydrogen",
            "h2o": "water",
        }
        resolved_name = alias_map.get(name_clean, name_clean)
        mol2_path = TEST_DATA_DIR / f"{resolved_name}.mol2"

        if not mol2_path.exists():
            # Try fuzzy match
            matched = list(TEST_DATA_DIR.glob(f"*{resolved_name}*.mol2"))
            if matched:
                mol2_path = matched[0]
            else:
                available = [p.stem for p in TEST_DATA_DIR.glob("*.mol2")]
                raise FileNotFoundError(
                    f"Material '{name}' not found in test_data/. Available materials: {available}"
                )

        # Parse .mol2
        lines = mol2_path.read_text().splitlines()
        sites = []
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
                    sigma = float(ff_entry.get("sigma_angstrom", 3.4))
                    eps_kcal = float(ff_entry.get("epsilon_kcal_mol", 0.1))
                    eps_k = float(ff_entry.get("epsilon_kelvin", 100.0))
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

        # Determine effective physical parameters based on primary non-zero interaction sites
        non_zero_sigmas = [s.sigma for s in sites if s.sigma > 0.5]
        eff_sigma = max(non_zero_sigmas) if non_zero_sigmas else (max((s.sigma for s in sites), default=3.405))

        non_zero_eps = [s.epsilon_k for s in sites if s.epsilon_k > 1.0]
        eff_eps_k = max(non_zero_eps) if non_zero_eps else (max((s.epsilon_k for s in sites), default=119.8))

        # System defaults
        temp = temperature_k if temperature_k is not None else 300.0
        # Typical dense liquid / gas density: water ~ 0.033 A^-3, argon ~ 0.021 A^-3
        if bulk_density_a3 is not None:
            density = bulk_density_a3
        elif resolved_name == "water":
            density = 0.0333  # SPC/E experimental liquid water at 300K
        else:
            density = 0.0210  # Standard liquid/dense fluid density

        dim_mode = cls.classify_dimension_mode(resolved_name, sites)

        mat = Material(
            name=resolved_name,
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
        """Returns all available materials in test_data/."""
        return sorted([p.stem for p in TEST_DATA_DIR.glob("*.mol2")])
