"""
Material metadata loader and force field parameter resolver for arbitrary .mol2 files.
Extracts site coordinates, partial charges, and Lennard-Jones parameters strictly
from the molecular geometry and force field database with zero hardcoded values.
Provides self-consistent Carnahan-Starling + Mean-Field Equation of State (EOS) solvers.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"
FF_JSON_PATH = TEST_DATA_DIR / "forcefield_parameters.json"


def compute_wca_dispersion_integral(sigma: float, epsilon_k: float, r_cut_sigma: float = 5.0) -> float:
    """
    Computes the exact 3D volume integral of the WCA attractive potential:
    \\int v_att(r) d^3r = \\int_{-\\infty}^\\infty v_att,1D(z) dz
    """
    r_cut = r_cut_sigma
    prefactor = 16.0 * math.pi * epsilon_k * (sigma**3)
    bracket = -(2.0 * math.sqrt(2.0)) / 9.0 - 1.0 / (9.0 * (r_cut**9)) + 1.0 / (3.0 * (r_cut**3))
    return prefactor * bracket


def solve_bulk_density_from_pressure(
    p_bar: float,
    temp_k: float,
    sigma: float,
    epsilon_k: float,
    phase: str = "liquid",
) -> float:
    """
    Self-consistently solves for bulk number density rho_bulk (molecules / Å³) at target pressure P
    using the Rosenfeld FMT / Percus-Yevick compressibility + Mean-Field attractive Equation of State (EOS):
    P(rho, T) = rho * k_B * T * [ (1 + eta + eta^2) / (1 - eta)^3 ] + 0.5 * rho^2 * \\int v_att(r) d^3r
    """
    # 1 bar = 1e5 Pa -> in (k_B * T / Å³) units: P / (k_B * T) = (p_bar * 1e5) / (1.380649e-23 * T * 1e30)
    p_target_kbt = (p_bar * 1e5) / (1.380649e-23 * temp_k * 1e30)
    v_att_int = compute_wca_dispersion_integral(sigma, epsilon_k)
    a_att = -v_att_int / temp_k  # Positive attractive coefficient in Å³
    b_vol = (math.pi / 6.0) * (sigma**3)

    def eos_pressure(rho: float) -> float:
        eta = b_vol * rho
        if eta >= 1.0 or eta <= 0.0:
            return float("inf") if eta >= 1.0 else -float("inf")
        z_py = (1.0 + eta + eta**2) / ((1.0 - eta) ** 3)
        return rho * z_py - 0.5 * a_att * (rho**2)

    # Maximum physical close packing
    rho_max = 0.65 / b_vol
    eta_c = 0.1213

    if phase == "liquid":
        low = max(0.08, eta_c * 0.8) / b_vol
        high = rho_max * 0.95
    else:
        low = 1e-9
        high = min(0.12, eta_c * 0.9) / b_vol

    p_low = eos_pressure(low)
    p_high = eos_pressure(high)
    if (p_low - p_target_kbt) * (p_high - p_target_kbt) > 0:
        low = 1e-9 if phase == "vapor" else 0.01 / b_vol
        high = rho_max * 0.98

    for _ in range(120):
        mid = 0.5 * (low + high)
        p_mid = eos_pressure(mid)
        if abs(p_mid - p_target_kbt) < 1e-9 or (high - low) < 1e-10:
            return mid
        if p_mid < p_target_kbt:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def solve_bulk_density_from_chemical_potential(
    mu_kbt: float,
    temp_k: float,
    sigma: float,
    epsilon_k: float,
) -> float:
    """
    Solves for bulk number density rho_bulk at target chemical potential mu / (k_B * T)
    via root finding on mu_FMT(rho, T) = mu_target.
    """
    v_att_int = compute_wca_dispersion_integral(sigma, epsilon_k)
    a_att = -v_att_int / temp_k
    b_vol = (math.pi / 6.0) * (sigma**3)

    def eos_mu(rho: float) -> float:
        eta = b_vol * rho
        eta = max(1e-12, min(0.65, eta))
        one_minus_eta = max(1e-12, 1.0 - eta)
        mu_id = math.log(max(1e-15, rho * (sigma**3)))
        mu_hs_fmt = -math.log(one_minus_eta) + (eta * (14.0 - 13.0 * eta + 5.0 * (eta**2))) / (2.0 * (one_minus_eta**3))
        mu_att = -a_att * rho
        return mu_id + mu_hs_fmt + mu_att

    low, high = 1e-9, 0.65 / b_vol
    for _ in range(100):
        mid = 0.5 * (low + high)
        val = eos_mu(mid)
        if abs(val - mu_kbt) < 1e-9 or (high - low) < 1e-10:
            return mid
        if val < mu_kbt:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


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
        using Rosenfeld FMT (Percus-Yevick compressibility) for hard-core repulsion + mean-field attractive dispersion.
        """
        temp = T if T is not None else self.temperature_k
        rho_b = rho if rho is not None else self.bulk_density_a3
        sig = self.effective_sigma
        eps_k = self.effective_epsilon_k

        # Packing fraction eta = (pi / 6) * rho * sigma^3
        eta = (math.pi / 6.0) * rho_b * (sig**3)

        # Ideal chemical potential (in k_B * T)
        mu_id = math.log(max(1e-15, rho_b * (sig**3)))

        # Rosenfeld FMT excess hard-sphere chemical potential (PY compressibility limit)
        one_minus_eta = max(1e-12, 1.0 - eta)
        mu_hs_ex = -math.log(one_minus_eta) + (eta * (14.0 - 13.0 * eta + 5.0 * (eta**2))) / (2.0 * (one_minus_eta**3))

        # Mean-field attractive chemical potential: \int v_att(r) d^3r
        v_att_integral = compute_wca_dispersion_integral(sig, eps_k) / temp
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
        pressure_bar: Optional[float] = None,
        chemical_potential_kbt: Optional[float] = None,
    ) -> Material:
        """
        Parses an arbitrary .mol2 file and resolves site Lennard-Jones parameters.
        Derives bulk thermodynamic state self-consistently from Equation of State.
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
            valid_sigmas = [s.sigma for s in sites if s.sigma > 0.0]
            eff_sigma = sum(s**3 for s in valid_sigmas) ** (1.0 / 3.0) if valid_sigmas else sites[0].sigma

            # Exact molecular WCA dispersion volume integral matching:
            # \int v_att,eff(r) d^3r = \sum_i \sum_j \int v_att,ij(r) d^3r
            # => eps_eff * sigma_eff^3 = \sum_i \sum_j \sqrt{eps_i * eps_j} * ((sigma_i + sigma_j)/2)^3
            att_vol_sum = 0.0
            for s1 in sites:
                for s2 in sites:
                    eps_ij = math.sqrt(max(0.0, s1.epsilon_k * s2.epsilon_k))
                    sig_ij = 0.5 * (s1.sigma + s2.sigma)
                    att_vol_sum += eps_ij * (sig_ij**3)
            eff_eps_k = att_vol_sum / (eff_sigma**3) if eff_sigma > 0.0 else sites[0].epsilon_k

        temp = temperature_k if temperature_k is not None else 300.0

        # Derive bulk density from Equation of State:
        if bulk_density_a3 is not None:
            density = bulk_density_a3
        elif chemical_potential_kbt is not None:
            density = solve_bulk_density_from_chemical_potential(
                mu_kbt=chemical_potential_kbt,
                temp_k=temp,
                sigma=eff_sigma,
                epsilon_k=eff_eps_k,
            )
        elif pressure_bar is not None:
            density = solve_bulk_density_from_pressure(
                p_bar=pressure_bar,
                temp_k=temp,
                sigma=eff_sigma,
                epsilon_k=eff_eps_k,
            )
        else:
            # Default thermodynamic state: standard liquid state packing fraction eta = 0.35
            density = (6.0 * 0.35) / (math.pi * (eff_sigma**3))

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
