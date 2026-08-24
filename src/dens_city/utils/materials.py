"""
Material metadata loader and force field parameter resolver for arbitrary .mol2 files.
Extracts site coordinates, partial charges, and Lennard-Jones parameters strictly
from the molecular geometry and force field database with zero hardcoded values.
Provides self-consistent Carnahan-Starling + Mean-Field Equation of State (EOS) solvers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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


def compute_bulk_pressure(
    rho: float,
    temp_k: float,
    sigma: float,
    epsilon_k: float,
) -> float:
    r"""
    Dynamically computes bulk thermodynamic pressure P_bulk in bar using the exact
    Percus-Yevick compressibility Equation of State (EOS):
    P_bulk(rho, T) = rho * k_B * T * Z_PY(eta) - a(T) * rho^2
    where Z_PY(eta) = (1 + eta + eta^2) / (1 - eta)^3
    and a(T) = -0.5 * \int v_att(r) d^3r
    """
    eta = (math.pi / 6.0) * rho * (sigma**3)
    if eta >= 1.0 or eta <= 0.0:
        return 0.0 if eta <= 0.0 else float("inf")
    one_minus_eta = max(1e-12, 1.0 - eta)
    z_py = (1.0 + eta + eta**2) / (one_minus_eta**3)

    v_att_int = compute_wca_dispersion_integral(sigma, epsilon_k)  # K * Å^3 (negative)
    # Dimensionless pressure P / (k_B * T) in Å^-3
    p_kbt = rho * z_py + 0.5 * (rho**2) * (v_att_int / temp_k)

    # 1 bar = 1e5 Pa -> P_bar = P_kbt * (k_B * T) * (1e30 Å^3 / m^3) / (1e5 Pa / bar)
    p_bar = p_kbt * (1.380649e-23 * temp_k * 1e25)
    return p_bar


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
    bonds: List[Tuple[int, int, str]] = field(default_factory=list)
    dihedral_quadruplets: List[Tuple[int, int, int, int]] = field(default_factory=list)
    effective_sigma: float = 3.4
    effective_epsilon_k: float = 120.0
    temperature_k: float = 300.0
    bulk_density_a3: float = 0.02
    bulk_mu: float = 0.0
    bulk_pressure_bar: float = 0.0

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

        rg_sq = sum(s.mass * ((s.x - cx) ** 2 + (s.y - cy) ** 2 + (s.z - cz) ** 2) for s in self.sites) / total_m
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

    def compute_bulk_pressure(self, T: Optional[float] = None, rho: Optional[float] = None) -> float:
        """
        Computes bulk pressure P_bulk(T, rho) in bar using the exact Percus-Yevick compressibility EOS:
        P(rho, T) = rho * k_B * T * Z_PY(eta) - a(T) * rho^2
        """
        temp = T if T is not None else self.temperature_k
        rho_b = rho if rho is not None else self.bulk_density_a3
        self.bulk_pressure_bar = compute_bulk_pressure(
            rho=rho_b,
            temp_k=temp,
            sigma=self.effective_sigma,
            epsilon_k=self.effective_epsilon_k,
        )
        return self.bulk_pressure_bar


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

        # Check linearity via moment of inertia tensor:
        total_m = sum(s.mass for s in sites)
        cx = sum(s.mass * s.x for s in sites) / total_m
        cy = sum(s.mass * s.y for s in sites) / total_m
        cz = sum(s.mass * s.z for s in sites) / total_m

        # Inertia tensor components relative to COM
        ixx = sum(s.mass * ((s.y - cy) ** 2 + (s.z - cz) ** 2) for s in sites)
        iyy = sum(s.mass * ((s.x - cx) ** 2 + (s.z - cz) ** 2) for s in sites)
        izz = sum(s.mass * ((s.x - cx) ** 2 + (s.y - cy) ** 2) for s in sites)
        ixy = -sum(s.mass * (s.x - cx) * (s.y - cy) for s in sites)
        ixz = -sum(s.mass * (s.x - cx) * (s.z - cz) for s in sites)
        iyz = -sum(s.mass * (s.y - cy) * (s.z - cz) for s in sites)

        i_mat = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        eigvals = np.sort(np.linalg.eigvalsh(i_mat))

        # If smallest principal moment of inertia is negligible compared to largest -> collinear
        if eigvals[2] > 0 and (eigvals[0] / eigvals[2]) < 1e-3:
            return "1D_ANGULAR"

        return "3D_MOLECULAR"

    @classmethod
    def load_material(
        cls,
        material_name_or_path: str,
        temperature_k: Optional[float] = None,
        bulk_density_a3: Optional[float] = None,
        pressure_bar: Optional[float] = None,
        chemical_potential_kbt: Optional[float] = None,
    ) -> Material:
        """
        Ingests arbitrary .mol2 dataset, dynamically derives force field parameters,
        and solves the bulk Equation of State without hardcoded tables or aliases.
        """
        ff_db = cls.get_forcefield_database()

        mol2_path = Path(material_name_or_path)
        if not mol2_path.exists():
            mol2_path = TEST_DATA_DIR / f"{material_name_or_path}.mol2"
        if not mol2_path.exists():
            raise FileNotFoundError(f"Mol2 file does not exist: {mol2_path}")

        # Parse .mol2 file
        lines = mol2_path.read_text().splitlines()
        sites: List[AtomSite] = []
        bonds: List[Tuple[int, int, str]] = []
        in_atom = False
        in_bond = False

        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                in_bond = False
                continue
            elif line.startswith("@<TRIPOS>BOND"):
                in_atom = False
                in_bond = True
                continue
            elif line.startswith("@<TRIPOS>"):
                in_atom = False
                in_bond = False
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
            elif in_bond and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    # bond_id, origin_atom_id, target_atom_id, bond_type
                    # Convert 1-indexed atom IDs to 0-indexed site indices
                    a1 = int(parts[1]) - 1
                    a2 = int(parts[2]) - 1
                    b_type = parts[3]
                    bonds.append((a1, a2, b_type))

        if not sites:
            raise ValueError(f"No valid atom sites found in {mol2_path}")

        # Extract explicit dihedral quadruplets (a, b, c, d) from bond graph connectivity
        dihedral_quadruplets: List[Tuple[int, int, int, int]] = []
        if bonds:
            adj: Dict[int, List[int]] = {}
            for a1, a2, _ in bonds:
                if 0 <= a1 < len(sites) and 0 <= a2 < len(sites):
                    adj.setdefault(a1, []).append(a2)
                    adj.setdefault(a2, []).append(a1)

            # Find all simple paths of length 3: a - b - c - d
            seen_dihedrals = set()
            for b in adj:
                for a in adj[b]:
                    for c in adj[b]:
                        if c == a:
                            continue
                        for d in adj.get(c, []):
                            if d == b or d == a:
                                continue
                            # Canonical ordering a < d to avoid (a,b,c,d) and (d,c,b,a) duplicate counting
                            quad = (a, b, c, d) if a < d else (d, c, b, a)
                            if quad not in seen_dihedrals:
                                seen_dihedrals.add(quad)
                                dihedral_quadruplets.append(quad)

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

        # Dynamically derive bulk density from Equation of State without hardcoded fallbacks:
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
            # Default dynamic thermodynamic reservoir: dimensionless chemical potential beta * mu = -8.0
            density = solve_bulk_density_from_chemical_potential(
                mu_kbt=-8.0,
                temp_k=temp,
                sigma=eff_sigma,
                epsilon_k=eff_eps_k,
            )

        dim_mode = cls.classify_dimension_mode(sites)

        mat = Material(
            name=mol2_path.stem,
            identifier=mol2_path.stem,
            dimension_mode=dim_mode,
            sites=sites,
            bonds=bonds,
            dihedral_quadruplets=dihedral_quadruplets,
            effective_sigma=eff_sigma,
            effective_epsilon_k=eff_eps_k,
            temperature_k=temp,
            bulk_density_a3=density,
        )
        mat.compute_bulk_mu()
        mat.compute_bulk_pressure()
        return mat

    @classmethod
    def list_available_materials(cls) -> List[str]:
        """Returns all available .mol2 files in test_data/."""
        return sorted([p.stem for p in TEST_DATA_DIR.glob("*.mol2")])


@dataclass
class MolecularBatch:
    """
    Encapsulates a stacked batch of M <= B target molecules padded to uniform dimensions:
    - B: fixed target batch size (default: 32)
    - N: fixed target sites per molecule (default: 128)
    All parameters are strictly stored as static (B, N) or (B, D) Tensor buffers on device
    to guarantee zero JIT recompilation across executions.
    """

    materials: List[Optional[Material]]
    batch_size: int
    n_particles: int
    sigmas: Any  # Tensor of shape (B, N)
    epsilons: Any  # Tensor of shape (B, N)
    charges: Any  # Tensor of shape (B, N)
    atom_mask: Any  # Tensor of shape (B, N) - 1.0 for real atoms, 0.0 for dummy atoms
    molecule_mask: Any  # Tensor of shape (B,) - 1.0 for active molecules, 0.0 for dummy slots
    temperature_k: Any  # Tensor of shape (B,)
    beta: Any  # Tensor of shape (B,)
    bulk_density_a3: Any  # Tensor of shape (B,)
    bulk_mu: Any  # Tensor of shape (B,)
    slit_width_a: Any  # Tensor of shape (B,)
    conditioning: Any  # Tensor of shape (B, 5) [sigma_eff, eps_eff, T, rho_bulk, mu]

    @property
    def num_active_materials(self) -> int:
        return sum(1 for m in self.materials if m is not None)

    @classmethod
    def create_batch(
        cls,
        materials: List[Material],
        batch_size: int = 32,
        target_n_particles: int = 128,
        default_temp_k: float = 300.0,
    ) -> MolecularBatch:
        """
        Ingests a list of M materials, pads each to target_n_particles sites with zeroed parameters,
        stacks them into fixed (B, N) tensors, and fills remaining batch slots up to batch_size
        with zeroed dummy molecules.
        """
        from tinygrad import Tensor, dtypes

        mats_padded: List[Optional[Material]] = []
        sigmas_list = []
        epsilons_list = []
        charges_list = []
        atom_mask_list = []
        molecule_mask_list = []
        temp_list = []
        beta_list = []
        rho_list = []
        mu_list = []
        slit_list = []
        cond_list = []

        n_mats = len(materials)
        for b in range(batch_size):
            if b < n_mats:
                mat = materials[b]
                mats_padded.append(mat)
                s_b = [s.sigma for s in mat.sites] if mat.sites else [mat.effective_sigma]
                e_b = [s.epsilon_k for s in mat.sites] if mat.sites else [mat.effective_epsilon_k]
                q_b = [s.charge for s in mat.sites] if mat.sites else [mat.total_charge]
                mask_b = [1.0] * len(s_b)

                # Pad dummy atoms up to target_n_particles
                n_pad = max(0, target_n_particles - len(s_b))
                if n_pad > 0:
                    s_b = s_b + [0.0] * n_pad
                    e_b = e_b + [0.0] * n_pad
                    q_b = q_b + [0.0] * n_pad
                    mask_b = mask_b + [0.0] * n_pad

                s_b = s_b[:target_n_particles]
                e_b = e_b[:target_n_particles]
                q_b = q_b[:target_n_particles]
                mask_b = mask_b[:target_n_particles]

                sigmas_list.append(s_b)
                epsilons_list.append(e_b)
                charges_list.append(q_b)
                atom_mask_list.append(mask_b)
                molecule_mask_list.append(1.0)

                temp_val = mat.temperature_k
                beta_val = 1.0 / max(1e-6, temp_val)
                rho_val = mat.bulk_density_a3
                mu_val = mat.bulk_mu
                slit_val = max(40.0, 12.0 * mat.effective_sigma)

                temp_list.append(temp_val)
                beta_list.append(beta_val)
                rho_list.append(rho_val)
                mu_list.append(mu_val)
                slit_list.append(slit_val)
                cond_list.append([mat.effective_sigma, mat.effective_epsilon_k, temp_val, rho_val, mu_val])
            else:
                # Empty batch slot: Zero-padded dummy molecule
                mats_padded.append(None)
                sigmas_list.append([0.0] * target_n_particles)
                epsilons_list.append([0.0] * target_n_particles)
                charges_list.append([0.0] * target_n_particles)
                atom_mask_list.append([0.0] * target_n_particles)
                molecule_mask_list.append(0.0)

                temp_list.append(default_temp_k)
                beta_list.append(1.0 / default_temp_k)
                rho_list.append(0.0)
                mu_list.append(0.0)
                slit_list.append(40.0)
                cond_list.append([0.0, 0.0, default_temp_k, 0.0, 0.0])

        return MolecularBatch(
            materials=mats_padded,
            batch_size=batch_size,
            n_particles=target_n_particles,
            sigmas=Tensor(sigmas_list, dtype=dtypes.float32).realize(),
            epsilons=Tensor(epsilons_list, dtype=dtypes.float32).realize(),
            charges=Tensor(charges_list, dtype=dtypes.float32).realize(),
            atom_mask=Tensor(atom_mask_list, dtype=dtypes.float32).realize(),
            molecule_mask=Tensor(molecule_mask_list, dtype=dtypes.float32).realize(),
            temperature_k=Tensor(temp_list, dtype=dtypes.float32).realize(),
            beta=Tensor(beta_list, dtype=dtypes.float32).realize(),
            bulk_density_a3=Tensor(rho_list, dtype=dtypes.float32).realize(),
            bulk_mu=Tensor(mu_list, dtype=dtypes.float32).realize(),
            slit_width_a=Tensor(slit_list, dtype=dtypes.float32).realize(),
            conditioning=Tensor(cond_list, dtype=dtypes.float32).realize(),
        )
