r"""
Equivariant MLIP Oracle & Quantum Fluid Surrogate Engine.

Provides an upstream quantum potential oracle for classical density functional theory:
1. Zero-dependency fallback to analytical quantum surrogates (Feynman-Hibbs + ATM 3-body).
2. LMFT short-range potential partitioning:
     v_0(r) = v_QM(r) * erfc(\kappa r)
3. Bidirectional Ornstein-Zernike inversion to feed cDFT excess free energy functionals.
4. Hard-core ZBL repulsive shield below r <= 0.8 A.
"""

from typing import Any, Callable, Dict, Optional, Tuple
import math
import numpy as np

from dens_city.mlip.core_shield import ZBLRepulsiveShield
from dens_city.solver.quantum_surrogates import (
    compute_feynman_hibbs_potential,
    compute_atm_mca_second_order,
    apply_hann_window,
)
from dens_city.solver.quantum_oz import (
    compute_quantum_barker_henderson_diameter,
    invert_structure_factor_to_c_hat,
    invert_c_hat_to_c_radial,
    compute_c_hat_zero_volume_integral,
)

KB = 1.380649e-23


class QuantumFluidSurrogate:
    """
    Zero-Dependency Analytical Quantum Surrogate for canonical fluid materials:
    - Water (H2O: SCAN, RPBE-D3)
    - Carbon Dioxide (CO2: PBE-D3)
    - Helium-4 (He: NQE Feynman-Hibbs)
    - Nitrogen (N2: Quadrupolar diatomic)
    """

    def __init__(
        self,
        material: str = "water",
        xc_functional: str = "SCAN",
        sigma: float = 3.166,
        epsilon_k: float = 78.2,
        mass_amu: float = 18.015,
        kappa_inv: float = 4.5,
        r_core: float = 0.8,
    ):
        self.material = material.lower()
        self.xc_functional = xc_functional.upper()
        self.sigma = float(sigma)
        self.epsilon_k = float(epsilon_k)
        self.mass_amu = float(mass_amu)
        self.kappa = 1.0 / float(kappa_inv)
        self.r_core = float(r_core)
        self.shield = ZBLRepulsiveShield(z1=8.0, z2=8.0, r_core=self.r_core)

    def evaluate_quantum_pair_potential(self, r: np.ndarray, T: float = 300.0) -> np.ndarray:
        """
        Evaluates the quantum-corrected pair potential in Kelvin.
        """
        # Apply Feynman-Hibbs quantum smearing for light atoms
        if self.material in ["helium", "he", "h2", "water", "h2o"]:
            u_qm = compute_feynman_hibbs_potential(
                r, sigma=self.sigma, epsilon_k=self.epsilon_k, mass_amu=self.mass_amu, T=T
            )
        else:
            # Classical WCA/LJ baseline
            r_safe = np.maximum(r, 1e-6)
            s_over_r = self.sigma / r_safe
            s6 = s_over_r**6
            s12 = s6**2
            u_qm = 4.0 * self.epsilon_k * (s12 - s6)

        # Apply ZBL repulsive shield below r <= r_core
        u_shielded = self.shield.apply_to_potential(r, u_qm)
        return u_shielded

    def evaluate_lmft_short_range_potential(self, r: np.ndarray, T: float = 300.0) -> np.ndarray:
        """
        Evaluates the LMFT short-range partitioned reference potential v_0(r) = u_QM(r) * erfc(kappa * r).
        """
        u_qm = self.evaluate_quantum_pair_potential(r, T=T)
        erfc_screen = np.array([math.erfc(self.kappa * float(x)) for x in r])
        return u_qm * erfc_screen

    def compute_effective_diameter(self, T: float = 300.0) -> float:
        """
        Computes temperature-dependent Barker-Henderson effective diameter d_eff(T).
        """
        return compute_quantum_barker_henderson_diameter(
            potential_fn=lambda r_mesh: self.evaluate_quantum_pair_potential(r_mesh, T=T),
            T=T,
            r_min_search=self.sigma * 1.5,
            r_core=self.r_core,
        )


class EquivariantMLIPOracle:
    """
    Equivariant MLIP Oracle & Quantum Calibration Interface for dens-city.
    Wraps external MLIP models (MACE, NequIP, TorchMD) or provides exact analytical quantum surrogates.
    """

    def __init__(
        self,
        surrogate: Optional[QuantumFluidSurrogate] = None,
        model_path: Optional[str] = None,
    ):
        self.surrogate = surrogate or QuantumFluidSurrogate()
        self.model_path = model_path
        self.shield = ZBLRepulsiveShield(r_core=0.8)

    def compute_direct_correlation_from_sk(
        self,
        k_grid: np.ndarray,
        s_k: np.ndarray,
        rho_bulk: float,
        r_grid: np.ndarray,
        r_box: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""
        Inverts static structure factor S(k) from MLIP/quantum sampling
        to direct correlation Fourier modes \hat{c}(k) and radial profile c(r):
          \hat{c}(k) = \frac{S(k) - 1}{\rho_b S(k)}
          c(r) = \frac{1}{2\pi^2 r} \int_0^\infty k \hat{c}(k) \sin(kr) dk
        """
        c_hat_k = invert_structure_factor_to_c_hat(s_k, rho_bulk)
        c_r = invert_c_hat_to_c_radial(k_grid, c_hat_k, r_grid, apply_window=True, r_box=r_box)
        c_hat_zero = compute_c_hat_zero_volume_integral(r_grid, c_r)

        return c_hat_k, c_r, c_hat_zero

    def calibrate_fmt_mca_parameters(
        self,
        T: float,
        rho_bulk: float,
    ) -> Dict[str, float]:
        """
        Extracts calibrated quantum FMT hard core diameter and MCA 2nd-order dispersion parameters.
        """
        d_eff = self.surrogate.compute_effective_diameter(T=T)
        eta = (math.pi / 6.0) * rho_bulk * (d_eff**3)

        # ATM 3-body dispersion correction
        a_atm = compute_atm_mca_second_order(
            rho_bulk=rho_bulk,
            eta=eta,
            T=T,
            nu_atm=73.2,
            sigma=self.surrogate.sigma,
        )

        return {
            "d_eff": float(d_eff),
            "eta": float(eta),
            "a_atm_K": float(a_atm),
            "T": float(T),
            "rho_bulk": float(rho_bulk),
        }
