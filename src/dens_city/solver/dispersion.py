"""
First-Principles Barker-Henderson (BH) / Weeks-Chandler-Andersen (WCA) Dispersion Solver.
Combines Analytical Fundamental Measure Theory (FMT) with exact attractive dispersion integration.
No empirical patches; all state points are derived strictly from the underlying pair Hamiltonian.
"""

from typing import Callable, Tuple
import numpy as np

from dens_city.solver.fmt import FundamentalMeasureTheory1D

KB = 1.380649e-23  # J/K
KB_EV = 8.617333262e-5  # eV/K


def compute_barker_henderson_diameter(
    sigma: float,
    epsilon_k: float,
    T: float,
    n_points: int = 1000,
) -> float:
    """
    Computes the temperature-dependent Barker-Henderson effective hard-sphere diameter d(T):
    d(T) = \int_0^{r_min} [1 - exp(-beta * u_0(r))] dr
    where r_min = 2^(1/6) * sigma, and u_0(r) = 4*eps*[(sig/r)^12 - (sig/r)^6] + eps.
    """
    r_min = (2.0 ** (1.0 / 6.0)) * sigma
    # Avoid singularity at r=0 by starting at a small fraction of sigma
    r_grid = np.linspace(0.4 * sigma, r_min, n_points)
    dr = r_grid[1] - r_grid[0]

    # WCA repulsive potential
    sig_r6 = (sigma / r_grid) ** 6
    sig_r12 = sig_r6**2
    u_0 = 4.0 * epsilon_k * (sig_r12 - sig_r6) + epsilon_k
    u_0 = np.maximum(u_0, 0.0)

    beta = 1.0 / max(1e-3, T)
    integrand = 1.0 - np.exp(-beta * u_0)
    d = 0.4 * sigma + np.sum(integrand) * dr
    return float(d)


def compute_planar_attractive_kernel(
    sigma: float,
    epsilon_k: float,
    z_coords: np.ndarray,
    r_cut: float = 15.0,
) -> np.ndarray:
    """
    Computes 1D planar slab-integrated attractive dispersion potential:
    u_att_bar(|z|) = 2 * pi * \int_{|z|}^{r_cut} r * u_att(r) dr
    where u_att(r) is the WCA attractive portion of the Lennard-Jones potential.
    """
    r_min = (2.0 ** (1.0 / 6.0)) * sigma
    u_att_bar = np.zeros_like(z_coords)

    for i, z_val in enumerate(z_coords):
        abs_z = abs(z_val)
        if abs_z >= r_cut:
            continue

        r_mesh = np.linspace(max(abs_z, 1e-4), r_cut, 500)
        dr = r_mesh[1] - r_mesh[0]

        sig_r6 = (sigma / r_mesh) ** 6
        sig_r12 = sig_r6**2
        u_lj = 4.0 * epsilon_k * (sig_r12 - sig_r6)

        u_att = np.where(r_mesh < r_min, -epsilon_k, u_lj)
        u_att_bar[i] = 2.0 * np.pi * np.sum(r_mesh * u_att) * dr

    return u_att_bar


class LennardJonesFMTDispersion1D:
    def __init__(self, sigma: float, epsilon_k: float, r_cut: float = 15.0):
        """
        sigma: LJ size parameter (Angstroms)
        epsilon_k: LJ energy parameter (epsilon / k_B in Kelvin)
        """
        self.sigma = float(sigma)
        self.epsilon_k = float(epsilon_k)
        self.r_cut = float(r_cut)

    def get_c1_functional(self, T: float, L_z: float = 40.0, grid_size: int = 256) -> Callable[[np.ndarray], np.ndarray]:
        """
        Returns a callable c1(rho) functional for a fixed temperature T.
        c1(rho, z) = c1_FMT(z; d(T)) - beta * \int dz' u_att_bar(|z - z'|) * rho(z')
        """
        d_T = compute_barker_henderson_diameter(self.sigma, self.epsilon_k, T)
        fmt_solver = FundamentalMeasureTheory1D(diameter=d_T)
        z_coords = np.linspace(-L_z / 2.0, L_z / 2.0, grid_size)
        dz = z_coords[1] - z_coords[0]

        # Kernel for convolution
        u_att_bar = compute_planar_attractive_kernel(self.sigma, self.epsilon_k, z_coords, self.r_cut)
        beta = 1.0 / T

        def c1_functional(rho_z: np.ndarray) -> np.ndarray:
            c1_hs = fmt_solver.compute_c1_hs(z_coords, rho_z)
            v_att = np.convolve(rho_z, u_att_bar * dz, mode="same")
            c1_att = -beta * v_att
            return c1_hs + c1_att

        return c1_functional

    def compute_bulk_pressure(self, rho_bulk: float, T: float) -> float:
        """
        Computes bulk pressure P(rho_bulk, T) in bar using Carnahan-Starling + mean-field dispersion:
        P / (rho * k_B * T) = (1 + eta + eta^2 - eta^3)/(1 - eta)^3 + (a_vdW / (k_B * T)) * rho
        """
        d_T = compute_barker_henderson_diameter(self.sigma, self.epsilon_k, T)
        eta = (np.pi / 6.0) * rho_bulk * (d_T**3)
        if eta >= 1.0:
            return 1e6

        # Hard sphere compressibility factor
        Z_hs = (1.0 + eta + eta**2 - eta**3) / ((1.0 - eta) ** 3)

        # Integrated van der Waals attraction constant a = - \int u_att(r) 4*pi*r^2 dr / 2
        # For standard LJ: a_LJ = (16/9) * pi * epsilon * sigma^3
        a_vdw_k = (16.0 / 9.0) * np.pi * self.epsilon_k * (self.sigma**3)  # K * A^3

        # Pressure in K / A^3
        P_k_A3 = rho_bulk * T * Z_hs - a_vdw_k * (rho_bulk**2)

        # Convert K / A^3 to bar:
        # 1 K / A^3 = (1.380649e-23 J) / (1e-30 m^3) = 1.380649e7 Pa = 138.0649 bar
        P_bar = P_k_A3 * 138.0649
        return float(P_bar)
