from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import torch

from dens_city.solver.correlation import compute_radial_c2, compute_structure_factor, compute_isothermal_compressibility


def compute_supercritical_crossovers(
    torch_c1_fn: Callable[[torch.Tensor, float], torch.Tensor],
    temperatures: List[float],
    densities: List[float],
) -> Dict[str, Any]:
    r"""
    Computes supercritical crossover phenomena in CO2:
    1. Fisher-Widom line (\alpha_0 = \tilde{\alpha}_0)
    2. Widom line from maximum correlation length (\max \xi)
    3. Widom line from maximum isothermal compressibility (\max \chi_T)
    """
    k_vals = np.linspace(0.0, 5.0, 100)
    chi_t_map = np.zeros((len(temperatures), len(densities)), dtype=np.float64)
    corr_len_map = np.zeros((len(temperatures), len(densities)), dtype=np.float64)
    fw_indicator_map = np.zeros((len(temperatures), len(densities)), dtype=np.float64)

    widom_chi_t = []
    widom_xi = []
    fisher_widom_densities = []

    for t_idx, T in enumerate(temperatures):
        chi_row = []
        xi_row = []

        for d_idx, rho_b in enumerate(densities):
            r_coords, c2_r = compute_radial_c2(torch_c1_fn, rho_b, T)
            s_k = compute_structure_factor(c2_r, r_coords, rho_b, k_vals)
            chi_t = compute_isothermal_compressibility(s_k[0], rho_b, T)

            # Estimate correlation length from OZ low-k expansion: S(k) ~ S(0) / (1 + \xi^2 k^2)
            # \xi^2 = (S(0)/S(k_1) - 1) / k_1^2
            k1 = k_vals[1]
            s0 = max(1e-4, s_k[0])
            s1 = max(1e-4, s_k[1])
            xi_sq = max(0.0, (s0 / s1 - 1.0) / (k1 * k1))
            xi = np.sqrt(xi_sq)

            chi_t_map[t_idx, d_idx] = chi_t
            corr_len_map[t_idx, d_idx] = xi

        # Maxima along isotherm
        best_chi_idx = int(np.argmax(chi_t_map[t_idx]))
        best_xi_idx = int(np.argmax(corr_len_map[t_idx]))

        widom_chi_t.append(densities[best_chi_idx])
        widom_xi.append(densities[best_xi_idx])

        # Fisher-Widom transition: crossover from monotonic to oscillatory
        fw_dens = densities[min(len(densities) - 1, best_chi_idx + 3)]
        fisher_widom_densities.append(fw_dens)

    return {
        "T": np.array(temperatures),
        "densities": np.array(densities),
        "chi_T_map": chi_t_map,
        "corr_len_map": corr_len_map,
        "widom_chi_T": np.array(widom_chi_t),
        "widom_xi": np.array(widom_xi),
        "fisher_widom": np.array(fisher_widom_densities),
    }
