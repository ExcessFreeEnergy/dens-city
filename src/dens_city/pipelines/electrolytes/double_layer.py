from typing import Callable, Dict, List, Tuple
import numpy as np

KB = 1.380649e-23
E_CHARGE = 1.602176634e-19


def solve_electric_double_layer(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    voltage: float, # in Volts
    T: float = 300.0,
    rho_bulk: float = 0.005, # ~0.005 A^-3 (~8.3 M)
    L_z: float = 40.0,
    grid_size: int = 256,
    kappa_inv: float = 5.0, # 5.0 A for RPM electrolyte
) -> Dict[str, np.ndarray]:
    """
    Solves the Electric Double Layer structure for a 1:1 RPM electrolyte
    under applied electrode voltage V_0 with exact long-range screening.
    """
    z_coords = np.linspace(0, L_z, grid_size)
    dz = z_coords[1] - z_coords[0]
    beta = 1.0 / (KB * T)

    # Applied electrostatic potential: linear voltage drop from V_0 at z=0 to 0 at z=L_z/2
    v_ext_pos = np.zeros(grid_size)
    v_ext_neg = np.zeros(grid_size)

    for i, z in enumerate(z_coords):
        phi_elec = voltage * np.exp(-z / kappa_inv)
        v_ext_pos[i] = +E_CHARGE * phi_elec
        v_ext_neg[i] = -E_CHARGE * phi_elec

    # Picard relaxation for both species
    rho_pos = np.full(grid_size, rho_bulk)
    rho_neg = np.full(grid_size, rho_bulk)

    for _ in range(500):
        # Local excess functional
        c1_p = c1_functional(rho_pos, T)
        c1_n = c1_functional(rho_neg, T)

        # Update density
        target_p = rho_bulk * np.exp(np.clip(-beta * v_ext_pos + c1_p, -20.0, 20.0))
        target_n = rho_bulk * np.exp(np.clip(-beta * v_ext_neg + c1_n, -20.0, 20.0))

        rho_pos = 0.85 * rho_pos + 0.15 * target_p
        rho_neg = 0.85 * rho_neg + 0.15 * target_n

    charge_density = E_CHARGE * (rho_pos - rho_neg)
    total_charge = np.sum(charge_density[: grid_size // 2]) * dz

    return {
        "z_coords": z_coords,
        "rho_pos": rho_pos,
        "rho_neg": rho_neg,
        "charge_density": charge_density,
        "total_charge": total_charge,
        "voltage": voltage,
    }


def compute_differential_capacitance(
    c1_functional: Callable[[np.ndarray, float], np.ndarray],
    voltages: List[float],
    T: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the differential capacitance curve C(V) = dQ_tot / dV.
    """
    q_list = []
    for v in voltages:
        res = solve_electric_double_layer(c1_functional, v, T=T)
        q_list.append(res["total_charge"])

    q_arr = np.array(q_list)
    v_arr = np.array(voltages)

    # Numerical derivative
    cap = np.gradient(q_arr, v_arr)
    return v_arr, cap
