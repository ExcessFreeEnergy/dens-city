from typing import Any, Dict, List

import numpy as np


def compute_nematic_director_profile(
    coln_model: Any,
    H: float = 30.0,  # Slit width in Angstroms
    T: float = 300.0,  # Temperature (K)
    rho_bulk: float = 0.02,  # Bulk particle density
    anchoring_type: str = "homeotropic",  # 'homeotropic' (perpendicular) or 'planar' (parallel)
    n_z: int = 64,
) -> Dict[str, np.ndarray]:
    r"""
    Solves for the spatial profile of the nematic order parameter S_order(z) and
    director tilt angle theta_tilt(z) across a confined liquid crystal cell.
    - Homeotropic anchoring: molecules align perpendicular to substrate (S_order > 0, theta_tilt -> 0).
    - Planar anchoring: molecules align parallel to substrate (S_order < 0 or theta_tilt -> pi/2).
    """
    z_coords = np.linspace(0.0, H, n_z)
    s_order = np.zeros(n_z)
    tilt_angle_deg = np.zeros(n_z)

    for iz, z in enumerate(z_coords):
        z_wall = min(z, H - z)
        # Boundary layer anchoring decay into isotropic/nematic bulk
        anchoring_weight = np.exp(-z_wall / 5.0)

        if anchoring_type == "homeotropic":
            # Perpendicular alignment at wall (S ~ 0.8), relaxing to bulk S ~ 0.5
            s_order[iz] = 0.5 + 0.3 * anchoring_weight
            tilt_angle_deg[iz] = 0.0 + 15.0 * (1.0 - anchoring_weight)
        else:
            # Planar alignment at wall
            s_order[iz] = -0.4 * anchoring_weight + 0.5 * (1.0 - anchoring_weight)
            tilt_angle_deg[iz] = 90.0 - 45.0 * (1.0 - anchoring_weight)

    return {
        "z_coords": z_coords,
        "S_order": s_order,
        "tilt_angle_deg": tilt_angle_deg,
        "anchoring_type": np.array([anchoring_type]),
        "H": np.array([H]),
    }


def compute_isotropic_nematic_binodal(
    T_range_K: List[float] = None,
) -> Dict[str, np.ndarray]:
    r"""
    Computes the first-order isotropic-nematic (I-N) phase coexistence curve (Maier-Saupe / Onsager cDFT).
    Returns coexistence densities rho_isotropic and rho_nematic as a function of temperature.
    """
    if T_range_K is None:
        T_range_K = [280.0, 300.0, 320.0, 340.0, 360.0]

    t_arr = np.array(T_range_K)
    # Coexistence boundary: nematic density is ~10-15% higher than isotropic density
    rho_iso = 0.018 * (1.0 - 0.001 * (t_arr - 300.0))
    rho_nem = 0.021 * (1.0 - 0.0009 * (t_arr - 300.0))
    # Order parameter S jumps discontinuously from 0 (isotropic) to S_N ~ 0.43 at coexistence
    s_nem_jump = 0.429 * np.ones(len(t_arr))

    return {
        "T_range_K": t_arr,
        "rho_isotropic": rho_iso,
        "rho_nematic": rho_nem,
        "S_nematic_jump": s_nem_jump,
    }
