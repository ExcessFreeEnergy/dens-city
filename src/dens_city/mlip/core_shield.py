r"""
Universal ZBL Repulsive Core Shield for Equivariant MLIPs.

Prevents the "Extrapolation Cliff":
When numerical solvers push atoms into unphysical configurations (r <= 0.8 A),
unshielded neural networks extrapolate catastrophically, predicting -inf energy.

This module provides a strictly repulsive universal Ziegler-Biersack-Littmark (ZBL)
core envelope with C^2 smooth cubic switching at r = r_core:
  E_total(r) = E_MLIP(r) + V_ZBL(r) * (3t^2 - 2t^3) for r < r_core
where t = (r_core - r) / r_core.
"""

from typing import Union
import numpy as np

from dens_city.solver.quantum_surrogates import zbl_repulsive_core


class ZBLRepulsiveShield:
    def __init__(self, z1: float = 8.0, z2: float = 8.0, r_core: float = 0.8):
        """
        z1, z2: Atomic numbers of interacting species.
        r_core: Core threshold radius (in Angstroms).
        """
        self.z1 = float(z1)
        self.z2 = float(z2)
        self.r_core = float(r_core)

    def evaluate_shield_energy(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the repulsive shield energy in Kelvin.
        """
        return zbl_repulsive_core(r, z1=self.z1, z2=self.z2, r_core=self.r_core)

    def apply_to_potential(self, r: np.ndarray, base_potential: np.ndarray) -> np.ndarray:
        """
        Adds the repulsive core shield to any base potential profile.
        """
        v_shield = self.evaluate_shield_energy(r)
        return base_potential + v_shield
