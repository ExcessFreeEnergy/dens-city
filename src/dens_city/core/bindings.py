import ctypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Load native shared library
_lib_path = Path(__file__).parent / "libdens_city_core.so"
_lib: Optional[ctypes.CDLL] = None

if _lib_path.exists():
    _lib = ctypes.CDLL(str(_lib_path))
else:
    # Try global lookup
    try:
        _lib = ctypes.CDLL("libdens_city_core.so")
    except OSError:
        _lib = None

if _lib is not None:
    _lib.dens_city_create.restype = ctypes.c_void_p
    _lib.dens_city_destroy.argtypes = [ctypes.c_void_p]

    _lib.dens_city_set_thermodynamics.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    _lib.dens_city_set_box.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    _lib.dens_city_set_moves.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]
    _lib.dens_city_set_molecule_type.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    _lib.dens_city_set_electrostatics.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int]

    _lib.dens_city_set_pair_potential.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double
    ]

    _lib.dens_city_set_external_potential.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]

    _lib.dens_city_step.argtypes = [ctypes.c_void_p]
    _lib.dens_city_run_steps.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.dens_city_get_molecule_count.argtypes = [ctypes.c_void_p]
    _lib.dens_city_get_molecule_count.restype = ctypes.c_int
    _lib.dens_city_get_total_energy.argtypes = [ctypes.c_void_p]
    _lib.dens_city_get_total_energy.restype = ctypes.c_double


class DensCityEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if _lib is None:
            raise RuntimeError("libdens_city_core.so not loaded. Please compile with nvcc/gcc first.")
        self._handle = _lib.dens_city_create()
        self.config = config or {}
        if config:
            self.configure(config)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle and _lib is not None:
            _lib.dens_city_destroy(self._handle)
            self._handle = None

    def configure(self, config: Dict[str, Any]):
        T = float(config.get("T", 300.0))
        mu1 = float(config.get("mu", config.get("mu1", 0.0)))
        mu2 = float(config.get("mu2", 0.0))
        _lib.dens_city_set_thermodynamics(self._handle, T, mu1, mu2)

        lx = float(config.get("box_x", config.get("lx", 20.0)))
        ly = float(config.get("box_y", config.get("ly", 20.0)))
        lz = float(config.get("box_z", config.get("lz", 20.0)))
        _lib.dens_city_set_box(self._handle, lx, ly, lz)

        # Move probabilities
        p_ins = float(config.get("prob_insert", 0.25))
        p_del = float(config.get("prob_delete", 0.25))
        p_disp = float(config.get("prob_displace", 0.25))
        p_rot = float(config.get("prob_rotate", 0.25))
        p_mut = float(config.get("prob_mutate", 0.0))
        maxdispl = float(config.get("maxdispl", 0.5))
        maxrot = float(config.get("maxrot", 0.2))
        _lib.dens_city_set_moves(self._handle, p_ins, p_del, p_disp, p_rot, p_mut, maxdispl, maxrot)

        # Molecule type
        mol_type_str = config.get("molecule_type", "single").lower()
        type_map = {
            "single": 1, "single_site": 1,
            "two_type": 2, "rpm": 2,
            "abc": 3, "dipole": 3,
            "water": 4, "h2o": 4, "spce": 4, "tip4p": 4,
            "co2": 5, "trappe": 5
        }
        mol_type = type_map.get(mol_type_str, 1)
        bond_len = float(config.get("bond_length", 1.0))
        _lib.dens_city_set_molecule_type(self._handle, mol_type, bond_len)

        # Electrostatics mode
        mode_str = config.get("electrostatics_mode", "short_range").lower()
        mode = 1 if mode_str in ("long_range", "ewald", "long_range_ewald") else 0
        ewald_alpha = float(config.get("ewald_alpha", 0.35))
        ewald_kmax = int(config.get("ewald_kmax", 4))
        _lib.dens_city_set_electrostatics(self._handle, mode, ewald_alpha, ewald_kmax)

    def set_pair_potential(
        self, type_i: int, type_j: int, kind: int,
        epsilon_lj: float = 0.0, sigma_lj: float = 0.0, rc: float = 10.0,
        epsilon_c: float = 0.0, q1: float = 0.0, q2: float = 0.0,
        kappa_inv: float = 4.5, diameter: float = 0.0, prefactor: float = 1.67101e-19,
        shift_lj: float = 0.0
    ):
        _lib.dens_city_set_pair_potential(
            self._handle, type_i, type_j, kind,
            epsilon_lj, sigma_lj, rc, epsilon_c, q1, q2,
            kappa_inv, diameter, prefactor, shift_lj
        )

    def set_external_potential(
        self, type_i: int, kind: int,
        low: float = 0.0, high: float = 20.0, width: float = 0.0, L: float = 20.0,
        epsilon: float = 0.0, sigma: float = 0.0, cutoff: float = 0.0, shift: float = 0.0, q: float = 0.0,
        A1: float = 0.0, A2: float = 0.0, A3: float = 0.0, A4: float = 0.0,
        phi1: float = 0.0, phi2: float = 0.0, phi3: float = 0.0, phi4: float = 0.0
    ):
        _lib.dens_city_set_external_potential(
            self._handle, type_i, kind,
            low, high, width, L, epsilon, sigma, cutoff, shift, q,
            A1, A2, A3, A4, phi1, phi2, phi3, phi4
        )

    def step(self):
        _lib.dens_city_step(self._handle)

    def run_steps(self, n_steps: int):
        _lib.dens_city_run_steps(self._handle, n_steps)

    @property
    def number(self) -> int:
        return _lib.dens_city_get_molecule_count(self._handle)

    def total_energy(self) -> float:
        return _lib.dens_city_get_total_energy(self._handle)
