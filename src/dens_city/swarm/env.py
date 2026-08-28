"""
Python interface and ctypes C bridge for the CDFT Swarm PufferLib Environment.
Exposes step, reset, action masking, 3D molecule export to Tripos .mol2, and RDKit conversion.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from dens_city.swarm.spec_loader import SwarmSpecLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OCEAN_SWARM_DIR = REPO_ROOT / "ocean" / "cdft_swarm"
LIB_SO_PATH = OCEAN_SWARM_DIR / "cdft_swarm_lib.so"

TOTAL_OBS_SIZE = 88
TOTAL_ACTION_MASK_SIZE = 29


def ensure_shared_library() -> Path:
    """Compiles cdft_swarm_lib.so if missing or outdated."""
    c_src = OCEAN_SWARM_DIR / "cdft_swarm_lib.c"
    if not c_src.exists():
        raise FileNotFoundError(f"Missing C source at {c_src}")

    c_deps = list(OCEAN_SWARM_DIR.glob("*.h")) + [c_src]
    recompile = False
    if not LIB_SO_PATH.exists():
        recompile = True
    else:
        so_mtime = LIB_SO_PATH.stat().st_mtime
        for dep in c_deps:
            if dep.stat().st_mtime > so_mtime:
                recompile = True
                break

    if recompile:
        cmd = [
            "gcc",
            "-O3",
            "-shared",
            "-fPIC",
            "-mavx2",
            "-mfma",
            "-fopenmp",
            f"-I{OCEAN_SWARM_DIR}",
            str(c_src),
            "-o",
            str(LIB_SO_PATH),
            "-lm",
        ]
        subprocess.run(cmd, check=True)

    return LIB_SO_PATH


class CDFTSwarmEnv:
    """Gymnasium / PufferLib compatible Python interface for the Stage 1 CDFT Swarm Agent."""

    def __init__(
        self,
        target_spec: Optional[Dict[str, Any]] = None,
        spec_yaml_path: Optional[str | Path] = None,
        seed: int = 42,
    ):
        self.lib_path = ensure_shared_library()
        self.lib = ctypes.CDLL(str(self.lib_path))

        # Setup ctypes signatures
        self.lib.env_create.argtypes = [ctypes.c_uint]
        self.lib.env_create.restype = ctypes.c_void_p

        self.lib.env_free.argtypes = [ctypes.c_void_p]
        self.lib.env_free.restype = None

        self.lib.env_set_targets.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
        ]
        self.lib.env_set_targets.restype = None

        self.lib.env_reset.argtypes = [ctypes.c_void_p]
        self.lib.env_reset.restype = None

        self.lib.env_step.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
        self.lib.env_step.restype = None

        self.lib.env_get_observations.argtypes = [ctypes.c_void_p]
        self.lib.env_get_observations.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.env_get_rewards.argtypes = [ctypes.c_void_p]
        self.lib.env_get_rewards.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.env_get_terminals.argtypes = [ctypes.c_void_p]
        self.lib.env_get_terminals.restype = ctypes.POINTER(ctypes.c_uint8)

        self.lib.env_get_action_mask.argtypes = [ctypes.c_void_p]
        self.lib.env_get_action_mask.restype = ctypes.POINTER(ctypes.c_uint8)

        self.lib.env_get_p_wall.argtypes = [ctypes.c_void_p]
        self.lib.env_get_p_wall.restype = ctypes.c_float

        self.lib.env_get_omega_solv.argtypes = [ctypes.c_void_p]
        self.lib.env_get_omega_solv.restype = ctypes.c_float

        self.lib.env_get_converged.argtypes = [ctypes.c_void_p]
        self.lib.env_get_converged.restype = ctypes.c_int

        self.lib.env_get_molecular_weight.argtypes = [ctypes.c_void_p]
        self.lib.env_get_molecular_weight.restype = ctypes.c_float

        self.lib.env_get_rotatable_fraction.argtypes = [ctypes.c_void_p]
        self.lib.env_get_rotatable_fraction.restype = ctypes.c_float

        self.lib.env_get_aromatic_density.argtypes = [ctypes.c_void_p]
        self.lib.env_get_aromatic_density.restype = ctypes.c_float

        self.lib.env_get_pmi_linearity.argtypes = [ctypes.c_void_p]
        self.lib.env_get_pmi_linearity.restype = ctypes.c_float

        self.lib.env_get_hbd_count.argtypes = [ctypes.c_void_p]
        self.lib.env_get_hbd_count.restype = ctypes.c_int

        self.lib.env_get_hba_count.argtypes = [ctypes.c_void_p]
        self.lib.env_get_hba_count.restype = ctypes.c_int

        self.lib.env_get_num_atoms.argtypes = [ctypes.c_void_p]
        self.lib.env_get_num_atoms.restype = ctypes.c_int

        self.lib.env_get_num_bonds.argtypes = [ctypes.c_void_p]
        self.lib.env_get_num_bonds.restype = ctypes.c_int

        self.lib.env_get_atom_pos_x.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_pos_x.restype = ctypes.c_float

        self.lib.env_get_atom_pos_y.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_pos_y.restype = ctypes.c_float

        self.lib.env_get_atom_pos_z.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_pos_z.restype = ctypes.c_float

        self.lib.env_get_atom_z.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_z.restype = ctypes.c_int

        self.lib.env_get_atom_is_aromatic.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_is_aromatic.restype = ctypes.c_int

        self.lib.env_get_atom_charge.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_charge.restype = ctypes.c_float

        self.lib.env_get_atom_sigma.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_sigma.restype = ctypes.c_float

        self.lib.env_get_atom_epsilon_k.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_atom_epsilon_k.restype = ctypes.c_float

        self.lib.env_get_atoms_block.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.env_get_atoms_block.restype = ctypes.c_int

        self.lib.env_get_bond_u.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_bond_u.restype = ctypes.c_int

        self.lib.env_get_bond_v.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_bond_v.restype = ctypes.c_int

        self.lib.env_get_bond_order.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.env_get_bond_order.restype = ctypes.c_int

        self.lib.env_get_atom_exclusions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.env_get_atom_exclusions.restype = ctypes.c_int

        # Create C environment
        self._env_ptr = self.lib.env_create(ctypes.c_uint(seed))

        # Set target specs
        if spec_yaml_path is not None:
            spec_data = SwarmSpecLoader.load_yaml(spec_yaml_path)
            target_spec = SwarmSpecLoader.derive_target_spec(spec_data)

        if target_spec is not None:
            self.set_targets(target_spec)

        self.reset()

    def set_targets(self, targets: Dict[str, float]) -> None:
        """Sets or updates the TargetSpec struct directly in C memory."""
        if not hasattr(self, "_env_ptr") or self._env_ptr is None:
            return
        self.lib.env_set_targets(
            self._env_ptr,
            ctypes.c_float(targets.get("target_elasticity", 0.25)),
            ctypes.c_float(targets.get("target_tensile", 0.25)),
            ctypes.c_float(targets.get("target_toughness", 0.25)),
            ctypes.c_float(targets.get("target_lightweight", 0.25)),
            ctypes.c_float(targets.get("max_solvation_kcal", -3.0)),
            ctypes.c_float(targets.get("min_wall_pressure_bar", 15.0)),
            ctypes.c_float(targets.get("max_molecular_weight", 850.0)),
            ctypes.c_int(targets.get("min_valency", 2)),
        )

    def close(self) -> None:
        """Frees C environment pointer."""
        if hasattr(self, "_env_ptr") and self._env_ptr:
            self.lib.env_free(self._env_ptr)
            self._env_ptr = None

    def __del__(self):
        self.close()

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and returns the initial observation."""
        self.lib.env_reset(self._env_ptr)
        obs_ptr = self.lib.env_get_observations(self._env_ptr)
        mask_ptr = self.lib.env_get_action_mask(self._env_ptr)
        obs = np.ctypeslib.as_array(obs_ptr, shape=(TOTAL_OBS_SIZE,)).copy()
        mask = np.ctypeslib.as_array(mask_ptr, shape=(TOTAL_ACTION_MASK_SIZE,)).copy()
        return obs, {"action_mask": mask}

    def step(self, action: Tuple[int, int] | np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single rigid-body assembly action."""
        self.lib.env_step(self._env_ptr, float(action[0]), float(action[1]))

        obs_ptr = self.lib.env_get_observations(self._env_ptr)
        mask_ptr = self.lib.env_get_action_mask(self._env_ptr)
        rew_ptr = self.lib.env_get_rewards(self._env_ptr)
        term_ptr = self.lib.env_get_terminals(self._env_ptr)

        obs = np.ctypeslib.as_array(obs_ptr, shape=(TOTAL_OBS_SIZE,)).copy()
        mask = np.ctypeslib.as_array(mask_ptr, shape=(TOTAL_ACTION_MASK_SIZE,)).copy()
        reward = float(rew_ptr[0])
        terminal = bool(term_ptr[0])

        info = {
            "action_mask": mask,
            "p_wall_bar": float(self.lib.env_get_p_wall(self._env_ptr)),
            "omega_solv_kcal": float(self.lib.env_get_omega_solv(self._env_ptr)),
            "molecular_weight": float(self.lib.env_get_molecular_weight(self._env_ptr)),
            "rotatable_fraction": float(self.lib.env_get_rotatable_fraction(self._env_ptr)),
            "pmi_linearity": float(self.lib.env_get_pmi_linearity(self._env_ptr)),
            "converged": bool(self.lib.env_get_converged(self._env_ptr)),
        }

        return obs, reward, terminal, False, info

    @property
    def num_ports(self) -> int:
        return int(self.lib.env_get_num_ports(self._env_ptr))

    @property
    def num_atoms(self) -> int:
        return int(self.lib.env_get_num_atoms(self._env_ptr))

    @property
    def num_bonds(self) -> int:
        return int(self.lib.env_get_num_bonds(self._env_ptr))

    @property
    def hbd_count(self) -> int:
        return int(self.lib.env_get_hbd_count(self._env_ptr))

    @property
    def hba_count(self) -> int:
        return int(self.lib.env_get_hba_count(self._env_ptr))

    @property
    def rotatable_fraction(self) -> float:
        return float(self.lib.env_get_rotatable_fraction(self._env_ptr))

    @property
    def aromatic_density(self) -> float:
        return float(self.lib.env_get_aromatic_density(self._env_ptr))

    @property
    def pmi_linearity(self) -> float:
        return float(self.lib.env_get_pmi_linearity(self._env_ptr))

    @property
    def molecular_weight(self) -> float:
        return float(self.lib.env_get_molecular_weight(self._env_ptr))

    def get_action_mask(self) -> np.ndarray:
        """Returns the binary action mask indicating allowable ports and fragments."""
        mask_ptr = self.lib.env_get_action_mask(self._env_ptr)
        return np.ctypeslib.as_array(mask_ptr, shape=(TOTAL_ACTION_MASK_SIZE,)).copy()

    def get_current_rdkit_mol(self) -> Optional[Chem.Mol]:
        """Converts the current C molecular graph into an RDKit 3D molecule."""
        num_atoms = int(self.lib.env_get_num_atoms(self._env_ptr))
        if num_atoms == 0:
            return None

        num_bonds = int(self.lib.env_get_num_bonds(self._env_ptr))
        mol = Chem.RWMol()

        for i in range(num_atoms):
            z = int(self.lib.env_get_atom_z(self._env_ptr, i))
            atom = Chem.Atom(z)
            mol.AddAtom(atom)

        for b in range(num_bonds):
            u = int(self.lib.env_get_bond_u(self._env_ptr, b))
            v = int(self.lib.env_get_bond_v(self._env_ptr, b))
            b_order = int(self.lib.env_get_bond_order(self._env_ptr, b))
            if u < num_atoms and v < num_atoms:
                b_type = Chem.BondType.SINGLE
                if b_order == 2:
                    b_type = Chem.BondType.DOUBLE
                elif b_order == 3:
                    b_type = Chem.BondType.TRIPLE
                elif b_order == 4:
                    b_type = Chem.BondType.AROMATIC
                mol.AddBond(u, v, b_type)

        res = mol.GetMol()
        try:
            Chem.SanitizeMol(res, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            pass

        conf = Chem.Conformer(num_atoms)
        for i in range(num_atoms):
            px = float(self.lib.env_get_atom_pos_x(self._env_ptr, i))
            py = float(self.lib.env_get_atom_pos_y(self._env_ptr, i))
            pz = float(self.lib.env_get_atom_pos_z(self._env_ptr, i))
            conf.SetAtomPosition(i, (px, py, pz))
        res.AddConformer(conf, assignId=True)

        return res

    def export_mol2_string(self, mol_name: str = "swarm_candidate") -> str:
        """Exports current pseudo-3D assembled molecule as a standard Tripos .mol2 string."""
        num_atoms = int(self.lib.env_get_num_atoms(self._env_ptr))
        num_bonds = int(self.lib.env_get_num_bonds(self._env_ptr))

        lines = [
            "@<TRIPOS>MOLECULE",
            mol_name,
            f"{num_atoms:>6}{num_bonds:>6}     1     0     0",
            "SMALL",
            "GAFF_CHARGES",
            "",
            "CDFT Swarm Candidate",
            "@<TRIPOS>ATOM",
        ]

        z_to_sym = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 16: "S", 17: "Cl", 35: "Br"}
        for i in range(num_atoms):
            z = int(self.lib.env_get_atom_z(self._env_ptr, i))
            is_arom = bool(self.lib.env_get_atom_is_aromatic(self._env_ptr, i))
            sym = z_to_sym.get(z, "C")
            s_name = f"{sym}{i + 1}"
            if z == 6:
                at_type = "C.ar" if is_arom else "C.3"
            elif z == 7:
                at_type = "N.ar" if is_arom else "N.3"
            elif z == 8:
                at_type = "O.3"
            elif z == 16:
                at_type = "S.3"
            else:
                at_type = sym
            px = float(self.lib.env_get_atom_pos_x(self._env_ptr, i))
            py = float(self.lib.env_get_atom_pos_y(self._env_ptr, i))
            pz = float(self.lib.env_get_atom_pos_z(self._env_ptr, i))
            charge = float(self.lib.env_get_atom_charge(self._env_ptr, i))
            lines.append(
                f"{i + 1:>7} {s_name:<6} {px:>12.4f} {py:>10.4f} {pz:>10.4f} {at_type:<8} 1 MOL {charge:>12.6f}"
            )

        lines.append("@<TRIPOS>BOND")
        for b in range(num_bonds):
            u = int(self.lib.env_get_bond_u(self._env_ptr, b))
            v = int(self.lib.env_get_bond_v(self._env_ptr, b))
            b_order = int(self.lib.env_get_bond_order(self._env_ptr, b))
            b_str = "ar" if b_order == 4 else str(b_order)
            lines.append(f"{b + 1:>6} {u + 1:>6} {v + 1:>6} {b_str:<4}")

        lines.append("@<TRIPOS>SUBSTRUCTURE\n     1 MOL         1 TEMP              0 ****  ****    0 ROOT\n")
        return "\n".join(lines)

    def get_raw_atom_arrays(self) -> Dict[str, Any]:
        """
        Extracts raw physical parameter arrays directly from C environment memory
        without intermediate file parsing.
        """
        num_atoms = int(self.lib.env_get_num_atoms(self._env_ptr))
        num_bonds = int(self.lib.env_get_num_bonds(self._env_ptr))
        if num_atoms == 0:
            return {
                "num_atoms": 0,
                "coords": np.zeros((0, 3), dtype=np.float32),
                "sigmas": np.zeros((0,), dtype=np.float32),
                "epsilons_k": np.zeros((0,), dtype=np.float32),
                "charges": np.zeros((0,), dtype=np.float32),
                "atomic_numbers": np.zeros((0,), dtype=np.int32),
                "bonds": [],
                "mw": 0.0,
                "p_wall": 0.0,
                "omega_solv": 0.0,
            }

        coords = np.zeros((num_atoms * 3,), dtype=np.float32)
        sigmas = np.zeros((num_atoms,), dtype=np.float32)
        epsilons_k = np.zeros((num_atoms,), dtype=np.float32)
        charges = np.zeros((num_atoms,), dtype=np.float32)
        atomic_numbers = np.zeros((num_atoms,), dtype=np.int32)
        exclusions = np.zeros((num_atoms * num_atoms,), dtype=np.float32)

        c_coords = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_sigmas = sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_epsilons = epsilons_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_charges = charges.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_z = atomic_numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        c_excl = exclusions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.env_get_atoms_block(self._env_ptr, c_coords, c_sigmas, c_epsilons, c_charges, c_z)
        self.lib.env_get_atom_exclusions(self._env_ptr, c_excl)

        bonds_list: List[Tuple[int, int, str]] = []
        for b in range(num_bonds):
            u = int(self.lib.env_get_bond_u(self._env_ptr, b))
            v = int(self.lib.env_get_bond_v(self._env_ptr, b))
            b_order = int(self.lib.env_get_bond_order(self._env_ptr, b))
            b_str = "ar" if b_order == 4 else str(b_order)
            bonds_list.append((u, v, b_str))

        return {
            "num_atoms": num_atoms,
            "coords": coords.reshape((num_atoms, 3)),
            "sigmas": sigmas,
            "epsilons_k": epsilons_k,
            "charges": charges,
            "atomic_numbers": atomic_numbers,
            "exclusions": exclusions.reshape((num_atoms, num_atoms)),
            "bonds": bonds_list,
            "mw": float(self.lib.env_get_molecular_weight(self._env_ptr)),
            "p_wall": float(self.lib.env_get_p_wall(self._env_ptr)),
            "omega_solv": float(self.lib.env_get_omega_solv(self._env_ptr)),
            "pmi_linearity": float(self.lib.env_get_pmi_linearity(self._env_ptr)),
            "aromatic_density": float(self.lib.env_get_aromatic_density(self._env_ptr)),
            "rotatable_fraction": float(self.lib.env_get_rotatable_fraction(self._env_ptr)),
        }
