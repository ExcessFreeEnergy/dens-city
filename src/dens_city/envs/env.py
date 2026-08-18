import ctypes
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Load native shared library
_env_lib_path = Path(__file__).parent / "libdens_city_env.so"
_env_lib: Optional[ctypes.CDLL] = None

if _env_lib_path.exists():
    _env_lib = ctypes.CDLL(str(_env_lib_path))
else:
    try:
        _env_lib = ctypes.CDLL("libdens_city_env.so")
    except OSError:
        _env_lib = None

CDFT_GRID_SIZE = 256


class CCdftEnvStruct(ctypes.Structure):
    _fields_ = [
        ("L_z", ctypes.c_float),
        ("dz", ctypes.c_float),
        ("T", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("mu_target", ctypes.c_float),
        ("rho_bulk", ctypes.c_float),
        ("kappa_inv", ctypes.c_float),
        ("phi_0", ctypes.c_float),
        ("mode_m", ctypes.c_float),
        ("v_bias", ctypes.c_float),
        ("target_filling", ctypes.c_float),
        ("z_coords", ctypes.c_float * CDFT_GRID_SIZE),
        ("rho", ctypes.c_float * CDFT_GRID_SIZE),
        ("n_charge", ctypes.c_float * CDFT_GRID_SIZE),
        ("V_ext", ctypes.c_float * CDFT_GRID_SIZE),
        ("phi_R", ctypes.c_float * CDFT_GRID_SIZE),
        ("c1_pred", ctypes.c_float * CDFT_GRID_SIZE),
        ("current_filling", ctypes.c_float),
        ("el_residual", ctypes.c_float),
        ("reward", ctypes.c_float),
        ("done", ctypes.c_bool),
        ("step_count", ctypes.c_int),
        ("max_steps", ctypes.c_int),
        ("observations", ctypes.c_void_p),
        ("actions", ctypes.c_void_p),
        ("rewards", ctypes.c_void_p),
        ("terminals", ctypes.c_void_p),
        ("rng_state", ctypes.c_uint64),
    ]


if _env_lib is not None:
    _env_lib.cdft_env_create.argtypes = [ctypes.c_int, ctypes.c_uint64]
    _env_lib.cdft_env_create.restype = ctypes.POINTER(CCdftEnvStruct)

    _env_lib.cdft_env_destroy.argtypes = [ctypes.POINTER(CCdftEnvStruct)]

    _env_lib.cdft_env_reset.argtypes = [ctypes.POINTER(CCdftEnvStruct), ctypes.c_int]
    _env_lib.cdft_env_step.argtypes = [ctypes.POINTER(CCdftEnvStruct), ctypes.c_int]
    _env_lib.cdft_env_compute_restructuring_phi_r.argtypes = [ctypes.POINTER(CCdftEnvStruct)]
    _env_lib.cdft_env_picard_relaxation_step.argtypes = [ctypes.POINTER(CCdftEnvStruct), ctypes.c_float]


class DensCityFluidEnv:
    """Gymnasium & PufferLib compliant environment for active cDFT fluid control."""

    def __init__(self, num_envs: int = 1, seed: int = 42):
        if _env_lib is None:
            raise RuntimeError("libdens_city_env.so not loaded. Compile with gcc first.")
        self.num_envs = num_envs
        self._envs_ptr = _env_lib.cdft_env_create(num_envs, seed)

    def __del__(self):
        if hasattr(self, "_envs_ptr") and self._envs_ptr and _env_lib is not None:
            _env_lib.cdft_env_destroy(self._envs_ptr)
            self._envs_ptr = None

    def reset(self, env_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        _env_lib.cdft_env_reset(self._envs_ptr, env_idx)
        return self.get_obs(env_idx), {}

    def step(self, action: np.ndarray, env_idx: int = 0) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        env = self._envs_ptr[env_idx]
        # Action is [phi_0_norm, mode_m_norm, v_bias_norm]
        phi_0 = float(action[0]) * 5.0  # Scale to [-5V, +5V]
        mode_m = max(1.0, round(float(action[1]) * 2.0 + 3.0))  # Scale to mode 1..5
        v_bias = float(action[2]) * 2.0  # Scale to [-2V, +2V]

        env.phi_0 = phi_0
        env.mode_m = mode_m
        env.v_bias = v_bias

        _env_lib.cdft_env_step(self._envs_ptr, env_idx)

        obs = self.get_obs(env_idx)
        reward = float(env.reward)
        done = bool(env.done)
        info = {
            "current_filling": float(env.current_filling),
            "target_filling": float(env.target_filling),
            "el_residual": float(env.el_residual),
            "T": float(env.T),
            "mu": float(env.mu_target),
        }
        return obs, reward, done, False, info

    def get_obs(self, env_idx: int = 0) -> np.ndarray:
        env = self._envs_ptr[env_idx]
        rho = np.ctypeslib.as_array(env.rho)
        v_ext = np.ctypeslib.as_array(env.V_ext)
        phi_r = np.ctypeslib.as_array(env.phi_R)
        scalars = np.array([env.T / 500.0, env.mu_target / 1e-19, env.target_filling], dtype=np.float32)
        return np.concatenate([rho, v_ext, phi_r, scalars]).astype(np.float32)

    # UI property bindings
    @property
    def current_filling(self) -> float:
        return float(self._envs_ptr[0].current_filling)

    @property
    def target_filling(self) -> float:
        return float(self._envs_ptr[0].target_filling)

    @target_filling.setter
    def target_filling(self, val: float):
        self._envs_ptr[0].target_filling = float(val)

    @property
    def phi_0(self) -> float:
        return float(self._envs_ptr[0].phi_0)

    @phi_0.setter
    def phi_0(self, val: float):
        self._envs_ptr[0].phi_0 = float(val)

    @property
    def mode_m(self) -> float:
        return float(self._envs_ptr[0].mode_m)

    @mode_m.setter
    def mode_m(self, val: float):
        self._envs_ptr[0].mode_m = float(val)

    @property
    def v_bias(self) -> float:
        return float(self._envs_ptr[0].v_bias)

    @v_bias.setter
    def v_bias(self, val: float):
        self._envs_ptr[0].v_bias = float(val)

    @property
    def rho(self) -> np.ndarray:
        return np.copy(np.ctypeslib.as_array(self._envs_ptr[0].rho))

    @property
    def phi_R(self) -> np.ndarray:
        return np.copy(np.ctypeslib.as_array(self._envs_ptr[0].phi_R))

    @property
    def V_ext(self) -> np.ndarray:
        return np.copy(np.ctypeslib.as_array(self._envs_ptr[0].V_ext))

    @property
    def z_coords(self) -> np.ndarray:
        return np.copy(np.ctypeslib.as_array(self._envs_ptr[0].z_coords))
