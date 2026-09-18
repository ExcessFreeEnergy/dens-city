"""
dens-city: High-Performance Molecular Classical Density Functional Theory (cDFT) Platform.
Powered by pure tinygrad tensor operations, autograd variational optimization, and JIT compilation.
"""

import os

# Prevent premature Tinygrad HCQ device hang watchdog timeouts during heavy compilation / large polyatomic batches
os.environ.setdefault("HCQDEV_WAIT_TIMEOUT_MS", "300000")


def _patch_tinygrad_nv_overflow():
    try:
        from tinygrad.runtime.ops_nv import QMD

        _orig_write = QMD.write

        def _safe_write(self, **kwargs):
            for k, val in kwargs.items():
                if k.upper() == "PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED" and isinstance(val, int):
                    kwargs[k] = val & 0xFFFFFFFF
            return _orig_write(self, **kwargs)

        QMD.write = _safe_write
    except Exception:
        pass


_patch_tinygrad_nv_overflow()

from dens_city.cdft import KernelBuilder, TinyCDFT
from dens_city.utils.materials import Material, MaterialLoader
from dens_city.utils.pipeline import (
    MaterialPipelineResult,
    MaterialPipelineTask,
    PipelineStatus,
    process_material_task,
    save_flow_weights,
    write_xyz_trajectory,
)

__all__ = [
    "TinyCDFT",
    "Material",
    "MaterialLoader",
    "KernelBuilder",
    "MaterialPipelineTask",
    "MaterialPipelineResult",
    "PipelineStatus",
    "process_material_task",
    "write_xyz_trajectory",
    "save_flow_weights",
]
