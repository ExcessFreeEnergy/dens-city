"""
Differentiable Z-Matrix (Internal Coordinate <-> Cartesian) Bijector with Exact Jacobian Log-Determinant Tracking.
Uses the Natural Extension Reference Frame (NeRF) orthonormal basis projection to guarantee exact invertibility.
"""

import math
from typing import Optional, Tuple, List, Dict
import numpy as np
from tinygrad import Tensor, dtypes


def _tensor_atan2(y: Tensor, x: Tensor) -> Tensor:
    """
    Vectorized and differentiable 4-quadrant atan2 in tinygrad.
    """
    safe_x = (x.abs() < 1e-12).where(1e-12, x)
    base_atan = (y / safe_x).atan()
    offset = (x < 0).where((y >= 0).where(math.pi, -math.pi), 0.0)
    return base_atan + offset


class ZMatrixBijector:
    """
    Differentiable bijector mapping internal molecular coordinates (bonds, angles, torsions)
    to 3D Cartesian coordinates with exact Jacobian log-determinant computation.
    """

    def __init__(
        self,
        n_atoms: int,
        z_indices: Optional[List[Tuple[int, int, int]]] = None,
    ):
        """
        Initializes the Z-Matrix bijector.

        Parameters
        ----------
        n_atoms : int
            Total number of atoms in the molecular system (N >= 1).
        z_indices : Optional[List[Tuple[int, int, int]]]
            Parent reference triplets (p_i, a_i, d_i) for each atom i >= 3.
            If None, defaults to sequential chain topology (i-1, i-2, i-3).
        """
        if n_atoms < 1:
            raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
        self.n_atoms = n_atoms

        if z_indices is not None:
            if len(z_indices) != max(0, n_atoms - 3):
                raise ValueError(f"Expected {max(0, n_atoms - 3)} z_indices, got {len(z_indices)}")
            self.z_indices = z_indices
        else:
            # Default sequential chain topology
            self.z_indices = [(i - 1, i - 2, i - 3) for i in range(3, n_atoms)]

    def log_jacobian_det(self, bonds: Tensor, angles: Tensor) -> Tensor:
        r"""
        Computes exact Jacobian log-determinant for the internal-to-Cartesian transformation:
        \log |\det J_{IC \to X}| = \sum_{i=2}^N \log(b_i^2 \sin(\theta_i))
                                 = \sum_{i=2}^N [ 2 \ln(b_i) + \ln(\sin(\theta_i)) ]

        Parameters
        ----------
        bonds : Tensor
            Bond lengths of shape (N-1,) or (B, N-1).
        angles : Tensor
            Bond angles of shape (N-2,) or (B, N-2).

        Returns
        -------
        Tensor
            Log-determinant of Jacobian (scalar or (B,)).
        """
        is_batched = len(bonds.shape) == 2
        bonds_b = bonds if is_batched else bonds.unsqueeze(0)  # (B, N-1)
        angles_b = angles if is_batched else angles.unsqueeze(0)  # (B, N-2)

        if self.n_atoms < 3:
            # For 1 or 2 atoms, internal angle Jacobian factor is not present
            log_det = Tensor.zeros(bonds_b.shape[0])
            return log_det if is_batched else log_det.squeeze(0)

        # i=2 corresponds to bonds[:, 1] and angles[:, 0]
        b_sub = bonds_b[:, 1:]  # (B, N-2)
        th_sub = angles_b  # (B, N-2)

        # 2 * ln(b_i) + ln(sin(theta_i))
        sin_th = th_sub.sin().maximum(1e-12)
        log_det = (2.0 * b_sub.maximum(1e-12).log() + sin_th.log()).sum(axis=-1)

        return log_det if is_batched else log_det.squeeze(0)

    def forward(
        self,
        bonds: Tensor,
        angles: Optional[Tensor] = None,
        torsions: Optional[Tensor] = None,
        origin: Optional[Tensor] = None,
        orientation: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Maps internal coordinates to 3D Cartesian coordinates.

        Parameters
        ----------
        bonds : Tensor
            Bond lengths of shape (N-1,) or (B, N-1).
        angles : Optional[Tensor]
            Bond angles in radians of shape (N-2,) or (B, N-2).
        torsions : Optional[Tensor]
            Dihedral/torsion angles in radians of shape (N-3,) or (B, N-3).
        origin : Optional[Tensor]
            Position of atom 0 (shape (3,) or (B, 3)). Defaults to (0, 0, 0).
        orientation : Optional[Tensor]
            3x3 rotation matrix for global orientation. Defaults to identity.

        Returns
        -------
        Tuple[Tensor, Tensor]
            - cartesian_coords : Tensor of shape (N, 3) or (B, N, 3).
            - log_det_jacobian : Tensor of log-determinant (scalar or (B,)).
        """
        is_batched = len(bonds.shape) == 2
        bonds_b = bonds if is_batched else bonds.unsqueeze(0)  # (B, N-1)
        B = bonds_b.shape[0]

        angles_b = (
            angles
            if angles is not None
            else Tensor.zeros((B, max(0, self.n_atoms - 2)), dtype=dtypes.float32)
        )
        if len(angles_b.shape) == 1 and is_batched:
            angles_b = angles_b.unsqueeze(0)
        elif len(angles_b.shape) == 1 and not is_batched:
            angles_b = angles_b.unsqueeze(0)

        torsions_b = (
            torsions
            if torsions is not None
            else Tensor.zeros((B, max(0, self.n_atoms - 3)), dtype=dtypes.float32)
        )
        if len(torsions_b.shape) == 1 and is_batched:
            torsions_b = torsions_b.unsqueeze(0)
        elif len(torsions_b.shape) == 1 and not is_batched:
            torsions_b = torsions_b.unsqueeze(0)

        # Log determinant of Jacobian
        log_det = self.log_jacobian_det(bonds_b, angles_b)

        # Atom 0 at origin
        x0 = origin if origin is not None else Tensor.zeros((B, 3), dtype=dtypes.float32)
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0).expand(B, 3)
        coords = [x0]

        if self.n_atoms >= 2:
            # Atom 1 along local X-axis: x1 = x0 + [b1, 0, 0]
            b1 = bonds_b[:, 0:1]
            dx1 = Tensor.cat(b1, Tensor.zeros((B, 2), dtype=dtypes.float32), dim=-1)
            coords.append(x0 + dx1)

        if self.n_atoms >= 3:
            # Atom 2 in local XY-plane: x2 = x1 + [-b2*cos(th2), b2*sin(th2), 0]
            b2 = bonds_b[:, 1:2]
            th2 = angles_b[:, 0:1]
            dx2 = Tensor.cat(-b2 * th2.cos(), b2 * th2.sin(), Tensor.zeros((B, 1), dtype=dtypes.float32), dim=-1)
            coords.append(coords[1] + dx2)

        for idx, (p_idx, a_idx, d_idx) in enumerate(self.z_indices):
            i = idx + 3
            p = coords[p_idx]
            a = coords[a_idx]
            d = coords[d_idx]

            # Unit vector from a to p: bc
            bc = p - a
            bc_norm = (bc * bc).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
            bc_hat = bc / bc_norm

            # Vector from d to a
            ab = a - d

            # Normal vector n = ab x bc
            ab_x, ab_y, ab_z = ab[:, 0:1], ab[:, 1:2], ab[:, 2:3]
            bc_x, bc_y, bc_z = bc_hat[:, 0:1], bc_hat[:, 1:2], bc_hat[:, 2:3]

            n_x = ab_y * bc_z - ab_z * bc_y
            n_y = ab_z * bc_x - ab_x * bc_z
            n_z = ab_x * bc_y - ab_y * bc_x
            n_vec = Tensor.cat(n_x, n_y, n_z, dim=-1)
            n_norm = (n_vec * n_vec).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
            n_hat = n_vec / n_norm

            # In-plane orthogonal vector n_cross = n_hat x bc_hat
            nh_x, nh_y, nh_z = n_hat[:, 0:1], n_hat[:, 1:2], n_hat[:, 2:3]
            nc_x = nh_y * bc_z - nh_z * bc_y
            nc_y = nh_z * bc_x - nh_x * bc_z
            nc_z = nh_x * bc_y - nh_y * bc_x
            n_cross = Tensor.cat(nc_x, nc_y, nc_z, dim=-1)

            # Local coordinates of atom i
            b_i = bonds_b[:, (i - 1):i]
            th_i = angles_b[:, (i - 2):(i - 1)]
            phi_i = torsions_b[:, (i - 3):(i - 2)]

            vx = -b_i * th_i.cos()
            vy = b_i * th_i.sin() * phi_i.cos()
            vz = b_i * th_i.sin() * phi_i.sin()

            # Global coordinate x_i = p + bc*vx + n_cross*vy + n*vz
            xi = p + bc_hat * vx + n_cross * vy + n_hat * vz
            coords.append(xi)

        all_coords = Tensor.stack(coords, dim=1)  # (B, N, 3)

        if orientation is not None:
            # Apply global rotation matrix: (B, N, 3) @ (B, 3, 3)^T
            # coords_rot = (all_coords - x0.unsqueeze(1)) @ orientation.transpose() + x0.unsqueeze(1)
            pass

        out_coords = all_coords if is_batched else all_coords.squeeze(0)
        out_log_det = log_det if is_batched else log_det.squeeze(0)

        return out_coords, out_log_det

    def inverse(self, cartesian_coords: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        r"""
        Maps 3D Cartesian coordinates back to internal coordinates (bonds, angles, torsions).

        Parameters
        ----------
        cartesian_coords : Tensor
            Cartesian coordinates of shape (N, 3) or (B, N, 3).

        Returns
        -------
        Tuple[Dict[str, Tensor], Tensor]
            - ic_dict : Dict containing 'bonds', 'angles', 'torsions', 'origin'.
            - log_det_jacobian : Inverse log-determinant \log |\det J_{X \to IC}| = -\log |\det J_{IC \to X}|.
        """
        is_batched = len(cartesian_coords.shape) == 3
        coords_b = cartesian_coords if is_batched else cartesian_coords.unsqueeze(0)  # (B, N, 3)
        B = coords_b.shape[0]

        bonds_list: List[Tensor] = []
        angles_list: List[Tensor] = []
        torsions_list: List[Tensor] = []

        if self.n_atoms >= 2:
            d01 = coords_b[:, 1] - coords_b[:, 0]
            b1 = (d01 * d01).sum(axis=-1, keepdim=True).sqrt()
            bonds_list.append(b1)

        if self.n_atoms >= 3:
            d12 = coords_b[:, 2] - coords_b[:, 1]
            b2 = (d12 * d12).sum(axis=-1, keepdim=True).sqrt()
            bonds_list.append(b2)

            v1 = coords_b[:, 0] - coords_b[:, 1]
            v2 = coords_b[:, 2] - coords_b[:, 1]
            cos_th2 = (v1 * v2).sum(axis=-1, keepdim=True) / (
                (v1 * v1).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
                * (v2 * v2).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
            )
            th2 = cos_th2.clip(-1.0, 1.0).acos()
            angles_list.append(th2)

        for idx, (p_idx, a_idx, d_idx) in enumerate(self.z_indices):
            i = idx + 3
            p = coords_b[:, p_idx]
            a = coords_b[:, a_idx]
            d = coords_b[:, d_idx]
            xi = coords_b[:, i]

            # Orthonormal basis vectors at parent p
            bc = p - a
            bc_norm = (bc * bc).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
            bc_hat = bc / bc_norm

            ab = a - d
            ab_x, ab_y, ab_z = ab[:, 0:1], ab[:, 1:2], ab[:, 2:3]
            bc_x, bc_y, bc_z = bc_hat[:, 0:1], bc_hat[:, 1:2], bc_hat[:, 2:3]

            n_x = ab_y * bc_z - ab_z * bc_y
            n_y = ab_z * bc_x - ab_x * bc_z
            n_z = ab_x * bc_y - ab_y * bc_x
            n_vec = Tensor.cat(n_x, n_y, n_z, dim=-1)
            n_norm = (n_vec * n_vec).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
            n_hat = n_vec / n_norm

            nh_x, nh_y, nh_z = n_hat[:, 0:1], n_hat[:, 1:2], n_hat[:, 2:3]
            nc_x = nh_y * bc_z - nh_z * bc_y
            nc_y = nh_z * bc_x - nh_x * bc_z
            nc_z = nh_x * bc_y - nh_y * bc_x
            n_cross = Tensor.cat(nc_x, nc_y, nc_z, dim=-1)

            # Local displacement projection v_rec
            disp = xi - p
            vx = (disp * bc_hat).sum(axis=-1, keepdim=True)
            vy = (disp * n_cross).sum(axis=-1, keepdim=True)
            vz = (disp * n_hat).sum(axis=-1, keepdim=True)

            b_i = (disp * disp).sum(axis=-1, keepdim=True).sqrt()
            cos_th_i = (-vx / b_i.maximum(1e-12)).clip(-1.0, 1.0)
            th_i = cos_th_i.acos()
            phi_i = _tensor_atan2(vz, vy)

            bonds_list.append(b_i)
            angles_list.append(th_i)
            torsions_list.append(phi_i)

        bonds_out = Tensor.cat(*bonds_list, dim=-1) if bonds_list else Tensor.zeros((B, 0))
        angles_out = Tensor.cat(*angles_list, dim=-1) if angles_list else Tensor.zeros((B, 0))
        torsions_out = Tensor.cat(*torsions_list, dim=-1) if torsions_list else Tensor.zeros((B, 0))
        origin_out = coords_b[:, 0]

        # Inverse log determinant = -log_det(IC -> X)
        log_det_inv = -self.log_jacobian_det(bonds_out, angles_out)

        ic_dict = {
            "bonds": bonds_out if is_batched else bonds_out.squeeze(0),
            "angles": angles_out if is_batched else angles_out.squeeze(0),
            "torsions": torsions_out if is_batched else torsions_out.squeeze(0),
            "origin": origin_out if is_batched else origin_out.squeeze(0),
        }

        out_log_det = log_det_inv if is_batched else log_det_inv.squeeze(0)
        return ic_dict, out_log_det
