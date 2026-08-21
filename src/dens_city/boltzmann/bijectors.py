"""
Differentiable Z-Matrix (Internal Coordinate <-> Cartesian) Bijector with Exact Jacobian Log-Determinant Tracking.
Uses the Natural Extension Reference Frame (NeRF) orthonormal basis projection to guarantee exact invertibility.
"""

import functools
from typing import Optional, Tuple, List, Dict, Callable, Union
import math
from tinygrad import Tensor, dtypes, nn


def _tensor_atan2(y: Tensor, x: Tensor) -> Tensor:
    """
    Vectorized and differentiable 4-quadrant atan2 in tinygrad.
    """
    safe_x = (x.abs() < 1e-12).where(1e-12, x)
    base_atan = (y / safe_x).atan()
    offset = (x < 0).where((y >= 0).where(math.pi, -math.pi), 0.0)
    return base_atan + offset


def _cross(u: Tensor, v: Tensor) -> Tensor:
    """
    Vectorized 3D cross product u x v for trailing dimension of size 3.
    """
    c0 = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    c1 = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    c2 = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return Tensor.stack(c0, c1, c2, dim=-1)


def _nerf_step(
    p: Tensor,
    a: Tensor,
    d: Tensor,
    b_i: Tensor,
    th_i: Tensor,
    phi_i: Tensor,
) -> Tensor:
    """
    Fused Natural Extension Reference Frame (NeRF) forward step.
    Orthonormal basis projection, cross products, and trigonometric displacement
    are fused into a single compound elementwise kernel graph.
    """
    bc = p - a
    bc_norm = (bc * bc).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
    bc_hat = bc / bc_norm

    ab = a - d
    n0 = ab[..., 1] * bc_hat[..., 2] - ab[..., 2] * bc_hat[..., 1]
    n1 = ab[..., 2] * bc_hat[..., 0] - ab[..., 0] * bc_hat[..., 2]
    n2 = ab[..., 0] * bc_hat[..., 1] - ab[..., 1] * bc_hat[..., 0]
    n_vec = Tensor.stack(n0, n1, n2, dim=-1)
    n_norm = (n_vec * n_vec).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
    n_hat = n_vec / n_norm

    c0 = n_hat[..., 1] * bc_hat[..., 2] - n_hat[..., 2] * bc_hat[..., 1]
    c1 = n_hat[..., 2] * bc_hat[..., 0] - n_hat[..., 0] * bc_hat[..., 2]
    c2 = n_hat[..., 0] * bc_hat[..., 1] - n_hat[..., 1] * bc_hat[..., 0]
    n_cross = Tensor.stack(c0, c1, c2, dim=-1)

    vx = -b_i * th_i.cos()
    vy = b_i * th_i.sin() * phi_i.cos()
    vz = b_i * th_i.sin() * phi_i.sin()

    return p + bc_hat * vx + n_cross * vy + n_hat * vz


def _nerf_inverse_step(
    p: Tensor,
    a: Tensor,
    d: Tensor,
    xi: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused Natural Extension Reference Frame (NeRF) inverse step.
    Computes bond length, planar angle, and dihedral torsion angle in a single compound kernel.
    """
    bc = p - a
    bc_norm = (bc * bc).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
    bc_hat = bc / bc_norm

    ab = a - d
    n0 = ab[..., 1] * bc_hat[..., 2] - ab[..., 2] * bc_hat[..., 1]
    n1 = ab[..., 2] * bc_hat[..., 0] - ab[..., 0] * bc_hat[..., 2]
    n2 = ab[..., 0] * bc_hat[..., 1] - ab[..., 1] * bc_hat[..., 0]
    n_vec = Tensor.stack(n0, n1, n2, dim=-1)
    n_norm = (n_vec * n_vec).sum(axis=-1, keepdim=True).sqrt().maximum(1e-12)
    n_hat = n_vec / n_norm

    c0 = n_hat[..., 1] * bc_hat[..., 2] - n_hat[..., 2] * bc_hat[..., 1]
    c1 = n_hat[..., 2] * bc_hat[..., 0] - n_hat[..., 0] * bc_hat[..., 2]
    c2 = n_hat[..., 0] * bc_hat[..., 1] - n_hat[..., 1] * bc_hat[..., 0]
    n_cross = Tensor.stack(c0, c1, c2, dim=-1)

    disp = xi - p
    vx = (disp * bc_hat).sum(axis=-1, keepdim=True)
    vy = (disp * n_cross).sum(axis=-1, keepdim=True)
    vz = (disp * n_hat).sum(axis=-1, keepdim=True)

    b_i = (disp * disp).sum(axis=-1, keepdim=True).sqrt()
    cos_th_i = (-vx / b_i.maximum(1e-12)).clip(-1.0 + 1e-6, 1.0 - 1e-6)
    th_i = cos_th_i.acos()
    phi_i = _tensor_atan2(vz, vy)

    return b_i, th_i, phi_i


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
        \log |\det J| = \sum_{i=2}^{N} 2 \ln b_i + \sum_{i=3}^{N} \ln \sin \theta_i
        """
        is_batched = len(bonds.shape) == 2
        bonds_b = bonds if is_batched else bonds.unsqueeze(0)
        angles_b = angles if is_batched else angles.unsqueeze(0)
        B = bonds_b.shape[0]

        log_det = Tensor.zeros(B, dtype=dtypes.float32)

        if self.n_atoms >= 2 and bonds_b.shape[-1] >= 1:
            # First bond b1 does not carry curvature term
            pass

        if self.n_atoms >= 3 and bonds_b.shape[-1] >= 2:
            # b2 contributes 2 ln b2
            b_rest = bonds_b[:, 1:]
            log_det = log_det + 2.0 * (b_rest.maximum(1e-12)).log().sum(axis=-1)

        if self.n_atoms >= 3 and angles_b.shape[-1] >= 1:
            # angles th2, th3, ... contribute ln sin(th)
            sin_angles = (angles_b.sin()).maximum(1e-12)
            log_det = log_det + sin_angles.log().sum(axis=-1)

        return log_det if is_batched else log_det.squeeze(0)

    def forward(
        self,
        internal_coords: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
        bonds: Optional[Tensor] = None,
        angles: Optional[Tensor] = None,
        torsions: Optional[Tensor] = None,
        origin: Optional[Tensor] = None,
        orientation: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Reconstructs 3D Cartesian coordinates from internal coordinates via NeRF.
        Supports both unified internal_coords (flat Tensor or dict) and explicit (bonds, angles, torsions) kwargs.
        """
        is_batched = False
        if internal_coords is not None:
            if isinstance(internal_coords, dict):
                bonds = internal_coords.get("bonds", Tensor.zeros((1, max(0, self.n_atoms - 1))))
                angles = internal_coords.get("angles", Tensor.zeros((1, max(0, self.n_atoms - 2))))
                torsions = internal_coords.get("torsions", Tensor.zeros((1, max(0, self.n_atoms - 3))))
                is_batched = len(bonds.shape) == 2
            else:
                is_batched = len(internal_coords.shape) == 2
                ic_b = internal_coords if is_batched else internal_coords.unsqueeze(0)
                n_bonds = max(0, self.n_atoms - 1)
                n_angles = max(0, self.n_atoms - 2)
                bonds = ic_b[:, :n_bonds]
                angles = ic_b[:, n_bonds : (n_bonds + n_angles)]
                torsions = ic_b[:, (n_bonds + n_angles) :]
        elif bonds is not None:
            is_batched = len(bonds.shape) == 2

        if bonds is None:
            bonds = Tensor.zeros((1, max(0, self.n_atoms - 1)), dtype=dtypes.float32)
        bonds_b = bonds if len(bonds.shape) == 2 else bonds.unsqueeze(0)
        B = bonds_b.shape[0]

        if angles is None:
            angles_b = Tensor.zeros((B, max(0, self.n_atoms - 2)), dtype=dtypes.float32)
        else:
            angles_b = angles if len(angles.shape) == 2 else angles.unsqueeze(0)

        if torsions is None:
            torsions_b = Tensor.zeros((B, max(0, self.n_atoms - 3)), dtype=dtypes.float32)
        else:
            torsions_b = torsions if len(torsions.shape) == 2 else torsions.unsqueeze(0)

        log_det = self.log_jacobian_det(bonds_b, angles_b)

        # Place origin atom 0
        if origin is not None:
            x0 = origin if len(origin.shape) == 2 else origin.unsqueeze(0)
        else:
            x0 = Tensor.zeros((B, 3), dtype=dtypes.float32)

        coords: List[Tensor] = [x0]

        if self.n_atoms >= 2:
            # Atom 1 along local X-axis: x1 = x0 + [b1, 0, 0]
            b1 = bonds_b[:, 0:1]
            dx1 = b1.pad(((0, 0), (0, 2)))
            coords.append(x0 + dx1)

        if self.n_atoms >= 3:
            # Atom 2 in local XY-plane: x2 = x1 + [-b2*cos(th2), b2*sin(th2), 0]
            b2 = bonds_b[:, 1:2]
            th2 = angles_b[:, 0:1]
            dx2_xy = Tensor.cat(-b2 * th2.cos(), b2 * th2.sin(), dim=-1)
            dx2 = dx2_xy.pad(((0, 0), (0, 1)))
            coords.append(coords[1] + dx2)

        for idx, (p_idx, a_idx, d_idx) in enumerate(self.z_indices):
            i = idx + 3
            p = coords[p_idx]
            a = coords[a_idx]
            d = coords[d_idx]

            b_i = bonds_b[:, (i - 1) : i]
            th_i = angles_b[:, (i - 2) : (i - 1)]
            phi_i = torsions_b[:, (i - 3) : (i - 2)]

            xi = _nerf_step(p, a, d, b_i, th_i, phi_i)
            coords.append(xi)

        all_coords = Tensor.cat(*[c.unsqueeze(1) for c in coords], dim=1)  # (B, N, 3)

        if orientation is not None:
            pass

        out_coords = all_coords if is_batched else all_coords.squeeze(0)
        out_log_det = log_det if is_batched else log_det.squeeze(0)

        return out_coords, out_log_det

    def inverse(self, cartesian_coords: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        r"""
        Maps 3D Cartesian coordinates back to internal coordinates (bonds, angles, torsions).
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
            th2 = cos_th2.clip(-1.0 + 1e-6, 1.0 - 1e-6).acos()
            angles_list.append(th2)

        for idx, (p_idx, a_idx, d_idx) in enumerate(self.z_indices):
            i = idx + 3
            p = coords_b[:, p_idx]
            a = coords_b[:, a_idx]
            d = coords_b[:, d_idx]
            xi = coords_b[:, i]

            b_i, th_i, phi_i = _nerf_inverse_step(p, a, d, xi)

            bonds_list.append(b_i)
            angles_list.append(th_i)
            torsions_list.append(phi_i)

        bonds_out = Tensor.cat(*bonds_list, dim=-1) if bonds_list else Tensor.zeros((B, 0), dtype=dtypes.float32)
        angles_out = Tensor.cat(*angles_list, dim=-1) if angles_list else Tensor.zeros((B, 0), dtype=dtypes.float32)
        torsions_out = Tensor.cat(*torsions_list, dim=-1) if torsions_list else Tensor.zeros((B, 0), dtype=dtypes.float32)
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


class AffineCouplingLayer:
    """
    Affine Coupling Layer with exact invertibility and analytical Jacobian log-determinant.
    Partitions input x into [x1, x2] and applies learned scale/shift transformations via Tinygrad NN.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, swap: bool = False):
        self.dim = dim
        self.swap = swap
        self.dim_a = max(1, dim // 2)
        self.dim_b = dim - self.dim_a

        # When swap=False: net takes dim_a, outputs dim_b * 2
        # When swap=True: net takes dim_b, outputs dim_a * 2
        self.in_dim = self.dim_b if swap else self.dim_a
        self.out_dim = self.dim_a if swap else self.dim_b

        self.net: list[Callable[[Tensor], Tensor]] = [
            nn.Linear(self.in_dim, hidden_dim),
            Tensor.relu,
            nn.Linear(hidden_dim, hidden_dim),
            Tensor.relu,
            nn.Linear(hidden_dim, self.out_dim * 2),
        ]

    def _net(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = x.sequential(self.net)
        return out[..., : self.out_dim].tanh(), out[..., self.out_dim :]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward affine coupling:
        y_A = x_A
        y_B = x_B * exp(s(x_A)) + t(x_A)
        log |det J| = sum(s(x_A))
        """
        is_batched = len(x.shape) == 2
        x_b = x if is_batched else x.unsqueeze(0)

        x1, x2 = x_b[..., :self.dim_a], x_b[..., self.dim_a:]

        if self.swap:
            s, t = self._net(x2)
            y1 = x1 * s.exp() + t
            y2 = x2
            y = Tensor.cat(y1, y2, dim=-1)
        else:
            s, t = self._net(x1)
            y1 = x1
            y2 = x2 * s.exp() + t
            y = Tensor.cat(y1, y2, dim=-1)

        log_det = s.sum(axis=-1)
        return (y if is_batched else y.squeeze(0)), (log_det if is_batched else log_det.squeeze(0))

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse affine coupling:
        x_A = y_A
        x_B = (y_B - t(y_A)) * exp(-s(y_A))
        log |det J_inv| = -sum(s(y_A))
        """
        is_batched = len(y.shape) == 2
        y_b = y if is_batched else y.unsqueeze(0)

        y1, y2 = y_b[..., :self.dim_a], y_b[..., self.dim_a:]

        if self.swap:
            s, t = self._net(y2)
            x1 = (y1 - t) * (-s).exp()
            x2 = y2
            x = Tensor.cat(x1, x2, dim=-1)
        else:
            s, t = self._net(y1)
            x1 = y1
            x2 = (y2 - t) * (-s).exp()
            x = Tensor.cat(x1, x2, dim=-1)

        log_det = (-s).sum(axis=-1)
        return (x if is_batched else x.squeeze(0)), (log_det if is_batched else log_det.squeeze(0))


class RealNVPFlow:
    """
    Stacked sequence of RealNVP Affine Coupling Layers with alternating partition masks.
    """

    def __init__(self, dim: int, n_layers: int = 5, hidden_dim: int = 64):
        self.dim = dim
        self.n_layers = n_layers
        self.layers: list[AffineCouplingLayer] = [
            AffineCouplingLayer(dim, hidden_dim=hidden_dim, swap=(i % 2 == 1))
            for i in range(n_layers)
        ]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward flow through all stacked coupling layers.
        """
        cur = x
        is_batched = len(x.shape) == 2
        log_dets = []
        for layer in self.layers:
            cur, ld = layer.forward(cur)
            log_dets.append(ld if is_batched else ld.unsqueeze(0))
        total_log_det = (
            functools.reduce(Tensor.add, log_dets)
            if log_dets
            else Tensor.zeros(cur.shape[0], dtype=dtypes.float32)
        )
        return cur, (total_log_det if is_batched else total_log_det.squeeze(0))

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Exact inverse flow in reverse layer order.
        """
        cur = y
        is_batched = len(y.shape) == 2
        log_dets = []
        for layer in reversed(self.layers):
            cur, ld = layer.inverse(cur)
            log_dets.append(ld if is_batched else ld.unsqueeze(0))
        total_log_det = (
            functools.reduce(Tensor.add, log_dets)
            if log_dets
            else Tensor.zeros(cur.shape[0], dtype=dtypes.float32)
        )
        return cur, (total_log_det if is_batched else total_log_det.squeeze(0))


class CompositeFlow:
    """
    Composite Invertible Normalizing Flow chaining RealNVPFlow with ZMatrixBijector.
    Transforms latent Gaussian noise z -> Internal Coordinates -> 3D Cartesian Coordinates x.

    Dimensionality partitioning for N atoms (N >= 3):
    - Bonds: N - 1
    - Angles: N - 2
    - Torsions: N - 3
    Total flow dimension: 3N - 6.
    """

    def __init__(
        self,
        n_atoms: int,
        flow: Optional[RealNVPFlow] = None,
        z_indices: Optional[List[Tuple[int, int, int]]] = None,
        n_layers: int = 5,
        hidden_dim: int = 64,
    ):
        if n_atoms < 1:
            raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
        self.n_atoms = n_atoms
        self.n_bonds = max(0, n_atoms - 1)
        self.n_angles = max(0, n_atoms - 2)
        self.n_torsions = max(0, n_atoms - 3)
        self.dim = self.n_bonds + self.n_angles + self.n_torsions

        self.zmat = ZMatrixBijector(n_atoms=n_atoms, z_indices=z_indices)
        self.flow = (
            flow
            if flow is not None
            else RealNVPFlow(dim=max(1, self.dim), n_layers=n_layers, hidden_dim=hidden_dim)
        )

    def forward(self, z: Tensor, origin: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Maps latent noise z to 3D Cartesian coordinates x:
        z (B, 3N-6) -> RealNVPFlow -> ic_flat (B, 3N-6) -> ZMatrixBijector -> x (B, N, 3)
        """
        is_batched = len(z.shape) == 2
        z_b = z if is_batched else z.unsqueeze(0)
        B = z_b.shape[0]

        if self.dim == 0:
            # Single atom at origin
            coords = origin if origin is not None else Tensor.zeros((B, 1, 3), dtype=dtypes.float32)
            if len(coords.shape) == 2:
                coords = coords.unsqueeze(1)
            log_det = Tensor.zeros(B, dtype=dtypes.float32)
            return (coords if is_batched else coords.squeeze(0)), (log_det if is_batched else log_det.squeeze(0))

        # 1. Flow transformation: z -> ic_flat
        ic_flat, log_det_flow = self.flow.forward(z_b)

        # 2. Slice flat internal coordinates into structural components
        bonds = ic_flat[:, :self.n_bonds]
        angles = (
            ic_flat[:, self.n_bonds : self.n_bonds + self.n_angles]
            if self.n_angles > 0
            else None
        )
        torsions = (
            ic_flat[:, self.n_bonds + self.n_angles : self.dim]
            if self.n_torsions > 0
            else None
        )

        # 3. Z-Matrix transformation: internal coordinates -> 3D Cartesian coordinates
        coords, log_det_zmat = self.zmat.forward(
            bonds=bonds, angles=angles, torsions=torsions, origin=origin
        )

        total_log_det = log_det_flow + log_det_zmat
        return (coords if is_batched else coords.squeeze(0)), (total_log_det if is_batched else total_log_det.squeeze(0))

    def inverse(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Maps 3D Cartesian coordinates x to latent noise z:
        x (B, N, 3) -> ZMatrixBijector -> ic_flat (B, 3N-6) -> RealNVPFlow.inverse -> z (B, 3N-6)
        """
        is_batched = len(coords.shape) == 3
        coords_b = coords if is_batched else coords.unsqueeze(0)
        B = coords_b.shape[0]

        if self.dim == 0:
            z_out = Tensor.zeros((B, 0), dtype=dtypes.float32)
            log_det = Tensor.zeros(B, dtype=dtypes.float32)
            return (z_out if is_batched else z_out.squeeze(0)), (log_det if is_batched else log_det.squeeze(0))

        # 1. Z-Matrix inverse transformation: 3D Cartesian coordinates -> internal coordinates
        ic_dict, log_det_zmat_inv = self.zmat.inverse(coords_b)

        # 2. Concatenate internal coordinates into flat tensor
        parts = [ic_dict["bonds"]]
        if self.n_angles > 0:
            parts.append(ic_dict["angles"])
        if self.n_torsions > 0:
            parts.append(ic_dict["torsions"])
        ic_flat = Tensor.cat(*parts, dim=-1)

        # 3. Flow inverse transformation: ic_flat -> z
        z, log_det_flow_inv = self.flow.inverse(ic_flat)

        total_log_det_inv = log_det_zmat_inv + log_det_flow_inv
        return (z if is_batched else z.squeeze(0)), (total_log_det_inv if is_batched else total_log_det_inv.squeeze(0))


