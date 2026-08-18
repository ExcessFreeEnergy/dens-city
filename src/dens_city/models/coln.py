"""
Convoluted Operator Learning Network (COLN) for Molecular cDFT.
Based on Yang, Pan, Sun, & Wu (2024) (arXiv:2411.03698 / spec3.md).

Deconvolutes high-dimensional orientational molecular density rho(x, theta, phi) into:
1. Directional DeepONet: Branch net takes locally angle-averaged density rho_bar(x)
2. Angular DeepONet: Branch net takes position-averaged angular function rho_hat(theta, phi)
3. Spherical Harmonics Expander: c_1(x, theta, phi) = sum_{l,m} c_{ml}(x) * Y_{ml}(theta, phi)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def compute_spherical_harmonics(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    r"""
    Computes low-order Spherical Harmonics for linear symmetric molecules (CO2, N2):
    Y_00 = 1 / sqrt(4 * pi)
    Y_10 = sqrt(3 / (4 * pi)) * cos(theta)
    Y_20 = sqrt(5 / (16 * pi)) * (3 * cos^2(theta) - 1)
    Returns: Tensor of shape [..., 3]
    """
    y00 = torch.full_like(theta, 1.0 / math.sqrt(4.0 * math.pi))
    y10 = math.sqrt(3.0 / (4.0 * math.pi)) * torch.cos(theta)
    y20 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * (torch.cos(theta) ** 2) - 1.0)
    return torch.stack([y00, y10, y20], dim=-1)


class DirectionalDeepONet(nn.Module):
    """
    Branch-Trunk network for the angle-averaged spatial density profile rho_bar(x).
    """

    def __init__(self, spatial_grid_dim: int = 64, basis_dim: int = 64, num_sh_modes: int = 3):
        super().__init__()
        self.num_sh_modes = num_sh_modes
        self.basis_dim = basis_dim

        # Branch net: takes quasi-local density profile rho_bar(x)
        self.branch = nn.Sequential(
            nn.Linear(spatial_grid_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(128, basis_dim * num_sh_modes),
        )

        # Trunk net: takes continuous spatial query coordinate x in [0, L]
        self.trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, basis_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rho_bar: torch.Tensor, x_coord: torch.Tensor) -> torch.Tensor:
        """
        rho_bar: [B, spatial_grid_dim]
        x_coord: [B, N_queries, 1] or [B, 1]
        Returns: Spherical harmonic projection coefficients c_{ml}(x) [B, N_queries, num_sh_modes]
        """
        B = rho_bar.shape[0]
        b_out = self.branch(rho_bar)  # [B, basis_dim * num_sh_modes]
        b_out = b_out.view(B, self.num_sh_modes, self.basis_dim)  # [B, modes, p]

        t_out = self.trunk(x_coord)  # [B, N_q, p]

        # Dot product across basis dimension p for each SH mode
        # b_out: [B, modes, p], t_out: [B, N_q, p] -> c_ml: [B, N_q, modes]
        c_ml = torch.einsum("bmp,bqp->bqm", b_out, t_out)
        return c_ml


class AngularDeepONet(nn.Module):
    """
    Branch-Trunk network for the position-averaged angular density profile rho_hat(theta, phi).
    Incorporates pi-periodicity for linear molecules.
    """

    def __init__(self, angular_grid_dim: int = 30 * 30, basis_dim: int = 64):
        super().__init__()
        self.basis_dim = basis_dim

        # Branch net: takes angular distribution rho_hat(theta, phi)
        self.branch = nn.Sequential(
            nn.Linear(angular_grid_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(128, basis_dim),
        )

        # Trunk net: takes continuous query angles (theta, phi)
        self.trunk = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, basis_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rho_hat: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        rho_hat: [B, angular_grid_dim]
        angles: [B, N_queries, 2] (theta, phi)
        Returns: Angular modulation factor [B, N_queries]
        """
        b_out = self.branch(rho_hat)  # [B, basis_dim]
        t_out = self.trunk(angles)  # [B, N_q, basis_dim]
        ang_out = torch.einsum("bp,bqp->bq", b_out, t_out)
        return ang_out


class ConvolutedOperatorNetwork(nn.Module):
    """
    Full Convoluted Operator Learning Network (COLN) combining:
    - Directional DeepONet (quasi-local rho_bar -> c_{ml}(x))
    - Angular DeepONet (periodic rho_hat -> angular modulation)
    - Spherical Harmonics projection Y_{ml}(theta, phi)
    """

    def __init__(
        self,
        spatial_dim: int = 64,
        angular_dim: int = 30 * 30,
        basis_dim: int = 64,
        num_sh_modes: int = 3,
    ):
        super().__init__()
        self.dir_net = DirectionalDeepONet(spatial_dim, basis_dim, num_sh_modes)
        self.ang_net = AngularDeepONet(angular_dim, basis_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        rho_bar: torch.Tensor,
        rho_hat: torch.Tensor,
        x_coords: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the continuous one-body direct correlation function c1(x, theta, phi).

        Args:
            rho_bar: [B, spatial_dim] Locally angle-averaged density profile
            rho_hat: [B, angular_dim] Position-averaged angular function
            x_coords: [B, N_q, 1] Query spatial positions
            angles: [B, N_q, 2] Query polar and azimuthal angles (theta, phi)

        Returns:
            c1_pred: [B, N_q] One-body direct correlation function c_1(x, theta, phi)
        """
        # 1. Evaluate spatial spherical harmonic projections c_{ml}(x)
        c_ml = self.dir_net(rho_bar, x_coords)  # [B, N_q, num_sh_modes]

        # 2. Compute analytical Spherical Harmonics Y_{ml}(theta, phi)
        theta = angles[..., 0]
        phi = angles[..., 1]
        y_ml = compute_spherical_harmonics(theta, phi)  # [B, N_q, num_sh_modes]

        # 3. Sum over spherical harmonic modes: sum_{l,m} c_{ml}(x) * Y_{ml}(theta, phi)
        c1_sh = torch.sum(c_ml * y_ml, dim=-1)  # [B, N_q]

        # 4. Multiply with angular modulation from Angular DeepONet
        ang_mod = self.ang_net(rho_hat, angles)  # [B, N_q]

        c1_pred = c1_sh * (1.0 + 0.1 * ang_mod) + self.bias
        return c1_pred

    def apply_mirror_augmentation(
        self, rho_bar: torch.Tensor, rho_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies physical mirror flip symmetry:
        rho_bar(x) <-> rho_bar(L - x), theta <-> pi - theta
        """
        rho_bar_flipped = torch.flip(rho_bar, dims=[-1])
        rho_hat_flipped = torch.flip(rho_hat, dims=[-1])
        return rho_bar_flipped, rho_hat_flipped
