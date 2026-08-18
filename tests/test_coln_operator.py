import math

import numpy as np
import torch

from dens_city.models.coln import (
    ConvolutedOperatorNetwork,
    compute_spherical_harmonics,
)
from dens_city.pipelines.co2.supercritical import compute_orientational_density_and_order


def test_spherical_harmonics_orthonormality():
    """Verify Spherical Harmonics calculations against analytical values."""
    theta = torch.tensor([0.0, math.pi / 2.0, math.pi])
    phi = torch.zeros_like(theta)

    y = compute_spherical_harmonics(theta, phi)  # [3, 3]
    assert y.shape == (3, 3)

    # Y_00 is constant: 1 / sqrt(4 * pi)
    expected_y00 = 1.0 / math.sqrt(4.0 * math.pi)
    assert torch.allclose(y[:, 0], torch.full_like(y[:, 0], expected_y00), atol=1e-5)

    # At theta = 0: cos(0) = 1 -> Y_10 = sqrt(3/(4pi))
    expected_y10_0 = math.sqrt(3.0 / (4.0 * math.pi))
    assert math.isclose(y[0, 1].item(), expected_y10_0, rel_tol=1e-4)

    # At theta = pi/2: cos(pi/2) = 0 -> Y_10 = 0
    assert math.isclose(y[1, 1].item(), 0.0, abs_tol=1e-5)


def test_coln_operator_forward_shape():
    """Verify ConvolutedOperatorNetwork forward pass and output dimensions."""
    B = 2
    spatial_dim = 64
    angular_dim = 30 * 30
    N_q = 10

    model = ConvolutedOperatorNetwork(spatial_dim=spatial_dim, angular_dim=angular_dim, basis_dim=32)

    rho_bar = torch.rand(B, spatial_dim)
    rho_hat = torch.rand(B, angular_dim)
    x_coords = torch.rand(B, N_q, 1)
    angles = torch.rand(B, N_q, 2) * math.pi

    c1_out = model(rho_bar, rho_hat, x_coords, angles)
    assert c1_out.shape == (B, N_q)
    assert not torch.isnan(c1_out).any()


def test_coln_mirror_flip_augmentation():
    """Verify mirror flip symmetry data augmentation."""
    model = ConvolutedOperatorNetwork(spatial_dim=32, angular_dim=64, basis_dim=16)

    rho_bar = torch.rand(2, 32)
    rho_hat = torch.rand(2, 64)

    rb_flip, rh_flip = model.apply_mirror_augmentation(rho_bar, rho_hat)
    assert rb_flip.shape == rho_bar.shape
    assert rh_flip.shape == rho_hat.shape
    assert torch.equal(rb_flip[0], torch.flip(rho_bar[0], dims=[0]))


def test_orientational_order_parameter():
    """Verify 3D orientational density and nematic order parameter in slit confinement."""
    model = ConvolutedOperatorNetwork(spatial_dim=64, angular_dim=30 * 30, basis_dim=32)

    res = compute_orientational_density_and_order(coln_model=model, H=20.0, T=400.0, rho_bulk=0.015, n_z=64, n_theta=30)

    assert "S_order" in res
    assert "rho_bar" in res
    assert "rho_z_theta" in res

    s_order = res["S_order"]
    assert len(s_order) == 64
    # Nematic order parameter S is bounded in [-0.5, 1.0]
    assert np.all(s_order >= -0.55)
    assert np.all(s_order <= 1.05)

    # In bulk center (z ~ 10 A), order parameter should approach isotropic bulk (S ~ 0)
    center_idx = 32
    assert abs(s_order[center_idx]) < 0.2
