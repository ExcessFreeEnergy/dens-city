"""
Boltzmann Generator: Deep Normalizing Flows trained on the exact Microscopic Hamiltonian
via variational Reverse Kullback-Leibler (KL) divergence minimization in pure tinygrad.
"""

import math
from typing import Callable, Optional, List, Union
import numpy as np
from tinygrad import Tensor, nn, dtypes

from dens_city.boltzmann.bijectors import RealNVPFlow
from dens_city.boltzmann.prior import CDFTBaseDistribution


class BoltzmannGenerator:
    """
    Learns the exact equilibrium Boltzmann distribution p(x) ~ exp(-beta * U(x))
    by training an invertible RealNVP normalizing flow on the exact microscopic potential energy.
    """

    def __init__(
        self,
        flow: RealNVPFlow,
        energy_fn: Callable[[Tensor], Tensor],
        prior: Optional[CDFTBaseDistribution] = None,
        temperature_k: float = 300.0,
        learning_rate: float = 0.01,
    ):
        """
        Initializes the Boltzmann Generator.

        Parameters
        ----------
        flow : RealNVPFlow
            Invertible normalizing flow network.
        energy_fn : Callable[[Tensor], Tensor]
            Exact microscopic Hamiltonian evaluating U(x) in Kelvin.
        prior : Optional[CDFTBaseDistribution]
            cDFT base spatial distribution. If None, uses standard isotropic Gaussian base distribution.
        temperature_k : float
            System thermodynamic temperature (in Kelvin).
        learning_rate : float
            Optimizer learning rate for Adam.
        """
        self.flow = flow
        self.energy_fn = energy_fn
        self.prior = prior
        self.temperature_k = float(temperature_k)
        # In dens-city, energies are in Kelvin, so beta = 1 / T
        self.beta = 1.0 / max(1e-6, self.temperature_k)
        self.dim = flow.dim

        # Optimizer over all flow parameters
        self.opt = nn.optim.Adam(nn.state.get_parameters(self.flow), lr=learning_rate)

    def compute_loss(self, z: Tensor) -> Tensor:
        r"""
        Evaluates the variational Reverse KL Divergence training loss:
        \mathcal{L}(\theta) = \mathbb{E}_{z \sim p_z} \left[ \beta U(f_\theta(z)) - \log p_z(z) - \log |\det J_{f_\theta}(z)| \right]

        Parameters
        ----------
        z : Tensor
            Base distribution samples of shape (B, dim) or (B, N, 3).

        Returns
        -------
        Tensor
            Scalar batch loss.
        """
        is_3d = len(z.shape) == 3
        B = z.shape[0]

        # Reshape to (B, dim) for flow if 3D particle positions
        z_flat = z.reshape(B, self.dim) if is_3d else z

        # Forward flow: z -> x
        x_flat, log_det = self.flow.forward(z_flat)
        x = x_flat.reshape(z.shape) if is_3d else x_flat

        # Evaluate exact microscopic potential energy
        u_exact = self.energy_fn(x)  # (B,)

        # Base distribution log probability log p_z(z)
        if self.prior is not None:
            log_pz = self.prior.log_prob(z)  # (B,)
        else:
            # Standard Gaussian log likelihood
            log_pz = -0.5 * (z_flat * z_flat + math.log(2.0 * math.pi)).sum(axis=-1)

        # Variational KL Loss
        loss_batch = self.beta * u_exact - log_pz - log_det
        return loss_batch.mean()

    def train_step(self, z: Tensor) -> float:
        """
        Executes a single gradient descent step on the flow parameters.

        Parameters
        ----------
        z : Tensor
            Batch of base distribution samples.

        Returns
        -------
        float
            Evaluated loss value.
        """
        self.opt.zero_grad()
        loss = self.compute_loss(z)
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(
        self,
        steps: int = 100,
        batch_size: int = 128,
        verbose: bool = False,
    ) -> List[float]:
        """
        Trains the Boltzmann generator for a specified number of optimization steps.

        Parameters
        ----------
        steps : int
            Number of gradient descent iterations.
        batch_size : int
            Number of base samples per batch.
        verbose : bool
            Whether to log loss progress.

        Returns
        -------
        List[float]
            Loss history across training steps.
        """
        losses = []
        for step in range(steps):
            if self.prior is not None:
                z = self.prior.sample(n_samples=batch_size)
            else:
                z = Tensor.randn(batch_size, self.dim)

            loss_val = self.train_step(z)
            losses.append(loss_val)

            if verbose and (step % 20 == 0 or step == steps - 1):
                print(f"[Step {step:4d}/{steps}] KL Loss: {loss_val:8.4f}")

        return losses

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Draws equilibrium configurations from the trained Boltzmann generator.

        Parameters
        ----------
        n_samples : int
            Number of configuration samples.

        Returns
        -------
        Tensor
            Generated particle coordinates.
        """
        if self.prior is not None:
            z = self.prior.sample(n_samples=n_samples)
            is_3d = len(z.shape) == 3
            z_flat = z.reshape(n_samples, self.dim) if is_3d else z
            x_flat, _ = self.flow.forward(z_flat)
            return x_flat.reshape(z.shape) if is_3d else x_flat
        else:
            z = Tensor.randn(n_samples, self.dim)
            x, _ = self.flow.forward(z)
            return x if n_samples > 1 else x.squeeze(0)

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Evaluates exact generated density log q_theta(x) = log p_z(f^{-1}(x)) + log |det J_{f^{-1}}(x)|.

        Parameters
        ----------
        x : Tensor
            Particle coordinates.

        Returns
        -------
        Tensor
            Log probabilities of samples.
        """
        is_3d = len(x.shape) == 3
        B = x.shape[0] if is_3d or len(x.shape) == 2 else 1
        x_flat = x.reshape(B, self.dim) if is_3d else (x if len(x.shape) == 2 else x.unsqueeze(0))

        z_flat, log_det_inv = self.flow.inverse(x_flat)
        z = z_flat.reshape(x.shape) if is_3d else z_flat

        if self.prior is not None:
            log_pz = self.prior.log_prob(z)
        else:
            log_pz = -0.5 * (z_flat * z_flat + math.log(2.0 * math.pi)).sum(axis=-1)

        log_qx = log_pz + log_det_inv
        return log_qx if (is_3d or len(x.shape) == 2) else log_qx.squeeze(0)
