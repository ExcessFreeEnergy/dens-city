"""
Boltzmann Generator: Deep Normalizing Flows trained on the exact Microscopic Hamiltonian
via variational Reverse Kullback-Leibler (KL) divergence minimization in pure tinygrad.
"""

import math
from typing import Callable, Optional, List, Union
from tinygrad import Tensor, TinyJit, nn, dtypes, GlobalCounters, Context
from tinygrad.helpers import getenv, trange

from dens_city.boltzmann.bijectors import RealNVPFlow, CompositeFlow
from dens_city.boltzmann.prior import CDFTBaseDistribution


class BoltzmannGenerator:
    """
    Learns the exact equilibrium Boltzmann distribution p(x) ~ exp(-beta * U(x))
    by training an invertible normalizing flow (RealNVPFlow or CompositeFlow)
    on the exact microscopic potential energy.
    """

    def __init__(
        self,
        flow: Union[RealNVPFlow, CompositeFlow],
        energy_fn: Callable[[Tensor], Tensor],
        prior: Optional[CDFTBaseDistribution] = None,
        temperature_k: float = 300.0,
        learning_rate: float = 0.01,
        batch_size: int = 128,
    ):
        self.flow = flow
        self.energy_fn = energy_fn
        self.prior = prior
        self.temperature_k = float(temperature_k)
        # In dens-city, energies are in Kelvin, so beta = 1 / T stored as realized device buffer
        self.beta = Tensor([1.0 / max(1e-6, self.temperature_k)], dtype=dtypes.float32).realize()
        self.log_2pi = Tensor([math.log(2.0 * math.pi)], dtype=dtypes.float32).realize()
        self.dim = flow.dim
        self.is_composite = isinstance(flow, CompositeFlow)
        self.batch_size = int(batch_size)
        pool_dim = 3 if self.is_composite else self.dim
        self.origin_pool = (
            self.prior.sample(n_samples=max(4096, self.batch_size * 16)).reshape(-1, pool_dim).realize()
            if self.prior is not None
            else None
        )

        # Realize all flow weights and biases on device
        for param in nn.state.get_parameters(self.flow):
            param.realize()

        # Optimizer over all flow parameters
        opt_type = nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam
        self.opt = opt_type(nn.state.get_parameters(self.flow), lr=learning_rate)
        self.train_step = TinyJit(self._train_step)

    def _forward_flow(self, z: Tensor, origin: Optional[Tensor] = None) -> Tensor:
        """
        Forward flow mapping latent noise z to 3D Cartesian coordinates.
        """
        if self.is_composite:
            x, _ = self.flow.forward(z, origin=origin)
            return x
        else:
            is_3d = len(z.shape) == 3
            B = z.shape[0]
            z_flat = z.reshape(B, self.dim) if is_3d else z
            x_flat, _ = self.flow.forward(z_flat)
            return x_flat.reshape(z.shape) if is_3d else x_flat

    def compute_loss(self, z: Tensor, origin: Optional[Tensor] = None) -> Tensor:
        r"""
        Evaluates the variational Reverse KL Divergence training loss:
        \mathcal{L}(\theta) = \mathbb{E}_{z \sim p_z} \left[ \beta U(f_\theta(z)) - \log p_z(z) - \log |\det J_{f_\theta}(z)| \right]
        """
        B = z.shape[0]

        if self.is_composite:
            z_flat = z.reshape(B, self.dim) if len(z.shape) > 2 else z
            x, log_det = self.flow.forward(z_flat, origin=origin)
            log_pz_internal = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            if self.prior is not None and origin is not None:
                log_pz_origin = self.prior.log_prob(origin)
                log_pz = log_pz_internal + log_pz_origin
            else:
                log_pz = log_pz_internal
        else:
            is_3d = len(z.shape) == 3
            z_flat = z.reshape(B, self.dim) if is_3d else z
            x_flat, log_det = self.flow.forward(z_flat)
            x = x_flat.reshape(z.shape) if is_3d else x_flat
            if self.prior is not None:
                log_pz = self.prior.log_prob(z)  # (B,)
            else:
                log_pz = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)

        # Evaluate exact microscopic potential energy
        u_exact = self.energy_fn(x)  # (B,)

        # Variational KL Loss with realized beta buffer
        loss_batch = self.beta * u_exact - log_pz - log_det
        return loss_batch.mean()

    def _train_step(self, origin_pool: Optional[Tensor] = None) -> Tensor:
        """
        Executes a single JIT-compiled gradient descent step on the flow parameters.
        Origin coordinates are randomly sampled from the preloaded device origin pool.
        """
        Tensor.training = True
        self.opt.zero_grad()
        if self.is_composite:
            z = Tensor.randn(self.batch_size, self.dim)
            if origin_pool is not None:
                idx = Tensor.randint(self.batch_size, high=origin_pool.shape[0])
                origin = origin_pool[idx].reshape(self.batch_size, 3)
            else:
                origin = None
            loss = self.compute_loss(z, origin=origin)
        else:
            if origin_pool is not None:
                idx = Tensor.randint(self.batch_size, high=origin_pool.shape[0])
                z = origin_pool[idx].reshape(self.batch_size, self.dim)
            else:
                z = Tensor.randn(self.batch_size, self.dim)
            loss = self.compute_loss(z)

        loss.backward()
        return loss.realize(*self.opt.schedule_step())

    def train(
        self,
        steps: int = 100,
        batch_size: int = 128,
        verbose: bool = False,
    ) -> List[float]:
        """
        Trains the Boltzmann generator for a specified number of optimization steps.
        """
        if self.train_step is None or self.batch_size != batch_size:
            self.batch_size = int(batch_size)
            if self.prior is not None and (self.origin_pool is None or self.origin_pool.shape[0] < self.batch_size):
                pool_dim = 3 if self.is_composite else self.dim
                self.origin_pool = self.prior.sample(n_samples=max(4096, self.batch_size * 16)).reshape(-1, pool_dim).realize()
            self.train_step = TinyJit(self._train_step)

        losses = []
        iterator = trange(steps) if verbose else range(steps)
        for i in iterator:
            GlobalCounters.reset()
            loss = self.train_step(self.origin_pool) if self.origin_pool is not None else self.train_step()
            loss_val = loss.item()
            losses.append(loss_val)

            if verbose and hasattr(iterator, "set_description") and (i % 20 == 0 or i == steps - 1):
                iterator.set_description(f"KL Loss: {loss_val:8.4f}")

        return losses

    def _sample_batch(self, n_samples: int) -> Tensor:
        """
        Generates equilibrium configurations mapping latent noise to 3D coordinates.
        """
        if self.is_composite:
            z = Tensor.randn(n_samples, self.dim)
            origin = (
                self.prior.sample(n_samples=n_samples).reshape(n_samples, 3)
                if self.prior is not None
                else None
            )
            return self._forward_flow(z, origin=origin)
        else:
            if self.prior is not None:
                z = self.prior.sample(n_samples=n_samples)
            else:
                z = Tensor.randn(n_samples, self.dim)
            return self._forward_flow(z)

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Draws equilibrium configurations from the trained Boltzmann generator.
        """
        Tensor.training = False
        out = self._sample_batch(n_samples)
        return (out if n_samples > 1 else out.squeeze(0)).realize()

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Evaluates exact generated density log q_theta(x) = log p_z(f^{-1}(x)) + log |det J_{f^{-1}}(x)|.
        """
        if self.is_composite:
            is_batched = len(x.shape) == 3
            x_b = x if is_batched else x.unsqueeze(0)
            z, log_det_inv = self.flow.inverse(x_b)
            if self.prior is not None:
                log_pz = self.prior.log_prob(z)
            else:
                log_pz = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(axis=-1)
            res = log_pz + log_det_inv
            return (res if is_batched else res.squeeze(0)).realize()
        else:
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
            res = log_qx if (is_3d or len(x.shape) == 2) else log_qx.squeeze(0)
            return res.realize()

