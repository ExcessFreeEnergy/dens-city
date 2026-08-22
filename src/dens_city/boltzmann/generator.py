"""
Boltzmann Generator: Deep Normalizing Flows trained on the exact Microscopic Hamiltonian
via variational Reverse Kullback-Leibler (KL) divergence minimization in pure tinygrad.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

from tinygrad import GlobalCounters, Tensor, TinyJit, dtypes, nn
from tinygrad.helpers import getenv, trange

from dens_city.boltzmann.bijectors import (
    Base2CartesianFlow,
    CompositeFlow,
    RealNVPFlow,
    compute_cartesian_torsion_loss,
    compute_torsion_rotamer_loss,
)
from dens_city.boltzmann.prior import CDFTBaseDistribution


class BoltzmannGenerator:
    """
    Learns the exact equilibrium Boltzmann distribution p(x) ~ exp(-beta * U(x))
    by training an invertible normalizing flow (Base2CartesianFlow, RealNVPFlow, or CompositeFlow)
    on the exact microscopic potential energy.
    """

    def __init__(
        self,
        flow: Union[Base2CartesianFlow, RealNVPFlow, CompositeFlow],
        energy_fn: Callable[[Tensor], Tensor],
        prior: Optional[CDFTBaseDistribution] = None,
        temperature_k: float = 300.0,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        w_torsion: float = 0.0,
        dihedral_quadruplets: Optional[Union[List[Tuple[int, int, int, int]], Tensor]] = None,
    ):
        self.flow = flow
        self.energy_fn = energy_fn
        self.prior = prior
        self.temperature_k = float(temperature_k)
        # In dens-city, energies are in Kelvin, so beta = 1 / T stored as realized device buffer
        self.beta = Tensor([1.0 / max(1e-6, self.temperature_k)], dtype=dtypes.float32).realize()
        self.log_2pi = Tensor([math.log(2.0 * math.pi)], dtype=dtypes.float32).realize()
        self.dim = flow.dim
        self.is_base2_cartesian = isinstance(flow, Base2CartesianFlow)
        self.is_composite = isinstance(flow, CompositeFlow)
        self.batch_size = int(batch_size)
        self.w_torsion = float(w_torsion)
        self.w_tor = Tensor([self.w_torsion], dtype=dtypes.float32).realize()

        if dihedral_quadruplets is not None and len(dihedral_quadruplets) > 0:
            if isinstance(dihedral_quadruplets, Tensor):
                self.dihedral_quadruplets = dihedral_quadruplets.realize()
            else:
                self.dihedral_quadruplets = Tensor(dihedral_quadruplets, dtype=dtypes.int32).realize()
        else:
            mat = getattr(self.energy_fn, "material", None)
            if mat is not None and getattr(mat, "dihedral_quadruplets", None):
                self.dihedral_quadruplets = Tensor(mat.dihedral_quadruplets, dtype=dtypes.int32).realize()
            else:
                self.dihedral_quadruplets = None

        if self.prior is not None:
            raw_pool = self.prior.sample(n_samples=max(4096, self.batch_size * 16))
            if self.is_base2_cartesian or self.is_composite:
                self.origin_pool = (
                    raw_pool[:, 0, :].reshape(-1, 3).realize()
                    if len(raw_pool.shape) == 3
                    else raw_pool.reshape(-1, 3).realize()
                )
            else:
                self.origin_pool = raw_pool.reshape(-1, self.dim).realize()
        else:
            self.origin_pool = None

        # Realize all flow weights and biases on device
        for param in nn.state.get_parameters(self.flow):
            param.realize()

        # Optimizer over all flow parameters
        opt_type = nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam
        self.opt = opt_type(nn.state.get_parameters(self.flow), lr=learning_rate)
        self.train_step = TinyJit(self._train_step)

    def _forward_flow(self, z: Tensor, origin: Optional[Tensor] = None) -> Tensor:
        """
        Forward flow mapping latent noise z to 3D Cartesian coordinates (B, N, 3) or flat coordinates.
        """
        if self.is_base2_cartesian:
            x, _ = self.flow.forward(z, origin=origin)
            return x
        elif self.is_composite:
            x, _ = self.flow.forward(z, origin=origin)
            return x
        else:
            B = z.shape[0]
            z_flat = z.reshape(B, self.dim)
            x_flat, _ = self.flow.forward(z_flat)
            if self.dim % 3 == 0:
                return x_flat.reshape(B, self.dim // 3, 3)
            return x_flat

    def compute_loss(self, z: Tensor, origin: Optional[Tensor] = None) -> Tensor:
        r"""
        Evaluates the variational Reverse KL Divergence training loss:
        \mathcal{L}(\theta) = \mathbb{E}_{z \sim p_z} \left[ \beta U(f_\theta(z)) - \log p_z(z) - \log |\det J_{f_\theta}(z)| + w_{\rm tor} J_{\rm tor} \right]
        """
        B = z.shape[0]
        j_tor = None

        if self.is_base2_cartesian:
            z_flat = z.reshape(B, self.dim)
            x, log_det = self.flow.forward(z_flat, origin=origin)
            log_pz_internal = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            if self.prior is not None and origin is not None:
                log_pz_origin = self.prior.log_prob(origin)
                log_pz = log_pz_internal + log_pz_origin
            else:
                log_pz = log_pz_internal

            if (
                self.w_torsion > 0.0
                and self.dihedral_quadruplets is not None
                and self.dihedral_quadruplets.shape[0] > 0
            ):
                j_tor = compute_cartesian_torsion_loss(x, self.dihedral_quadruplets)

        elif self.is_composite:
            z_flat = z.reshape(B, self.dim) if len(z.shape) > 2 else z
            ic_flat, log_det_flow = self.flow.flow.forward(z_flat)
            n_bonds = self.flow.n_bonds
            n_angles = self.flow.n_angles
            n_torsions = self.flow.n_torsions
            bonds = ic_flat[:, :n_bonds]
            angles = ic_flat[:, n_bonds : (n_bonds + n_angles)] if n_angles > 0 else None
            torsions = ic_flat[:, (n_bonds + n_angles) : self.flow.dim] if n_torsions > 0 else None
            x, log_det_zmat = self.flow.zmat.forward(bonds=bonds, angles=angles, torsions=torsions, origin=origin)
            log_det = log_det_flow + log_det_zmat

            log_pz_internal = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            if self.prior is not None and origin is not None:
                log_pz_origin = self.prior.log_prob(origin)
                log_pz = log_pz_internal + log_pz_origin
            else:
                log_pz = log_pz_internal

            if self.w_torsion > 0.0 and torsions is not None and torsions.shape[-1] > 0:
                j_tor = compute_torsion_rotamer_loss(torsions)

        else:
            z_flat = z.reshape(B, self.dim)
            x_flat, log_det = self.flow.forward(z_flat)
            if self.dim % 3 == 0:
                n_particles = self.dim // 3
                x = x_flat.reshape(B, n_particles, 3)
                if self.prior is not None:
                    z_3d = z.reshape(B, n_particles, 3)
                    log_pz = self.prior.log_prob(z_3d)  # (B,)
                else:
                    log_pz = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            else:
                x = x_flat
                if self.prior is not None:
                    log_pz = self.prior.log_prob(z)
                else:
                    log_pz = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)

            if (
                self.w_torsion > 0.0
                and self.dihedral_quadruplets is not None
                and self.dihedral_quadruplets.shape[0] > 0
                and len(x.shape) == 3
            ):
                j_tor = compute_cartesian_torsion_loss(x, self.dihedral_quadruplets)

        # Evaluate exact microscopic potential energy
        u_exact = self.energy_fn(x)  # (B,)

        # Variational KL Loss with realized beta buffer and optional torsional penalty
        loss_batch = self.beta * u_exact - log_pz - log_det
        if j_tor is not None:
            loss_batch = loss_batch + self.w_tor * j_tor
        return loss_batch.mean()

    def _train_step(self, origin_pool: Optional[Tensor] = None) -> Tensor:
        """
        Executes a single JIT-compiled gradient descent step on the flow parameters.
        Origin coordinates are randomly sampled from the preloaded device origin pool.
        """
        Tensor.training = True
        self.opt.zero_grad()
        if self.is_base2_cartesian:
            z = Tensor.randn(self.batch_size, self.dim)
            if origin_pool is not None:
                idx = Tensor.randint(self.batch_size, high=origin_pool.shape[0])
                origin = origin_pool[idx].reshape(self.batch_size, 3)
            else:
                origin = None
            loss = self.compute_loss(z, origin=origin)
        elif self.is_composite:
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
        batch_size: int = 64,
        verbose: bool = False,
    ) -> List[float]:
        """
        Trains the Boltzmann generator for a specified number of optimization steps.
        """
        if self.train_step is None or self.batch_size != batch_size:
            self.batch_size = int(batch_size)
            if self.prior is not None and (self.origin_pool is None or self.origin_pool.shape[0] < self.batch_size):
                raw_pool = self.prior.sample(n_samples=max(4096, self.batch_size * 16))
                if self.is_base2_cartesian or self.is_composite:
                    self.origin_pool = (
                        raw_pool[:, 0, :].reshape(-1, 3).realize()
                        if len(raw_pool.shape) == 3
                        else raw_pool.reshape(-1, 3).realize()
                    )
                else:
                    self.origin_pool = raw_pool.reshape(-1, self.dim).realize()
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
        if self.is_base2_cartesian:
            z = Tensor.randn(n_samples, self.dim)
            if self.prior is not None:
                p_samp = self.prior.sample(n_samples=n_samples)
                origin = (
                    p_samp[:, 0, :].reshape(n_samples, 3) if len(p_samp.shape) == 3 else p_samp.reshape(n_samples, 3)
                )
            else:
                origin = None
            return self._forward_flow(z, origin=origin)
        elif self.is_composite:
            z = Tensor.randn(n_samples, self.dim)
            if self.prior is not None:
                p_samp = self.prior.sample(n_samples=n_samples)
                origin = (
                    p_samp[:, 0, :].reshape(n_samples, 3) if len(p_samp.shape) == 3 else p_samp.reshape(n_samples, 3)
                )
            else:
                origin = None
            return self._forward_flow(z, origin=origin)
        else:
            if self.prior is not None:
                z = self.prior.sample(n_samples=n_samples).reshape(n_samples, self.dim)
            else:
                z = Tensor.randn(n_samples, self.dim)
            return self._forward_flow(z)

    def sample_relaxed(
        self,
        n_samples: int = 1,
        mcmc_steps: int = 5,
        mcmc_step_size: float = 0.1,
        origin: Optional[Tensor] = None,
        return_all_pad: bool = False,
        return_stats: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """
        Relaxes generated configurations via Vectorized Latent Space Metropolis Monte Carlo.
        Draws initial latent z ~ N(0, I) and performs K Metropolis moves:
            z' = z + s * N(0, I)
            p_acc = min(1, exp(-Delta E)) = exp(min(0, -Delta E))
        where Delta E = E_eff(z') - E_eff(z) and E_eff(z) = beta * U(f(z)) - log p_z(z) - log |det J(z)|.
        Protects against inf overflow on large downhill steps using (-delta_e).minimum(0.0).exp().
        Operates without gradient computation or graph freezing.
        """
        Tensor.training = False
        B = int(n_samples)
        step_size = float(mcmc_step_size)

        # 1. Initialize latent points and origins
        if self.is_base2_cartesian or self.is_composite:
            z_curr = Tensor.randn(B, self.dim)
            if origin is not None:
                orig_curr = origin.reshape(B, 3)
            elif self.prior is not None:
                p_samp = self.prior.sample(n_samples=B)
                orig_curr = p_samp[:, 0, :].reshape(B, 3) if len(p_samp.shape) == 3 else p_samp.reshape(B, 3)
            else:
                orig_curr = None
        else:
            if self.prior is not None:
                z_curr = self.prior.sample(n_samples=B).reshape(B, self.dim)
            else:
                z_curr = Tensor.randn(B, self.dim)
            orig_curr = None

        # 2. Initial state forward flow and energy evaluation
        if self.is_base2_cartesian:
            x_curr, log_det_curr = self.flow.forward(z_curr, origin=orig_curr)
            log_pz_curr = -0.5 * (z_curr * z_curr + self.log_2pi).sum(axis=-1)
            if self.prior is not None and orig_curr is not None:
                log_pz_curr = log_pz_curr + self.prior.log_prob(orig_curr)
        elif self.is_composite:
            z_flat = z_curr.reshape(B, self.dim) if len(z_curr.shape) > 2 else z_curr
            ic_flat, log_det_flow = self.flow.flow.forward(z_flat)
            n_bonds = self.flow.n_bonds
            n_angles = self.flow.n_angles
            n_torsions = self.flow.n_torsions
            bonds = ic_flat[:, :n_bonds]
            angles = ic_flat[:, n_bonds : (n_bonds + n_angles)] if n_angles > 0 else None
            torsions = ic_flat[:, (n_bonds + n_angles) : self.flow.dim] if n_torsions > 0 else None
            x_curr, log_det_zmat = self.flow.zmat.forward(
                bonds=bonds, angles=angles, torsions=torsions, origin=orig_curr
            )
            log_det_curr = log_det_flow + log_det_zmat
            log_pz_curr = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            if self.prior is not None and orig_curr is not None:
                log_pz_curr = log_pz_curr + self.prior.log_prob(orig_curr)
        else:
            z_flat = z_curr.reshape(B, self.dim)
            x_flat, log_det_curr = self.flow.forward(z_flat)
            if self.dim % 3 == 0:
                n_particles = self.dim // 3
                x_curr = x_flat.reshape(B, n_particles, 3)
                if self.prior is not None:
                    log_pz_curr = self.prior.log_prob(z_curr.reshape(B, n_particles, 3))
                else:
                    log_pz_curr = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            else:
                x_curr = x_flat
                if self.prior is not None:
                    log_pz_curr = self.prior.log_prob(z_curr)
                else:
                    log_pz_curr = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)

        u_curr = self.energy_fn(x_curr)
        e_curr = self.beta * u_curr - log_pz_curr - log_det_curr

        initial_u_mean = float(u_curr.mean().item())
        accepted_count = 0.0

        # 3. Vectorized Latent Space Metropolis Monte Carlo loop (unjitted for dynamic RNG)
        for _ in range(mcmc_steps):
            eta = Tensor.randn(B, self.dim)
            z_prop = z_curr + step_size * eta

            if self.is_base2_cartesian:
                x_prop, log_det_prop = self.flow.forward(z_prop, origin=orig_curr)
                log_pz_prop = -0.5 * (z_prop * z_prop + self.log_2pi).sum(axis=-1)
                if self.prior is not None and orig_curr is not None:
                    log_pz_prop = log_pz_prop + self.prior.log_prob(orig_curr)
            elif self.is_composite:
                z_flat_p = z_prop.reshape(B, self.dim) if len(z_prop.shape) > 2 else z_prop
                ic_flat_p, log_det_flow_p = self.flow.flow.forward(z_flat_p)
                n_bonds = self.flow.n_bonds
                n_angles = self.flow.n_angles
                n_torsions = self.flow.n_torsions
                bonds_p = ic_flat_p[:, :n_bonds]
                angles_p = ic_flat_p[:, n_bonds : (n_bonds + n_angles)] if n_angles > 0 else None
                torsions_p = ic_flat_p[:, (n_bonds + n_angles) : self.flow.dim] if n_torsions > 0 else None
                x_prop, log_det_zmat_p = self.flow.zmat.forward(
                    bonds=bonds_p, angles=angles_p, torsions=torsions_p, origin=orig_curr
                )
                log_det_prop = log_det_flow_p + log_det_zmat_p
                log_pz_prop = -0.5 * (z_flat_p * z_flat_p + self.log_2pi).sum(axis=-1)
                if self.prior is not None and orig_curr is not None:
                    log_pz_prop = log_pz_prop + self.prior.log_prob(orig_curr)
            else:
                z_flat_p = z_prop.reshape(B, self.dim)
                x_flat_p, log_det_prop = self.flow.forward(z_flat_p)
                if self.dim % 3 == 0:
                    n_particles = self.dim // 3
                    x_prop = x_flat_p.reshape(B, n_particles, 3)
                    if self.prior is not None:
                        log_pz_prop = self.prior.log_prob(z_prop.reshape(B, n_particles, 3))
                    else:
                        log_pz_prop = -0.5 * (z_flat_p * z_flat_p + self.log_2pi).sum(axis=-1)
                else:
                    x_prop = x_flat_p
                    if self.prior is not None:
                        log_pz_prop = self.prior.log_prob(z_prop)
                    else:
                        log_pz_prop = -0.5 * (z_flat_p * z_flat_p + self.log_2pi).sum(axis=-1)

            u_prop = self.energy_fn(x_prop)
            e_prop = self.beta * u_prop - log_pz_prop - log_det_prop

            # Vectorized Metropolis Acceptance Criterion with IEEE 754 overflow capping
            delta_e = e_prop - e_curr
            accept_prob = (-delta_e).minimum(0.0).exp()
            rand_u = Tensor.rand(B)
            accept_mask = rand_u < accept_prob

            accepted_count += float(accept_mask.float().mean().item())

            # Batch updates via where()
            mask_z = accept_mask.reshape(B, *([1] * (len(z_curr.shape) - 1)))
            z_curr = mask_z.where(z_prop, z_curr).realize()

            mask_x = accept_mask.reshape(B, *([1] * (len(x_curr.shape) - 1)))
            x_curr = mask_x.where(x_prop, x_curr).realize()

            e_curr = accept_mask.where(e_prop, e_curr).realize()
            u_curr = accept_mask.where(u_prop, u_curr).realize()

        final_u_mean = float(u_curr.mean().item())
        acceptance_rate = (accepted_count / max(1, mcmc_steps)) if mcmc_steps > 0 else 1.0

        out = x_curr
        n_real = getattr(self.energy_fn, "n_real_particles", None)
        n_pad = getattr(self.energy_fn, "n_particles", None)
        if (
            not return_all_pad
            and n_real is not None
            and n_pad is not None
            and n_pad > n_real
            and len(out.shape) == 3
            and n_real < out.shape[1]
        ):
            out = out[:, :n_real, :]

        res = (out if n_samples > 1 else out.squeeze(0)).realize()
        if return_stats:
            stats = {
                "initial_energy_mean": initial_u_mean,
                "final_energy_mean": final_u_mean,
                "acceptance_rate": acceptance_rate,
                "mcmc_steps": mcmc_steps,
                "mcmc_step_size": mcmc_step_size,
            }
            return res, stats
        return res

    def sample(
        self,
        n_samples: int = 1,
        return_all_pad: bool = False,
        mcmc_steps: int = 0,
        mcmc_step_size: float = 0.1,
        origin: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Draws equilibrium configurations from the trained Boltzmann generator.
        Automatically slices padded dummy sites to return real molecular atoms.
        If mcmc_steps > 0, performs latent space Metropolis Monte Carlo relaxation.
        """
        if mcmc_steps > 0:
            return self.sample_relaxed(
                n_samples=n_samples,
                mcmc_steps=mcmc_steps,
                mcmc_step_size=mcmc_step_size,
                origin=origin,
                return_all_pad=return_all_pad,
                return_stats=False,
            )

        Tensor.training = False
        out = self._sample_batch(n_samples)
        n_real = getattr(self.energy_fn, "n_real_particles", None)
        n_pad = getattr(self.energy_fn, "n_particles", None)
        if (
            not return_all_pad
            and n_real is not None
            and n_pad is not None
            and n_pad > n_real
            and len(out.shape) == 3
            and n_real < out.shape[1]
        ):
            out = out[:, :n_real, :]
        return (out if n_samples > 1 else out.squeeze(0)).realize()

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Evaluates exact generated density log q_theta(x) = log p_z(f^{-1}(x)) + log |det J_{f^{-1}}(x)|.
        """
        if self.is_base2_cartesian:
            is_batched = len(x.shape) >= 2
            x_b = x if is_batched else x.unsqueeze(0)
            z, log_det_inv = self.flow.inverse(x_b)
            z_flat = z.reshape(z.shape[0], self.dim)
            log_pz = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)
            res = log_pz + log_det_inv
            return (res if is_batched else res.squeeze(0)).realize()
        elif self.is_composite:
            is_batched = len(x.shape) == 3
            x_b = x if is_batched else x.unsqueeze(0)
            z, log_det_inv = self.flow.inverse(x_b)
            if self.prior is not None:
                log_pz = self.prior.log_prob(z)
            else:
                log_pz = -0.5 * (z * z + self.log_2pi).sum(axis=-1)
            res = log_pz + log_det_inv
            return (res if is_batched else res.squeeze(0)).realize()
        else:
            is_batched = len(x.shape) == 3 or (len(x.shape) == 2 and self.dim % 3 != 0)
            x_b = x if is_batched else x.unsqueeze(0)
            B = x_b.shape[0]
            x_flat = x_b.reshape(B, self.dim)

            z_flat, log_det_inv = self.flow.inverse(x_flat)
            if self.dim % 3 == 0:
                N = self.dim // 3
                z = z_flat.reshape(B, N, 3)
            else:
                z = z_flat

            if self.prior is not None:
                log_pz = self.prior.log_prob(z)
            else:
                log_pz = -0.5 * (z_flat * z_flat + self.log_2pi).sum(axis=-1)

            log_qx = log_pz + log_det_inv
            return (log_qx if is_batched else log_qx.squeeze(0)).realize()
