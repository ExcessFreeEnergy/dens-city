"""
Batched Quasi-Newton (L-BFGS) GPU Molecular Geometry Relaxation Solver for dens-city.

Implements vectorized L-BFGS minimization with two-loop recursion, rolling history buffers,
reverse-mode automatic differentiation in tinygrad for exact Cartesian force evaluation,
batched Armijo backtracking line search, and per-molecule SIMD convergence masking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes


@dataclass
class LBFGSResult:
    """Results container for batched L-BFGS geometry optimization."""

    x_relaxed: np.ndarray  # (B, N, 3) float32
    final_energies: np.ndarray  # (B,) float32
    initial_energies: np.ndarray  # (B,) float32
    iterations_taken: int
    converged: np.ndarray  # (B,) bool
    rms_forces: np.ndarray  # (B,) float32
    energy_history: List[np.ndarray]  # list of (B,) arrays


class BatchedLBFGS:
    """
    Batched L-BFGS Quasi-Newton Geometry Optimizer.

    Minimizes the potential energy surface U(X) for a batch of molecules (B, N, 3)
    on the GPU using reverse-mode autograd forces and O(m * 3N) two-loop recursion.
    """

    def __init__(
        self,
        m: int = 6,
        max_iter: int = 75,
        grad_tol: float = 1e-3,
        lr: float = 1.0,
        c1: float = 1e-4,
        backtrack_factor: float = 0.5,
        max_line_search_steps: int = 8,
        verbose: bool = False,
        use_jit: bool = True,
    ):
        """
        Args:
            m: History size for L-BFGS displacement and gradient difference vectors.
            max_iter: Maximum optimization iterations.
            grad_tol: RMS force convergence threshold per active atom (in units of K/Å or reduced).
            lr: Base initial step size.
            c1: Armijo condition sufficient decrease parameter.
            backtrack_factor: Step reduction multiplier during line search.
            max_line_search_steps: Maximum backtracking evaluations per iteration.
            verbose: If True, prints iteration progress.
            use_jit: If True, compiles energy and analytical force evaluation into a TinyJit execution graph.
        """
        self.m = max(1, int(m))
        self.max_iter = max(1, int(max_iter))
        self.grad_tol = float(grad_tol)
        self.lr = float(lr)
        self.c1 = float(c1)
        self.backtrack_factor = float(backtrack_factor)
        self.max_line_search_steps = int(max_line_search_steps)
        self.verbose = bool(verbose)
        self.use_jit = bool(use_jit)

    def _eval_energy_and_grad(
        self,
        energy_fn: Callable[[Tensor], Tensor],
        x_flat: np.ndarray,
        atom_mask_3d: np.ndarray,
        shape_3d: Tuple[int, int, int],
        jit_eval_fn: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
        x_buf: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates potential energy U(X) and analytical autograd Cartesian forces g = grad(U).

        Args:
            energy_fn: MicroscopicEnergy callable accepting Tensor of shape (B, N, 3).
            x_flat: NumPy array of shape (B, 3N).
            atom_mask_3d: NumPy array of shape (B, N, 3).
            shape_3d: Tuple (B, N, 3).
            jit_eval_fn: Optional pre-compiled TinyJit evaluator mapping Tensor(x_3d) -> (u, grad).
            x_buf: Optional pre-allocated persistent static device buffer for in-place .assign().

        Returns:
            Tuple of (energies (B,), gradients (B, 3N)).
        """
        B, N, _ = shape_3d
        x_3d = x_flat.reshape(B, N, 3)

        if jit_eval_fn is not None and x_buf is not None:
            # Explicitly drop previous gradient reference prior to .assign() and backward pass
            x_buf.grad = None
            x_buf.assign(Tensor(x_3d)).realize()
            x_buf.requires_grad = True
            u_b, g_tensor = jit_eval_fn(x_buf)
            energies = u_b.numpy().astype(np.float32)
            grad_3d = g_tensor.numpy() if g_tensor is not None else np.zeros_like(x_3d)
            x_buf.grad = None
        elif jit_eval_fn is not None:
            x_tensor = Tensor(x_3d)
            x_tensor.requires_grad = True
            u_b, g_tensor = jit_eval_fn(x_tensor)
            energies = u_b.numpy().astype(np.float32)
            grad_3d = g_tensor.numpy() if g_tensor is not None else np.zeros_like(x_3d)
            x_tensor.grad = None
        else:
            x_tensor = Tensor(x_3d)
            x_tensor.requires_grad = True
            u_b = energy_fn(x_tensor)  # (B,)
            u_sum = u_b.sum()
            u_sum.backward()
            energies = u_b.numpy().astype(np.float32)
            grad_3d = x_tensor.grad.numpy() if x_tensor.grad is not None else np.zeros_like(x_3d)
            x_tensor.grad = None

        # Strictly mask out forces on dummy/padding atoms
        grad_3d = grad_3d * atom_mask_3d
        # Replace any potential NaNs/Infs with zero to preserve optimizer robustness
        grad_3d = np.nan_to_num(grad_3d, nan=0.0, posinf=0.0, neginf=0.0)

        grad_flat = grad_3d.reshape(B, -1).astype(np.float32)
        return energies, grad_flat

    def minimize(
        self,
        energy_fn: Callable[[Tensor], Tensor],
        x_init: Union[np.ndarray, Tensor],
        atom_mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> LBFGSResult:
        """
        Executes batched L-BFGS geometry optimization on 3D coordinates.

        Args:
            energy_fn: MicroscopicEnergy callable accepting (B, N, 3) Tensor.
            x_init: Initial 3D Cartesian coordinates (B, N, 3) or (N, 3).
            atom_mask: Atom presence mask (B, N) or (N,); 1.0 for real atoms, 0.0 for dummy.

        Returns:
            LBFGSResult containing relaxed coordinates, energies, and convergence info.
        """
        # 1. Normalize input tensors and dimensions
        if isinstance(x_init, Tensor):
            x_arr = x_init.numpy().copy().astype(np.float32)
        else:
            x_arr = np.asarray(x_init, dtype=np.float32).copy()

        if len(x_arr.shape) == 2:
            x_arr = x_arr[np.newaxis, ...]  # (1, N, 3)

        B, N, D = x_arr.shape
        dim = N * D

        if atom_mask is not None:
            if isinstance(atom_mask, Tensor):
                mask_2d = atom_mask.numpy().astype(np.float32)
            else:
                mask_2d = np.asarray(atom_mask, dtype=np.float32)
            if len(mask_2d.shape) == 1:
                mask_2d = mask_2d[np.newaxis, ...]
        else:
            mask_2d = np.ones((B, N), dtype=np.float32)

        atom_mask_3d = mask_2d[:, :, np.newaxis].repeat(3, axis=-1)  # (B, N, 3)
        atom_mask_flat = atom_mask_3d.reshape(B, dim)
        n_real_per_mol = np.maximum(1.0, mask_2d.sum(axis=-1, keepdims=True))  # (B, 1)

        # 2. Setup persistent static device buffers for TinyJit execution
        x_buf = None
        x_trial_buf = None
        jit_eval_grad = None
        jit_eval_trial = None
        if self.use_jit:
            x_buf = Tensor.zeros(B, N, D, dtype=dtypes.float32).realize()
            x_trial_buf = Tensor.zeros(B, N, D, dtype=dtypes.float32).realize()

            def _step_grad(xt: Tensor) -> Tuple[Tensor, Tensor]:
                u = energy_fn(xt)
                u.sum().backward()
                g = xt.grad if xt.grad is not None else Tensor.zeros_like(xt)
                return u.realize(), g.realize()

            jit_eval_grad = TinyJit(_step_grad)

            def _step_trial(xt: Tensor) -> Tensor:
                u = energy_fn(xt)
                return u.realize()

            jit_eval_trial = TinyJit(_step_trial)

        x_curr = x_arr.reshape(B, dim)
        u_curr, g_curr = self._eval_energy_and_grad(
            energy_fn, x_curr, atom_mask_3d, (B, N, D), jit_eval_fn=jit_eval_grad, x_buf=x_buf
        )
        u_init = u_curr.copy()

        energy_history = [u_curr.copy()]

        # Rolling history buffers for two-loop recursion: deque of length m
        s_history: deque[np.ndarray] = deque(maxlen=self.m)
        y_history: deque[np.ndarray] = deque(maxlen=self.m)
        rho_history: deque[np.ndarray] = deque(maxlen=self.m)

        converged = np.zeros(B, dtype=bool)
        rms_forces = np.zeros(B, dtype=np.float32)

        # 3. Main Quasi-Newton Optimization Loop
        total_steps_taken = 0
        for step in range(self.max_iter):
            total_steps_taken = step + 1

            # Compute RMS force per active atom: sqrt(sum(g_i^2) / N_real)
            g_sq_sum = (g_curr**2).sum(axis=-1, keepdims=True)  # (B, 1)
            rms_f = np.sqrt(g_sq_sum / n_real_per_mol).reshape(B)  # (B,)
            rms_forces = rms_f.copy()

            # Update per-molecule convergence mask
            converged = rms_f < self.grad_tol
            if np.all(converged):
                if self.verbose:
                    print(
                        f"  [L-BFGS] All {B} molecules converged at step {step + 1} (Max RMS Force: {rms_f.max():.2e})"
                    )
                break

            active_mask = (~converged).astype(np.float32)[:, np.newaxis]  # (B, 1)

            # 4. Two-Loop L-BFGS Recursion to construct descent direction p_k = -H_k * g_k
            q = g_curr.copy()
            alphas: List[np.ndarray] = []

            # Backward loop over history
            for s_i, y_i, rho_i in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
                # alpha_i = rho_i * (s_i^T * q)
                alpha_i = rho_i * (s_i * q).sum(axis=-1, keepdims=True)  # (B, 1)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i

            # Initial Hessian scaling: gamma_k = (s_{k-1}^T * y_{k-1}) / (y_{k-1}^T * y_{k-1})
            if len(s_history) > 0:
                s_last = s_history[-1]
                y_last = y_history[-1]
                sy = (s_last * y_last).sum(axis=-1, keepdims=True)
                yy = (y_last * y_last).sum(axis=-1, keepdims=True)
                gamma = np.where(yy > 1e-12, sy / (yy + 1e-12), 1.0)
                # Keep gamma positive and well-conditioned
                gamma = np.clip(gamma, 1e-4, 1e4)
            else:
                gamma = np.ones((B, 1), dtype=np.float32)

            r = gamma * q

            # Forward loop over history
            for s_i, y_i, rho_i, alpha_i in zip(s_history, y_history, rho_history, reversed(alphas)):
                # beta_i = rho_i * (y_i^T * r)
                beta_i = rho_i * (y_i * r).sum(axis=-1, keepdims=True)
                r = r + s_i * (alpha_i - beta_i)

            descent_dir = -r * atom_mask_flat

            # Verify descent property: g_k^T * d_k must be strictly negative
            g_dot_d = (g_curr * descent_dir).sum(axis=-1, keepdims=True)  # (B, 1)
            bad_dir = (g_dot_d >= 0.0) | np.isnan(g_dot_d)
            if np.any(bad_dir):
                # Fallback to steepest descent for non-descent batch items
                descent_dir = np.where(bad_dir, -g_curr * atom_mask_flat, descent_dir)
                g_dot_d = (g_curr * descent_dir).sum(axis=-1, keepdims=True)

            # Trust-region clamp: limit maximum single-atom displacement to 0.2 Å
            max_disp_per_mol = np.linalg.norm(descent_dir.reshape(B, N, D), axis=-1).max(
                axis=-1, keepdims=True
            )  # (B, 1)
            disp_scale = np.minimum(1.0, 0.20 / np.maximum(1e-6, max_disp_per_mol))
            descent_dir = descent_dir * disp_scale
            g_dot_d = (g_curr * descent_dir).sum(axis=-1, keepdims=True)

            # 5. Batched Armijo Backtracking Line Search
            step_sizes = np.full((B, 1), self.lr, dtype=np.float32)
            accepted = np.zeros(B, dtype=bool)
            best_u_trial = u_curr.copy()
            best_x_trial = x_curr.copy()

            for _ in range(self.max_line_search_steps):
                x_trial = x_curr + (step_sizes * descent_dir * active_mask)
                # Evaluate trial energy (without autograd graph)
                x_trial_3d = x_trial.reshape(B, N, D)
                if jit_eval_trial is not None and x_trial_buf is not None:
                    x_trial_buf.assign(Tensor(x_trial_3d)).realize()
                    u_trial = jit_eval_trial(x_trial_buf).numpy().astype(np.float32)
                elif jit_eval_trial is not None:
                    u_trial = jit_eval_trial(Tensor(x_trial_3d)).numpy().astype(np.float32)
                else:
                    u_trial = energy_fn(Tensor(x_trial_3d)).numpy().astype(np.float32)

                # Sufficient decrease: U(x + alpha * d) <= U(x) + c1 * alpha * (g^T * d)
                target_u = u_curr + (self.c1 * step_sizes.reshape(B) * g_dot_d.reshape(B))
                sufficient_decrease = (u_trial <= target_u) | converged

                # Record best decreasing state
                improved = (u_trial < best_u_trial) & (~converged)
                if np.any(improved):
                    best_u_trial = np.where(improved, u_trial, best_u_trial)
                    best_x_trial = np.where(improved[:, np.newaxis], x_trial, best_x_trial)

                accepted = accepted | sufficient_decrease

                if np.all(accepted):
                    best_x_trial = x_trial
                    break

                # Backtrack step size for non-accepted active molecules
                backtrack_mask = (~accepted)[:, np.newaxis]
                step_sizes = np.where(backtrack_mask, step_sizes * self.backtrack_factor, step_sizes)

            # Use best discovered decreasing position (guaranteeing monotonic non-increase)
            x_next = np.where(active_mask, best_x_trial, x_curr)

            # Re-evaluate new energy and gradient at accepted position
            u_next, g_next = self._eval_energy_and_grad(
                energy_fn, x_next, atom_mask_3d, (B, N, D), jit_eval_fn=jit_eval_grad, x_buf=x_buf
            )

            # 6. Update rolling history buffers for active molecules
            s_k = (x_next - x_curr) * atom_mask_flat
            y_k = (g_next - g_curr) * atom_mask_flat
            sy_k = (s_k * y_k).sum(axis=-1, keepdims=True)  # (B, 1)

            # Only store curvature pairs if positive curvature is satisfied: s^T * y > 1e-8
            valid_curv = (sy_k > 1e-8) & active_mask.astype(bool)
            rho_k = np.where(valid_curv, 1.0 / (sy_k + 1e-12), 0.0).astype(np.float32)

            s_history.append(s_k)
            y_history.append(y_k)
            rho_history.append(rho_k)

            # Advance state
            x_curr = x_next
            u_curr = u_next
            g_curr = g_next
            energy_history.append(u_curr.copy())

            if self.verbose and (step % 10 == 0 or step == self.max_iter - 1):
                print(
                    f"  [L-BFGS Step {step + 1:03d}/{self.max_iter}] "
                    f"Mean U: {u_curr.mean():.2f} K | Min U: {u_curr.min():.2f} K | "
                    f"Max RMS Force: {rms_forces.max():.2e} | Converged: {converged.sum()}/{B}"
                )

        x_relaxed_3d = x_curr.reshape(B, N, D)
        return LBFGSResult(
            x_relaxed=x_relaxed_3d,
            final_energies=u_curr,
            initial_energies=u_init,
            iterations_taken=total_steps_taken,
            converged=converged,
            rms_forces=rms_forces,
            energy_history=energy_history,
        )
