"""
Equivariant Graph Neural Network (EGNN) Machine Learned Force Field (MLFF).
Predicts total potential energy U(x) with Density Functional Theory (DFT) level accuracy
and derives exact conservative forces F = -∇_x U with strict E(n) rotational and translational invariance.

Adheres strictly to tinygrad best practices and power-of-2 vector alignments (N=128, F=128, L=7).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from tinygrad import Tensor, TinyJit, dtypes, nn


class EGNNLayer:
    """
    Single E(n)-invariant message passing layer.
    Computes edge interaction messages from node embeddings, relative squared distances,
    and edge masks, then updates node embeddings with residual connections.
    Uses decomposed linear projections to avoid materializing huge (B, N, N, 2F+2) tensors in memory.
    """

    def __init__(self, hidden_dim: int = 128, edge_in_dim: Optional[int] = None):
        self.hidden_dim = hidden_dim

        # Decomposed first linear projection of edge features
        self.edge_hi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_hj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_d = nn.Linear(1, hidden_dim, bias=False)
        self.edge_a = nn.Linear(1, hidden_dim, bias=True)

        # Second linear layer of edge MLP
        self.edge_l2 = nn.Linear(hidden_dim, hidden_dim)

        # Node Update MLP: phi_h: R^(256) -> R^(128) -> R^(128) with Swish (SiLU)
        self.node_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, hidden_dim),
        ]

    def __call__(
        self,
        h: Tensor,
        d_sq: Tensor,
        edge_mask: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass for a single message-passing step.
        h: (B, N, F) node features
        d_sq: (B, N, N, 1) pairwise squared distances ||x_i - x_j||^2
        edge_mask: (B, N, N, 1) edge validity mask
        atom_mask: (B, N, 1) atom validity mask
        """
        B, N, F = h.shape

        # 1. Project node features directly on (B, N, F) before spatial broadcast
        h_i_proj = self.edge_hi(h).reshape(B, N, 1, F)
        h_j_proj = self.edge_hj(h).reshape(B, 1, N, F)
        d_proj = self.edge_d(d_sq)
        a_proj = self.edge_a(edge_mask)

        # 2. Sum linear projections and apply SiLU
        e_hidden = (h_i_proj + h_j_proj + d_proj + a_proj).silu()

        # 3. Message generation via second linear layer & masking
        m_ij = self.edge_l2(e_hidden).silu() * edge_mask

        # 4. Message aggregation over neighbors
        m_i = m_ij.sum(axis=2)

        # 5. Node update with residual connection
        node_inputs = Tensor.cat(h, m_i, dim=-1)
        h_delta = node_inputs.sequential(self.node_mlp)
        h_next = (h + h_delta) * atom_mask

        return h_next


class EGNNForceField:
    """
    7-layer Equivariant Graph Neural Network (EGNN) Potential Energy Surface & Force Field.
    Ingests atomic numbers Z and Cartesian coordinates x, predicting scalar potential energy U(x).
    Derives analytical conservative forces via reverse-mode autograd: F = -∇_x U(x).
    """

    def __init__(
        self,
        num_layers: int = 7,
        hidden_dim: int = 128,
        max_atomic_number: int = 128,
        n_particles: int = 128,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_atomic_number = max_atomic_number
        self.n_particles = n_particles

        # Linear embedding layer for atomic numbers Z -> h^0 in R^(128)
        self.embedding = nn.Linear(max_atomic_number, hidden_dim)

        # 7 sequential message-passing layers
        self.layers = [EGNNLayer(hidden_dim=hidden_dim) for _ in range(num_layers)]

        # Readout MLP: node-wise energy contribution eps_i
        self.readout_mlp: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(hidden_dim, hidden_dim),
            Tensor.silu,
            nn.Linear(hidden_dim, 1),
        ]

    def _prepare_inputs(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Validates and broadcasts input dimensions to static (B, 128, ...) shapes."""
        # Ensure x is (B, N, 3)
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 3)
        B, N, _ = x.shape

        # Ensure atomic_numbers is (B, N)
        if len(atomic_numbers.shape) == 1:
            atomic_numbers = atomic_numbers.reshape(B, N)
        elif len(atomic_numbers.shape) == 3:
            atomic_numbers = atomic_numbers.reshape(B, N)

        # Default masks if not supplied
        if atom_mask is None:
            atom_mask = (atomic_numbers > 0).cast(dtypes.float32).reshape(B, N, 1)
        elif len(atom_mask.shape) == 2:
            atom_mask = atom_mask.reshape(B, N, 1)

        if molecule_mask is None:
            molecule_mask = Tensor.ones(B, dtype=dtypes.float32)
        elif len(molecule_mask.shape) == 2:
            molecule_mask = molecule_mask.reshape(B)

        # Edge mask a_ij = mask_i * mask_j * (1 - delta_ij)
        mask_i = atom_mask.reshape(B, N, 1, 1)
        mask_j = atom_mask.reshape(B, 1, N, 1)
        diag_zero = (1.0 - Tensor.eye(N, dtype=dtypes.float32)).reshape(1, N, N, 1)
        edge_mask = (mask_i * mask_j * diag_zero).realize()

        return x, atomic_numbers, atom_mask, molecule_mask, edge_mask

    def compute_energy(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes total potential energy U(x) for batch across all molecules.
        Returns: Tensor of shape (B,) in internal energy units (kcal/mol or reduced units).
        """
        x, atomic_numbers, atom_mask, molecule_mask, edge_mask = self._prepare_inputs(
            x, atomic_numbers, atom_mask, molecule_mask
        )
        B, N, _ = x.shape

        # 1. One-hot encode atomic numbers Z -> (B, N, max_atomic_number) -> h^0 (B, N, 128)
        z_clamped = atomic_numbers.cast(dtypes.int32)
        z_one_hot = Tensor.one_hot(z_clamped, num_classes=self.max_atomic_number)
        h = self.embedding(z_one_hot) * atom_mask

        # 2. Evaluate Pairwise Relative Squared Distances d_ij^2 = ||x_i - x_j||^2
        x_i = x.reshape(B, N, 1, 3)
        x_j = x.reshape(B, 1, N, 3)
        diff = x_i - x_j
        d_sq = (diff * diff).sum(axis=-1, keepdim=True)  # (B, N, N, 1)

        # 3. Pass through L=7 message-passing layers
        for layer in self.layers:
            h = layer(h, d_sq, edge_mask, atom_mask)

        # 4. Readout: Map node embeddings to atomic energies eps_i -> sum over atoms
        eps_i = self.readout_mlp[0](h)
        eps_i = self.readout_mlp[1](eps_i)
        eps_i = self.readout_mlp[2](eps_i) * atom_mask  # (B, N, 1)

        u_total = eps_i.sum(axis=(1, 2)) * molecule_mask
        return u_total

    def compute_forces(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Evaluates exact conservative forces F = -∇_x U(x) via reverse-mode autograd.
        Returns: Tensor of shape (B, N, 3).
        """
        _, forces = self.compute_energy_and_forces(
            x=x,
            atomic_numbers=atomic_numbers,
            atom_mask=atom_mask,
            molecule_mask=molecule_mask,
        )
        return forces

    def compute_energy_and_forces(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        molecule_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluates total potential energy U(x) and conservative forces F = -∇_x U(x)
        via reverse-mode autograd in a single unified graph traversal.
        Returns: Tuple of (energy (B,), forces (B, N, 3)).
        """
        x_in = x.detach()
        x_in.requires_grad = True

        u_total = self.compute_energy(
            x=x_in,
            atomic_numbers=atomic_numbers,
            atom_mask=atom_mask,
            molecule_mask=molecule_mask,
        )

        loss = u_total.sum()
        loss.backward()

        _, _, a_mask, _, _ = self._prepare_inputs(x_in, atomic_numbers, atom_mask, molecule_mask)
        grad = x_in.grad if x_in.grad is not None else Tensor.zeros_like(x_in)
        forces = (-grad * a_mask).realize()
        return u_total.realize(), forces

    def get_jit_evaluator(self) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns a lazily-instantiated TinyJit compiled evaluator function
        mapping (x, atomic_numbers, atom_mask, molecule_mask) -> (u_total, forces).
        Traces the execution graph once and replays directly from GPU command queue.
        """
        if getattr(self, "_jit_evaluator", None) is None:
            # Ensure parameters are realized on device before tracing
            for p in nn.state.get_parameters(self):
                p.realize()

            def _step(x_in: Tensor, z_in: Tensor, a_mask: Tensor, m_mask: Tensor) -> Tuple[Tensor, Tensor]:
                u = self.compute_energy(x=x_in, atomic_numbers=z_in, atom_mask=a_mask, molecule_mask=m_mask)
                u.sum().backward()
                grad = x_in.grad if x_in.grad is not None else Tensor.zeros_like(x_in)
                f = (-grad * a_mask).realize()
                return u.realize(), f

            self._jit_evaluator = TinyJit(_step)
        return self._jit_evaluator
