"""
Universal Generalized Born (GB) Electrostatic Solvation Engine.
Evaluates context-dependent dielectric hydration free energy ΔG_GB directly on device
using Hawkins-Cramer-Truhlar / Grycuk effective Born radii and Still's pairwise equation.

Adheres strictly to tinygrad tensor-native operations and zero-sync GPU gather execution.
"""

from __future__ import annotations

from typing import Optional

from tinygrad import Tensor, dtypes


def _build_bondi_radii_table() -> list[float]:
    """Builds static Bondi intrinsic van der Waals radii lookup table up to Z=118."""
    radii = [2.00] * 119
    radii[0] = 0.00  # Padding / dummy
    radii[1] = 1.20  # H
    radii[2] = 1.40  # He
    radii[3] = 1.82  # Li
    radii[4] = 1.53  # Be
    radii[5] = 1.92  # B
    radii[6] = 1.70  # C
    radii[7] = 1.55  # N
    radii[8] = 1.52  # O
    radii[9] = 1.47  # F
    radii[10] = 1.54  # Ne
    radii[11] = 2.27  # Na
    radii[12] = 1.73  # Mg
    radii[13] = 1.84  # Al
    radii[14] = 2.10  # Si
    radii[15] = 1.80  # P
    radii[16] = 1.80  # S
    radii[17] = 1.75  # Cl
    radii[18] = 1.88  # Ar
    radii[19] = 2.75  # K
    radii[20] = 2.31  # Ca
    radii[35] = 1.85  # Br
    radii[53] = 1.98  # I
    return radii


_BONDI_RADII_TENSOR: Optional[Tensor] = None


def get_bondi_radii_tensor() -> Tensor:
    """Lazily allocates and realizes the static Bondi radii tensor on GPU device."""
    global _BONDI_RADII_TENSOR
    if _BONDI_RADII_TENSOR is None:
        _BONDI_RADII_TENSOR = Tensor(_build_bondi_radii_table(), dtype=dtypes.float32).realize()
    return _BONDI_RADII_TENSOR


# Coulomb electrostatic constant in kcal * Å / (e^2 * mol)
COULOMB_KCAL_A = 332.06371


class GeneralizedBornSolvation:
    """
    Tensor-native Universal Generalized Born (GB) Solvation Functional.
    Computes electrostatic polarization free energy in arbitrary dielectric solvent
    from dynamic quantum partial charges q(x), atomic numbers Z, and Cartesian coordinates x.
    """

    def __init__(
        self,
        dielectric_constant: float = 78.4,
        solute_dielectric: float = 1.0,
        radius_offset_a: float = 0.09,
    ):
        self.dielectric_constant = dielectric_constant
        self.solute_dielectric = solute_dielectric
        self.radius_offset_a = radius_offset_a

    def compute_born_radii(
        self,
        x: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """
        Computes effective Born radii alpha_i(x) using Hawkins/Grycuk pairwise volume descreening:
        1 / alpha_i = 1 / rho_i - 0.5 * sum_{j != i} (sigma_j^3 / (r_ij^4 + soft))
        x: (B, N, 3)
        atomic_numbers: (B, N)
        atom_mask: (B, N, 1)
        Returns: Tensor of shape (B, N, 1) in Ångströms.
        """
        B, N, _ = x.shape

        # 1. Zero-sync O(1) GPU Bondi radii gather
        bondi_tensor = get_bondi_radii_tensor()
        z_clamped = atomic_numbers.cast(dtypes.int32).clip(0, 118)
        sigma = bondi_tensor[z_clamped].reshape(B, N, 1) * atom_mask  # (B, N, 1)
        rho_i = (sigma - self.radius_offset_a).maximum(0.8) * atom_mask  # (B, N, 1)

        # 2. Pairwise distances d_ij^2 = ||x_i - x_j||^2
        x_i = x.reshape(B, N, 1, 3)
        x_j = x.reshape(B, 1, N, 3)
        diff = x_i - x_j
        d_sq = (diff * diff).sum(axis=-1, keepdim=True)  # (B, N, N, 1)
        d_dist = d_sq.maximum(1e-4).sqrt()

        # 3. Off-diagonal pair validity mask
        mask_i = atom_mask.reshape(B, N, 1, 1)
        mask_j = atom_mask.reshape(B, 1, N, 1)
        diag_zero = (1.0 - Tensor.eye(N, dtype=dtypes.float32)).reshape(1, N, N, 1)
        pair_mask = mask_i * mask_j * diag_zero

        # 4. Grycuk smooth volume descreening: alpha_i = rho_i * (1 + sum_{j != i} 0.12 * sigma_j^3 / (r_ij^3 + rho_i^3))
        sigma_j = sigma.reshape(B, 1, N, 1)
        rho_i_4d = rho_i.reshape(B, N, 1, 1)
        denom = (d_dist * d_sq) + (rho_i_4d**3)
        descreen = ((0.12 * (sigma_j**3)) / denom.maximum(0.5)) * pair_mask  # (B, N, N, 1)
        integral_i = descreen.sum(axis=2) * atom_mask  # (B, N, 1)

        # 5. Effective Born radius alpha_i >= rho_i
        alpha_i = rho_i * (1.0 + integral_i.minimum(1.5)) * atom_mask
        return alpha_i.maximum(0.8 * atom_mask).realize()

    def compute_solvation_free_energy(
        self,
        x: Tensor,
        charges: Tensor,
        atomic_numbers: Tensor,
        atom_mask: Optional[Tensor] = None,
        dielectric_constant: Optional[float] = None,
    ) -> Tensor:
        """
        Computes total Generalized Born electrostatic solvation free energy ΔG_GB in kcal/mol.
        x: (B, N, 3)
        charges: (B, N) or (B, N, 1)
        atomic_numbers: (B, N)
        atom_mask: Optional (B, N, 1) or (B, N)
        dielectric_constant: Optional solvent dielectric constant (overrides instance default)
        Returns: Tensor of shape (B,) in kcal/mol.
        """
        if len(x.shape) == 2:
            x = x.reshape(1, -1, 3)
        B, N, _ = x.shape

        if len(charges.shape) == 2:
            q = charges.reshape(B, N, 1)
        else:
            q = charges.reshape(B, N, 1)

        if len(atomic_numbers.shape) == 1:
            atomic_numbers = atomic_numbers.reshape(B, N)

        if atom_mask is None:
            atom_mask = (atomic_numbers > 0).cast(dtypes.float32).reshape(B, N, 1)
        elif len(atom_mask.shape) == 2:
            atom_mask = atom_mask.reshape(B, N, 1)

        eps_solv = dielectric_constant if dielectric_constant is not None else self.dielectric_constant
        eps_solv = max(1.0, float(eps_solv))

        # Dielectric screening prefactor: -0.5 * (1/eps_in - 1/eps_out)
        eps_factor = 0.5 * ((1.0 / self.solute_dielectric) - (1.0 / eps_solv))

        # 1. Evaluate Effective Born Radii alpha_i
        alpha = self.compute_born_radii(x, atomic_numbers, atom_mask)  # (B, N, 1)

        # 2. Self-solvation energy: sum_i q_i^2 / alpha_i
        self_energy = ((q * q) / alpha.maximum(0.5)).sum(axis=(1, 2))  # (B,)

        # 3. Pairwise Still interaction equation: sum_{i != j} q_i q_j / f_GB(r_ij, alpha_i, alpha_j)
        x_i = x.reshape(B, N, 1, 3)
        x_j = x.reshape(B, 1, N, 3)
        diff = x_i - x_j
        d_sq = (diff * diff).sum(axis=-1, keepdim=True)  # (B, N, N, 1)

        q_i = q.reshape(B, N, 1, 1)
        q_j = q.reshape(B, 1, N, 1)
        q_prod = q_i * q_j  # (B, N, N, 1)

        alpha_i = alpha.reshape(B, N, 1, 1)
        alpha_j = alpha.reshape(B, 1, N, 1)
        a_prod = alpha_i * alpha_j

        # f_GB(r_ij) = sqrt(r_ij^2 + alpha_i * alpha_j * exp(-r_ij^2 / (4 * alpha_i * alpha_j)))
        exp_damp = (-d_sq / (4.0 * a_prod.maximum(0.25))).exp()
        f_gb = (d_sq + a_prod * exp_damp).maximum(0.25).sqrt()

        # Pair mask (excluding diagonal)
        mask_i = atom_mask.reshape(B, N, 1, 1)
        mask_j = atom_mask.reshape(B, 1, N, 1)
        diag_zero = (1.0 - Tensor.eye(N, dtype=dtypes.float32)).reshape(1, N, N, 1)
        pair_mask = mask_i * mask_j * diag_zero

        pair_energy = ((q_prod / f_gb) * pair_mask).sum(axis=(1, 2, 3))  # (B,)

        # Total Born Solvation Free Energy in kcal/mol
        total_gb_kcal = -eps_factor * COULOMB_KCAL_A * (self_energy + pair_energy)
        return total_gb_kcal.realize()
