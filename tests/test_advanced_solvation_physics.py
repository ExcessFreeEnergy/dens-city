"""
Unit and integration tests for:
1. Long alkyl chain conformational ensembling (s=48 depth) and Rg contraction.
2. Dielectric-dependent open/closed conformer weighting for intramolecular H-bonding pseudorings.
3. First-principles neat protic liquid self-association parameterization.
"""

import numpy as np
from tinygrad import Tensor, dtypes

from dens_city.boltzmann.egnn import EGNNForceField
from dens_city.utils.materials import (
    compute_neat_liquid_self_association_correction,
    generate_conformer_rotamer_diversity,
)


def test_long_chain_ensemble_depth_and_rg_contraction():
    """
    Verifies that for flexible molecules (N_rot >= 10), generate_conformer_rotamer_diversity
    generates s=48 conformers and includes coiled/contracted states with reduced radius of gyration.
    """
    N = 28
    coords = np.zeros((N, 3), dtype=np.float32)
    coords[:, 0] = np.linspace(0, 1.54 * (N - 1), N)  # Extended zigzag-like chain
    atomic_numbers = [6] * N
    bonds = [(i, i + 1, 1.0) for i in range(N - 1)]

    n_rot = 25  # octacosane has 25 rotatable C-C bonds
    n_conf = 48

    confs = generate_conformer_rotamer_diversity(
        coords=coords,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
        n_rot=n_rot,
        n_conf=n_conf,
        seed=123,
    )

    assert confs.shape == (48, N, 3)

    # Conformer 0 is the ground state extended conformation
    rg_0 = np.sqrt(np.mean(np.sum((confs[0] - np.mean(confs[0], axis=0)) ** 2, axis=-1)))

    # Conformer 30 (contracted globule) must have strictly smaller radius of gyration
    rg_contracted = np.sqrt(np.mean(np.sum((confs[30] - np.mean(confs[30], axis=0)) ** 2, axis=-1)))

    assert rg_contracted < rg_0 * 0.90, f"Expected contracted Rg < {rg_0 * 0.90:.2f}, got {rg_contracted:.2f}"
    assert np.all(np.isfinite(confs))


def test_ortho_polar_open_rotamer_generation():
    """
    Verifies that for molecules with ortho-donor-acceptor pairs (e.g. 2-hydroxybenzaldehyde),
    generate_conformer_rotamer_diversity detects the intramolecular H-bond and creates
    an open conformer (slot 1) rotating the OH group by 180 degrees.
    """
    # Synthetic 2-hydroxybenzaldehyde fragment:
    # 0: C_ar (ring), 1: O (phenol), 2: H (phenol), 3: C_ar (ortho carbonyl)
    # 4: C (formyl), 5: O (carbonyl acceptor)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # C_ar (O)
            [1.36, 0.0, 0.0],  # O_phenol
            [1.70, 0.85, 0.0],  # H_phenol (pointing toward formyl oxygen)
            [-0.70, 1.20, 0.0],  # C_ar (formyl)
            [-0.30, 2.50, 0.0],  # C_formyl
            [0.85, 2.80, 0.0],  # O_formyl (acceptor within 2.5 A of H)
        ],
        dtype=np.float32,
    )
    atomic_numbers = [6, 8, 1, 6, 6, 8]
    bonds = [(0, 1, 1.0), (1, 2, 1.0), (0, 3, 1.5), (3, 4, 1.0), (4, 5, 2.0)]

    confs = generate_conformer_rotamer_diversity(
        coords=coords,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
        n_rot=2,
        n_conf=8,
        seed=42,
    )

    d_initial = float(np.linalg.norm(coords[2] - coords[5]))
    d_open = float(np.linalg.norm(confs[1, 2] - confs[1, 5]))

    assert d_initial < 2.5, f"Initial H should be close to acceptor, got {d_initial:.2f} Å"
    assert d_open > d_initial + 1.0, f"Open rotamer H should be rotated away from acceptor, got {d_open:.2f} Å"


def test_dielectric_dependent_boltzmann_weighting():
    """
    Verifies that compute_ensembled_solvation_readouts uses solution-phase potential of mean force:
    In water (dielectric = 78.4), an open conformer with higher gas-phase internal energy but
    more negative Born solvation free energy receives increased Boltzmann weight.
    """
    egnn = EGNNForceField(num_layers=2, hidden_dim=64, n_particles=32)

    B, s, N = 1, 2, 32
    coords = np.zeros((B, s, N, 3), dtype=np.float32)

    # Closed conformer k=0: atoms clustered together
    coords[0, 0, 0] = [0.0, 0.0, 0.0]
    coords[0, 0, 1] = [1.2, 0.0, 0.0]
    coords[0, 0, 2] = [0.0, 1.2, 0.0]
    coords[0, 0, 3] = [1.0, 1.0, 0.0]

    # Open conformer k=1: atoms extended
    coords[0, 1, 0] = [0.0, 0.0, 0.0]
    coords[0, 1, 1] = [2.5, 0.0, 0.0]
    coords[0, 1, 2] = [0.0, 2.5, 0.0]
    coords[0, 1, 3] = [3.5, 3.5, 0.0]

    z = Tensor([[8, 8, 1, 1] + [0] * 28], dtype=dtypes.float32)
    bq = Tensor([[-0.45, -0.45, 0.45, 0.45] + [0.0] * 28], dtype=dtypes.float32)
    mask = Tensor(np.array([[1.0, 1.0, 1.0, 1.0] + [0.0] * 28], dtype=np.float32)).reshape(1, 32, 1)

    # Gas phase internal energy: k=1 is +4.0 kcal/mol higher (disfavored in vacuum)
    e_int = Tensor([[0.0, 4.0]], dtype=dtypes.float32)

    # In vacuum / nonpolar (dielectric = 2.0): k=0 dominates
    _, solv_vac, gb_vac = egnn.compute_ensembled_solvation_readouts(
        x_ensemble=Tensor(coords),
        atomic_numbers=z,
        atom_mask=mask,
        base_charges=bq,
        internal_energies=e_int,
        dielectric_constant=2.0,
        temperature_k=298.15,
    )

    # In water (dielectric = 78.4): Born electrostatics favors open conformer k=1
    _, solv_aq, gb_aq = egnn.compute_ensembled_solvation_readouts(
        x_ensemble=Tensor(coords),
        atomic_numbers=z,
        atom_mask=mask,
        base_charges=bq,
        internal_energies=e_int,
        dielectric_constant=78.4,
        temperature_k=298.15,
    )

    assert float(gb_aq.numpy()[0]) < float(gb_vac.numpy()[0]) - 2.0


def test_neat_liquid_self_association_correction():
    """
    Verifies that compute_neat_liquid_self_association_correction computes proper
    stabilization for neat protic liquids (formamide, 2-ethylhexanol) and zero for others.
    """
    # Formamide in formamide (neat protic, alpha=0.71, beta=0.48, eta=0.45)
    dg_formamide = compute_neat_liquid_self_association_correction(
        solute_name="FORMAMIDE",
        solvent_name="FORMAMIDE",
        alpha_s=0.71,
        beta_s=0.48,
        packing_fraction=0.45,
    )
    assert -3.0 < dg_formamide < -0.8, f"Expected formamide self-association ~ -1.8 kcal/mol, got {dg_formamide}"

    # 2-Ethylhexanol in 2-ethylhexanol (neat protic alcohol, alpha=0.33, beta=0.68, eta=0.42)
    dg_2eh = compute_neat_liquid_self_association_correction(
        solute_name="2-ETHYLHEXANOL",
        solvent_name="2-ETHYLHEXANOL",
        alpha_s=0.33,
        beta_s=0.68,
        packing_fraction=0.42,
    )
    assert -2.5 < dg_2eh < -0.5, f"Expected 2-ethylhexanol self-association ~ -1.4 kcal/mol, got {dg_2eh}"

    # Formamide in Hexane (not neat liquid: solute != solvent) -> strictly 0.0
    dg_mix = compute_neat_liquid_self_association_correction(
        solute_name="FORMAMIDE",
        solvent_name="HEXANE",
        alpha_s=0.0,
        beta_s=0.0,
        packing_fraction=0.40,
    )
    assert dg_mix == 0.0

    # Hexane in Hexane (neat alkane: alpha=0, beta=0) -> strictly 0.0
    dg_hexane = compute_neat_liquid_self_association_correction(
        solute_name="HEXANE",
        solvent_name="HEXANE",
        alpha_s=0.0,
        beta_s=0.0,
        packing_fraction=0.42,
    )
    assert dg_hexane == 0.0
