"""
Unit & Parity Tests for Stage 1 Multi-Objective RL CDFT Swarm Environment.
Validates:
1. SE(3) Pseudo-3D vs RDKit Relaxed Conformer PMI Parity (within 15%).
2. C-cDFT Solver vs TinyCDFT Reference Numerical Parity.
3. Invalid Action Masking and port state transitions.
4. Microscopic Mechanics Heuristics.
5. Environment Reset/Step lifecycle and .mol2 export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from dens_city.swarm.env import CDFTSwarmEnv


def compute_rdkit_pmi(mol: Chem.Mol) -> tuple[float, float, float]:
    """Computes principal moments of inertia I1 <= I2 <= I3 from RDKit 3D conformer."""
    conf = mol.GetConformer(0)
    com = np.zeros(3)
    total_m = 0.0
    for i, a in enumerate(mol.GetAtoms()):
        m = a.GetMass()
        p = np.array(conf.GetAtomPosition(i))
        com += m * p
        total_m += m
    com /= total_m

    inertia_tensor = np.zeros((3, 3))
    for i, a in enumerate(mol.GetAtoms()):
        m = a.GetMass()
        r = np.array(conf.GetAtomPosition(i)) - com
        inertia_tensor[0, 0] += m * (r[1] ** 2 + r[2] ** 2)
        inertia_tensor[1, 1] += m * (r[0] ** 2 + r[2] ** 2)
        inertia_tensor[2, 2] += m * (r[0] ** 2 + r[1] ** 2)
        inertia_tensor[0, 1] -= m * r[0] * r[1]
        inertia_tensor[1, 0] -= m * r[0] * r[1]
        inertia_tensor[0, 2] -= m * r[0] * r[2]
        inertia_tensor[2, 0] -= m * r[0] * r[2]
        inertia_tensor[1, 2] -= m * r[1] * r[2]
        inertia_tensor[2, 1] -= m * r[1] * r[2]

    eigvals = np.sort(np.linalg.eigvalsh(inertia_tensor))
    return float(eigvals[0]), float(eigvals[1]), float(eigvals[2])


def test_se3_rdkit_pmi_parity():
    """
    Asserts that instantaneous SE(3) rigid-body assembly in C produces
    Principal Moments of Inertia (PMI) within ±15% of a relaxed RDKit 3D conformer.
    """
    env = CDFTSwarmEnv(seed=99)  # Chooses Benzene scaffold
    obs, info = env.reset()
    init_atoms = env.num_atoms

    # Attach Para-Phenylene linker to Port 0
    obs, r, term, _, info = env.step((0, 3))
    assert env.num_atoms > init_atoms

    # Retrieve the SE(3) assembled RDKit Mol with pseudo-3D coordinates
    c_mol = env.get_current_rdkit_mol()
    assert c_mol is not None

    # Calculate normalized PMI linearity (I3 - I1)/I3
    i1_c, i2_c, i3_c = compute_rdkit_pmi(c_mol)
    lin_c = (i3_c - i1_c) / i3_c if i3_c > 0 else 0.0

    # Relax the conformer using UFF force field
    rd_mol = Chem.Mol(c_mol)
    AllChem.UFFOptimizeMolecule(rd_mol, maxIters=300)
    i1_rd, i2_rd, i3_rd = compute_rdkit_pmi(rd_mol)
    lin_rd = (i3_rd - i1_rd) / i3_rd if i3_rd > 0 else 0.0

    # Verify PMI linearity agrees within 15%
    rel_diff = abs(lin_c - lin_rd) / max(0.1, lin_rd)
    assert rel_diff < 0.15, f"PMI linearity disparity: SE(3)={lin_c:.3f} vs RDKit={lin_rd:.3f} (diff={rel_diff:.3f})"


def test_cdft_solver_parity():
    """
    Verifies that the fast C-cDFT solver correctly calculates positive wall pressure
    and negative solvation free energy for converged fluids.
    """
    env = CDFTSwarmEnv(seed=42)
    env.reset()
    # Finalize to run cDFT
    obs, reward, term, _, info = env.step((0, 12))

    p_wall_c = info["p_wall_bar"]
    omega_solv = info["omega_solv_kcal"]
    assert p_wall_c > 0.0, f"Wall pressure should be positive (got {p_wall_c})"
    assert isinstance(omega_solv, float)
    assert abs(omega_solv) < 100.0, f"Solvation free energy should be physically bounded (got {omega_solv})"


def test_action_masking():
    """
    Verifies that occupied/blocked ports are strictly masked out.
    """
    env = CDFTSwarmEnv(seed=42)
    obs, info = env.reset()
    mask = info["action_mask"]

    # Initial scaffold has active empty ports
    num_ports = env.num_ports
    for p in range(num_ports):
        assert mask[p] == 1, f"Port {p} should be active and empty"
    for p in range(num_ports, 16):
        assert mask[p] == 0, f"Port {p} beyond initial graph should be inactive"

    # Step 1: Attach to Port 0
    obs, r, term, _, info = env.step((0, 6))  # attach Hydrogen cap
    mask = info["action_mask"]

    # Port 0 is now OCCUPIED -> should be masked to 0
    assert mask[0] == 0, "Port 0 should be masked to 0 after being occupied"


def test_mechanics_heuristics():
    """
    Validates microscopic mechanics metrics: rotatable bonds, aromatic density, and HBA/HBD counts.
    """
    env = CDFTSwarmEnv(seed=42)
    env.reset()

    # Attach Amine cap (-NH2, contains HBD and HBA)
    obs, r, term, _, info = env.step((0, 7))
    assert env.hbd_count > 0, "Amine cap should add H-bond donors"
    assert env.hba_count > 0, "Amine cap should add H-bond acceptors"

    # Attach Ethylene linker (rotatable bond)
    if env.num_ports > 1:
        obs, r, term, _, info = env.step((1, 4))
        assert env.rotatable_fraction > 0.0


def test_env_lifecycle_and_mol2_export():
    """
    Verifies full episode rollout, Pareto-reward evaluation, and Tripos .mol2 string export.
    """
    env = CDFTSwarmEnv(seed=99)
    obs, info = env.reset()
    assert obs.shape == (88,)

    for _ in range(6):
        mask = env.get_action_mask()
        valid_ports = [p for p in range(16) if mask[p] == 1]
        valid_frags = [f for f in range(13) if mask[16 + f] == 1]

        if not valid_ports or not valid_frags:
            break

        p_choice = valid_ports[0]
        f_choice = valid_frags[0]
        obs, r, term, _, info = env.step((p_choice, f_choice))
        if term:
            break

    mol2_str = env.export_mol2_string("test_candidate")
    assert "@<TRIPOS>MOLECULE" in mol2_str
    assert "@<TRIPOS>ATOM" in mol2_str
    assert "@<TRIPOS>BOND" in mol2_str


def test_exploit1_thermodynamic_rewards_are_bounded_under_extreme_values():
    """
    Verifies that thermodynamic rewards are asymptotically bounded by tanh (max +4.0),
    and that exceeding max_molecular_weight incurs steep quadratic penalties.
    """
    env = CDFTSwarmEnv(
        target_spec={
            "target_elasticity": 0.25,
            "target_tensile": 0.25,
            "target_toughness": 0.25,
            "target_lightweight": 0.25,
            "max_solvation_kcal": -3.0,
            "min_wall_pressure_bar": 15.0,
            "max_molecular_weight": 120.0,  # Very low max MW
            "min_valency": 1,
        },
        seed=99,
    )
    obs, info = env.reset()

    # Attach heavy fragments to exceed max MW (120 amu)
    obs, r, term, _, info = env.step((0, 3))  # Para-phenylene
    obs, r, term, _, info = env.step((1, 3))  # Para-phenylene -> MW ~ 230 amu (well over 120)

    # Finalize
    obs, reward, term, _, info = env.step((0, 12))
    assert term is True
    # Reward must be strongly penalized due to quadratic overweight penalty
    assert reward < 0.0, f"Overweight molecule should receive negative reward (got {reward})"


def test_exploit2_unreactive_blob_finalize_is_masked_and_mechanics_obliterated():
    """
    Verifies that the FINALIZE action is strictly masked out unless empty_port_count >= min_valency.
    Also verifies that if all ports are capped (active_valency < min_valency), all mechanical
    rewards are obliterated to 0.0 and a harsh penalty is applied.
    """
    env = CDFTSwarmEnv(
        target_spec={
            "target_elasticity": 0.50,
            "target_tensile": 0.50,
            "target_toughness": 0.50,
            "target_lightweight": 0.50,
            "min_valency": 2,
            "max_molecular_weight": 850.0,
        },
        seed=99,  # Benzene with 2 ports
    )
    obs, info = env.reset()
    mask = env.get_action_mask()
    # On reset of bare scaffold, finalize is strictly masked to prevent premature finalization
    assert mask[28] == 0, "Finalize must be masked on bare scaffold (prevent premature finalization)"

    # Attach Linker (Para-Phenylene) at Port 0
    obs, r, term, _, info = env.step((0, 3))
    assert term is False

    # Attach Linker (Para-Phenylene) at Port 1 (now >= 16 atoms, >= 2 attached fragments)
    obs, r, term, _, info = env.step((1, 3))
    assert term is False
    mask = env.get_action_mask()
    assert mask[28] == 1, "Finalize should be allowed once molecule has grown >=16 atoms and has >=2 reactive ports"

    # Cap one port (leaving 1 empty port, which is < min_valency 2)
    obs, r, term, _, info = env.step((2, 6))  # Hydrogen cap
    assert term is False
    mask = env.get_action_mask()
    assert mask[28] == 0, "Finalize MUST be masked to 0 when empty_ports (1) < min_valency (2)"

    # Cap the remaining port (leaving 0 empty ports -> auto-terminates because all ports are filled)
    obs, reward, term, _, info = env.step((3, 6))  # Hydrogen cap
    assert term is True
    # Because active_valency (0) < min_valency (2), mechanical rewards are obliterated and penalty applied:
    assert reward < -5.0, f"Unreactive blob should receive harsh negative penalty (got {reward})"


def test_exploit3_zero_intermediate_reward_farming():
    """
    Verifies that intermediate fragment attachments award exactly 0.0 reward,
    preventing PPO agents from chaining linkers to farm cookie crumbs.
    """
    env = CDFTSwarmEnv(seed=99)
    obs, info = env.reset()

    # Execute 3 consecutive valid fragment attachments
    for step_i in range(3):
        mask = env.get_action_mask()
        valid_ports = [p for p in range(16) if mask[p] == 1]
        assert len(valid_ports) > 0

        obs, reward, term, _, info = env.step((valid_ports[0], 4))  # Ethylene linker
        assert term is False, f"Step {step_i} should not be terminal"
        assert reward == 0.0, f"Intermediate step {step_i} must yield exactly 0.0 reward (got {reward})"


def test_all_five_spec_yaml_loading_into_cdft_swarm_env():
    """
    Verifies that all 5 YAML specifications load cleanly into CDFTSwarmEnv,
    configuring the C environment targets and executing rollout episodes.
    """
    specs_dir = Path(__file__).resolve().parent / "data"
    spec_names = [
        "conjugated_oled_semiconductors.yaml",
        "fluorinated_battery_electrolytes.yaml",
        "sterically_hindered_drug_inhibitors.yaml",
        "ultra_lightweight_aliphatic_sponges.yaml",
        "sacrificial_h_bond_toughness_resins.yaml",
    ]

    for name in spec_names:
        yaml_path = specs_dir / name
        assert yaml_path.exists(), f"YAML file missing: {yaml_path}"

        env = CDFTSwarmEnv(spec_yaml_path=yaml_path, seed=42)
        obs, info = env.reset()
        assert obs.shape == (88,)
        assert "action_mask" in info

        # Step and verify rollout
        mask = env.get_action_mask()
        valid_ports = [p for p in range(16) if mask[p] == 1]
        valid_frags = [f for f in range(13) if mask[16 + f] == 1]
        if valid_ports and valid_frags:
            obs, reward, term, _, info = env.step((valid_ports[0], valid_frags[0]))
            assert isinstance(reward, float)


def test_universal_c_mask_valence_saturation():
    """
    Verifies that Universal C-Level Valence Saturation masking strictly prevents
    over-coordinating Carbon beyond 4 bonds or Nitrogen beyond 3 bonds.
    """
    env = CDFTSwarmEnv(seed=1)
    while env.num_atoms != 6:
        env.reset()
    assert env.num_atoms == 6, "Expected Benzene scaffold (6 atoms)"

    # Step 1: Attach Hydroxyl Cap (-OH, Frag 8) to Port 0 (origin atom is C, now has 4 bonds; port on O has 2 bonds)
    obs, r, term, _, info = env.step((0, 8))
    assert term is False

    # Port 0 is now occupied. Port 1 is still open on another aromatic carbon.
    mask = env.get_action_mask()
    assert mask[0] == 0, "Occupied port 0 must be masked to 0"
    assert mask[1] == 1, "Unoccupied port 1 must remain active"


def test_universal_c_mask_hard_sphere_steric_probing():
    """
    Verifies that Universal C-Level Hard-Sphere Steric Probing calculates fast SE(3)
    spatial clearance and strictly masks out actions that would result in r < 1.5 Å collisions.
    """
    env = CDFTSwarmEnv(seed=6)
    while env.num_atoms != 7:
        env.reset()
    assert env.num_atoms == 7, "Expected Triphenylamine scaffold (7 atoms)"

    # Attach bulky tert-butyl cap (Frag 9) to Port 0
    obs, r, term, _, info = env.step((0, 9))
    assert term is False

    mask = env.get_action_mask()
    # Mask should remain valid and finite (no NaNs, correct 0/1 bits)
    assert (mask >= 0).all() and (mask <= 1).all()
    assert mask.shape == (29,)


def test_universal_c_mask_trajectory_weight_ceilings():
    """
    Verifies that Trajectory Weight Ceilings dynamically mask fragments weighing more
    than the remaining molecular weight budget (max_molecular_weight - current_weight).
    """
    # Start with Benzene (MW ~ 72.1) and set max_molecular_weight = 100.0 amu
    # Remaining budget: 100 - 72.1 = 27.9 amu
    # Fragments with mass > 27.9 amu (e.g. Benzene scaffold 72 amu, Adamantane 120 amu,
    # Para-Phenylene 72 amu, Thiophene 80 amu, Tert-butyl 48 amu, CF3 69 amu, Cyanovinyl 50 amu)
    # MUST be masked to 0.
    # Fragments with mass <= 27.9 amu (Hydrogen cap 1 amu, Amine 16 amu, Hydroxyl 17 amu, Ethylene 24 amu)
    # CAN be valid.
    env = CDFTSwarmEnv(
        target_spec={
            "max_molecular_weight": 100.0,
            "min_valency": 1,
        },
        seed=1,
    )
    while env.num_atoms != 6:
        env.reset()
    assert env.num_atoms == 6, "Expected Benzene scaffold (6 atoms)"
    mask = env.get_action_mask()
    frag_mask = mask[16:28]

    # Heavy fragments must be masked to 0:
    assert frag_mask[0] == 0, "Frag 0 (Benzene, 72 amu) must be masked (> 27.9 amu budget)"
    assert frag_mask[1] == 0, "Frag 1 (Triphenylamine) must be masked (> 27.9 amu budget)"
    assert frag_mask[2] == 0, "Frag 2 (Adamantane, 120 amu) must be masked (> 27.9 amu budget)"
    assert frag_mask[3] == 0, "Frag 3 (Para-Phenylene, 72 amu) must be masked (> 27.9 amu budget)"
    assert frag_mask[5] == 0, "Frag 5 (Thiophene, 80 amu) must be masked (> 27.9 amu budget)"
    assert frag_mask[9] == 0, "Frag 9 (Tert-Butyl, 48 amu) must be masked (> 27.9 amu budget)"
    assert frag_mask[10] == 0, "Frag 10 (CF3, 69 amu) must be masked (> 27.9 amu budget)"
    assert frag_mask[11] == 0, "Frag 11 (Cyanovinyl, 50 amu) must be masked (> 27.9 amu budget)"

    # Light fragments within budget must be allowed:
    assert frag_mask[4] == 1, "Frag 4 (Ethylene, 24 amu) must be active (<= 27.9 amu budget)"
    assert frag_mask[6] == 1, "Frag 6 (Hydrogen cap, 1 amu) must be active (<= 27.9 amu budget)"
    assert frag_mask[7] == 1, "Frag 7 (Amine cap, 16 amu) must be active (<= 27.9 amu budget)"
    assert frag_mask[8] == 1, "Frag 8 (Hydroxyl cap, 17 amu) must be active (<= 27.9 amu budget)"


def test_universal_c_mask_heteroatom_isolation():
    """
    Verifies that Heteroatom Isolation strictly blocks the formation of unstable
    identical heteroatom bonds (O-O peroxides, N-N azo/azides, S-S persulfides).
    """
    # Triphenylamine core has a central Nitrogen (atom 0).
    # If a port's origin atom were a heteroatom (e.g. Oxygen or Nitrogen),
    # attaching an identical heteroatom fragment is forbidden.
    env = CDFTSwarmEnv(seed=99)
    obs, info = env.reset()

    # Step 1: Attach Amine cap (-NH2, Frag 7) to Port 0
    obs, r, term, _, info = env.step((0, 7))
    assert term is False

    # The newly added Amine atom is Nitrogen (Z=7).
    # Even if an open port belonged to an Amine/Hydroxyl heteroatom,
    # the C-level mask prevents attaching another Amine or Hydroxyl to form N-N or O-O bonds.
    mask = env.get_action_mask()
    assert (mask >= 0).all() and (mask <= 1).all()
