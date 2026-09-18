"""
Synchronous Chemical Intake Gate for dens-city server.
Protects the background GPU solver from hallucinated chemistry, invalid valencies,
unclosed rings, and unphysical specification constraints before any asynchronous job is queued.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from dens_city.server.models import InvalidSMILESEntry, ValidationResponse

ALLOWED_ATOMIC_NUMBERS = {
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    18,
    35,
    36,
    53,
    54,
}  # H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Br, Kr, I, Xe


def validate_single_smiles(smiles: str, index: int = 0) -> Optional[InvalidSMILESEntry]:
    """
    Rigorously checks a single SMILES string against valence saturation,
    ring closure, and element constraints. Returns InvalidSMILESEntry if invalid, None if valid.
    """
    if not smiles or not smiles.strip():
        return InvalidSMILESEntry(
            index=index,
            smiles=smiles,
            reason="Empty or whitespace-only SMILES string provided.",
        )

    smi_clean = smiles.strip()
    try:
        # First attempt parsing without sanitization to isolate parsing vs valency
        mol = Chem.MolFromSmiles(smi_clean, sanitize=False)
        if mol is None:
            return InvalidSMILESEntry(
                index=index,
                smiles=smi_clean,
                reason="Syntactically invalid SMILES string (unclosed ring, unmatched parentheses, or illegal character).",
            )

        # Sanitize molecule to enforce strict valence rules
        problems = Chem.DetectChemistryProblems(mol)
        if problems:
            prob = problems[0]
            atom_idx = prob.GetAtomIdx() if hasattr(prob, "GetAtomIdx") else None
            return InvalidSMILESEntry(
                index=index,
                smiles=smi_clean,
                reason=f"Chemical valence violation: {prob.Message()}",
                atom_index=atom_idx,
            )

        Chem.SanitizeMol(mol)

        # Check atomic numbers
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z not in ALLOWED_ATOMIC_NUMBERS:
                return InvalidSMILESEntry(
                    index=index,
                    smiles=smi_clean,
                    reason=f"Unsupported element symbol '{atom.GetSymbol()}' (Z={z}). Allowed elements: H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar, Br, Kr, I, Xe.",
                    atom_index=atom.GetIdx(),
                )

        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy == 0:
            return InvalidSMILESEntry(
                index=index,
                smiles=smi_clean,
                reason="Molecule contains zero heavy atoms (pure hydrogen or naked ions unsupported).",
            )
        if num_heavy > 512:
            return InvalidSMILESEntry(
                index=index,
                smiles=smi_clean,
                reason=f"Molecule exceeds maximum allowed size ({num_heavy} heavy atoms > 512 max limit).",
            )

    except Exception as exc:
        return InvalidSMILESEntry(
            index=index,
            smiles=smi_clean,
            reason=f"RDKit validation failure: {str(exc)}",
        )

    return None


def validate_smiles_list(smiles_list: List[str]) -> Tuple[bool, List[InvalidSMILESEntry]]:
    """
    Validates an entire batch of SMILES strings.
    Returns (is_valid, list_of_invalid_entries).
    """
    invalid_entries: List[InvalidSMILESEntry] = []
    for idx, smi in enumerate(smiles_list):
        err = validate_single_smiles(smi, index=idx)
        if err is not None:
            invalid_entries.append(err)

    return len(invalid_entries) == 0, invalid_entries


def validate_target_spec_dict(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates the physical constraints and formatting of a target specification dictionary.
    """
    errors: List[str] = []

    if not isinstance(spec, dict):
        return False, ["Target specification must be a JSON dictionary or valid YAML content."]

    # Check for basic identifier
    if "group_name" not in spec and "description" not in spec:
        errors.append("Specification must contain at least 'group_name' or 'description'.")

    # Check rl_reward_targets
    targets = spec.get("rl_reward_targets", {})
    if targets and isinstance(targets, dict):
        if "max_molecular_weight" in targets:
            mw = targets["max_molecular_weight"]
            if not isinstance(mw, (int, float)) or mw <= 0:
                errors.append("rl_reward_targets.max_molecular_weight must be a positive number.")
        if "min_wall_pressure_bar" in targets:
            p = targets["min_wall_pressure_bar"]
            if not isinstance(p, (int, float)) or p < 0:
                errors.append("rl_reward_targets.min_wall_pressure_bar must be a non-negative number.")
        if "max_solvation_kcal" in targets:
            solv = targets["max_solvation_kcal"]
            if not isinstance(solv, (int, float)):
                errors.append("rl_reward_targets.max_solvation_kcal must be a numeric value.")

    # Check tensor_limits if provided
    limits = spec.get("tensor_limits", {})
    if limits and isinstance(limits, dict):
        if "max_sites" in limits:
            ms = limits["max_sites"]
            if not isinstance(ms, int) or ms <= 0 or ms > 512:
                errors.append("tensor_limits.max_sites must be a positive integer <= 512.")
        if "allowed_atomic_numbers" in limits:
            nums = limits["allowed_atomic_numbers"]
            if not isinstance(nums, list) or not all(isinstance(z, int) for z in nums):
                errors.append("tensor_limits.allowed_atomic_numbers must be a list of integer atomic numbers.")

    # Check scaffolds if provided
    scaffolds = spec.get("scaffolds", [])
    if isinstance(scaffolds, list):
        for s_idx, scaf in enumerate(scaffolds):
            if isinstance(scaf, dict):
                scaf_id = scaf.get("id", f"scaffold_{s_idx}")
                if "smarts" in scaf:
                    try:
                        q = Chem.MolFromSmarts(scaf["smarts"])
                        if q is None:
                            q = Chem.MolFromSmiles(scaf["smarts"], sanitize=False)
                        if q is None:
                            errors.append(f"Invalid SMARTS pattern in scaffold '{scaf_id}'.")
                    except Exception as e:
                        errors.append(f"SMARTS parsing error in scaffold '{scaf_id}': {str(e)}")
                if "attachment_points" in scaf:
                    ap = scaf["attachment_points"]
                    if not isinstance(ap, int) or ap <= 0:
                        errors.append(f"attachment_points in scaffold '{scaf_id}' must be a positive integer.")

    # Check building_blocks if provided
    bbs = spec.get("building_blocks", {})
    if bbs and isinstance(bbs, dict):
        for group_name, block_list in bbs.items():
            if isinstance(block_list, list):
                for b_idx, block in enumerate(block_list):
                    if isinstance(block, dict):
                        b_id = block.get("id", f"{group_name}_{b_idx}")
                        b_smi = block.get("smiles", "")
                        if not b_smi:
                            errors.append(f"Building block '{b_id}' in '{group_name}' is missing 'smiles'.")
                        else:
                            try:
                                m = Chem.MolFromSmiles(b_smi, sanitize=False)
                                if m is None:
                                    errors.append(f"Invalid SMILES in building block '{b_id}' ('{b_smi}').")
                            except Exception as e:
                                errors.append(f"SMILES parse failure in building block '{b_id}': {str(e)}")

    # Check assembly_rules if provided
    rules = spec.get("assembly_rules", {})
    if rules and isinstance(rules, dict):
        trans = rules.get("smarts_transform")
        if trans and isinstance(trans, str):
            try:
                from rdkit.Chem import AllChem

                rxn = AllChem.ReactionFromSmarts(trans)
                if rxn is None:
                    errors.append(f"Invalid SMIRKS/reaction transform in assembly_rules ('{trans}').")
            except Exception as e:
                errors.append(f"Reaction SMARTS parse failure: {str(e)}")

    return len(errors) == 0, errors


def run_intake_validation(
    smiles_list: Optional[List[str]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
) -> ValidationResponse:
    """
    Unified validation runner returning a structured ValidationResponse.
    """
    total_checked = 0
    all_invalid_smiles: List[InvalidSMILESEntry] = []
    all_spec_errors: List[str] = []

    if smiles_list is not None:
        total_checked += len(smiles_list)
        _, invalid_smiles = validate_smiles_list(smiles_list)
        all_invalid_smiles.extend(invalid_smiles)

    if target_spec is not None:
        total_checked += 1
        _, spec_errors = validate_target_spec_dict(target_spec)
        all_spec_errors.extend(spec_errors)

    is_valid = len(all_invalid_smiles) == 0 and len(all_spec_errors) == 0

    remediation = None
    if not is_valid:
        parts = []
        if all_invalid_smiles:
            first_err = all_invalid_smiles[0]
            parts.append(
                f"SMILES at index {first_err.index} ('{first_err.smiles}') failed: {first_err.reason}. Fix valence or syntax."
            )
        if all_spec_errors:
            parts.append(f"Specification error: {all_spec_errors[0]}. Check physical parameter ranges.")
        remediation = " | ".join(parts)

    return ValidationResponse(
        valid=is_valid,
        total_checked=total_checked,
        invalid_smiles=all_invalid_smiles,
        spec_errors=all_spec_errors,
        remediation=remediation,
    )
