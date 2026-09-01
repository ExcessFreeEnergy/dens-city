"""
Skill & Rule Proposer Agent: Wiki-Informed Skill Evolution for WikiSkill.

Formulates atomic, auditable modifications to skills and rules:
- Consults wiki/index.md and relevant pattern pages.
- Consults wiki/skill-impact.md to strictly avoid repeating rejected interventions.
- Creates or patches skills (with SKILL.md and PURPOSE.md) and rules.
- Generates proposal diffs ready for gating and validation.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dens_city.wikiskill.wiki_manager import PatchOperation, WikiManager


class ProposalAction(str, Enum):
    CREATE = "create"
    PATCH = "patch"
    NO_ACTION = "no_action"


@dataclass
class Proposal:
    proposal_id: str
    action: ProposalAction
    target_name: str  # skill or rule name
    target_file: Path
    skill_content: Optional[str] = None  # Full content for create or new state
    purpose_content: Optional[str] = None  # PURPOSE.md content for skills
    edits: List[PatchOperation] = field(default_factory=list)
    motivating_patterns: List[str] = field(default_factory=list)
    rationale: str = ""
    diff: str = ""


class SkillProposer:
    """Proposes atomic skill or rule updates guided by the Wiki."""

    def __init__(self, wiki_manager: WikiManager, workspace_root: Optional[Path | str] = None) -> None:
        self.wiki = wiki_manager
        self.workspace_root = Path(workspace_root) if workspace_root else wiki_manager.workspace_root
        self.skills_dir = self.workspace_root / ".agents" / "skills"
        self.rules_dir = self.workspace_root / ".agents" / "rules"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.rules_dir.mkdir(parents=True, exist_ok=True)

    def propose_from_pattern(self, pattern_name: str) -> Optional[Proposal]:
        """Synthesizes a proposed skill/rule based on a specific wiki pattern."""
        pattern_text = self.wiki.get_pattern(pattern_name)
        if not pattern_text:
            return None

        # Check if the pattern is already addressed in an existing skill
        skill_name = pattern_name.replace("pattern_", "").replace("_", "-")
        skill_dir = self.skills_dir / skill_name
        skill_file = skill_dir / "SKILL.md"

        # Check if a proposal for this was previously rejected
        reject_reason = self.wiki.is_proposal_previously_rejected(skill_name, pattern_text)
        if reject_reason:
            # Check if we should find an alternative approach or skip
            return None

        if not skill_file.exists():
            # Create new skill
            title = re.search(r"# Pattern:\s*(.+)", pattern_text)
            title_str = title.group(1).strip() if title else skill_name.replace("-", " ").title()

            prob_m = re.search(r"-\s+\*\*Problem\*\*:\s*(.+)", pattern_text)
            fix_m = re.search(r"-\s+\*\*Actionable Fix\*\*:\s*(.+)", pattern_text)
            prob = prob_m.group(1).strip() if prob_m else "Physical/numerical failure mode"
            fix = fix_m.group(1).strip() if fix_m else "Verified physical solution"

            skill_md = f"""---
name: {skill_name}
description: Procedural knowledge and physics invariants for {title_str}. Use when working on cDFT or simulation modules relating to this domain.
---

# {title_str}

## When to Apply
- When implementing or modifying routines involving {title_str.lower()}.
- When addressing errors relating to: {prob}.

## When NOT to Apply
- Standard high-level API calls that do not modify numerical physics equations.

## Instructions & Verified Rules
1. **Core Invariant**: {fix}
2. **Implementation Reference**: Consult [wiki/patterns/{pattern_name}.md](file://{self.wiki.patterns_dir}/{pattern_name}.md) for deep root cause analysis.
"""

            purpose_md = f"""# Purpose & Evolution History: {skill_name}

- **Origin**: Evolved from WikiSkill pattern `{pattern_name}`.
- **Motivating Patterns Addressed**:
  - `patterns/{pattern_name}.md`
- **Initial Rationale**: Codify the verified solution for {title_str} to prevent re-implementing known errors.
"""
            # Diff against empty file
            diff = "".join(
                difflib.unified_diff(
                    [],
                    skill_md.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=str(skill_file),
                )
            )

            import time

            prop_id = f"prop_{int(time.time() * 1000) % 1000000:06d}_{skill_name[:15]}"
            return Proposal(
                proposal_id=prop_id,
                action=ProposalAction.CREATE,
                target_name=skill_name,
                target_file=skill_file,
                skill_content=skill_md,
                purpose_content=purpose_md,
                motivating_patterns=[pattern_name],
                rationale=f"Created skill `{skill_name}` to enforce verified fix for {title_str}.",
                diff=diff,
            )

        return None

    def apply_proposal(self, proposal: Proposal) -> Dict[str, Any]:
        """Applies the proposal to disk and returns file paths modified."""
        if proposal.action == ProposalAction.NO_ACTION:
            return {"status": "no_action"}

        if proposal.action == ProposalAction.CREATE:
            proposal.target_file.parent.mkdir(parents=True, exist_ok=True)
            proposal.target_file.write_text(proposal.skill_content or "", encoding="utf-8")
            if proposal.purpose_content:
                purpose_file = proposal.target_file.parent / "PURPOSE.md"
                purpose_file.write_text(proposal.purpose_content, encoding="utf-8")
            return {
                "status": "created",
                "target_file": str(proposal.target_file),
                "proposal_id": proposal.proposal_id,
            }

        if proposal.action == ProposalAction.PATCH:
            content = proposal.target_file.read_text(encoding="utf-8")
            for edit in proposal.edits:
                if edit.op == "append":
                    content = content.rstrip() + "\n\n" + edit.content.strip() + "\n"
                elif edit.op == "replace" and edit.target:
                    content = content.replace(edit.target, edit.content, 1)
                elif edit.op == "insert_after" and edit.target:
                    pos = content.find(edit.target) + len(edit.target)
                    content = content[:pos] + "\n" + edit.content + content[pos:]

            proposal.target_file.write_text(content, encoding="utf-8")
            return {
                "status": "patched",
                "target_file": str(proposal.target_file),
                "proposal_id": proposal.proposal_id,
            }

        return {"status": "unknown"}
