"""
Wiki Manager: Persistent, Compounding Knowledge Base Layer for WikiSkill.

Manages:
- .agents/wikiskill/wiki/index.md (Catalog of patterns: PROBLEM + ROOT CAUSE + FIX)
- .agents/wikiskill/wiki/logs.md (Chronological iteration history)
- .agents/wikiskill/wiki/skill-impact.md (Audit tracker of all proposals and outcomes)
- .agents/wikiskill/wiki/patterns/*.md (In-depth pattern pages with root cause analyses)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PatchOperation:
    op: str  # "append", "replace", "insert_after"
    content: str
    target: Optional[str] = None  # Exact text to match for replace or insert_after

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target is not None:
            d["target"] = self.target
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PatchOperation:
        return cls(
            op=data["op"],
            content=data["content"],
            target=data.get("target"),
        )


@dataclass
class WikiPattern:
    name: str  # filename without .md
    title: str
    problem: str
    root_cause: str
    actionable_fix: str
    anti_patterns: List[str] = field(default_factory=list)
    code_example: Optional[str] = None
    related_skills: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    raw_content: Optional[str] = None

    def render_markdown(self) -> str:
        if self.raw_content:
            return self.raw_content

        anti_pat_str = "\n".join(f"- ❌ **Anti-Pattern**: {ap}" for ap in self.anti_patterns)
        skills_str = ", ".join(self.related_skills) if self.related_skills else "None"
        code_block = f"\n```python\n{self.code_example}\n```\n" if self.code_example else ""

        return f"""# Pattern: {self.title}

## Summary
- **Problem**: {self.problem}
- **Root Cause**: {self.root_cause}
- **Actionable Fix**: {self.actionable_fix}
- **Related Skills / Modules**: {skills_str}

## Deep Root Cause Analysis
{self.root_cause}

## Verified Solution & Action Rules
{self.actionable_fix}
{code_block}
## Anti-Patterns to Avoid
{anti_pat_str if anti_pat_str else "- None documented."}
"""


class WikiManager:
    """Manages reading, updating, and patching the persistent Wiki layer."""

    def __init__(self, workspace_root: Optional[Path | str] = None) -> None:
        if workspace_root is None:
            self.workspace_root = Path.cwd()
        else:
            self.workspace_root = Path(workspace_root)

        self.wiki_dir = self.workspace_root / ".agents" / "wikiskill" / "wiki"
        self.patterns_dir = self.wiki_dir / "patterns"
        self.index_file = self.wiki_dir / "index.md"
        self.log_file = self.wiki_dir / "logs.md"
        self.impact_file = self.wiki_dir / "skill-impact.md"

        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_core_files()

    def _ensure_core_files(self) -> None:
        """Ensures index.md, logs.md, and skill-impact.md exist."""
        if not self.index_file.exists():
            self.index_file.write_text(
                "# Wiki Pattern Catalog\n\n"
                "This catalog documents known physics invariants, algorithmic patterns, failure modes, and verified solutions for `dens-city`.\n\n"
                "## Active Patterns\n\n",
                encoding="utf-8",
            )
        if not self.log_file.exists():
            self.log_file.write_text(
                "# WikiSkill Evolution Log\n\n"
                "Chronological log of evolution iterations, test passes/failures, and diagnostic findings.\n\n",
                encoding="utf-8",
            )
        if not self.impact_file.exists():
            self.impact_file.write_text(
                "# Skill Impact & Proposal Audit Tracker\n\n"
                "Record of all proposed skill/code modifications, evaluation outcomes, and diffs.\n"
                "**Crucial Rule**: Consult this file before proposing changes. NEVER repeat rejected approaches.\n\n"
                "| Timestamp | Proposal ID | Target Skill / File | Score Change | Outcome | Summary |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
                encoding="utf-8",
            )

    def save_pattern(self, pattern: WikiPattern) -> Path:
        """Saves a pattern document to wiki/patterns/<name>.md and updates index."""
        p_path = self.patterns_dir / f"{pattern.name}.md"
        content = pattern.render_markdown()
        p_path.write_text(content, encoding="utf-8")
        self.rebuild_index()
        return p_path

    def get_pattern(self, name: str) -> Optional[str]:
        """Reads the full markdown content of a pattern."""
        clean_name = name.replace(".md", "")
        p_path = self.patterns_dir / f"{clean_name}.md"
        if not p_path.exists():
            return None
        return p_path.read_text(encoding="utf-8")

    def patch_pattern(self, name: str, edits: List[PatchOperation]) -> bool:
        """Applies patch operations to an existing pattern file."""
        clean_name = name.replace(".md", "")
        p_path = self.patterns_dir / f"{clean_name}.md"
        if not p_path.exists():
            return False

        content = p_path.read_text(encoding="utf-8")
        for edit in edits:
            if edit.op == "append":
                content = content.rstrip() + "\n\n" + edit.content.strip() + "\n"
            elif edit.op == "replace":
                if not edit.target:
                    continue
                if edit.target not in content:
                    raise ValueError(f"Replace target text not found in pattern {clean_name}: '{edit.target[:50]}...'")
                content = content.replace(edit.target, edit.content, 1)
            elif edit.op == "insert_after":
                if not edit.target:
                    continue
                if edit.target not in content:
                    raise ValueError(f"Insert target text not found in pattern {clean_name}: '{edit.target[:50]}...'")
                pos = content.find(edit.target) + len(edit.target)
                content = content[:pos] + "\n" + edit.content + content[pos:]

        p_path.write_text(content, encoding="utf-8")
        self.rebuild_index()
        return True

    def list_patterns(self) -> List[str]:
        """Lists all pattern names."""
        return sorted([f.stem for f in self.patterns_dir.glob("*.md")])

    def rebuild_index(self) -> str:
        """Rebuilds wiki/index.md by parsing pattern files for summary lines."""
        lines = [
            "# Wiki Pattern Catalog\n",
            "This catalog documents known physics invariants, algorithmic patterns, failure modes, and verified solutions for `dens-city`.\n",
            "## Active Patterns\n",
        ]

        for p_file in sorted(self.patterns_dir.glob("*.md")):
            content = p_file.read_text(encoding="utf-8")
            stem = p_file.stem
            # Parse problem and fix from summary
            prob_match = re.search(r"-\s+\*\*Problem\*\*:\s*(.+)", content)
            fix_match = re.search(r"-\s+\*\*Actionable Fix\*\*:\s*(.+)", content)
            cause_match = re.search(r"-\s+\*\*Root Cause\*\*:\s*(.+)", content)

            prob = prob_match.group(1).strip() if prob_match else "Problem documented"
            cause = cause_match.group(1).strip() if cause_match else ""
            fix = fix_match.group(1).strip() if fix_match else "Fix documented"

            rel_link = f"patterns/{p_file.name}"
            summary_line = f"- [{stem}]({rel_link}): **Problem**: {prob} | **Cause**: {cause} | **Fix**: {fix}"
            lines.append(summary_line)

        new_index = "\n".join(lines) + "\n"
        self.index_file.write_text(new_index, encoding="utf-8")
        return new_index

    def get_index(self) -> str:
        """Returns the content of wiki/index.md."""
        if not self.index_file.exists():
            return self.rebuild_index()
        return self.index_file.read_text(encoding="utf-8")

    def append_log(
        self,
        summary: str,
        iteration: Optional[int] = None,
        test_score: Optional[float] = None,
        findings: Optional[str] = None,
    ) -> None:
        """Appends a chronological entry to wiki/logs.md."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        iter_str = f"Iteration {iteration}" if iteration is not None else "Manual Step"
        score_str = f"Score: {test_score:.4f}" if test_score is not None else "Score: N/A"

        entry = f"""
### [{ts}] {iter_str} ({score_str})
- **Summary**: {summary}
"""
        if findings:
            entry += f"- **Diagnostic Findings**:\n{findings.strip()}\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def record_skill_impact(
        self,
        proposal_id: str,
        target_skill_or_file: str,
        diff: str,
        score_before: float,
        score_after: float,
        outcome: str,  # "Accepted" or "Rejected"
        rationale: str,
    ) -> None:
        """Appends a proposal and outcome record to wiki/skill-impact.md."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        delta = score_after - score_before
        delta_str = f"{score_before:.3f} -> {score_after:.3f} ({delta:+.3f})"

        table_row = (
            f"| {ts} | `{proposal_id}` | `{target_skill_or_file}` | {delta_str} | **{outcome}** | {rationale} |\n"
        )

        detail_section = f"""
<details>
<summary>Proposal <code>{proposal_id}</code> ({outcome}) - {target_skill_or_file}</summary>

- **Timestamp**: {ts}
- **Target**: `{target_skill_or_file}`
- **Outcome**: {outcome}
- **Score Change**: {delta_str}
- **Rationale**: {rationale}

```diff
{diff}
```
</details>
"""
        content = self.impact_file.read_text(encoding="utf-8")
        if table_row not in content:
            # Append table row and details
            with open(self.impact_file, "a", encoding="utf-8") as f:
                f.write(table_row)
                f.write(detail_section)

    def get_skill_impact_history(self) -> str:
        """Returns the full content of wiki/skill-impact.md."""
        return self.impact_file.read_text(encoding="utf-8")

    def is_proposal_previously_rejected(self, target_name: str, candidate_text: str) -> Optional[str]:
        """Scans skill-impact.md to check if an identical or very similar approach was previously rejected."""
        history = self.get_skill_impact_history()
        if "Rejected" not in history:
            return None

        clean_cand = re.sub(r"\s+", " ", candidate_text).strip().lower()
        if len(clean_cand) < 10:
            return None

        # Check sections marked with Rejected
        rejected_blocks = re.findall(
            r"<details>.*?<summary>Proposal <code>(.*?)</code> \(Rejected\) - (.*?)</summary>.*?```diff(.*?)```.*?</details>",
            history,
            re.DOTALL,
        )

        for prop_id, tgt, diff in rejected_blocks:
            if target_name.lower() in tgt.lower() or tgt.strip().lower() == target_name.strip().lower():
                clean_diff = re.sub(r"\s+", " ", diff).strip().lower()
                # Substring check for significant snippet
                for diff_line in diff.splitlines():
                    cleaned_line = diff_line.lstrip("+- ").strip().lower()
                    if len(cleaned_line) > 15 and cleaned_line in clean_cand:
                        return f"Proposal contains line `{cleaned_line}` from rejected proposal `{prop_id}`."

                # Token overlap check
                words_cand = set(re.findall(r"\w+", clean_cand))
                words_diff = set(re.findall(r"\w+", clean_diff))
                if words_cand and words_diff:
                    overlap = len(words_cand.intersection(words_diff)) / len(words_cand)
                    if overlap > 0.40:
                        return f"Proposal matches rejected proposal `{prop_id}` ({overlap * 100:.1f}% keyword overlap)."

        return None
