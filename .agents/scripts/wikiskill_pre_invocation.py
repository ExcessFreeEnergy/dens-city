#!/usr/bin/env python3
"""
Antigravity PreInvocation Hook for WikiSkill.

Triggered before the LLM model is called on every turn.
1. Error Monitoring: If recent command failures are recorded in raw/traces/,
   injects a high-priority warning to review wiki/index.md and skill-impact.md.
2. Semantic / Keyword Pattern Routing: Inspects the latest user input from
   transcript.jsonl and automatically surfaces the top relevant Wiki patterns
   directly into the model's context as an ephemeral hint.
"""

import json
import re
import sys
from pathlib import Path


def extract_latest_user_text(transcript_path: Path) -> str:
    """Extracts the latest user message text from the conversation transcript."""
    if not transcript_path.exists():
        return ""
    try:
        user_texts = []
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "USER_INPUT":
                        content = entry.get("content", "")
                        if content:
                            user_texts.append(str(content))
                except Exception:
                    continue
        return user_texts[-1] if user_texts else ""
    except Exception:
        return ""


def find_matching_patterns(user_text: str, patterns_dir: Path) -> list[tuple[str, str]]:
    """Matches user query against pattern filenames, summaries, and keywords."""
    if not user_text or not patterns_dir.exists():
        return []

    tokens = set(re.findall(r"[a-zA-Z0-9_]{3,}", user_text.lower()))
    if not tokens:
        return []

    matches = []
    for pattern_file in sorted(patterns_dir.glob("pattern_*.md")):
        name = pattern_file.stem
        # Read first 15 lines of pattern file to find Problem and Summary
        summary_lines = []
        try:
            with open(pattern_file, "r", encoding="utf-8") as f:
                for _ in range(15):
                    line = f.readline()
                    if not line:
                        break
                    summary_lines.append(line.strip())
        except Exception:
            continue

        text_content = (name + " " + " ".join(summary_lines)).lower()
        # Count overlapping tokens
        score = sum(1 for t in tokens if t in text_content)
        if score > 0:
            # Extract Problem or summary
            problem = ""
            for s_line in summary_lines:
                if s_line.startswith("- **Problem**:") or s_line.startswith("- **Fix**:"):
                    problem = s_line
                    break

            matches.append((score, name, problem, pattern_file))

    # Sort by relevance score descending
    matches.sort(key=lambda x: x[0], reverse=True)
    return [(m[1], m[2]) for m in matches[:2] if m[0] >= 2 or (m[0] >= 1 and len(tokens) <= 3)]


def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except Exception:
        payload = {}

    inject_steps = []
    try:
        workspace_paths = payload.get("workspacePaths", [])
        if workspace_paths:
            workspace_root = Path(workspace_paths[0])
        else:
            workspace_root = Path(__file__).resolve().parent.parent.parent

        # 1. Check for recent test or command failures in raw/traces/
        traces_dir = workspace_root / ".agents" / "wikiskill" / "raw" / "traces"
        if traces_dir.exists():
            trace_files = sorted(traces_dir.glob("trace_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if trace_files:
                latest = trace_files[0]
                with open(latest, "r", encoding="utf-8") as fp:
                    data = json.load(fp)

                if not data.get("passed", True):
                    fail_summary = data.get("summary", "Command failed")
                    inject_steps.append(
                        {
                            "ephemeralMessage": (
                                f"⚠️ [WikiSkill Notice] Recent failure recorded: {fail_summary[:100]}.\n"
                                "Before modifying code, check `.agents/wikiskill/wiki/index.md` for known patterns "
                                "and `.agents/wikiskill/wiki/skill-impact.md` to avoid repeating previously rejected approaches."
                            )
                        }
                    )

        # 2. Semantic Pattern Routing via Transcript User Query
        transcript_path_str = payload.get("transcriptPath")
        if transcript_path_str:
            user_query = extract_latest_user_text(Path(transcript_path_str))
            if user_query:
                patterns_dir = workspace_root / ".agents" / "wikiskill" / "wiki" / "patterns"
                matched = find_matching_patterns(user_query, patterns_dir)
                if matched:
                    hints = "\n".join(
                        f"- `{name}`: {summary} (File: `.agents/wikiskill/wiki/patterns/{name}.md`)"
                        for name, summary in matched
                    )
                    inject_steps.append(
                        {
                            "ephemeralMessage": (
                                f"💡 [WikiSkill Pattern Hint] The user query matches known wiki patterns:\n{hints}\n"
                                "Consult these pattern files to enforce verified physics invariants and compiler idioms."
                            )
                        }
                    )

    except Exception:
        pass

    print(json.dumps({"injectSteps": inject_steps}))


if __name__ == "__main__":
    main()
