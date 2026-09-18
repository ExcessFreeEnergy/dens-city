#!/usr/bin/env bash
# ==============================================================================
# Solvatum E2E Pre-Commit & Pre-Merge Ratchet Gate Hook
#
# Enforces that all future merges into `master` and commits on `master`
# run the full 5,952-material Solvatum benchmark and meet or exceed the
# established speed and accuracy baseline standard.
#
# Branch scoping:
# - Feature branch commits (e.g. feat/*, fix/*) are skipped instantly (exit 0).
# - Commits on `master` or merges targeting `master` trigger full verification.
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CURRENT_BRANCH="${GIT_BRANCH_OVERRIDE:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")}"
GIT_DIR="$(git rev-parse --git-dir 2>/dev/null || echo "$REPO_ROOT/.git")"

# Check if an explicit bypass is requested
if [ "${SKIP_SOLVATUM_GATE:-0}" = "1" ]; then
    echo "⚠️  [Solvatum Ratchet Gate] Warning: SKIP_SOLVATUM_GATE=1 detected. Bypassing ratchet check."
    exit 0
fi

# Detect whether we are committing on master or merging into master
IS_MERGE=false
if [ -f "$GIT_DIR/MERGE_HEAD" ]; then
    IS_MERGE=true
fi

TARGET_IS_MASTER=false
if [ "$CURRENT_BRANCH" = "master" ] || [ "$CURRENT_BRANCH" = "main" ]; then
    TARGET_IS_MASTER=true
fi

# If on a feature branch and not merging into master, allow fast micro-commits!
if [ "$TARGET_IS_MASTER" = false ]; then
    echo "ℹ️  [Solvatum Ratchet Gate] Active branch is '$CURRENT_BRANCH' (not master). Skipping full E2E gate check."
    exit 0
fi

echo "================================================================================"
echo "  Executing Solvatum E2E Pre-Commit & Pre-Merge Ratchet Gate on branch: $CURRENT_BRANCH"
echo "================================================================================"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v uv >/dev/null 2>&1; then
        PYTHON_BIN="uv run python"
    else
        PYTHON_BIN="python3"
    fi
fi

# Determine whether to use an existing results directory or run e2e
GATE_ARGS=()
if [ -n "$SOLVATUM_RESULTS_DIR" ] && [ -d "$SOLVATUM_RESULTS_DIR" ]; then
    echo "Using pre-computed results directory: $SOLVATUM_RESULTS_DIR"
    GATE_ARGS+=(--results-dir "$SOLVATUM_RESULTS_DIR")
else
    echo "Running complete Solvatum end-to-end simulation across 5,952 materials..."
    GATE_ARGS+=(--run-e2e --batch-size 64)
fi

# Ensure Tinygrad hardware queue watchdog allows full graph compilation on large polyatomics
export HCQDEV_WAIT_TIMEOUT_MS="${HCQDEV_WAIT_TIMEOUT_MS:-300000}"

set +e
$PYTHON_BIN -m dens_city.utils.ratchet_gate "${GATE_ARGS[@]}"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ [Solvatum Ratchet Gate] MERGE BLOCKED!"
    echo "   The simulation failed to meet the speed and accuracy ratchet baseline."
    echo "   See audit report at data/solvatum_ratchet_gate_report.md for exact deltas."
    echo ""
    exit 1
else
    echo ""
    echo "✅ [Solvatum Ratchet Gate] Standard verified successfully. Merge approved."
    echo ""
    exit 0
fi
