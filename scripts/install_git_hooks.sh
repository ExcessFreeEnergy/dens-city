#!/usr/bin/env bash
# ==============================================================================
# Installer for dens-city Git Quality & Performance Ratchet Hooks
#
# Installs:
# - .git/hooks/pre-commit
# - .git/hooks/pre-merge-commit
# - .git/hooks/pre-push
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GIT_DIR="$(git rev-parse --git-dir 2>/dev/null || echo "$REPO_ROOT/.git")"
HOOKS_DIR="$GIT_DIR/hooks"

mkdir -p "$HOOKS_DIR"

GATE_SCRIPT="$REPO_ROOT/scripts/pre_commit_solvatum_gate.sh"
chmod +x "$GATE_SCRIPT"

# 1. Install pre-commit hook
PRE_COMMIT_HOOK="$HOOKS_DIR/pre-commit"
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/usr/bin/env bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
exec "$REPO_ROOT/scripts/pre_commit_solvatum_gate.sh" "$@"
EOF
chmod +x "$PRE_COMMIT_HOOK"
echo "✅ Installed .git/hooks/pre-commit"

# 2. Install pre-merge-commit hook
PRE_MERGE_HOOK="$HOOKS_DIR/pre-merge-commit"
cat > "$PRE_MERGE_HOOK" << 'EOF'
#!/usr/bin/env bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
exec "$REPO_ROOT/scripts/pre_commit_solvatum_gate.sh" "$@"
EOF
chmod +x "$PRE_MERGE_HOOK"
echo "✅ Installed .git/hooks/pre-merge-commit"

# 3. Install pre-push hook (intercepts git push origin master)
PRE_PUSH_HOOK="$HOOKS_DIR/pre-push"
cat > "$PRE_PUSH_HOOK" << 'EOF'
#!/usr/bin/env bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
REMOTE="$1"
URL="$2"

# Read standard input for ref update information
TARGETS_MASTER=false
while read -r local_ref local_oid remote_ref remote_oid; do
    if [[ "$remote_ref" =~ refs/heads/master$ ]] || [[ "$remote_ref" =~ refs/heads/main$ ]]; then
        TARGETS_MASTER=true
        break
    fi
done

if [ "$TARGETS_MASTER" = true ]; then
    echo "🔍 [pre-push] Push targets 'master'. Triggering Solvatum Ratchet Gate verification..."
    exec "$REPO_ROOT/scripts/pre_commit_solvatum_gate.sh"
else
    exit 0
fi
EOF
chmod +x "$PRE_PUSH_HOOK"
echo "✅ Installed .git/hooks/pre-push"

echo ""
echo "🎉 All git quality hooks successfully installed and active!"
echo "   - Feature branch commits skip immediately."
echo "   - Commits, merges, and pushes targeting 'master' enforce the Solvatum ratchet gate."
