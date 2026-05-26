#!/usr/bin/env bash
# WorktreeRemove hook for gradient-server.
#
# Mirrors worktree-create.sh: drops the per-worktree Postgres database, then
# removes the git worktree. Falls back to `git worktree prune` + rm -rf if
# `git worktree remove` fails (e.g. file locks).
#
# Input (stdin JSON): { "worktree_path": "<absolute-path>", ... }

set -euo pipefail

PG_SERVICE="postgres"
PG_USER="gradient"

log() { printf '[wt-remove] %s\n' "$*" >&2; }

command -v jq >/dev/null || { log "ERROR: jq required"; exit 1; }

INPUT="$(cat)"
WT_PATH="$(printf '%s' "$INPUT" | jq -r '.worktree_path // empty')"
if [ -z "$WT_PATH" ] || [ "$WT_PATH" = "null" ]; then
  log "no worktree_path provided, nothing to do"
  exit 0
fi

# --- Resolve MAIN repo (for docker compose) ---
MAIN="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$MAIN" ] && [ -d "$WT_PATH/.git" -o -f "$WT_PATH/.git" ]; then
  COMMON="$(git -C "$WT_PATH" rev-parse --git-common-dir 2>/dev/null || true)"
  if [ -n "$COMMON" ]; then
    MAIN="$(cd "$WT_PATH" && cd "$COMMON" && cd .. && pwd)"
  fi
fi
[ -n "$MAIN" ] || MAIN="/Users/rcantu/Code/gradient-server"

# --- Derive slug → DB name from worktree dir basename ---
NAME="$(basename "$WT_PATH")"
SLUG="$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')"
DB_NAME="gradient_${SLUG}"
log "worktree=$WT_PATH db=$DB_NAME"

# --- Drop per-worktree database ---
# Terminate other sessions first; DROP fails if connections remain.
if [ -n "$SLUG" ] && [ "$SLUG" != "gradient" ]; then
  log "dropping database $DB_NAME"
  (cd "$MAIN" && docker compose exec -T "$PG_SERVICE" psql -U "$PG_USER" -d postgres \
    -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${DB_NAME}' AND pid <> pg_backend_pid();" \
    >/dev/null 2>&1 || true)
  (cd "$MAIN" && docker compose exec -T "$PG_SERVICE" psql -U "$PG_USER" -d postgres \
    -c "DROP DATABASE IF EXISTS \"${DB_NAME}\";" >&2) \
    || log "WARN: DROP DATABASE $DB_NAME failed (continuing)"
else
  log "refusing to drop DB for empty/main slug ($SLUG)"
fi

# --- Remove the worktree ---
if [ ! -d "$WT_PATH" ]; then
  log "worktree dir already gone"
  exit 0
fi

if (cd "$MAIN" && git worktree remove "$WT_PATH" --force >&2 2>/dev/null); then
  log "removed worktree $WT_PATH"
else
  log "git worktree remove failed, falling back to manual cleanup"
  (cd "$MAIN" && git worktree prune >&2 2>/dev/null) || true
  rm -rf "$WT_PATH" 2>/dev/null || true
fi

exit 0
