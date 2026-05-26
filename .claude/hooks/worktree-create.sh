#!/usr/bin/env bash
# WorktreeCreate hook for gradient-server.
#
# Replaces Claude Code's default git worktree creation to:
#   1. Use feature/<name> branch naming from origin/dev (gitflow).
#   2. Copy files listed in .worktreeinclude (gitignored locals: .env, data/, ...).
#   3. Create a per-worktree Postgres database (gradient_<slug>) in MAIN's
#      docker-compose postgres container — no extra container per worktree.
#   4. Rewrite the copied .env so POSTGRES_DB / DATABASE_URL point at it.
#   5. Run `alembic upgrade head` against the new DB.
#
# Input  (stdin JSON): { "name": "<slug>", "cwd": "<project-root>", ... }
# Output (stdout):     Absolute path to the created worktree directory.
# All informational output goes to stderr (stdout is reserved for the path).

set -euo pipefail

PG_SERVICE="postgres"
PG_USER="gradient"
PG_PASSWORD_VAR="gradient_secret"
DEFAULT_BASE_BRANCH="dev"

log() { printf '[wt-create] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

command -v jq >/dev/null || die "jq required (brew install jq)"

# --- Parse hook input ---
INPUT="$(cat)"
NAME="$(printf '%s' "$INPUT" | jq -r '.name // empty')"
CWD="$(printf '%s' "$INPUT" | jq -r '.cwd // empty')"
[ -n "$NAME" ] || die "missing .name on stdin"

# --- Resolve MAIN repo root ---
if [ -n "$CWD" ] && [ -d "$CWD" ]; then
  MAIN="$CWD"
elif [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  MAIN="$CLAUDE_PROJECT_DIR"
else
  MAIN="$(git rev-parse --show-toplevel)"
fi

WT_DIR="$MAIN/.claude/worktrees/$NAME"
BRANCH_NAME="$NAME"

# --- Slug for DB name (lowercase, [a-z0-9_] only) ---
SLUG="$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')"
[ -n "$SLUG" ] || die "slug empty after sanitization"
DB_NAME="gradient_${SLUG}"
log "name=$NAME slug=$SLUG db=$DB_NAME wt=$WT_DIR"

# --- Ensure .claude/worktrees is ignored ---
# Repo already gitignores .claude/, so worktrees nested under it inherit that.
# Add an explicit entry only if .claude/ itself is somehow not ignored.
if ! (cd "$MAIN" && git check-ignore -q .claude/worktrees 2>/dev/null); then
  if ! grep -qF '.claude/worktrees' "$MAIN/.gitignore" 2>/dev/null; then
    [ -s "$MAIN/.gitignore" ] && [ -n "$(tail -c 1 "$MAIN/.gitignore")" ] && echo "" >> "$MAIN/.gitignore"
    echo ".claude/worktrees" >> "$MAIN/.gitignore"
    log "added .claude/worktrees to .gitignore"
  fi
fi

# --- Determine base branch ---
# Prefer DEFAULT_BASE_BRANCH (dev) over origin/HEAD, which points at main.
# Resolve to a concrete start-point ref, preferring origin/<branch> when it
# exists, else the local branch (origin/dev is not always pushed).
BASE_BRANCH=""
BASE_REF=""
for candidate in "$DEFAULT_BASE_BRANCH" main master; do
  if (cd "$MAIN" && git show-ref --verify --quiet "refs/remotes/origin/$candidate" 2>/dev/null); then
    BASE_BRANCH="$candidate"
    BASE_REF="origin/$candidate"
    break
  elif (cd "$MAIN" && git show-ref --verify --quiet "refs/heads/$candidate" 2>/dev/null); then
    BASE_BRANCH="$candidate"
    BASE_REF="$candidate"
    break
  fi
done
BASE_BRANCH="${BASE_BRANCH:-$DEFAULT_BASE_BRANCH}"
BASE_REF="${BASE_REF:-$DEFAULT_BASE_BRANCH}"

# --- Create worktree ---
mkdir -p "$(dirname "$WT_DIR")"

if [ -d "$WT_DIR" ]; then
  die "worktree dir already exists: $WT_DIR (run worktree-remove.sh first)"
fi

if (cd "$MAIN" && git show-ref --verify --quiet "refs/heads/$BRANCH_NAME" 2>/dev/null); then
  log "using existing branch $BRANCH_NAME"
  (cd "$MAIN" && git worktree add "$WT_DIR" "$BRANCH_NAME") >&2
else
  log "creating branch $BRANCH_NAME from $BASE_REF"
  (cd "$MAIN" && git worktree add -b "$BRANCH_NAME" "$WT_DIR" "$BASE_REF") >&2
fi

# --- Copy .worktreeinclude files ---
# Uses git's gitignore pattern engine via --exclude-from. All gitignore rules
# (globs, **, !, trailing /) work natively. Bulk-copies via tar pipe.
INCLUDE_FILE="$MAIN/.worktreeinclude"
if [ -f "$INCLUDE_FILE" ]; then
  file_list="$(cd "$MAIN" && git ls-files --others --ignored --exclude-from="$INCLUDE_FILE" 2>/dev/null || true)"
  if [ -z "$file_list" ]; then
    log "no files matched .worktreeinclude"
  else
    count="$(printf '%s\n' "$file_list" | wc -l | tr -d ' ')"
    log "copying $count file(s) from .worktreeinclude"
    (cd "$MAIN" && git ls-files -z --others --ignored --exclude-from="$INCLUDE_FILE" 2>/dev/null) \
      | tar -C "$MAIN" --null -T - -cf - \
      | tar -C "$WT_DIR" -xf -
    printf '%s\n' "$file_list" | awk -F/ '{print ($2 ? $1"/" : $0)}' | sort | uniq -c \
      | while read -r cnt path; do
          if [ "$cnt" -eq 1 ] && [ "${path%/}" = "$path" ]; then
            log "  + $path"
          else
            log "  + $path ($cnt files)"
          fi
        done
  fi
else
  log "no .worktreeinclude found — skipping file copy"
fi

# --- Per-worktree Postgres database ---
ENV_FILE="$WT_DIR/.env"
[ -f "$ENV_FILE" ] || die ".env not copied — add it to .worktreeinclude"

log "creating database $DB_NAME"
EXISTS="$(cd "$MAIN" && docker compose exec -T "$PG_SERVICE" psql -U "$PG_USER" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" 2>/dev/null || true)"
if [ "$EXISTS" = "1" ]; then
  log "database $DB_NAME already exists, skipping CREATE"
else
  (cd "$MAIN" && docker compose exec -T "$PG_SERVICE" psql -U "$PG_USER" -d postgres \
    -c "CREATE DATABASE \"${DB_NAME}\" OWNER \"${PG_USER}\";") >&2 \
    || die "CREATE DATABASE failed"
fi

# --- Rewrite copied .env ---
log "rewriting $ENV_FILE → POSTGRES_DB=$DB_NAME"
sed -i '' -E "s|^POSTGRES_DB=.*|POSTGRES_DB=${DB_NAME}|" "$ENV_FILE"
sed -i '' -E \
  "s|^DATABASE_URL=.*|DATABASE_URL=postgresql+asyncpg://${PG_USER}:${PG_PASSWORD_VAR}@localhost:5432/${DB_NAME}|" \
  "$ENV_FILE"

# --- Run migrations ---
log "alembic upgrade head"
if [ -x "$MAIN/.venv/bin/alembic" ]; then
  (cd "$WT_DIR" && set -a && . "$ENV_FILE" && set +a && "$MAIN/.venv/bin/alembic" upgrade head) >&2 \
    || die "alembic upgrade head failed"
else
  (cd "$WT_DIR" && set -a && . "$ENV_FILE" && set +a && uv run --project "$MAIN" alembic upgrade head) >&2 \
    || die "alembic upgrade head failed"
fi

# --- Output worktree path for Claude Code to consume ---
echo "$WT_DIR"
