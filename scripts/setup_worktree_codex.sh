#!/usr/bin/env bash
set -euo pipefail

SOURCE_DB="${SOURCE_DB:-gradient}"
ENV_FILE="${ENV_FILE:-.env}"
WORKTREE_PATH="${CODEX_WORKTREE_PATH:-$(pwd)}"
SOURCE_TREE_PATH="${CODEX_SOURCE_TREE_PATH:-}"

cd "$WORKTREE_PATH"

if [ -n "$SOURCE_TREE_PATH" ]; then
  if [ ! -d "$SOURCE_TREE_PATH" ]; then
    echo "CODEX_SOURCE_TREE_PATH does not exist: ${SOURCE_TREE_PATH}" >&2
    exit 1
  fi

  if [ -d "$SOURCE_TREE_PATH/data" ]; then
    echo "Copying data/ from ${SOURCE_TREE_PATH} to ${WORKTREE_PATH}..."
    mkdir -p "$WORKTREE_PATH/data"
    cp -R "$SOURCE_TREE_PATH/data/." "$WORKTREE_PATH/data/"
  else
    echo "No data/ directory found in ${SOURCE_TREE_PATH}; skipping data copy."
  fi
fi

if [ -n "$SOURCE_TREE_PATH" ] && [ -f "$SOURCE_TREE_PATH/.env" ]; then
  ENV_SOURCE_FILE="$SOURCE_TREE_PATH/.env"
else
  ENV_SOURCE_FILE="$WORKTREE_PATH/.env.example"
fi

if [ ! -f "$ENV_SOURCE_FILE" ]; then
  echo "Environment source file does not exist: ${ENV_SOURCE_FILE}" >&2
  exit 1
fi

echo "Creating ${ENV_FILE} from ${ENV_SOURCE_FILE}..."
if [ "$ENV_SOURCE_FILE" != "$WORKTREE_PATH/$ENV_FILE" ]; then
  cp "$ENV_SOURCE_FILE" "$ENV_FILE"
fi

BRANCH_NAME="${BRANCH_NAME:-$(git branch --show-current)}"
if [ -z "$BRANCH_NAME" ]; then
  BRANCH_NAME="$(git rev-parse --short HEAD)"
fi

DB_SUFFIX="$(
  printf '%s' "$BRANCH_NAME" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9_]+/_/g; s/^_+//; s/_+$//'
)"
if [ -z "$DB_SUFFIX" ]; then
  DB_SUFFIX="worktree"
fi

TARGET_DB="${TARGET_DB:-gradient_${DB_SUFFIX}}"

for db_name in "$SOURCE_DB" "$TARGET_DB"; do
  if ! printf '%s' "$db_name" | grep -Eq '^[a-zA-Z_][a-zA-Z0-9_]*$'; then
    echo "Invalid database name: ${db_name}" >&2
    echo "Use only letters, numbers, and underscores; the first character must not be a number." >&2
    exit 1
  fi
done

env_value() {
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE"
}

set_env_value() {
  key="$1"
  value="$2"
  tmp_env="$(mktemp)"
  awk -v key="$key" -v value="$value" '
    BEGIN { wrote = 0 }
    $0 ~ "^[#[:space:]]*" key "=" {
      print key "=" value
      wrote = 1
      next
    }
    { print }
    END {
      if (!wrote) print key "=" value
    }
  ' "$ENV_FILE" > "$tmp_env"
  mv "$tmp_env" "$ENV_FILE"
}

POSTGRES_USER="${POSTGRES_USER:-$(env_value POSTGRES_USER)}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(env_value POSTGRES_PASSWORD)}"
HOST_POSTGRES_PORT="${HOST_POSTGRES_PORT:-$(env_value HOST_POSTGRES_PORT)}"
HOST_POSTGRES_PORT="${HOST_POSTGRES_PORT:-5432}"

if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
  echo "Missing POSTGRES_USER or POSTGRES_PASSWORD in ${ENV_FILE}" >&2
  exit 1
fi

docker compose up -d postgres

echo "Waiting for Postgres..."
for _ in $(seq 1 60); do
  if docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$SOURCE_DB" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$SOURCE_DB" >/dev/null 2>&1; then
  echo "Postgres database ${SOURCE_DB} was not ready after 60 seconds." >&2
  exit 1
fi

db_exists="$(
  docker compose exec -T postgres psql -U "$POSTGRES_USER" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '${TARGET_DB}'"
)"

if [ "$db_exists" = "1" ]; then
  if [ "${RESET_BRANCH_DB:-0}" != "1" ]; then
    echo "Database ${TARGET_DB} already exists; leaving it unchanged."
  else
    echo "Resetting existing database ${TARGET_DB}..."
    docker compose exec -T postgres dropdb -U "$POSTGRES_USER" --force "$TARGET_DB"
    docker compose exec -T postgres createdb -U "$POSTGRES_USER" -O "$POSTGRES_USER" "$TARGET_DB"
  fi
else
  echo "Creating database ${TARGET_DB}..."
  docker compose exec -T postgres createdb -U "$POSTGRES_USER" -O "$POSTGRES_USER" "$TARGET_DB"
fi

if [ "$db_exists" != "1" ] || [ "${RESET_BRANCH_DB:-0}" = "1" ]; then
  echo "Cloning ${SOURCE_DB} into ${TARGET_DB}..."
  docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" --clean --if-exists --no-owner --no-privileges "$SOURCE_DB" \
    | docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$TARGET_DB" -v ON_ERROR_STOP=1 >/dev/null
fi

DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${HOST_POSTGRES_PORT}/${TARGET_DB}"
tmp_env="$(mktemp)"
awk -v db="$TARGET_DB" -v url="$DATABASE_URL" '
  BEGIN { wrote_db = 0; wrote_url = 0 }
  /^POSTGRES_DB=/ {
    print "POSTGRES_DB=" db
    wrote_db = 1
    next
  }
  /^DATABASE_URL=/ {
    print "DATABASE_URL=" url
    wrote_url = 1
    next
  }
  { print }
  END {
    if (!wrote_db) print "POSTGRES_DB=" db
    if (!wrote_url) print "DATABASE_URL=" url
  }
' "$ENV_FILE" > "$tmp_env"
mv "$tmp_env" "$ENV_FILE"

for secret_key in \
  OPENAI_API_KEY \
  COACH_TOKEN \
  NOTION_API_TOKEN; do
  secret_value="${!secret_key-}"
  if [ -n "$secret_value" ]; then
    set_env_value "$secret_key" "$secret_value"
  fi
done

echo "Wrote ${ENV_FILE} for ${TARGET_DB}."

alembic upgrade head