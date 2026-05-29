#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
ENV_FILE="${1:-$ROOT/.env}"

[ -f "$ENV_FILE" ] || {
  echo "env file not found: $ENV_FILE" >&2
  exit 1
}

set -a
source "$ENV_FILE"
set +a

: "${POSTGRES_DB:?POSTGRES_DB is required in $ENV_FILE}"
POSTGRES_USER="${POSTGRES_USER:-gradient}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"

case "$POSTGRES_DB" in
  postgres|template0|template1)
    echo "refusing to drop protected database: $POSTGRES_DB" >&2
    exit 1
    ;;
esac

CONTAINER_ID="$(cd "$ROOT" && docker compose ps -q "$POSTGRES_SERVICE")"
[ -n "$CONTAINER_ID" ] || {
  echo "postgres container not found for service: $POSTGRES_SERVICE" >&2
  exit 1
}

echo "Dropping database: $POSTGRES_DB"
docker exec -i "$CONTAINER_ID" psql -U "$POSTGRES_USER" -d postgres \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$POSTGRES_DB' AND pid <> pg_backend_pid();" \
  -c "DROP DATABASE IF EXISTS \"$POSTGRES_DB\";"
