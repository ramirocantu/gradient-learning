"""Shared SQLite-cache base for LLM result caches.

Each LLM caller (categorizer, feature extractor, synthesizer) owns its
own SQLite file, table, and payload shape. The base owns connection
lifecycle, schema-create, stats, clear, lookup_cost, close, and the
context-manager protocol. Subclasses override the class-level config
(TABLE_NAME, PAYLOAD_COLUMN_DDL, EXTRA_COLUMNS_DDL, INDEX_NAME) and the
per-cache `get` / `put` pair.

The schema prelude is: cache_key PK, <payload>, extractor_version, model,
input_tokens, output_tokens, <extras>, cost_estimate_usd, created_at.
`CREATE TABLE IF NOT EXISTS` is idempotent against the live on-disk
cache files in `backend/data/` — the column order produced here matches
the pre-R.2d modules column-for-column.

Sync, not async — async callers wrap reads/writes in `asyncio.to_thread`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class LlmCacheBase:
    """Base class for the LLM SQLite result caches.

    Subclasses provide `TABLE_NAME`, `PAYLOAD_COLUMN_DDL`, optionally
    `EXTRA_COLUMNS_DDL`, and a unique `INDEX_NAME`, then override
    `get` and `put`.
    """

    # Subclass overrides. PAYLOAD_COLUMN_DDL is e.g. "result_json TEXT
    # NOT NULL" or "markdown TEXT NOT NULL". EXTRA_COLUMNS_DDL is inserted
    # between output_tokens and cost_estimate_usd (categorizer needs
    # cached_input_tokens; the others leave it empty). INDEX_NAME must
    # match the name on disk in the live cache file.
    TABLE_NAME: str = "override_me"
    PAYLOAD_COLUMN_DDL: str = "override_me"
    EXTRA_COLUMNS_DDL: tuple[str, ...] = ()
    INDEX_NAME: str = "override_me"

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so a single instance can be shared across
        # threads (sync FastAPI endpoints, ThreadPoolExecutor, etc.). All
        # writes happen via short-lived cursor()/commit() pairs so contention
        # is bounded.
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._ensure_schema()
        # Enable WAL for safer concurrent reads.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass

    def _ensure_schema(self) -> None:
        """Create table + extractor_version index if missing (idempotent)."""
        prelude = [
            "cache_key TEXT PRIMARY KEY",
            self.PAYLOAD_COLUMN_DDL,
            "extractor_version TEXT NOT NULL",
            "model TEXT NOT NULL",
            "input_tokens INTEGER NOT NULL",
            "output_tokens INTEGER NOT NULL",
            *self.EXTRA_COLUMNS_DDL,
            "cost_estimate_usd REAL NOT NULL",
            "created_at TEXT NOT NULL DEFAULT (datetime('now'))",
        ]
        cols_sql = ",\n  ".join(prelude)
        self._conn.executescript(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (\n  {cols_sql}\n);\n"
            f"CREATE INDEX IF NOT EXISTS {self.INDEX_NAME} "
            f"ON {self.TABLE_NAME} (extractor_version);\n"
        )

    @property
    def path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> "LlmCacheBase":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def lookup_cost(self, cache_key: str) -> float:
        """Return the original LLM cost stored for this key (0.0 if missing)."""
        row = self._conn.execute(
            f"SELECT cost_estimate_usd FROM {self.TABLE_NAME} WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}").fetchone()[0]
        by_version = {
            row[0]: row[1]
            for row in self._conn.execute(
                f"SELECT extractor_version, COUNT(*) "
                f"FROM {self.TABLE_NAME} GROUP BY extractor_version"
            )
        }
        by_model = {
            row[0]: row[1]
            for row in self._conn.execute(
                f"SELECT model, COUNT(*) FROM {self.TABLE_NAME} GROUP BY model"
            )
        }
        total_cost = self._conn.execute(
            f"SELECT COALESCE(SUM(cost_estimate_usd), 0) FROM {self.TABLE_NAME}"
        ).fetchone()[0]
        return {
            "total_entries": int(total),
            "by_version": by_version,
            "by_model": by_model,
            "total_cost_saved_usd": float(total_cost),
        }

    def clear(self, *, extractor_version: str | None = None) -> int:
        if extractor_version is None:
            cur = self._conn.execute(f"DELETE FROM {self.TABLE_NAME}")
        else:
            cur = self._conn.execute(
                f"DELETE FROM {self.TABLE_NAME} WHERE extractor_version = ?",
                (extractor_version,),
            )
        self._conn.commit()
        return cur.rowcount

    def get(self, cache_key: str, extractor_version: str) -> Any:
        raise NotImplementedError("Subclasses must implement get()")

    def put(self, cache_key: str, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Subclasses must implement put()")
