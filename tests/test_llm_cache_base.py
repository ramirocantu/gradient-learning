"""Tests for the shared LlmCacheBase (Phase 9.5 R.2d).

The base class subsumes the connection lifecycle + maintenance ops that
were once shared across the concrete LLM caches. Those concrete caches
(CategorizerCache, FeatureExtractorCache, SynthesizerCache) were deleted
in T53 along with the legacy categorizer/analyzer; the base survives and
is exercised here directly via a minimal subclass.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services.llm.cache import LlmCacheBase


class _DummyCache(LlmCacheBase):
    """Minimal subclass for exercising the base directly."""

    TABLE_NAME = "_test_dummy_cache"
    PAYLOAD_COLUMN_DDL = "payload TEXT NOT NULL"
    INDEX_NAME = "idx_test_dummy_version"

    def get(self, cache_key: str, extractor_version: str) -> str | None:
        row = self._conn.execute(
            f"SELECT payload, extractor_version FROM {self.TABLE_NAME} WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None or row[1] != extractor_version:
            return None
        return row[0]

    def put(
        self,
        cache_key: str,
        payload: str,
        extractor_version: str,
        *,
        model: str,
        cost: float = 0.0,
    ) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_NAME} "
            "(cache_key, payload, extractor_version, model, "
            " input_tokens, output_tokens, cost_estimate_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (cache_key, payload, extractor_version, model, 0, 0, cost),
        )
        self._conn.commit()


def test_base_creates_table_with_subclass_config(tmp_path: Path):
    """`_ensure_schema` lands the prelude + payload column on disk."""
    cache = _DummyCache(tmp_path / "dummy.db")
    try:
        # Confirm via a sibling sqlite3 connection that the table + columns exist.
        conn = sqlite3.connect(str(cache.path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(_test_dummy_cache)")}
        finally:
            conn.close()
        # Prelude columns + the subclass's payload column.
        assert cols == {
            "cache_key",
            "payload",
            "extractor_version",
            "model",
            "input_tokens",
            "output_tokens",
            "cost_estimate_usd",
            "created_at",
        }
        # `stats()` works on the empty table and returns the documented shape.
        stats = cache.stats()
        assert stats == {
            "total_entries": 0,
            "by_version": {},
            "by_model": {},
            "total_cost_saved_usd": 0.0,
        }
    finally:
        cache.close()


def test_subclasses_share_clear_and_lookup_cost(tmp_path: Path):
    """`clear(extractor_version=...)` and `lookup_cost()` come from the base."""
    cache = _DummyCache(tmp_path / "dummy.db")
    try:
        cache.put("k1", "p1", "v1", model="m1", cost=0.01)
        cache.put("k2", "p2", "v1", model="m1", cost=0.02)
        cache.put("k3", "p3", "v2", model="m1", cost=0.03)

        # lookup_cost returns the stored float; missing key returns 0.0.
        assert cache.lookup_cost("k1") == 0.01
        assert cache.lookup_cost("k3") == 0.03
        assert cache.lookup_cost("missing") == 0.0

        # clear by extractor_version returns deleted row count.
        deleted = cache.clear(extractor_version="v1")
        assert deleted == 2
        # v2 entry survives.
        assert cache.get("k3", "v2") == "p3"
        # v1 entries are gone.
        assert cache.get("k1", "v1") is None
    finally:
        cache.close()
