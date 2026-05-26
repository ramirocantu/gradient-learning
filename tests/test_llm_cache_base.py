"""Tests for the shared LlmCacheBase (Phase 9.5 R.2d).

The base class subsumes the connection lifecycle + maintenance ops
shared across CategorizerCache, FeatureExtractorCache, and
SynthesizerCache. Per-cache `get`/`put` overrides are exercised by the
existing pre-R.2d behavior tests (test_categorizer_cache.py,
test_feature_extractor.py, test_synthesizer.py) — those remain the
canonical regression guard for serialization behavior.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.synthesizer_cache import SynthesizerCache
from app.services.categorizer.cache import CategorizerCache
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


def test_categorizer_cache_inherits_base(tmp_path: Path):
    """CategorizerCache inherits from LlmCacheBase; round-trip still works.

    The existing test_categorizer_cache.py is the deeper regression guard;
    this case only confirms the inheritance shape + that the base's
    maintenance methods are reachable via a subclass instance.
    """
    assert issubclass(CategorizerCache, LlmCacheBase)
    cache = CategorizerCache(tmp_path / "cat.db")
    try:
        # The base-owned maintenance methods are reachable.
        assert cache.stats()["total_entries"] == 0
        assert cache.lookup_cost("anything") == 0.0
        assert cache.clear() == 0
        # The extras column from EXTRA_COLUMNS_DDL landed on disk.
        conn = sqlite3.connect(str(cache.path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_categorizer_cache)")}
        finally:
            conn.close()
        assert "cached_input_tokens" in cols
        assert "result_json" in cols
    finally:
        cache.close()


def test_extractor_cache_inherits_base(tmp_path: Path):
    """FeatureExtractorCache inherits from LlmCacheBase; no extras column."""
    assert issubclass(FeatureExtractorCache, LlmCacheBase)
    cache = FeatureExtractorCache(tmp_path / "fc.db")
    try:
        assert cache.stats()["total_entries"] == 0
        assert cache.lookup_cost("anything") == 0.0
        conn = sqlite3.connect(str(cache.path))
        try:
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(llm_feature_extractor_cache)")
            }
        finally:
            conn.close()
        # The feature-extractor table omits cached_input_tokens (R.0 §2.b).
        assert "cached_input_tokens" not in cols
        assert "result_json" in cols
    finally:
        cache.close()


def test_synthesizer_cache_inherits_base(tmp_path: Path):
    """SynthesizerCache inherits; payload column is `markdown`, not `result_json`."""
    assert issubclass(SynthesizerCache, LlmCacheBase)
    cache = SynthesizerCache(tmp_path / "synth.db")
    try:
        assert cache.stats()["total_entries"] == 0
        assert cache.lookup_cost("anything") == 0.0
        conn = sqlite3.connect(str(cache.path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_synthesizer_cache)")}
        finally:
            conn.close()
        assert "markdown" in cols
        assert "result_json" not in cols
    finally:
        cache.close()


def test_existing_sqlite_file_still_readable(tmp_path: Path):
    """A pre-R.2d-shape SQLite file opens cleanly under the refactored cache.

    Construct a file with the literal pre-R.2d categorizer DDL (no
    `DEFAULT 0` on cached_input_tokens), then point a refactored
    CategorizerCache at it. The base's `_ensure_schema` must be a no-op
    (CREATE TABLE IF NOT EXISTS), and reads/writes must still work.
    """
    db_path = tmp_path / "preexisting-categorizer-cache.db"
    # Literal pre-R.2d DDL from the previous categorizer/cache.py.
    pre_r2d_ddl = """
    CREATE TABLE IF NOT EXISTS llm_categorizer_cache (
      cache_key TEXT PRIMARY KEY,
      result_json TEXT NOT NULL,
      extractor_version TEXT NOT NULL,
      model TEXT NOT NULL,
      input_tokens INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      cached_input_tokens INTEGER NOT NULL,
      cost_estimate_usd REAL NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_extractor_version
      ON llm_categorizer_cache (extractor_version);
    """
    pre_conn = sqlite3.connect(str(db_path))
    pre_conn.executescript(pre_r2d_ddl)
    # Insert a row directly so we can confirm post-refactor reads see it.
    pre_conn.execute(
        "INSERT INTO llm_categorizer_cache "
        "(cache_key, result_json, extractor_version, model, "
        " input_tokens, output_tokens, cached_input_tokens, "
        " cost_estimate_usd, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (
            "pre-existing-key",
            '{"suggestions":[],"primary_aamc_section":null}',
            "v1",
            "m1",
            100,
            50,
            0,
            0.0125,
        ),
    )
    pre_conn.commit()
    pre_conn.close()

    # The refactored CategorizerCache must open without raising and see
    # the row via the base's stats() + lookup_cost().
    cache = CategorizerCache(db_path)
    try:
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["by_version"] == {"v1": 1}
        assert stats["by_model"] == {"m1": 1}
        assert abs(stats["total_cost_saved_usd"] - 0.0125) < 1e-9
        assert cache.lookup_cost("pre-existing-key") == 0.0125
    finally:
        cache.close()
