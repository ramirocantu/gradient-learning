"""SQLite-backed persistent cache for LLM insight synthesis results.

Lives at `settings.SYNTHESIZER_CACHE_PATH` (defaults to
`backend/data/synthesizer-cache.db`). Cache key: SHA-256 of canonical JSON
of InsightReport content + model name.

Shared boilerplate (connection lifecycle, schema-create, stats, clear,
lookup_cost, close, context-manager) lives in `LlmCacheBase`. Only `get`
and `put` here, because this cache stores rendered markdown (not a
JSON-serialized result object) and `put` carries explicit token/cost
kwargs rather than a result object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from app.services.llm.cache import LlmCacheBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedSynthesisResult:
    markdown: str
    input_tokens: int
    output_tokens: int
    cost_estimate_usd: float
    cached_at: datetime


class SynthesizerCache(LlmCacheBase):
    """Persistent SQLite cache for LLM synthesis results."""

    TABLE_NAME = "llm_synthesizer_cache"
    PAYLOAD_COLUMN_DDL = "markdown TEXT NOT NULL"
    INDEX_NAME = "idx_synthesizer_version"

    def get(self, cache_key: str, extractor_version: str) -> CachedSynthesisResult | None:
        row = self._conn.execute(
            "SELECT markdown, extractor_version, input_tokens, output_tokens, "
            "cost_estimate_usd, created_at "
            "FROM llm_synthesizer_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        markdown, stored_version, input_tokens, output_tokens, cost, created_at = row
        if stored_version != extractor_version:
            return None
        try:
            cached_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            cached_at = datetime.utcnow()
        return CachedSynthesisResult(
            markdown=markdown,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost_estimate_usd=float(cost),
            cached_at=cached_at,
        )

    def put(
        self,
        cache_key: str,
        markdown: str,
        extractor_version: str,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_estimate_usd: float,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_synthesizer_cache "
            "(cache_key, markdown, extractor_version, model, "
            " input_tokens, output_tokens, cost_estimate_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                cache_key,
                markdown,
                extractor_version,
                model,
                input_tokens,
                output_tokens,
                cost_estimate_usd,
            ),
        )
        self._conn.commit()
