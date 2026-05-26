"""SQLite-backed persistent cache for LLM categorization results.

Lives at `settings.CATEGORIZER_CACHE_PATH` (defaults to
`backend/data/categorizer-cache.db`). Cache key includes the model name
so Sonnet and Haiku results don't collide. `extractor_version` is stored
and checked on lookup; bumping it invalidates without deleting (clean up
via `clear(extractor_version=...)`).

Shared boilerplate (connection lifecycle, schema-create, stats, clear,
lookup_cost, close, context-manager) lives in `LlmCacheBase`. Only
`get` and `put` here, because the payload (`CategorizeResult`) is
categorizer-specific.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from app.services.categorizer.llm import CategorizeResult, LlmTagSuggestion
from app.services.llm.cache import LlmCacheBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedResult:
    result: CategorizeResult
    cached_at: datetime


def _serialize_result(result: CategorizeResult) -> str:
    payload = {
        "suggestions": [
            {
                "kind": s.kind,
                "identifier": s.identifier,
                "under_content_category": s.under_content_category,
                "confidence": s.confidence,
                "rationale": s.rationale,
            }
            for s in result.suggestions
        ],
        "primary_aamc_section": result.primary_aamc_section,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "extractor_version": result.extractor_version,
        "parse_warnings": list(result.parse_warnings),
    }
    return json.dumps(payload, separators=(",", ":"))


def _deserialize_result(raw_json: str) -> CategorizeResult:
    payload = json.loads(raw_json)
    suggestions = [
        LlmTagSuggestion(
            kind=s["kind"],
            identifier=s["identifier"],
            under_content_category=s.get("under_content_category"),
            confidence=float(s.get("confidence", 0.0)),
            rationale=s.get("rationale", ""),
        )
        for s in payload.get("suggestions", [])
    ]
    return CategorizeResult(
        suggestions=suggestions,
        primary_aamc_section=payload.get("primary_aamc_section"),
        cache_hit=True,  # Caller is treating this as a hit.
        input_tokens=int(payload.get("input_tokens", 0)),
        output_tokens=int(payload.get("output_tokens", 0)),
        estimated_cost_usd=0.0,  # No spend on a hit.
        extractor_version=payload.get("extractor_version", ""),
        parse_warnings=list(payload.get("parse_warnings", [])),
    )


class CategorizerCache(LlmCacheBase):
    """Persistent SQLite cache for LLM categorization results."""

    TABLE_NAME = "llm_categorizer_cache"
    PAYLOAD_COLUMN_DDL = "result_json TEXT NOT NULL"
    # The cached_input_tokens column is dead — `put` hard-codes it to 0 —
    # but it lives in the on-disk schema, so the DDL must mention it for
    # `CREATE TABLE IF NOT EXISTS` to stay column-compatible.
    EXTRA_COLUMNS_DDL = ("cached_input_tokens INTEGER NOT NULL",)
    INDEX_NAME = "idx_extractor_version"

    def get(self, cache_key: str, extractor_version: str) -> CategorizeResult | None:
        row = self._conn.execute(
            "SELECT result_json, extractor_version, cost_estimate_usd "
            "FROM llm_categorizer_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        raw_json, stored_version, _ = row
        if stored_version != extractor_version:
            return None
        return _deserialize_result(raw_json)

    def put(
        self,
        cache_key: str,
        result: CategorizeResult,
        extractor_version: str,
        *,
        model: str,
    ) -> None:
        raw_json = _serialize_result(result)
        cached_input_tokens = max(0, result.input_tokens - result.output_tokens)
        # The CategorizeResult carries combined input+cache_create+cache_read
        # in input_tokens. We don't have a clean split here, so store the
        # combined value as cached_input_tokens=0 default and the rest as
        # input_tokens. Cost was already computed; both are stored.
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_categorizer_cache "
            "(cache_key, result_json, extractor_version, model, "
            " input_tokens, output_tokens, cached_input_tokens, "
            " cost_estimate_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                cache_key,
                raw_json,
                extractor_version,
                model,
                result.input_tokens,
                result.output_tokens,
                0,
                result.estimated_cost_usd,
            ),
        )
        self._conn.commit()
        # Suppress unused-var warning while keeping the future hook.
        del cached_input_tokens
