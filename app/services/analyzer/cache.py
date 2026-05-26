"""SQLite-backed persistent cache for LLM feature-extraction results.

Lives at `settings.FEATURE_EXTRACTOR_CACHE_PATH` (defaults to
`backend/data/feature-extractor-cache.db`). Cache key includes the model
name; `extractor_version` is stored and checked on lookup.

Shared boilerplate (connection lifecycle, schema-create, stats, clear,
lookup_cost, close, context-manager) lives in `LlmCacheBase`. Only
`get` and `put` here, because the payload (`ExtractFeaturesResult`) is
extractor-specific.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from app.services.analyzer.feature_extractor import (
    ExtractFeaturesResult,
    LlmJudgmentFeatures,
)
from app.services.llm.cache import LlmCacheBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedFeatureResult:
    result: ExtractFeaturesResult
    cached_at: datetime


def _serialize_result(result: ExtractFeaturesResult) -> str:
    f = result.features
    payload = {
        "features": {
            "reasoning_type": f.reasoning_type,
            "requires_calculation": f.requires_calculation,
            "calculation_steps": f.calculation_steps,
            "passage_type": f.passage_type,
            "distractor_difficulty": f.distractor_difficulty,
            "trap_distractor_present": f.trap_distractor_present,
            "common_misconception": f.common_misconception,
            "jargon_density": f.jargon_density,
            "key_concept_summary": f.key_concept_summary,
            "involves_graph_or_figure": f.involves_graph_or_figure,
            "involves_data_table": f.involves_data_table,
        },
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "extractor_version": result.extractor_version,
        "model": result.model,
        "parse_warnings": list(result.parse_warnings),
    }
    return json.dumps(payload, separators=(",", ":"))


def _deserialize_result(raw_json: str) -> ExtractFeaturesResult:
    payload = json.loads(raw_json)
    f = payload.get("features", {})
    features = LlmJudgmentFeatures(
        reasoning_type=f.get("reasoning_type", "application"),
        requires_calculation=bool(f.get("requires_calculation", False)),
        calculation_steps=int(f.get("calculation_steps", 0)),
        passage_type=f.get("passage_type"),
        distractor_difficulty=f.get("distractor_difficulty", "medium"),
        trap_distractor_present=bool(f.get("trap_distractor_present", False)),
        common_misconception=f.get("common_misconception"),
        jargon_density=f.get("jargon_density", "medium"),
        key_concept_summary=f.get("key_concept_summary", ""),
        involves_graph_or_figure=bool(f.get("involves_graph_or_figure", False)),
        involves_data_table=bool(f.get("involves_data_table", False)),
    )
    return ExtractFeaturesResult(
        features=features,
        cache_hit=True,
        cost_saved_usd=0.0,
        input_tokens=int(payload.get("input_tokens", 0)),
        output_tokens=int(payload.get("output_tokens", 0)),
        estimated_cost_usd=0.0,
        extractor_version=payload.get("extractor_version", ""),
        model=payload.get("model", ""),
        parse_warnings=list(payload.get("parse_warnings", [])),
    )


class FeatureExtractorCache(LlmCacheBase):
    """Persistent SQLite cache for LLM feature-extraction results."""

    TABLE_NAME = "llm_feature_extractor_cache"
    PAYLOAD_COLUMN_DDL = "result_json TEXT NOT NULL"
    INDEX_NAME = "idx_features_extractor_version"

    def get(self, cache_key: str, extractor_version: str) -> ExtractFeaturesResult | None:
        row = self._conn.execute(
            "SELECT result_json, extractor_version "
            "FROM llm_feature_extractor_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        raw_json, stored_version = row
        if stored_version != extractor_version:
            return None
        return _deserialize_result(raw_json)

    def put(
        self,
        cache_key: str,
        result: ExtractFeaturesResult,
        extractor_version: str,
        *,
        model: str,
    ) -> None:
        raw_json = _serialize_result(result)
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_feature_extractor_cache "
            "(cache_key, result_json, extractor_version, model, "
            " input_tokens, output_tokens, cost_estimate_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                cache_key,
                raw_json,
                extractor_version,
                model,
                result.input_tokens,
                result.output_tokens,
                result.estimated_cost_usd,
            ),
        )
        self._conn.commit()
