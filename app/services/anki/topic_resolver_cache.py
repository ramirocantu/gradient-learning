"""SQLite cache for Anki topic-resolver results (SPEC §T32).

Mirrors `app/services/categorizer/cache.py`. Lives at
`settings.ANKI_TOPIC_RESOLVER_CACHE_PATH`. Payload is a JSON list of
`TopicPick` dicts (post-§V25 amendment — multi-topic per card per CC).
"""

from __future__ import annotations

import json

from app.services.anki.topic_resolver import TopicPick
from app.services.llm.cache import LlmCacheBase


def _serialize(picks: list[TopicPick]) -> str:
    return json.dumps(
        [
            {
                "topic_path": p.topic_path,
                "confidence": p.confidence,
                "rationale": p.rationale,
            }
            for p in picks
        ],
        separators=(",", ":"),
    )


def _deserialize(raw_json: str) -> list[TopicPick]:
    payload = json.loads(raw_json)
    # Back-compat: pre-v4 cache rows are a single dict, not a list.
    # Old rows are invalidated by extractor_version mismatch in get() anyway,
    # but we defend in depth.
    if isinstance(payload, dict):
        payload = [payload]
    return [
        TopicPick(
            topic_path=item["topic_path"],
            confidence=float(item["confidence"]),
            rationale=item.get("rationale", ""),
        )
        for item in payload
    ]


class AnkiTopicResolverCache(LlmCacheBase):
    TABLE_NAME = "anki_topic_resolver_cache"
    PAYLOAD_COLUMN_DDL = "result_json TEXT NOT NULL"
    INDEX_NAME = "idx_anki_topic_resolver_extractor_version"

    def get(self, cache_key: str, extractor_version: str) -> list[TopicPick] | None:
        row = self._conn.execute(
            f"SELECT result_json, extractor_version FROM {self.TABLE_NAME} WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        raw_json, stored_version = row
        if stored_version != extractor_version:
            return None
        return _deserialize(raw_json)

    def put(
        self,
        cache_key: str,
        picks: list[TopicPick],
        extractor_version: str,
        *,
        model: str,
        cost: float,
    ) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_NAME} "
            f"(cache_key, result_json, extractor_version, model, "
            f" input_tokens, output_tokens, cost_estimate_usd, created_at) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (cache_key, _serialize(picks), extractor_version, model, 0, 0, cost),
        )
        self._conn.commit()
