"""Anki topic-resolver worker — T13 stub.

The PoC ran an Anthropic-Claude resolver that walked notes parsed as
`aamc_cc`, asked the LLM for a topic under that CC, and wrote back an
`aamc_topic` `AnkiNoteTag` carrying `topic_id` + `content_category_id`. Those
columns are gone (T2 collapsed the 3-target shape to `node_id`), the
`aamc_cc`/`aamc_topic` parsed_kind CHECK is gone, the AAMC CC codes are gone,
and the LLM stack pivots Anthropic → OpenAI in T4 with the structured-output
rework in T6.

TODO(T4 + T6 + T14): rebuild the resolver — OpenAI structured output,
node_id resolution via `OutlineLookup.node_id_by_path`, write canonical
`AnkiNoteTag(source='llm', node_id=..., confidence=..., manual_review=...)`.
This stub keeps the public surface so `app/scheduler.py` and admin routes
keep importing/loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


@dataclass
class ResolverSummary:
    processed: int = 0
    persisted: int = 0
    skipped_low_confidence: int = 0
    declined_by_llm: int = 0
    cache_hits: int = 0
    total_cost_usd: float = 0.0
    total_cost_saved_usd: float = 0.0
    error: str | None = None
    partial_failure: bool = False


async def run(session: AsyncSession, **_kwargs) -> ResolverSummary:
    """Stub — TODO(T4/T6/T14). Returns an empty summary."""
    logger.warning(
        "anki topic_resolver_worker.run stub: returns empty until T4 (openai) + "
        "T6 (structured output) + T14 (node_id) ports land"
    )
    return ResolverSummary()


def make_summary_text(summary: ResolverSummary) -> str:
    return (
        f"resolver stub: processed={summary.processed} persisted={summary.persisted} "
        f"(pending T4/T6/T14)"
    )
