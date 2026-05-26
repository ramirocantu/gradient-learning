"""Anki topic-resolver batch — T13 stub.

The PoC's batch adapter built Anthropic Messages-Batch requests for the same
aamc_cc → aamc_topic LLM resolution as `topic_resolver_worker`. Stubbed for
the same reasons (T2 dropped topic_id/content_category_id, T4 swaps to
OpenAI, T6 reworks structured output, T14 ports the readers).

TODO(T4 + T6 + T14): rebuild the batch flow on the OpenAI Batch API +
node_id canonical tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


@dataclass
class BatchBuildSummary:
    requests_built: int = 0
    notes_in_scope: int = 0
    cache_hits: int = 0
    skipped_low_text: int = 0


@dataclass
class BatchPersistSummary:
    processed: int = 0
    persisted: int = 0
    skipped_low_confidence: int = 0
    declined_by_llm: int = 0
    errored: int = 0
    custom_id_decode_errors: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


async def build_topic_resolver_batch_requests(session: AsyncSession, **_kwargs) -> tuple[list[Any], BatchBuildSummary]:
    logger.warning("build_topic_resolver_batch_requests stub: pending T4/T6/T14")
    return [], BatchBuildSummary()


async def persist_topic_resolver_batch_results(session: AsyncSession, **_kwargs) -> BatchPersistSummary:
    logger.warning("persist_topic_resolver_batch_results stub: pending T4/T6/T14")
    return BatchPersistSummary()


async def submit_topic_resolver_batch(session: AsyncSession, **_kwargs) -> dict[str, Any]:
    logger.warning("submit_topic_resolver_batch stub: pending T4/T6/T14")
    return {}


async def finalize_topic_resolver_batch(session: AsyncSession, **_kwargs) -> BatchPersistSummary:
    logger.warning("finalize_topic_resolver_batch stub: pending T4/T6/T14")
    return BatchPersistSummary()
