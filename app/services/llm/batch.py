"""Generic Anthropic Message Batches API wrapper (SPEC §T51).

Submit a list of `BatchRequestItem`s → returns a batch id. Poll status with
`get_batch_status`. Stream per-request results with `iter_batch_results`. The
service is extractor-agnostic — per-extractor adapters (e.g.
`app.services.anki.topic_resolver_batch`) build the request list and persist
results into their own DB schemas.

Anthropic's Batches API offers 50% off input + output tokens, supports the
same `cache_control` semantics as synchronous calls, and has a 24h SLA
(typically minutes). Max 100k requests per batch, 256 MB total payload.

Costs persist in `llm_batch_runs` (see §T51 migration + `app.models.llm_batch`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types.messages import (
    MessageBatch,
    MessageBatchIndividualResponse,
)

logger = logging.getLogger(__name__)


# Anthropic enforces these limits per request — keep adapters honest.
MAX_REQUESTS_PER_BATCH = 100_000
# Local soft cap so a single bad adapter can't generate 1M items by accident.
_DEFAULT_SAFETY_CAP = 50_000


@dataclass(frozen=True)
class BatchRequestItem:
    """One per-call payload in a batch.

    `custom_id` is opaque to Anthropic — adapters use it to correlate results
    back to domain objects (e.g. `f"{anki_card_id}:{cc_code}"`). Must be
    unique within the batch and ≤64 chars.

    `params` mirrors the `messages.create` kwargs minus the SDK envelope —
    `{model, max_tokens, system, tools, tool_choice, messages, ...}`. The
    same dict is what Anthropic will replay later when processing.
    """

    custom_id: str
    params: dict[str, Any]


async def submit_batch(
    client: AsyncAnthropic,
    items: list[BatchRequestItem],
    *,
    safety_cap: int = _DEFAULT_SAFETY_CAP,
) -> MessageBatch:
    """Submit a batch and return the Anthropic `MessageBatch` envelope.

    Raises `ValueError` when over the per-batch cap (defensive — adapters
    should split before calling). Empty list raises too: Anthropic rejects
    zero-item batches.
    """
    if not items:
        raise ValueError("submit_batch: empty items list")
    if len(items) > safety_cap:
        raise ValueError(
            f"submit_batch: {len(items)} items exceeds local safety cap "
            f"({safety_cap}); split or raise safety_cap explicitly"
        )
    if len(items) > MAX_REQUESTS_PER_BATCH:
        raise ValueError(
            f"submit_batch: {len(items)} items exceeds Anthropic limit ({MAX_REQUESTS_PER_BATCH})"
        )

    # SDK takes an Iterable[Request]; Request is a TypedDict so a plain
    # dict per item works without importing the type at call time.
    requests = [{"custom_id": it.custom_id, "params": it.params} for it in items]
    batch = await client.messages.batches.create(requests=requests)
    logger.info(
        "submitted Anthropic batch id=%s status=%s items=%d",
        batch.id,
        batch.processing_status,
        len(items),
    )
    return batch


async def get_batch_status(client: AsyncAnthropic, batch_id: str) -> MessageBatch:
    """Retrieve current batch envelope (counts + processing_status)."""
    return await client.messages.batches.retrieve(batch_id)


async def iter_batch_results(
    client: AsyncAnthropic, batch_id: str
) -> AsyncIterator[MessageBatchIndividualResponse]:
    """Stream per-request results from a finished batch. Yields
    `MessageBatchIndividualResponse` items, each carrying `custom_id` and a
    `result` union (succeeded/errored/canceled/expired). Adapters dispatch
    on `result.type` and parse the inner message when succeeded.
    """
    stream = await client.messages.batches.results(batch_id)
    async for item in stream:
        yield item
