"""LLM batches wrapper — RETIRED with the OpenAI pivot (P0, T4).

The PoC's Anthropic Message-Batches API is gone. OpenAI has its own batch
API but P0 does not need it: the categorizer + topic-resolver synchronous
paths cover the workload. A future task can port `submit_batch` /
`iter_batch_results` to `openai.batches.*` if a 50% discount becomes load-
bearing again.

Keep the module + types as shims so callers that still import
`BatchRequestItem` (e.g. the topic-resolver batch adapter) don't crash at
import time. The runtime path raises immediately — there is no live
Anthropic client to fall back to.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


MAX_REQUESTS_PER_BATCH = 100_000
_DEFAULT_SAFETY_CAP = 50_000


@dataclass(frozen=True)
class BatchRequestItem:
    """Shim — see module docstring."""

    custom_id: str
    params: dict[str, Any]


class _Retired:
    """Sentinel for retired Anthropic batch entry points."""


async def submit_batch(*_args: Any, **_kwargs: Any) -> _Retired:
    raise NotImplementedError(
        "Anthropic Message-Batches retired in T4. Port to openai.batches.* before re-enabling."
    )


async def get_batch_status(*_args: Any, **_kwargs: Any) -> _Retired:
    raise NotImplementedError(
        "Anthropic Message-Batches retired in T4. Port to openai.batches.* before re-enabling."
    )


async def iter_batch_results(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
    raise NotImplementedError(
        "Anthropic Message-Batches retired in T4. Port to openai.batches.* before re-enabling."
    )
    if False:  # pragma: no cover — make this an AsyncIterator for typing
        yield None
