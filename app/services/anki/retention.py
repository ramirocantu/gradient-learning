"""Windowed Anki "true retention" per CC + per topic — FENCED (T18, V-RB2, V-O5).

The PoC's raw-SQL `retention_for_cc` / `retention_for_topic` joined
`anki_card_reviews` → `anki_cards` → `anki_note_tags` → the dropped
`topics` + `content_categories` tables. All four references are gone
(T1/T2), so the live SQL is removed entirely from this file.

FENCED because the consuming endpoint (`/api/v1/anki/performance`) is
route-disabled in `app/api/v1/anki.py` per T18. Both `retention_for_cc`
and `retention_for_topic` return an empty `RetentionSummary` so direct
imports do not crash. Restoration depends on a node_id subtree-set
rollup against `outline_subtree.subtree_node_ids` +
`anki_note_tags.node_id`; it is not on the PKM critical path and is
tracked post-P0.5.

Dataclasses `RetentionWindow` and `RetentionSummary` remain real (pure
shape; no legacy joins).

This file is FENCED, not a stub: behavior is deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


DEFAULT_WINDOWS: tuple[int, ...] = (7, 30, 0)


_FENCED_MSG = (
    "anki.retention is FENCED (T18, V-RB2) — /api/v1/anki/performance "
    "route unmounted; restoration pending node_id subtree port"
)


@dataclass(frozen=True)
class RetentionWindow:
    window_days: int
    pass_count: int
    fail_count: int

    @property
    def total(self) -> int:
        return self.pass_count + self.fail_count

    @property
    def retention(self) -> float | None:
        return None if self.total == 0 else self.pass_count / self.total


@dataclass(frozen=True)
class RetentionSummary:
    scope: str
    windows: dict[int, RetentionWindow]


def _empty(scope: str, windows: tuple[int, ...]) -> RetentionSummary:
    return RetentionSummary(
        scope=scope,
        windows={n: RetentionWindow(window_days=n, pass_count=0, fail_count=0) for n in windows},
    )


async def retention_for_cc(
    session: AsyncSession,
    *,
    cc_code: str,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> RetentionSummary:
    """FENCED — returns empty RetentionSummary. See module docstring."""
    logger.warning(_FENCED_MSG)
    return _empty(f"cc:{cc_code}", windows)


async def retention_for_topic(
    session: AsyncSession,
    *,
    topic_id: int,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> RetentionSummary:
    """FENCED — returns empty RetentionSummary. See module docstring."""
    logger.warning(_FENCED_MSG)
    return _empty(f"topic:{topic_id}", windows)
