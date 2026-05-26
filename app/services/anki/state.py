"""Subtree Anki state counts per CC + per topic — FENCED (T18, V-RB2, V-O5).

The PoC's raw-SQL `state_for_cc` / `state_for_topic` joined `anki_cards`
→ `anki_note_tags` → the dropped legacy outline tables via the legacy
topic-id / content-category-id columns. All four references are gone
(T1/T2), so the live SQL is removed entirely from this file.

FENCED because the consuming endpoint (`/api/v1/anki/performance`) is
route-disabled in `app/api/v1/anki.py` per T18. Both `state_for_cc` and
`state_for_topic` return an empty `StateCounts` so direct imports do
not crash. Restoration depends on a node_id subtree-set rollup via
`outline_subtree.subtree_node_ids` + `anki_note_tags.node_id`; it is
not on the PKM critical path and is tracked post-P0.5.

Dataclass `StateCounts` + the `unlock_pct` property remain real (pure
shape; no legacy joins).

This file is FENCED, not a stub: behavior is deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


_FENCED_MSG = (
    "anki.state is FENCED (T18, V-RB2) — /api/v1/anki/performance "
    "route unmounted; restoration pending node_id subtree port"
)


@dataclass(frozen=True)
class StateCounts:
    scope: str
    total_cards: int
    assigned: int
    suspended: int
    new: int
    learning: int
    young: int
    mature: int

    @property
    def unlock_pct(self) -> float | None:
        return None if self.total_cards == 0 else self.assigned / self.total_cards


def _empty(scope: str) -> StateCounts:
    return StateCounts(
        scope=scope,
        total_cards=0,
        assigned=0,
        suspended=0,
        new=0,
        learning=0,
        young=0,
        mature=0,
    )


async def state_for_cc(session: AsyncSession, *, cc_code: str) -> StateCounts:
    """FENCED — returns empty StateCounts. See module docstring."""
    logger.warning(_FENCED_MSG)
    return _empty(f"cc:{cc_code}")


async def state_for_topic(session: AsyncSession, *, topic_id: int) -> StateCounts:
    """FENCED — returns empty StateCounts. See module docstring."""
    logger.warning(_FENCED_MSG)
    return _empty(f"topic:{topic_id}")
