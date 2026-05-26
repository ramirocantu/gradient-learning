"""Windowed accuracy trajectory per CC + per topic — T14 stub.

The PoC's raw-SQL trajectory walked `attempts → question_tags →
topics/content_categories` with a recursive CTE over `topics.parent_topic_id`.
All three of those tables are gone (T1), so the recursive CTE moves to
`outline_nodes.parent_id` via `outline_subtree.subtree_node_ids` and the
`question_tags.node_id` canonical column.

Stub keeps the public dataclasses + entry points so mastery's "(last 10 vs
prior 10)" trend block keeps importing; both `trajectory_for_*` return an
empty summary until the node_id port lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


MIN_WINDOW_SIZE = 5
ARROW_THRESHOLD = 0.10


@dataclass(frozen=True)
class TrajectoryWindow:
    n: int
    correct: int

    @property
    def accuracy(self) -> float | None:
        return None if self.n == 0 else self.correct / self.n


@dataclass(frozen=True)
class TrajectorySummary:
    scope: str
    last: TrajectoryWindow
    prior: TrajectoryWindow

    @property
    def delta(self) -> float | None:
        if self.last.n < MIN_WINDOW_SIZE or self.prior.n < MIN_WINDOW_SIZE:
            return None
        last_acc = self.last.accuracy
        prior_acc = self.prior.accuracy
        assert last_acc is not None and prior_acc is not None
        return last_acc - prior_acc

    @property
    def arrow(self) -> str | None:
        d = self.delta
        if d is None:
            return None
        if d >= ARROW_THRESHOLD:
            return "↑"
        if d <= -ARROW_THRESHOLD:
            return "↓"
        return "→"


def _empty(scope: str) -> TrajectorySummary:
    z = TrajectoryWindow(n=0, correct=0)
    return TrajectorySummary(scope=scope, last=z, prior=z)


async def trajectory_for_cc(session: AsyncSession, *, cc_code: str) -> TrajectorySummary:
    """Stub — TODO(T14 follow-up): port to node_id subtree."""
    logger.warning("trajectory_for_cc stub: empty pending node_id port")
    return _empty(f"cc:{cc_code}")


async def trajectory_for_topic(session: AsyncSession, *, topic_id: int) -> TrajectorySummary:
    """Stub — TODO(T14 follow-up): port to node_id subtree."""
    logger.warning("trajectory_for_topic stub: empty pending node_id port")
    return _empty(f"topic:{topic_id}")
