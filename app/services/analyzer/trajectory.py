"""Windowed accuracy trajectory per CC + per topic — FENCED (T17, V-RB1, V-O5).

Trajectory consumes the same dashboard mastery surfaces that T17 has
fenced. Public dataclasses + entry points stay importable so the mastery
service (also FENCED) does not raise on import; both `trajectory_for_*`
return an empty summary. Restoration depends on a node_id subtree rollup
and is tracked post-P0.5.
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
    """FENCED — see module docstring. Returns an empty summary."""
    logger.warning(
        "analyzer.trajectory.trajectory_for_cc is FENCED (T17, V-RB1)"
    )
    return _empty(f"cc:{cc_code}")


async def trajectory_for_topic(session: AsyncSession, *, topic_id: int) -> TrajectorySummary:
    """FENCED — see module docstring. Returns an empty summary."""
    logger.warning(
        "analyzer.trajectory.trajectory_for_topic is FENCED (T17, V-RB1)"
    )
    return _empty(f"topic:{topic_id}")
