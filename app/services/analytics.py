"""Topic-level analytics rollups — FENCED (T17, V-RB1, V-O5).

Mastery rollups are off the PKM critical path per the 2026-05-26 rescope.
The PoC's `compute_mastery` walked Section/FC/CC/Topic + the 3-target
`QuestionTag(topic_id/content_category_id/skill)`. All four outline tables
are gone (T1) and the 3-target columns are gone (T2).

Restoration depends on a node_id subtree rollup via
`app.services.outline_subtree` and is tracked separately (post-P0.5,
candidate for P3 or P4 once the dashboard SPA reassessment at T34 picks
which mastery surfaces survive). Until then:

  - the `/api/v1/analytics/*` router is unmounted in `app/main.py`,
  - `compute_mastery` returns an empty `MasteryReport` so any direct
    import does not crash,
  - related tests are collect-ignored in `tests/conftest.py`.

This file is FENCED, not a stub: behavior is deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from math import sqrt
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


_FENCED_MSG = (
    "analytics.compute_mastery is FENCED (T17, V-RB1) — route unmounted; "
    "restoration pending post-P0.5 port to OutlineNode + outline_subtree"
)


AccuracyKind = Literal["section", "content_category", "topic", "skill"]


@dataclass(frozen=True)
class AccuracyStat:
    label: str
    code: str | None
    kind: AccuracyKind
    target_id: int | None
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float


@dataclass(frozen=True)
class TimingStat:
    median_seconds_discrete: float | None
    median_seconds_passage_based: float | None
    questions_over_target_discrete: int
    questions_over_target_passage: int


@dataclass(frozen=True)
class TrendPoint:
    period_start: date
    accuracy: float
    attempts: int


@dataclass(frozen=True)
class MasteryReport:
    by_section: list[AccuracyStat] = field(default_factory=list)
    by_content_category: list[AccuracyStat] = field(default_factory=list)
    by_topic: list[AccuracyStat] = field(default_factory=list)
    by_skill: list[AccuracyStat] = field(default_factory=list)
    timing: TimingStat = field(
        default_factory=lambda: TimingStat(
            median_seconds_discrete=None,
            median_seconds_passage_based=None,
            questions_over_target_discrete=0,
            questions_over_target_passage=0,
        )
    )
    trend_7d: list[TrendPoint] = field(default_factory=list)
    trend_30d: list[TrendPoint] = field(default_factory=list)
    uncategorized_question_count: int = 0
    total_attempts: int = 0
    total_questions: int = 0


def wilson_lower(correct: int, attempts: int, z: float = 1.96) -> float:
    """95% Wilson score lower bound on the success-rate proportion. (Pure math.)"""
    if attempts == 0:
        return 0.0
    p = correct / attempts
    denominator = 1 + z**2 / attempts
    center = p + z**2 / (2 * attempts)
    margin = sqrt(p * (1 - p) / attempts + z**2 / (4 * attempts**2)) * z
    return max(0.0, (center - margin) / denominator)


async def compute_mastery(session: AsyncSession) -> MasteryReport:
    """FENCED — see module docstring. Returns an empty report."""
    logger.warning(_FENCED_MSG)
    return MasteryReport()
