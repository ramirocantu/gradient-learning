"""Topic-level analytics rollups.

T14 stub. The PoC's `compute_mastery` walked Section/FC/CC/Topic + the
3-target `QuestionTag(topic_id/content_category_id/skill)`. All four outline
tables are gone (T1) and the 3-target columns are gone (T2). Restoring
mastery needs a node_id subtree rollup via `app.services.outline_subtree`.

This stub keeps the public surface — dataclasses + `wilson_lower` (pure math)
+ `compute_mastery` returning an empty `MasteryReport` — so the API/dashboard
routes load. Real rollup lands when the read services are reimplemented
against canonical `QuestionTag.node_id` + `outline_subtree.subtree_node_ids`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from math import sqrt
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


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
    """Stub — TODO(T14 follow-up): node_id rollup via outline_subtree."""
    logger.warning(
        "compute_mastery stub: returns empty MasteryReport until the node_id "
        "rollup port lands (see app.services.outline_subtree.subtree_node_ids)"
    )
    return MasteryReport()
