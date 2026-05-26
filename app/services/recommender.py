"""Study-next recommender — FENCED (T17, V-RB1, V-O5).

Per §G (2026-05-26 rescope) the study-plan / recommender surface is
candidate-for-cut: not part of the PKM critical loop
(question review → discriminator factors → atomic facts → Notion).
The PoC scorer was tied to the dropped 3-target tag shape
(`QuestionTag.topic_id / .content_category_id`) and the
Section/FC/CC/Topic outline tables; rebuilding it would need a node_id
mastery rollup + feature-pattern analysis + AAMC weighting rework.

Until a decision is made (post-P0.5; see T34 reassessment), this module
is FENCED:

  - the `/api/v1/recommendations/*` router is unmounted in `app/main.py`,
  - the dashboard recommendations route is unmounted in
    `app/web/dashboard/main.py`,
  - `recommend()` returns an empty result so any direct import does not
    crash,
  - related tests are collect-ignored in `tests/conftest.py`.

This file is FENCED, not a stub: behavior is deliberate, not in-progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


_FENCED_MSG = (
    "recommender.recommend is FENCED (T17, V-RB1) — route unmounted; "
    "rescope candidate-for-cut pending T34 reassessment"
)


MIN_ATTEMPTS = 3


RecommendationKind = Literal["topic_weakness", "feature_pattern"]


@dataclass(frozen=True)
class StudyRecommendation:
    kind: RecommendationKind
    label: str | None
    code: str | None
    target_id: int | None
    accuracy: float | None
    wilson_lower: float | None
    attempts: int | None
    feature_name: str | None
    feature_value: str | None
    accuracy_with: float | None
    accuracy_without: float | None
    priority_score: float
    reason: str
    representative_qids: list[str]


@dataclass
class RecommendationResult:
    recommendations: list[StudyRecommendation]
    total_candidates_scored: int


async def recommend(session: AsyncSession, *, n: int = 5) -> RecommendationResult:
    """FENCED — see module docstring. Returns an empty result."""
    logger.warning(_FENCED_MSG)
    return RecommendationResult(recommendations=[], total_candidates_scored=0)
