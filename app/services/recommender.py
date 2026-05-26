"""Study-next recommender.

TODO(T14): the original PoC scorer is gone — it was tied to the 3-target tag
shape (QuestionTag.topic_id / .content_category_id) and the section/cc/topic
tables. Restoring it needs:
  - node_id mastery rollup via the subtree-set helper (V-O1),
  - feature-pattern analysis ported off the old outline,
  - recency + AAMC weighting reworked without section codes.

This stub keeps the import surface (so the API endpoint loads) and returns
an empty result; T14 reimplements `recommend` against the canonical node_id
tags + ported analytics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

logger = logging.getLogger(__name__)


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
    """Stub — TODO(T14) port to node_id mastery rollup. Returns empty."""
    logger.warning(
        "recommender.recommend stub: returns empty until T14 ports analytics + "
        "scorer onto node_id canonical tags"
    )
    return RecommendationResult(recommendations=[], total_candidates_scored=0)
