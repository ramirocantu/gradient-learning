"""Feature extractor — T14 stub.

The PoC's `extract_features_for_question` resolved a question's outline
ancestry via the dropped `Topic`/`ContentCategory`/`FoundationalConcept`/
`Section` tables (used to skip CARS via `_is_cars_question` walking
topic→cc→fc→section) and called Anthropic for LLM judgment features.

Stubbed because:
  - Outline walk is dead (T1).
  - Anthropic SDK pivots to OpenAI in T4.
  - CARS detection by section code is gone (V-O3 — codes are dropped); the
    follow-up port detects domain-blind via outline node attributes or a
    domain-pack flag.

Public surface preserved so the scheduler + insights routes still import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

from app.services.analyzer.feature_extractor import (
    EXTRACTOR_VERSION,
    ExtractFeaturesResult,
    LlmJudgmentFeatures,
)
from app.services.analyzer.mechanical_features import MechanicalFeatures

logger = logging.getLogger(__name__)


__all__ = [
    "CARS_SKIPPED_REASON",
    "EXTRACTOR_VERSION",
    "ExtractFeaturesResult",
    "FeatureExtractionResult",
    "LlmJudgmentFeatures",
    "MechanicalFeatures",
    "QuestionNotFoundError",
    "extract_features_for_question",
]


CARS_SKIPPED_REASON = "cars_deferred_to_4_1b"
_T14_PORT_PENDING_REASON = "t14_port_pending"


class QuestionNotFoundError(LookupError):
    """Raised when extract_features_for_question is called with an unknown id."""


@dataclass(frozen=True)
class FeatureExtractionResult:
    question_id: int
    qid: str
    persisted: bool
    skipped_reason: str | None
    cache_hit: bool
    cost_estimate_usd: float
    cost_saved_usd: float
    extractor_version: str
    mechanical: MechanicalFeatures | None
    features: LlmJudgmentFeatures | None


async def extract_features_for_question(
    question_id: int,
    session: AsyncSession,
    **_kwargs,
) -> FeatureExtractionResult:
    """Stub — TODO(T4 + T14 follow-up)."""
    logger.warning(
        "extract_features_for_question stub: returns skipped result until T4 "
        "(openai) + T14 (node_id CARS detection) ports land"
    )
    return FeatureExtractionResult(
        question_id=question_id,
        qid="",
        persisted=False,
        skipped_reason=_T14_PORT_PENDING_REASON,
        cache_hit=False,
        cost_estimate_usd=0.0,
        cost_saved_usd=0.0,
        extractor_version=EXTRACTOR_VERSION,
        mechanical=None,
        features=None,
    )
