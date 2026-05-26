"""Feature extractor — FENCED (T17, V-RB1, V-O5).

The PoC's `extract_features_for_question` resolved a question's outline
ancestry via the dropped `Topic`/`ContentCategory`/`FoundationalConcept`/
`Section` tables (used to skip CARS via `_is_cars_question` walking
topic→cc→fc→section) and called Anthropic for LLM judgment features.

FENCED because:
  - outline walk is dead (T1) — domain-blind CARS detection requires a
    domain-pack flag or outline-node attribute (post-P0.5),
  - Anthropic→OpenAI SDK pivot lives in T35 follow-up for this surface,
  - `/api/v1/analyzer/*` router is unmounted in `app/main.py`,
  - dashboard `insights` route is unmounted in `app/web/dashboard/main.py`,
  - `run_feature_extraction_job` scheduler entry is unregistered in
    `app/scheduler.py`,
  - related tests are collect-ignored in `tests/conftest.py`.

`extract_features_for_question` returns a skipped result so callers that
still hold a reference do not crash. Public surface preserved.
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
    "EXTRACTOR_VERSION",
    "ExtractFeaturesResult",
    "FeatureExtractionResult",
    "LlmJudgmentFeatures",
    "MechanicalFeatures",
    "QuestionNotFoundError",
    "extract_features_for_question",
]


_FENCED_REASON = "fenced_t17"
_FENCED_MSG = (
    "analyzer.extract_features_for_question is FENCED (T17, V-RB1) — "
    "route + scheduler entry unmounted; restoration pending post-P0.5"
)


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
    """FENCED — see module docstring. Returns a skipped result."""
    logger.warning(_FENCED_MSG)
    return FeatureExtractionResult(
        question_id=question_id,
        qid="",
        persisted=False,
        skipped_reason=_FENCED_REASON,
        cache_hit=False,
        cost_estimate_usd=0.0,
        cost_saved_usd=0.0,
        extractor_version=EXTRACTOR_VERSION,
        mechanical=None,
        features=None,
    )
