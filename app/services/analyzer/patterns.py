"""Pattern aggregator — FENCED (T17, V-RB1, V-O5).

The PoC's `analyze` built feature-pattern findings by joining `attempts` →
`questions` → `question_tags(topic_id|content_category_id|skill)` →
`topics`/`content_categories`/`sections` to filter and group. All four
outline tables are gone (T1) and the 3-target tag columns are gone (T2).

FENCED because the consuming routes (`/api/v1/analyzer/patterns` and the
dashboard `insights` route) are unmounted per T17. `analyze` returns an
empty report so direct imports do not crash. Restoration depends on a
node_id subtree rollup and is tracked post-P0.5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from math import sqrt

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401 — kept for signature

from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION

logger = logging.getLogger(__name__)


def wilson_upper(correct: int, attempts: int, z: float = 1.96) -> float:
    """95% Wilson upper bound. (Pure math.)"""
    if attempts == 0:
        return 1.0
    p = correct / attempts
    denominator = 1 + z**2 / attempts
    center = p + z**2 / (2 * attempts)
    margin = sqrt(p * (1 - p) / attempts + z**2 / (4 * attempts**2)) * z
    return min(1.0, (center + margin) / denominator)


@dataclass(frozen=True)
class AnalysisFilter:
    section_code: str | None = None
    content_category_code: str | None = None
    topic_id: int | None = None
    skill: int | None = None
    since: date | None = None
    until: date | None = None
    min_sample_size: int = 10


@dataclass(frozen=True)
class FeatureFinding:
    feature_name: str
    feature_value: str
    accuracy_with: float
    accuracy_without: float
    attempts_with: int
    attempts_without: int
    correct_with: int
    correct_without: int
    accuracy_delta: float
    wilson_lower_with: float
    wilson_lower_without: float
    confident_delta: float
    representative_missed_qids: list[str]


@dataclass(frozen=True)
class CoverageStats:
    questions_with_features: int
    questions_without_features: int
    feature_extractor_version: str


@dataclass(frozen=True)
class InsightReport:
    filter_applied: AnalysisFilter
    total_attempts_in_scope: int = 0
    total_questions_in_scope: int = 0
    baseline_accuracy: float = 0.0
    baseline_wilson_lower: float = 0.0
    findings: list[FeatureFinding] = field(default_factory=list)
    coverage: CoverageStats = field(
        default_factory=lambda: CoverageStats(
            questions_with_features=0,
            questions_without_features=0,
            feature_extractor_version=EXTRACTOR_VERSION,
        )
    )


async def analyze(filter: AnalysisFilter, session: AsyncSession) -> InsightReport:
    """FENCED — see module docstring. Returns an empty report."""
    logger.warning(
        "analyzer.patterns.analyze is FENCED (T17, V-RB1) — route unmounted"
    )
    return InsightReport(filter_applied=filter)
