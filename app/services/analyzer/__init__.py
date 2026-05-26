"""Analyzer orchestrator (Ticket 4.2).

Loads a Question + Passage, computes mechanical features in Python, calls
the LLM for judgment-call features, and UPSERTs the result into the
question_features table. CARS questions are skipped pending Ticket 4.1b.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.captures import Passage, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic
from app.services.analyzer import feature_extractor as feature_extractor_module
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.feature_extractor import (
    EXTRACTOR_VERSION,
    ExtractFeaturesResult,
    LlmJudgmentFeatures,
    extract_judgment_features,
)
from app.services.analyzer.mechanical_features import (
    MechanicalFeatures,
    compute_mechanical_features,
)

logger = logging.getLogger(__name__)


__all__ = [
    "EXTRACTOR_VERSION",
    "ExtractFeaturesResult",
    "FeatureExtractionResult",
    "LlmJudgmentFeatures",
    "MechanicalFeatures",
    "QuestionNotFoundError",
    "compute_mechanical_features",
    "extract_features_for_question",
    "extract_judgment_features",
]


CARS_SKIPPED_REASON = "cars_deferred_to_4_1b"


class QuestionNotFoundError(LookupError):
    """Raised when extract_features_for_question is called with an unknown question_id."""


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


async def _is_cars_question(session: AsyncSession, question_id: int) -> bool:
    """True iff any QuestionTag for this question points at the CARS section.

    Walks topic.cc.fc.section OR cc.fc.section. Skill tags (1-4) are
    section-agnostic and ignored — a skill-only tagged question is not CARS.
    """
    topic_cars = (
        await session.execute(
            select(QuestionTag.id)
            .join(Topic, Topic.id == QuestionTag.topic_id)
            .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
            .join(
                FoundationalConcept,
                FoundationalConcept.id == ContentCategory.foundational_concept_id,
            )
            .join(Section, Section.id == FoundationalConcept.section_id)
            .where(QuestionTag.question_id == question_id, Section.code == "CARS")
            .limit(1)
        )
    ).scalar_one_or_none()
    if topic_cars is not None:
        return True

    cc_cars = (
        await session.execute(
            select(QuestionTag.id)
            .join(
                ContentCategory,
                ContentCategory.id == QuestionTag.content_category_id,
            )
            .join(
                FoundationalConcept,
                FoundationalConcept.id == ContentCategory.foundational_concept_id,
            )
            .join(Section, Section.id == FoundationalConcept.section_id)
            .where(QuestionTag.question_id == question_id, Section.code == "CARS")
            .limit(1)
        )
    ).scalar_one_or_none()
    return cc_cars is not None


async def _upsert_features(
    session: AsyncSession,
    *,
    question_id: int,
    mechanical: MechanicalFeatures,
    judgment: LlmJudgmentFeatures,
    extractor_version: str,
) -> None:
    """INSERT ... ON CONFLICT (question_id) DO UPDATE — replace prior extraction."""
    values = {
        "question_id": question_id,
        "question_format": mechanical.question_format,
        "reasoning_type": judgment.reasoning_type,
        "requires_calculation": judgment.requires_calculation,
        "calculation_steps": judgment.calculation_steps,
        "involves_graph_or_figure": judgment.involves_graph_or_figure,
        "involves_data_table": judgment.involves_data_table,
        "has_negative_phrasing": mechanical.has_negative_phrasing,
        "passage_length_bucket": mechanical.passage_length_bucket,
        "passage_type": judgment.passage_type,
        "distractor_difficulty": judgment.distractor_difficulty,
        "trap_distractor_present": judgment.trap_distractor_present,
        "common_misconception": judgment.common_misconception,
        "jargon_density": judgment.jargon_density,
        "key_concept_summary": judgment.key_concept_summary,
        "extractor_version": extractor_version,
    }
    stmt = pg_insert(QuestionFeatures).values(**values)
    update_columns = {k: stmt.excluded[k] for k in values if k != "question_id"}
    update_columns["extracted_at"] = func.now()
    stmt = stmt.on_conflict_do_update(
        index_elements=[QuestionFeatures.question_id],
        set_=update_columns,
    )
    await session.execute(stmt)


async def extract_features_for_question(
    question_id: int,
    session: AsyncSession,
    *,
    anthropic_client: AsyncAnthropic,
    cache: FeatureExtractorCache | None = None,
) -> FeatureExtractionResult:
    """Extract content-agnostic features for one question and UPSERT into question_features.

    1. Load Question + Passage.
    2. Skip CARS questions (skipped_reason=cars_deferred_to_4_1b).
    3. Compute mechanical features in Python.
    4. Call extract_judgment_features (LLM + cache).
    5. UPSERT QuestionFeatures (UNIQUE on question_id).
    """
    question = (
        await session.execute(
            select(Question)
            .options(selectinload(Question.passage))
            .where(Question.id == question_id)
        )
    ).scalar_one_or_none()
    if question is None:
        raise QuestionNotFoundError(f"question_id={question_id} not found")

    if await _is_cars_question(session, question_id):
        logger.info(
            "extract_features qid=%s: CARS detected — skipping (deferred to 4.1b)",
            question.qid,
        )
        return FeatureExtractionResult(
            question_id=question_id,
            qid=question.qid,
            persisted=False,
            skipped_reason=CARS_SKIPPED_REASON,
            cache_hit=False,
            cost_estimate_usd=0.0,
            cost_saved_usd=0.0,
            extractor_version=feature_extractor_module.EXTRACTOR_VERSION,
            mechanical=None,
            features=None,
        )

    passage: Passage | None = question.passage
    mechanical = compute_mechanical_features(question, passage)

    extraction = await extract_judgment_features(
        question,
        passage,
        mechanical,
        anthropic_client=anthropic_client,
        cache=cache,
        extractor_version=feature_extractor_module.EXTRACTOR_VERSION,
    )

    await _upsert_features(
        session,
        question_id=question_id,
        mechanical=mechanical,
        judgment=extraction.features,
        extractor_version=extraction.extractor_version,
    )
    await session.flush()

    return FeatureExtractionResult(
        question_id=question_id,
        qid=question.qid,
        persisted=True,
        skipped_reason=None,
        cache_hit=extraction.cache_hit,
        cost_estimate_usd=extraction.estimated_cost_usd,
        cost_saved_usd=extraction.cost_saved_usd,
        extractor_version=extraction.extractor_version,
        mechanical=mechanical,
        features=extraction.features,
    )
