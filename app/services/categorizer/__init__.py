"""Categorizer orchestrator: glue between llm.categorize and QuestionTag persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from anthropic import AsyncAnthropic
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag
from app.services.categorizer import llm as llm_module
from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.llm import (
    EXTRACTOR_VERSION,  # re-exported for callers; bind reflects import-time value
    CategorizeResult,
    LlmTagSuggestion,
    categorize,
)

# Quiet ruff; orchestrator reads via `llm_module.EXTRACTOR_VERSION` at call time,
# but `EXTRACTOR_VERSION` is re-exported for backwards-compat with existing imports.
__all__ = [
    "EXTRACTOR_VERSION",
    "CategorizeResult",
    "LlmTagSuggestion",
    "TagQuestionResult",
    "QuestionNotFoundError",
    "tag_question",
    "serializable_suggestions",
]
from app.services.categorizer.outline_lookup import OutlineLookup

logger = logging.getLogger(__name__)


class QuestionNotFoundError(LookupError):
    """Raised when tag_question is called with an unknown question_id."""


@dataclass(frozen=True)
class TagQuestionResult:
    question_id: int
    qid: str
    targets_persisted: int
    targets_replaced: int
    suggestions_unresolved: int
    manual_tags_preserved: int
    cache_hit: bool
    cost_estimate_usd: float
    cost_saved_usd: float
    extractor_version: str
    categorize_result: CategorizeResult


def _resolved_target(
    suggestion: LlmTagSuggestion, lookup: OutlineLookup
) -> tuple[str, int | None, int | None, int | None]:
    """Resolve a single LLM suggestion to (kind, topic_id, content_category_id, skill).

    Returns (kind, topic_id, cc_id, skill) — exactly one of the three id fields
    is non-None when resolution succeeds. All three are None when resolution
    fails (e.g., topic name not in outline).
    """
    if suggestion.kind == "skill":
        if isinstance(suggestion.identifier, int) and 1 <= suggestion.identifier <= 4:
            return ("skill", None, None, suggestion.identifier)
        return ("skill", None, None, None)
    if suggestion.kind == "content_category":
        code = str(suggestion.identifier).strip()
        cc_id = lookup.content_category_id(code)
        return ("content_category", None, cc_id, None)
    if suggestion.kind == "topic":
        path = str(suggestion.identifier).strip()
        topic_id = lookup.topic_id_by_path(path)
        return ("topic", topic_id, None, None)
    return (suggestion.kind, None, None, None)


def _row_from(
    question_id: int,
    topic_id: int | None,
    cc_id: int | None,
    skill: int | None,
    *,
    confidence: float,
    rationale: str,
) -> QuestionTag:
    return QuestionTag(
        question_id=question_id,
        topic_id=topic_id,
        content_category_id=cc_id,
        skill=skill,
        confidence=confidence,
        source="llm",
        rationale=rationale or None,
        extractor_version=llm_module.EXTRACTOR_VERSION,
    )


async def _count_tags_by_source(session: AsyncSession, question_id: int) -> dict[str, int]:
    rows = (
        await session.execute(
            select(QuestionTag.source, func.count())
            .where(QuestionTag.question_id == question_id)
            .group_by(QuestionTag.source)
        )
    ).all()
    return {source: count for source, count in rows}


async def tag_question(
    question_id: int,
    session: AsyncSession,
    *,
    lookup: OutlineLookup,
    anthropic_client: AsyncAnthropic,
    cache: CategorizerCache | None = None,
) -> TagQuestionResult:
    """Categorize a question via the LLM and persist QuestionTag rows.

    Semantics (idempotent for source='llm'):
      - DELETE existing source='llm' rows for this question (re-runs are fresh).
      - Call llm.categorize() with the question's UWorld tags + stem + explanation.
      - Resolve each LlmTagSuggestion to a real DB target via OutlineLookup.
      - INSERT one QuestionTag row per resolved target, with source='llm',
        the LLM's confidence/rationale, and extractor_version stamped.
      - Set question.needs_categorization=False. Bump question.last_updated_at.

    Preserves source='manual' rows untouched.

    Raises QuestionNotFoundError on missing question_id.
    """
    question = (
        await session.execute(select(Question).where(Question.id == question_id))
    ).scalar_one_or_none()
    if question is None:
        raise QuestionNotFoundError(f"question_id={question_id} not found")

    counts = await _count_tags_by_source(session, question_id)
    targets_replaced = counts.get("llm", 0)
    manual_preserved = counts.get("manual", 0)

    if targets_replaced:
        await session.execute(
            delete(QuestionTag).where(
                QuestionTag.question_id == question_id,
                QuestionTag.source == "llm",
            )
        )

    # Read EXTRACTOR_VERSION via the `llm` module so runtime overrides (e.g.
    # an eval/smoke script that mutates `llm.EXTRACTOR_VERSION`) take effect.
    cat = await categorize(
        question,
        anthropic_client=anthropic_client,
        outline_lookup=lookup,
        cache=cache,
        extractor_version=llm_module.EXTRACTOR_VERSION,
    )

    seen: set[tuple[str, int]] = set()
    persisted = 0
    unresolved = 0
    for suggestion in cat.suggestions:
        kind, topic_id, cc_id, skill = _resolved_target(suggestion, lookup)
        target_value = topic_id or cc_id or skill
        if target_value is None:
            unresolved += 1
            logger.warning(
                "tag_question qid=%s: dropping unresolved suggestion kind=%s ident=%r",
                question.qid,
                suggestion.kind,
                suggestion.identifier,
            )
            continue
        key = (kind, target_value)
        if key in seen:
            continue
        seen.add(key)
        session.add(
            _row_from(
                question_id,
                topic_id,
                cc_id,
                skill,
                confidence=suggestion.confidence,
                rationale=suggestion.rationale,
            )
        )
        persisted += 1

    question.needs_categorization = False
    question.last_updated_at = datetime.now(tz=timezone.utc)

    await session.flush()

    return TagQuestionResult(
        question_id=question_id,
        qid=question.qid,
        targets_persisted=persisted,
        targets_replaced=targets_replaced,
        suggestions_unresolved=unresolved,
        manual_tags_preserved=manual_preserved,
        cache_hit=cat.cache_hit,
        cost_estimate_usd=cat.estimated_cost_usd,
        cost_saved_usd=cat.cost_saved_usd,
        extractor_version=cat.extractor_version,
        categorize_result=cat,
    )


def serializable_suggestions(cat: CategorizeResult) -> list[dict]:
    """For the recategorize endpoint's JSON response."""
    return [
        {
            "kind": s.kind,
            "identifier": s.identifier,
            "under_content_category": s.under_content_category,
            "confidence": s.confidence,
            "rationale": s.rationale,
        }
        for s in cat.suggestions
    ]
