"""Persist calibrated grounded-tag decisions to the canonical tag tables (T30).

T29's :func:`app.services.llm.grounded.generate_grounded_tags` returns
calibrated tag *decisions* in memory (``GroundedResult``); this seam writes
them durably:

- entity_kind ``'question'``  → ``question_tags`` rows.
- entity_kind ``'atomic_fact'`` → ``atomic_fact_tags`` rows, plus the
  denormalized ``atomic_facts.node_id`` primary pick.

Both follow the V-T2 re-run pattern: ``DELETE WHERE <target>_id=X AND
source='llm'`` then ``INSERT`` the fresh calibrated rows. ``manual`` and
``schema_map`` rows are never touched. The persisted ``confidence`` is the
V69-calibrated logprob grade (the model self-report is no longer collected);
``<0.5`` ⇒ ``manual_review=true`` — flagged for review, ⊥ silently dropped
(V-T3). ``extractor_version`` is stamped from the ``GroundedResult``; the
``embedding_version`` stamp lives on the ``content_embeddings`` rows the
recall layer consumed (V-E1) — the canonical tag shape has no such column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question, QuestionTag
from app.services.llm.grounded import GroundedResult

logger = logging.getLogger("app.services.kb.persist_tags")

QUESTION = "question"
ATOMIC_FACT = "atomic_fact"
_VALID_ENTITY_KINDS = (QUESTION, ATOMIC_FACT)


class EntityNotFoundError(LookupError):
    """Raised when the target question / atomic_fact id does not exist."""


@dataclass(frozen=True)
class PersistResult:
    entity_kind: str
    entity_id: int
    persisted: int                 # source='llm' rows inserted
    replaced: int                  # prior source='llm' rows deleted
    manual_review_flagged: int     # of persisted, how many Conf<0.5
    primary_node_id: int | None    # atomic_facts.node_id set (atomic_fact only)


def _quantize_conf(value: float) -> Decimal:
    """Match the ``Numeric(3,2)`` column precision so the V-T3 CHECK is
    evaluated against the value actually stored, not a wider float.
    """

    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


async def _count_llm_tags_question(session: AsyncSession, question_id: int) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(QuestionTag)
                .where(
                    QuestionTag.question_id == question_id,
                    QuestionTag.source == "llm",
                )
            )
        ).scalar_one()
    )


async def _count_llm_tags_fact(session: AsyncSession, atomic_fact_id: int) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(AtomicFactTag)
                .where(
                    AtomicFactTag.atomic_fact_id == atomic_fact_id,
                    AtomicFactTag.source == "llm",
                )
            )
        ).scalar_one()
    )


def _dedupe_by_node(result: GroundedResult):
    """First decision per node_id wins (candidates are already unique by
    node, but a defensive dedupe keeps the UQ(target,node,source) safe).
    """

    seen: set[int] = set()
    for tag in result.tags:
        if tag.node_id in seen:
            continue
        seen.add(tag.node_id)
        yield tag


async def _persist_question(
    session: AsyncSession, *, question_id: int, result: GroundedResult
) -> PersistResult:
    question = (
        await session.execute(select(Question).where(Question.id == question_id))
    ).scalar_one_or_none()
    if question is None:
        raise EntityNotFoundError(f"question id={question_id} not found")

    replaced = await _count_llm_tags_question(session, question_id)
    if replaced:
        await session.execute(
            delete(QuestionTag).where(
                QuestionTag.question_id == question_id,
                QuestionTag.source == "llm",
            )
        )

    persisted = 0
    flagged = 0
    for tag in _dedupe_by_node(result):
        session.add(
            QuestionTag(
                question_id=question_id,
                node_id=tag.node_id,
                source="llm",
                confidence=_quantize_conf(tag.calibrated_confidence),
                rationale=tag.rationale or None,
                extractor_version=result.extractor_version,
                manual_review=tag.manual_review,
            )
        )
        persisted += 1
        if tag.manual_review:
            flagged += 1

    await session.flush()
    return PersistResult(
        entity_kind=QUESTION,
        entity_id=question_id,
        persisted=persisted,
        replaced=replaced,
        manual_review_flagged=flagged,
        primary_node_id=None,
    )


async def _persist_atomic_fact(
    session: AsyncSession, *, atomic_fact_id: int, result: GroundedResult
) -> PersistResult:
    fact = (
        await session.execute(
            select(AtomicFact).where(AtomicFact.id == atomic_fact_id)
        )
    ).scalar_one_or_none()
    if fact is None:
        raise EntityNotFoundError(f"atomic_fact id={atomic_fact_id} not found")

    replaced = await _count_llm_tags_fact(session, atomic_fact_id)
    if replaced:
        await session.execute(
            delete(AtomicFactTag).where(
                AtomicFactTag.atomic_fact_id == atomic_fact_id,
                AtomicFactTag.source == "llm",
            )
        )

    persisted = 0
    flagged = 0
    # Track the best non-review pick for the denormalized primary node_id.
    best_node_id: int | None = None
    best_conf = float("-inf")
    for tag in _dedupe_by_node(result):
        session.add(
            AtomicFactTag(
                atomic_fact_id=atomic_fact_id,
                node_id=tag.node_id,
                source="llm",
                confidence=_quantize_conf(tag.calibrated_confidence),
                rationale=tag.rationale or None,
                extractor_version=result.extractor_version,
                manual_review=tag.manual_review,
            )
        )
        persisted += 1
        if tag.manual_review:
            flagged += 1
        elif tag.calibrated_confidence > best_conf:
            best_conf = tag.calibrated_confidence
            best_node_id = tag.node_id

    # Denormalized primary node: highest-confidence non-review pick. ⊥ auto-
    # assign a low-confidence node — leave NULL so the fact stays for review.
    fact.node_id = best_node_id

    await session.flush()
    return PersistResult(
        entity_kind=ATOMIC_FACT,
        entity_id=atomic_fact_id,
        persisted=persisted,
        replaced=replaced,
        manual_review_flagged=flagged,
        primary_node_id=best_node_id,
    )


async def persist_grounded_tags(
    session: AsyncSession,
    *,
    entity_kind: str,
    entity_id: int,
    result: GroundedResult,
) -> PersistResult:
    """Persist a ``GroundedResult`` to the canonical tag table for
    ``entity_kind`` (V-T2 re-run, V-T3 calibration + manual_review).

    Args:
        entity_kind: ``'question'`` or ``'atomic_fact'``.
        entity_id: the question / atomic_fact id being tagged.
        result: calibrated decisions from
            :func:`app.services.llm.grounded.generate_grounded_tags`.

    Raises:
        ValueError: unknown ``entity_kind``.
        EntityNotFoundError: target row does not exist.
    """

    if entity_kind == QUESTION:
        out = await _persist_question(
            session, question_id=entity_id, result=result
        )
    elif entity_kind == ATOMIC_FACT:
        out = await _persist_atomic_fact(
            session, atomic_fact_id=entity_id, result=result
        )
    else:
        raise ValueError(
            f"unknown entity_kind {entity_kind!r}; expected one of {_VALID_ENTITY_KINDS}"
        )

    logger.info(
        "persist_grounded_tags: kind=%s id=%d persisted=%d replaced=%d "
        "flagged=%d primary_node=%s extractor=%s",
        out.entity_kind,
        out.entity_id,
        out.persisted,
        out.replaced,
        out.manual_review_flagged,
        out.primary_node_id,
        result.extractor_version,
    )
    return out
