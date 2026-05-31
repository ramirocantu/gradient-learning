"""Discriminator-factor persistence — tutor/MCP write seam (T31, V-M1, V-M3).

The MCP host runs the Socratic dialogue and decides the factor text; this
seam only *persists* it (V-M1 — data persist, ⊥ verdicts/heuristics in the
signature). Append-only (V-M3): a `(question_id, factor_text)` pair is
deduped — re-writing the same factor returns the existing row rather than
inserting a duplicate or erroring, so question ↔ factor links survive
re-writes. The Notion mirror (`notion_block_id`, V-N1/V-N2) is filled by the
T32 write-back; T31 leaves it NULL.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question
from app.models.discriminator_factor import DiscriminatorFactor
from app.models.outline import OutlineNode

logger = logging.getLogger("app.services.tutor.discriminators")


class QuestionNotFoundError(LookupError):
    """Raised when the target question_id does not exist."""


class NodeNotFoundError(LookupError):
    """Raised when a provided node_id does not exist."""


async def _find_existing(
    session: AsyncSession, *, question_id: int, factor_text: str
) -> DiscriminatorFactor | None:
    return (
        await session.execute(
            select(DiscriminatorFactor).where(
                DiscriminatorFactor.question_id == question_id,
                DiscriminatorFactor.factor_text == factor_text,
            )
        )
    ).scalar_one_or_none()


async def write_discriminator_factor(
    session: AsyncSession,
    *,
    question_id: int,
    factor_text: str,
    node_id: int | None = None,
) -> DiscriminatorFactor:
    """Persist a discriminator factor, append-only + deduped (V-M3).

    Returns the existing row when ``(question_id, factor_text)`` already
    exists (idempotent re-write); otherwise inserts a new row. Carries no
    verdict/grade — the host reasons, this seam only persists (V-M1).

    Raises:
        ValueError: ``factor_text`` is empty or whitespace-only.
        QuestionNotFoundError: ``question_id`` does not exist.
        NodeNotFoundError: ``node_id`` given but does not exist.
    """

    factor_text = factor_text.strip()
    if factor_text == "":
        raise ValueError("factor_text must not be empty or whitespace-only")

    existing = await _find_existing(session, question_id=question_id, factor_text=factor_text)
    if existing is not None:
        return existing

    question = (
        await session.execute(select(Question).where(Question.id == question_id))
    ).scalar_one_or_none()
    if question is None:
        raise QuestionNotFoundError(f"question id={question_id} not found")

    if node_id is not None:
        node = (
            await session.execute(select(OutlineNode).where(OutlineNode.id == node_id))
        ).scalar_one_or_none()
        if node is None:
            raise NodeNotFoundError(f"node id={node_id} not found")

    row = DiscriminatorFactor(
        question_id=question_id,
        factor_text=factor_text,
        node_id=node_id,
    )
    session.add(row)
    try:
        await session.flush()
    except IntegrityError:
        # Race: another writer inserted the same (question_id, factor_text)
        # between our SELECT and INSERT. Append-only + idempotent → roll back
        # and return the winner (V-M3, ⊥ duplicate).
        await session.rollback()
        winner = await _find_existing(session, question_id=question_id, factor_text=factor_text)
        if winner is None:
            raise
        return winner

    logger.info(
        "write_discriminator_factor: qid_pk=%d node_id=%s len=%d",
        question_id,
        node_id,
        len(factor_text),
    )
    return row
