"""Manual-tag service.

Pure async functions for inserting and removing `QuestionTag` rows.
Used by the backend's admin endpoints and by the dashboard directly;
the dashboard does not self-call the HTTP endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag


class QuestionNotFoundError(LookupError):
    """Raised when the target question_id does not exist."""


class ManualTagValidationError(ValueError):
    """Raised when the requested tag does not specify exactly one target."""


class ManualTagConflictError(ValueError):
    """Raised when the tag would violate a uniqueness or check constraint."""


class TagNotFoundError(LookupError):
    """Raised when tag_id does not exist."""


class TagDeleteForbiddenError(PermissionError):
    """Raised when the tag source cannot be deleted (e.g. uworld_map)."""


async def create_manual_tag(
    session: AsyncSession,
    question_id: int,
    *,
    topic_id: int | None = None,
    content_category_id: int | None = None,
    skill: int | None = None,
    rationale: str | None = None,
) -> QuestionTag:
    """Insert a single manual tag.

    Enforces the exactly-one-target invariant before touching the DB so callers
    get a clean ``ManualTagValidationError`` instead of an opaque
    ``IntegrityError`` from the ``ck_question_tags_exactly_one_target`` check.

    Raises:
        QuestionNotFoundError: ``question_id`` not in ``questions`` table.
        ManualTagValidationError: zero or 2+ targets supplied.
        ManualTagConflictError: row already exists or violates a constraint.
    """
    provided = [v for v in (topic_id, content_category_id, skill) if v is not None]
    if len(provided) != 1:
        raise ManualTagValidationError(
            "exactly one of topic_id, content_category_id, skill must be provided"
            f" (got {len(provided)})"
        )

    question = (
        await session.execute(select(Question).where(Question.id == question_id))
    ).scalar_one_or_none()
    if question is None:
        raise QuestionNotFoundError(f"question_id={question_id} not found")

    row = QuestionTag(
        question_id=question_id,
        topic_id=topic_id,
        content_category_id=content_category_id,
        skill=skill,
        confidence=1.0,
        source="manual",
        rationale=(rationale.strip() or None) if rationale else None,
    )
    session.add(row)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise ManualTagConflictError(
            "manual tag conflicts with existing row or violates schema constraint"
        ) from exc
    return row


async def delete_tag(session: AsyncSession, tag_id: int) -> None:
    """Remove or soft-delete a tag.

    Manual tags are hard-deleted. LLM tags are soft-deleted via is_overridden.
    Raises TagNotFoundError if the tag does not exist.
    Raises TagDeleteForbiddenError for uworld_map tags.
    """
    row = (
        await session.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
    ).scalar_one_or_none()
    if row is None:
        raise TagNotFoundError(f"tag_id={tag_id} not found")
    if row.source == "manual":
        await session.delete(row)
    elif row.source == "llm":
        row.is_overridden = True
        row.overridden_at = datetime.now(timezone.utc)
    else:
        raise TagDeleteForbiddenError(f"tag source={row.source!r} cannot be removed")
    await session.flush()
