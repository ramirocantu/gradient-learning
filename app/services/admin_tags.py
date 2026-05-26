"""Manual-tag service.

Pure async functions for inserting and removing `QuestionTag` rows.
Used by the backend's admin endpoints and by the dashboard.

T14 port: the PoC's 3-target signature (`topic_id` / `content_category_id` /
`skill`) is retired (T2: V-T1 canonical shape is `node_id`-only). The new
signature takes a single `node_id` already resolved by the caller via
`OutlineLookup.node_id_by_path`. `delete_tag` updates the forbidden-source
list to the new V-T2 enum (schema_map cannot be deleted; manual is
hard-deleted; llm is soft-deleted via `is_overridden`).
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
    """Raised when the requested tag's payload is invalid."""


class ManualTagConflictError(ValueError):
    """Raised when the tag would violate a uniqueness or check constraint."""


class TagNotFoundError(LookupError):
    """Raised when tag_id does not exist."""


class TagDeleteForbiddenError(PermissionError):
    """Raised when the tag source cannot be deleted (e.g. schema_map)."""


async def create_manual_tag(
    session: AsyncSession,
    question_id: int,
    *,
    node_id: int,
    rationale: str | None = None,
) -> QuestionTag:
    """Insert a manual `(question_id, node_id, source='manual')` tag.

    V-T3: confidence is NULL for source='manual'.

    Raises:
        QuestionNotFoundError: question_id not in `questions`.
        ManualTagConflictError: row already exists or violates a constraint.
    """
    question = (
        await session.execute(select(Question).where(Question.id == question_id))
    ).scalar_one_or_none()
    if question is None:
        raise QuestionNotFoundError(f"question_id={question_id} not found")

    row = QuestionTag(
        question_id=question_id,
        node_id=node_id,
        source="manual",
        confidence=None,
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

    Manual tags hard-delete; LLM tags soft-delete via `is_overridden`;
    schema_map tags are forbidden (deterministic — re-run the importer).
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
