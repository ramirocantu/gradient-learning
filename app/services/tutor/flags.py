"""Tutor flag-for-review query — T14 partial port.

The topic-name join (`QuestionTag.topic_id → Topic.name`) is gone; the
`topics` list is left empty until the node_id port resolves labels via
`OutlineLookup.path_of`.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question


async def get_flagged_attempts(session: AsyncSession, *, limit: int = 20) -> list[dict[str, Any]]:
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    rows = (
        await session.execute(
            select(
                Attempt.id.label("attempt_id"),
                Attempt.question_id,
                Question.qid,
                Question.stem_plain,
                AttemptNote.note_text,
                AttemptNote.created_at,
            )
            .join(AttemptNote, AttemptNote.attempt_id == Attempt.id)
            .join(Question, Question.id == Attempt.question_id)
            .where(AttemptNote.flag_for_review.is_(True))
            .order_by(AttemptNote.created_at.desc())
            .limit(limit)
        )
    ).all()

    if not rows:
        return []

    base = settings.BACKEND_BASE_URL
    out: list[dict[str, Any]] = []
    for r in rows:
        relative_path = f"/questions/{r.question_id}"
        dashboard_url = f"{base.rstrip('/')}{relative_path}" if base else relative_path
        out.append(
            {
                "attempt_id": r.attempt_id,
                "qid": r.qid,
                "stem_preview": (r.stem_plain or "")[:240],
                # TODO(T14 follow-up): resolve QuestionTag.node_id → path_of().
                "topics": [],
                "note_text": r.note_text,
                "flagged_at": r.created_at.isoformat(),
                "dashboard_path": relative_path,
                "dashboard_url": dashboard_url,
            }
        )
    return out
