from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Topic


async def get_recent_captures(session: AsyncSession, *, n: int = 5) -> list[dict[str, Any]]:
    if n < 1:
        n = 1
    if n > 50:
        n = 50

    rows = (
        await session.execute(
            select(Attempt, Question.qid, Question.stem_plain)
            .join(Question, Question.id == Attempt.question_id)
            .order_by(Attempt.attempted_at.desc())
            .limit(n)
        )
    ).all()

    attempt_ids = [a.id for a, _, _ in rows]
    question_ids = [a.question_id for a, _, _ in rows]

    topic_rows = (
        (
            await session.execute(
                select(QuestionTag.question_id, Topic.name)
                .join(Topic, Topic.id == QuestionTag.topic_id)
                .where(QuestionTag.question_id.in_(question_ids))
                .where(QuestionTag.is_overridden.is_(False))
                .where(QuestionTag.topic_id.is_not(None))
            )
        ).all()
        if question_ids
        else []
    )
    topics_by_q: dict[int, list[str]] = {}
    for qid, name in topic_rows:
        topics_by_q.setdefault(qid, []).append(name)

    note_rows = (
        (
            await session.execute(
                select(AttemptNote)
                .where(AttemptNote.attempt_id.in_(attempt_ids))
                .order_by(AttemptNote.created_at.asc())
            )
        ).all()
        if attempt_ids
        else []
    )
    notes_by_attempt: dict[int, list[dict[str, Any]]] = {}
    for (note,) in note_rows:
        notes_by_attempt.setdefault(note.attempt_id, []).append(
            {
                "id": note.id,
                "text": note.note_text,
                "flag_for_review": note.flag_for_review,
                "source": note.source,
                "created_at": note.created_at.isoformat(),
            }
        )

    out: list[dict[str, Any]] = []
    for attempt, qid, stem in rows:
        out.append(
            {
                "attempt_id": attempt.id,
                "question_id": attempt.question_id,
                "qid": qid,
                "stem_preview": (stem or "")[:240],
                "attempted_at": attempt.attempted_at.isoformat(),
                "is_correct": attempt.is_correct,
                "selected_choice": attempt.selected_choice,
                "flagged": attempt.flagged,
                "uworld_test_id": attempt.uworld_test_id,
                "topics": topics_by_q.get(attempt.question_id, []),
                "notes": notes_by_attempt.get(attempt.id, []),
            }
        )
    return out
