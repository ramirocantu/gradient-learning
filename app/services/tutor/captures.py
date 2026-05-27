from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question, QuestionTag
from app.services.tutor.outline import resolve_node_labels


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

    # T38 (V-T1, V-O5): resolve each question's canonical node tags → label +
    # `>>` path. Non-overridden tags only; dedup by node_id, deterministic order.
    tag_rows = (
        (
            await session.execute(
                select(QuestionTag.question_id, QuestionTag.node_id)
                .where(QuestionTag.question_id.in_(question_ids))
                .where(QuestionTag.is_overridden.is_(False))
            )
        ).all()
        if question_ids
        else []
    )
    labels = await resolve_node_labels(session, {nid for _, nid in tag_rows})
    topics_by_q: dict[int, dict[int, dict[str, Any]]] = {}
    for q_id, node_id in tag_rows:
        label = labels.get(node_id)
        if label is not None:
            topics_by_q.setdefault(q_id, {})[node_id] = label

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
                "topics": [
                    topics_by_q[attempt.question_id][nid]
                    for nid in sorted(topics_by_q.get(attempt.question_id, {}))
                ],
                "notes": notes_by_attempt.get(attempt.id, []),
            }
        )
    return out
