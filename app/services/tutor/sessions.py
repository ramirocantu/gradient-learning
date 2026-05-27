from __future__ import annotations

from typing import Any

from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question, QuestionTag
from app.services.tutor.outline import resolve_node_labels


class SessionNotFoundError(Exception):
    pass


async def get_latest_session_id(session: AsyncSession) -> str | None:
    row = (
        await session.execute(
            select(Attempt.uworld_test_id)
            .where(Attempt.uworld_test_id.is_not(None))
            .order_by(desc(Attempt.attempted_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


async def get_recent_sessions(session: AsyncSession, *, n: int = 5) -> list[dict[str, Any]]:
    if n < 1:
        n = 1
    if n > 50:
        n = 50

    rows = (
        await session.execute(
            select(
                Attempt.uworld_test_id.label("test_id"),
                func.count(Attempt.id).label("attempt_count"),
                func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct_count"),
                func.min(Attempt.attempted_at).label("started_at"),
                func.max(Attempt.attempted_at).label("ended_at"),
            )
            .where(Attempt.uworld_test_id.is_not(None))
            .group_by(Attempt.uworld_test_id)
            .order_by(func.max(Attempt.attempted_at).desc())
            .limit(n)
        )
    ).all()

    out: list[dict[str, Any]] = []
    for r in rows:
        attempts = int(r.attempt_count)
        correct = int(r.correct_count or 0)
        out.append(
            {
                "test_id": r.test_id,
                "attempt_count": attempts,
                "correct_count": correct,
                "accuracy": (correct / attempts) if attempts else 0.0,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            }
        )
    return out


async def get_session_summary(session: AsyncSession, *, test_id: str) -> dict[str, Any]:
    summary = (
        await session.execute(
            select(
                func.count(Attempt.id).label("attempt_count"),
                func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct_count"),
                func.min(Attempt.attempted_at).label("started_at"),
                func.max(Attempt.attempted_at).label("ended_at"),
            ).where(Attempt.uworld_test_id == test_id)
        )
    ).one()
    if not summary.attempt_count:
        raise SessionNotFoundError(test_id)

    attempts_total = int(summary.attempt_count)
    correct_total = int(summary.correct_count or 0)
    accuracy = (correct_total / attempts_total) if attempts_total else 0.0

    # T38 (V-O1, V-T1, V-O5): per-node breakdown. Each session attempt's
    # question carries canonical node tags; a question tagged to N nodes counts
    # in each (set membership rollup, V-O1 — ⊥ summed once). Non-overridden
    # tags only. `by_topic` is sorted by node_id (deterministic); `top_topics`
    # ranks by attempt volume (data ordering, ⊥ verdict — V-M1), capped at 5.
    tag_rows = (
        await session.execute(
            select(QuestionTag.node_id, Attempt.is_correct)
            .join(Attempt, Attempt.question_id == QuestionTag.question_id)
            .where(Attempt.uworld_test_id == test_id)
            .where(QuestionTag.is_overridden.is_(False))
        )
    ).all()
    per_node: dict[int, dict[str, int]] = {}
    for node_id, is_correct in tag_rows:
        bucket = per_node.setdefault(node_id, {"attempt_count": 0, "correct_count": 0})
        bucket["attempt_count"] += 1
        if is_correct:
            bucket["correct_count"] += 1

    labels = await resolve_node_labels(session, per_node.keys())
    by_topic = [
        {
            **labels[node_id],
            "attempt_count": counts["attempt_count"],
            "correct_count": counts["correct_count"],
            "accuracy": (
                counts["correct_count"] / counts["attempt_count"]
                if counts["attempt_count"]
                else 0.0
            ),
        }
        for node_id, counts in sorted(per_node.items())
        if node_id in labels
    ]
    top_topics = sorted(
        by_topic,
        key=lambda t: (-t["attempt_count"], t["node_id"]),
    )[:5]

    flagged_rows = (
        await session.execute(
            select(Attempt.id, Question.qid, Question.stem_plain)
            .join(Question, Question.id == Attempt.question_id)
            .join(AttemptNote, AttemptNote.attempt_id == Attempt.id)
            .where(Attempt.uworld_test_id == test_id)
            .where(AttemptNote.flag_for_review.is_(True))
            .distinct()
        )
    ).all()
    flagged = [
        {"attempt_id": r.id, "qid": r.qid, "stem_preview": (r.stem_plain or "")[:240]}
        for r in flagged_rows
    ]

    note_rows = (
        await session.execute(
            select(AttemptNote, Question.qid)
            .join(Attempt, Attempt.id == AttemptNote.attempt_id)
            .join(Question, Question.id == Attempt.question_id)
            .where(Attempt.uworld_test_id == test_id)
            .order_by(AttemptNote.created_at.asc())
        )
    ).all()
    notes = [
        {
            "id": n.id,
            "attempt_id": n.attempt_id,
            "qid": qid,
            "text": n.note_text,
            "flag_for_review": n.flag_for_review,
            "source": n.source,
            "created_at": n.created_at.isoformat(),
        }
        for (n, qid) in note_rows
    ]

    return {
        "test_id": test_id,
        "attempt_count": attempts_total,
        "correct_count": correct_total,
        "accuracy": accuracy,
        "started_at": summary.started_at.isoformat() if summary.started_at else None,
        "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
        "by_topic": by_topic,
        "top_topics": top_topics,
        "flagged_attempts": flagged,
        "notes": notes,
    }
