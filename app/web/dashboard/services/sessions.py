"""Data-fetching helpers for the Sessions dashboard view — T14 stub.

The PoC's session rollups joined `QuestionTag.topic_id → Topic.name` for the
"top topics" lists rendered on each session row. Topic + topic_id columns are
gone (T1/T2); restoring per-session topic breakdowns needs the node_id
resolver (`OutlineLookup.path_of`) and the V-O1 subtree rollup.

Stub keeps the public surface so `app/web/dashboard/routes/sessions.py`
loads. Session rows still surface accurate attempt/correct/flag/note counts
(those are outline-free) — only `top_topics` / `topic_labels` are empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionRowTopic:
    topic_id: int | None
    label: str
    attempt_count: int


@dataclass(frozen=True)
class SessionRow:
    uworld_test_id: str | None
    started_at: datetime | None
    ended_at: datetime | None
    attempt_count: int
    correct_count: int
    flag_count: int
    note_count: int
    top_topics: list[SessionRowTopic]


@dataclass(frozen=True)
class SessionAttempt:
    attempt_id: int
    question_id: int
    qid: str
    attempted_at: datetime
    selected_choice: str
    is_correct: bool
    flagged: bool
    topic_labels: list[str]
    note_count: int


@dataclass(frozen=True)
class SessionDetail:
    uworld_test_id: str | None
    started_at: datetime | None
    ended_at: datetime | None
    attempt_count: int
    correct_count: int
    flag_count: int
    note_count: int
    top_topics: list[SessionRowTopic]
    attempts: list[SessionAttempt]


async def list_sessions(session: AsyncSession, *, limit: int = 20) -> list[SessionRow]:
    """Per-`uworld_test_id` rollup. `top_topics` empty until node_id port lands."""
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    rows = (
        await session.execute(
            select(
                Attempt.uworld_test_id.label("test_id"),
                func.count(Attempt.id).label("attempt_count"),
                func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct"),
                func.min(Attempt.attempted_at).label("started_at"),
                func.max(Attempt.attempted_at).label("ended_at"),
                func.sum(case((Attempt.flagged.is_(True), 1), else_=0)).label("flag_count"),
            )
            .group_by(Attempt.uworld_test_id)
            .order_by(func.max(Attempt.attempted_at).desc())
            .limit(limit)
        )
    ).all()

    note_count_by_test = await _notes_counts_by_test_id(session)

    return [
        SessionRow(
            uworld_test_id=r.test_id,
            started_at=r.started_at,
            ended_at=r.ended_at,
            attempt_count=int(r.attempt_count or 0),
            correct_count=int(r.correct or 0),
            flag_count=int(r.flag_count or 0),
            note_count=note_count_by_test.get(r.test_id, 0),
            top_topics=[],
        )
        for r in rows
    ]


async def _notes_counts_by_test_id(session: AsyncSession) -> dict[str | None, int]:
    rows = (
        await session.execute(
            select(
                Attempt.uworld_test_id.label("test_id"),
                func.count(AttemptNote.id).label("n"),
            )
            .join(AttemptNote, AttemptNote.attempt_id == Attempt.id)
            .group_by(Attempt.uworld_test_id)
        )
    ).all()
    return {r.test_id: int(r.n or 0) for r in rows}


async def get_session_detail(
    session: AsyncSession, *, test_id: str | None
) -> SessionDetail | None:
    """Per-session detail. `top_topics` + `topic_labels` empty until node_id port."""
    summary = (
        await session.execute(
            select(
                func.count(Attempt.id).label("n"),
                func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct"),
                func.min(Attempt.attempted_at).label("started_at"),
                func.max(Attempt.attempted_at).label("ended_at"),
                func.sum(case((Attempt.flagged.is_(True), 1), else_=0)).label("flag_count"),
            ).where(Attempt.uworld_test_id == test_id)
        )
    ).one()
    if not summary.n:
        return None

    note_total = (
        await session.execute(
            select(func.count(AttemptNote.id))
            .join(Attempt, Attempt.id == AttemptNote.attempt_id)
            .where(Attempt.uworld_test_id == test_id)
        )
    ).scalar_one()

    attempt_rows = (
        await session.execute(
            select(Attempt, Question.qid)
            .join(Question, Question.id == Attempt.question_id)
            .where(Attempt.uworld_test_id == test_id)
            .order_by(Attempt.attempted_at.asc())
        )
    ).all()
    note_count_by_attempt = {
        row[0]: int(row[1])
        for row in (
            await session.execute(
                select(AttemptNote.attempt_id, func.count(AttemptNote.id))
                .join(Attempt, Attempt.id == AttemptNote.attempt_id)
                .where(Attempt.uworld_test_id == test_id)
                .group_by(AttemptNote.attempt_id)
            )
        ).all()
    }

    attempts = [
        SessionAttempt(
            attempt_id=a.id,
            question_id=a.question_id,
            qid=qid,
            attempted_at=a.attempted_at,
            selected_choice=a.selected_choice,
            is_correct=a.is_correct,
            flagged=a.flagged,
            topic_labels=[],
            note_count=note_count_by_attempt.get(a.id, 0),
        )
        for (a, qid) in attempt_rows
    ]

    return SessionDetail(
        uworld_test_id=test_id,
        started_at=summary.started_at,
        ended_at=summary.ended_at,
        attempt_count=int(summary.n or 0),
        correct_count=int(summary.correct or 0),
        flag_count=int(summary.flag_count or 0),
        note_count=int(note_total or 0),
        top_topics=[],
        attempts=attempts,
    )
