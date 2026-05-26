"""Data-fetching helpers for the Sessions dashboard view (Ticket 6.9d).

A "session" is the set of `attempts` rows sharing a `uworld_test_id`. Rows
with `uworld_test_id IS NULL` are aggregated into a single "Unsessioned"
pseudo-row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Topic


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


_CARS_LABEL = "CARS (skill)"
_TOP_TOPICS_LIMIT = 3


async def _top_topics_for(
    session: AsyncSession,
    *,
    test_ids: list[str],
    include_null: bool,
) -> dict[str | None, list[SessionRowTopic]]:
    """Compute top-3 topics for each session bucket, attempt-count weighted.

    Returns a dict keyed by `uworld_test_id` (with `None` for the unsessioned
    bucket when ``include_null=True``).
    """
    if not test_ids and not include_null:
        return {}

    # Topic-tagged attempts.
    topic_stmt = (
        select(
            Attempt.uworld_test_id,
            QuestionTag.topic_id,
            Topic.name.label("topic_name"),
            func.count(Attempt.id).label("attempt_count"),
        )
        .join(QuestionTag, QuestionTag.question_id == Attempt.question_id)
        .join(Topic, Topic.id == QuestionTag.topic_id)
        .where(QuestionTag.is_overridden.is_(False))
        .where(QuestionTag.topic_id.is_not(None))
    )

    # CARS attempts: questions with a skill-only QuestionTag (no topic_id).
    cars_stmt = (
        select(
            Attempt.uworld_test_id,
            func.count(Attempt.id).label("attempt_count"),
        )
        .join(QuestionTag, QuestionTag.question_id == Attempt.question_id)
        .where(QuestionTag.is_overridden.is_(False))
        .where(QuestionTag.skill.is_not(None))
        .where(QuestionTag.topic_id.is_(None))
    )

    conds = []
    if test_ids:
        conds.append(Attempt.uworld_test_id.in_(test_ids))
    if include_null:
        conds.append(Attempt.uworld_test_id.is_(None))
    if not conds:
        return {}

    from sqlalchemy import or_

    topic_stmt = topic_stmt.where(or_(*conds)).group_by(
        Attempt.uworld_test_id, QuestionTag.topic_id, Topic.name
    )
    cars_stmt = cars_stmt.where(or_(*conds)).group_by(Attempt.uworld_test_id)

    topic_rows = (await session.execute(topic_stmt)).all()
    cars_rows = (await session.execute(cars_stmt)).all()

    by_key: dict[str | None, list[SessionRowTopic]] = {}
    for r in topic_rows:
        by_key.setdefault(r.uworld_test_id, []).append(
            SessionRowTopic(
                topic_id=r.topic_id,
                label=r.topic_name,
                attempt_count=int(r.attempt_count),
            )
        )
    for r in cars_rows:
        by_key.setdefault(r.uworld_test_id, []).append(
            SessionRowTopic(
                topic_id=None,
                label=_CARS_LABEL,
                attempt_count=int(r.attempt_count),
            )
        )

    out: dict[str | None, list[SessionRowTopic]] = {}
    for k, rows in by_key.items():
        rows.sort(key=lambda t: (-t.attempt_count, t.label))
        out[k] = rows[:_TOP_TOPICS_LIMIT]
    return out


async def _notes_counts_by_test_id(
    session: AsyncSession,
    *,
    test_ids: list[str],
    include_null: bool,
) -> tuple[dict[str | None, int], dict[str | None, int]]:
    """Return (note_count_by_test_id, flag_count_by_test_id).

    Notes are joined to attempts so the aggregation rolls up to the session
    bucket via `attempts.uworld_test_id`.
    """
    if not test_ids and not include_null:
        return {}, {}

    from sqlalchemy import or_

    conds = []
    if test_ids:
        conds.append(Attempt.uworld_test_id.in_(test_ids))
    if include_null:
        conds.append(Attempt.uworld_test_id.is_(None))

    stmt = (
        select(
            Attempt.uworld_test_id,
            func.count(AttemptNote.id).label("note_count"),
            func.sum(case((AttemptNote.flag_for_review.is_(True), 1), else_=0)).label("flag_count"),
        )
        .join(AttemptNote, AttemptNote.attempt_id == Attempt.id)
        .where(or_(*conds))
        .group_by(Attempt.uworld_test_id)
    )

    rows = (await session.execute(stmt)).all()
    notes: dict[str | None, int] = {}
    flags: dict[str | None, int] = {}
    for r in rows:
        notes[r.uworld_test_id] = int(r.note_count or 0)
        flags[r.uworld_test_id] = int(r.flag_count or 0)
    return notes, flags


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


async def list_sessions(session: AsyncSession, *, limit: int = 20) -> list[SessionRow]:
    """Recent sessions, sorted by latest attempt DESC, plus an unsessioned bucket.

    The unsessioned bucket — a single aggregation across all rows with
    `uworld_test_id IS NULL` — is always appended at the end of the list
    when any null-test_id attempts exist. It does not count against ``limit``.
    """
    agg_stmt = (
        select(
            Attempt.uworld_test_id.label("uworld_test_id"),
            func.min(Attempt.attempted_at).label("started_at"),
            func.max(Attempt.attempted_at).label("ended_at"),
            func.count(Attempt.id).label("attempt_count"),
            func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct_count"),
        )
        .where(Attempt.uworld_test_id.is_not(None))
        .group_by(Attempt.uworld_test_id)
        .order_by(func.max(Attempt.attempted_at).desc())
        .limit(limit)
    )
    rows = (await session.execute(agg_stmt)).all()

    null_stmt = select(
        func.min(Attempt.attempted_at).label("started_at"),
        func.max(Attempt.attempted_at).label("ended_at"),
        func.count(Attempt.id).label("attempt_count"),
        func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct_count"),
    ).where(Attempt.uworld_test_id.is_(None))
    null_row = (await session.execute(null_stmt)).one()

    test_ids = [r.uworld_test_id for r in rows]
    has_unsessioned = (null_row.attempt_count or 0) > 0

    topics_by_id = await _top_topics_for(session, test_ids=test_ids, include_null=has_unsessioned)
    notes_by_id, flags_by_id = await _notes_counts_by_test_id(
        session, test_ids=test_ids, include_null=has_unsessioned
    )

    out: list[SessionRow] = []
    for r in rows:
        out.append(
            SessionRow(
                uworld_test_id=r.uworld_test_id,
                started_at=r.started_at,
                ended_at=r.ended_at,
                attempt_count=int(r.attempt_count),
                correct_count=int(r.correct_count or 0),
                flag_count=int(flags_by_id.get(r.uworld_test_id, 0)),
                note_count=int(notes_by_id.get(r.uworld_test_id, 0)),
                top_topics=topics_by_id.get(r.uworld_test_id, []),
            )
        )

    if has_unsessioned:
        out.append(
            SessionRow(
                uworld_test_id=None,
                started_at=null_row.started_at,
                ended_at=null_row.ended_at,
                attempt_count=int(null_row.attempt_count),
                correct_count=int(null_row.correct_count or 0),
                flag_count=int(flags_by_id.get(None, 0)),
                note_count=int(notes_by_id.get(None, 0)),
                top_topics=topics_by_id.get(None, []),
            )
        )

    return out


async def get_session_detail(session: AsyncSession, *, test_id: str | None) -> SessionDetail | None:
    """Single-session detail. ``test_id=None`` → the unsessioned aggregation."""
    if test_id is None:
        condition = Attempt.uworld_test_id.is_(None)
    else:
        condition = Attempt.uworld_test_id == test_id

    summary = (
        await session.execute(
            select(
                func.min(Attempt.attempted_at).label("started_at"),
                func.max(Attempt.attempted_at).label("ended_at"),
                func.count(Attempt.id).label("attempt_count"),
                func.sum(case((Attempt.is_correct.is_(True), 1), else_=0)).label("correct_count"),
            ).where(condition)
        )
    ).one()

    if not summary.attempt_count:
        return None

    if test_id is None:
        topics_map = await _top_topics_for(session, test_ids=[], include_null=True)
        notes_by_id, flags_by_id = await _notes_counts_by_test_id(
            session, test_ids=[], include_null=True
        )
        top_topics = topics_map.get(None, [])
        note_count = notes_by_id.get(None, 0)
        flag_count = flags_by_id.get(None, 0)
    else:
        topics_map = await _top_topics_for(session, test_ids=[test_id], include_null=False)
        notes_by_id, flags_by_id = await _notes_counts_by_test_id(
            session, test_ids=[test_id], include_null=False
        )
        top_topics = topics_map.get(test_id, [])
        note_count = notes_by_id.get(test_id, 0)
        flag_count = flags_by_id.get(test_id, 0)

    attempt_rows = (
        await session.execute(
            select(Attempt, Question.qid)
            .join(Question, Question.id == Attempt.question_id)
            .where(condition)
            .order_by(Attempt.attempted_at.asc(), Attempt.id.asc())
        )
    ).all()

    attempt_ids = [a.id for a, _ in attempt_rows]
    question_ids = list({a.question_id for a, _ in attempt_rows})

    # Per-attempt note count.
    notes_per_attempt: dict[int, int] = {}
    if attempt_ids:
        rows = (
            await session.execute(
                select(AttemptNote.attempt_id, func.count(AttemptNote.id))
                .where(AttemptNote.attempt_id.in_(attempt_ids))
                .group_by(AttemptNote.attempt_id)
            )
        ).all()
        notes_per_attempt = {a_id: int(n) for a_id, n in rows}

    # Topic labels per question (with CARS bucket for skill-only tags).
    topic_labels_by_q: dict[int, list[str]] = {q: [] for q in question_ids}
    if question_ids:
        topic_rows = (
            await session.execute(
                select(QuestionTag.question_id, Topic.name)
                .join(Topic, Topic.id == QuestionTag.topic_id)
                .where(QuestionTag.question_id.in_(question_ids))
                .where(QuestionTag.is_overridden.is_(False))
                .where(QuestionTag.topic_id.is_not(None))
            )
        ).all()
        for q_id, t_name in topic_rows:
            if t_name not in topic_labels_by_q[q_id]:
                topic_labels_by_q[q_id].append(t_name)

        cars_question_ids = (
            await session.execute(
                select(QuestionTag.question_id)
                .where(QuestionTag.question_id.in_(question_ids))
                .where(QuestionTag.is_overridden.is_(False))
                .where(QuestionTag.skill.is_not(None))
                .where(QuestionTag.topic_id.is_(None))
            )
        ).scalars()
        for q_id in cars_question_ids:
            if _CARS_LABEL not in topic_labels_by_q[q_id]:
                topic_labels_by_q[q_id].append(_CARS_LABEL)

    attempts_out: list[SessionAttempt] = []
    for attempt, qid in attempt_rows:
        attempts_out.append(
            SessionAttempt(
                attempt_id=attempt.id,
                question_id=attempt.question_id,
                qid=qid,
                attempted_at=attempt.attempted_at,
                selected_choice=attempt.selected_choice,
                is_correct=attempt.is_correct,
                flagged=attempt.flagged,
                topic_labels=topic_labels_by_q.get(attempt.question_id, []),
                note_count=notes_per_attempt.get(attempt.id, 0),
            )
        )

    return SessionDetail(
        uworld_test_id=test_id,
        started_at=summary.started_at,
        ended_at=summary.ended_at,
        attempt_count=int(summary.attempt_count),
        correct_count=int(summary.correct_count or 0),
        flag_count=int(flag_count),
        note_count=int(note_count),
        top_topics=top_topics,
        attempts=attempts_out,
    )
