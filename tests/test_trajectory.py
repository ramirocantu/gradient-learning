"""Tests for SPEC §T40 — last-10 vs prior-10 accuracy trajectory.

Covers:
- §V36 — last vs prior 10 attempts; delta NULL when either window <5;
  arrow thresholds ±0.10.
- §V31 — subtree-membership rollup; multi-tag attempts dedupe within
  a single scope.
- §C — `Attempt.time_seconds` MUST NOT enter the computation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, Topic
from app.services.analyzer.trajectory import (
    ARROW_THRESHOLD,
    TrajectorySummary,
    trajectory_for_cc,
    trajectory_for_topic,
)


def _new_qid() -> str:
    return f"q-{uuid.uuid4().hex[:10]}"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_question() -> Question:
    return Question(
        qid=_new_qid(),
        passage_id=None,
        stem_html="<p>s</p>",
        stem_plain="s",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_content_hashes": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_content_hashes": []},
        ],
        correct_choice="A",
        explanation_html=None,
        explanation_plain="why",
        uworld_aamc_tags=[],
        needs_categorization=False,
    )


def _attempt(*, question_id: int, is_correct: bool, attempted_at: datetime) -> Attempt:
    return Attempt(
        question_id=question_id,
        attempted_at=attempted_at,
        selected_choice="A" if is_correct else "B",
        is_correct=is_correct,
        time_seconds=None,
        flagged=False,
    )


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _second_cc(session: AsyncSession) -> ContentCategory:
    rows = (await session.execute(select(ContentCategory).limit(2))).scalars().all()
    return rows[1]


async def _make_topic_tree(
    session: AsyncSession, cc: ContentCategory, *, label: str
) -> tuple[Topic, Topic, Topic]:
    parent = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T40 parent {label}",
        disciplines=[],
        depth=0,
        position=910,
    )
    sibling = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T40 sibling {label}",
        disciplines=[],
        depth=0,
        position=911,
    )
    session.add_all([parent, sibling])
    await session.flush()
    child = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent.id,
        name=f"T40 child {label}",
        disciplines=[],
        depth=1,
        position=912,
    )
    session.add(child)
    await session.flush()
    return parent, child, sibling


async def _add_tagged_attempts(
    session: AsyncSession,
    *,
    cc: ContentCategory | None = None,
    topic: Topic | None = None,
    pattern: list[bool],
    base_time: datetime,
    spacing: timedelta = timedelta(minutes=1),
) -> list[Attempt]:
    """Create one question per `pattern` entry, tag it at `cc` or `topic`,
    add one attempt per question with the given correctness, spaced by
    `spacing` starting at `base_time` (index 0 = oldest).
    """
    assert (cc is None) ^ (topic is None), "tag at exactly one of cc/topic"
    attempts: list[Attempt] = []
    for i, is_correct in enumerate(pattern):
        q = _make_question()
        session.add(q)
        await session.flush()
        if cc is not None:
            session.add(
                QuestionTag(
                    question_id=q.id,
                    content_category_id=cc.id,
                    confidence=0.9,
                    source="llm",
                )
            )
        else:
            assert topic is not None
            session.add(
                QuestionTag(
                    question_id=q.id,
                    topic_id=topic.id,
                    confidence=0.9,
                    source="llm",
                )
            )
        att = _attempt(
            question_id=q.id,
            is_correct=is_correct,
            attempted_at=base_time + spacing * i,
        )
        session.add(att)
        attempts.append(att)
    await session.flush()
    return attempts


# --- §V36: window math ---


async def test_cc_last_and_prior_windows_compute_delta(
    db_session: AsyncSession,
) -> None:
    """Last 10 mostly-correct (8/10), prior 10 mostly-wrong (2/10) → big +Δ."""
    cc = await _first_cc(db_session)
    # 10 prior (oldest), then 10 last (newest). attempted_at increases with i,
    # so the last 10 chronologically are indices 10..19.
    prior = [True, True] + [False] * 8  # 2 / 10 correct
    last = [True] * 8 + [False] * 2  # 8 / 10 correct
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        pattern=prior + last,
        base_time=_now() - timedelta(hours=1),
    )

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.last.n == 10
    assert s.last.correct == 8
    assert s.prior.n == 10
    assert s.prior.correct == 2
    assert s.delta == pytest.approx(0.6)
    assert s.arrow == "↑"
    assert s.scope == f"cc:{cc.code}"


async def test_delta_none_when_last_window_below_5(
    db_session: AsyncSession,
) -> None:
    """Only 4 total attempts → last_n=4, prior_n=0; delta + arrow NULL."""
    cc = await _first_cc(db_session)
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        pattern=[True] * 4,
        base_time=_now() - timedelta(hours=1),
    )

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.last.n == 4
    assert s.prior.n == 0
    assert s.delta is None
    assert s.arrow is None


async def test_delta_none_when_prior_window_below_5(
    db_session: AsyncSession,
) -> None:
    """14 total → last_n=10, prior_n=4; delta + arrow NULL (prior insufficient)."""
    cc = await _first_cc(db_session)
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        pattern=[True] * 14,
        base_time=_now() - timedelta(hours=1),
    )

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.last.n == 10
    assert s.prior.n == 4
    assert s.delta is None
    assert s.arrow is None


async def test_boundary_minimum_15_total_surfaces_delta(
    db_session: AsyncSession,
) -> None:
    """Minimum to satisfy V36 (both windows ≥5): 15 total attempts.

    Last fills the newest 10; prior takes the next 5. With <15 total
    the prior window cannot reach 5 and delta stays NULL.
    """
    cc = await _first_cc(db_session)
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        pattern=[False] * 5 + [True] * 10,  # oldest 5 wrong, newest 10 correct
        base_time=_now() - timedelta(hours=1),
    )

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.last.n == 10 and s.last.correct == 10
    assert s.prior.n == 5 and s.prior.correct == 0
    assert s.delta == pytest.approx(1.0)
    assert s.arrow == "↑"


async def test_arrow_down_threshold(db_session: AsyncSession) -> None:
    """Δ ≤ −0.10 → ↓."""
    cc = await _first_cc(db_session)
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        # prior 10/10, last 2/10 → delta = -0.8
        pattern=[True] * 10 + [True] * 2 + [False] * 8,
        base_time=_now() - timedelta(hours=1),
    )
    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.delta is not None and s.delta <= -ARROW_THRESHOLD
    assert s.arrow == "↓"


async def test_arrow_flat_within_threshold(db_session: AsyncSession) -> None:
    """|Δ| < 0.10 → →."""
    cc = await _first_cc(db_session)
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        # prior 5/10, last 5/10 → delta = 0.0
        pattern=[True, False] * 5 + [True, False] * 5,
        base_time=_now() - timedelta(hours=1),
    )
    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.delta == pytest.approx(0.0)
    assert s.arrow == "→"


# --- §V31: subtree rollup ---


async def test_topic_subtree_rolls_up_child_attempts(
    db_session: AsyncSession,
) -> None:
    """Parent topic trajectory aggregates child attempts; sibling excluded."""
    cc = await _first_cc(db_session)
    parent, child, sibling = await _make_topic_tree(db_session, cc, label="rollup")

    # 10 parent attempts (all correct, recent) + 5 child attempts (all wrong,
    # older) = 15 in subtree → last=10 (all correct, parent), prior=5 (all
    # wrong, child). Sibling attempts MUST NOT bleed in.
    base = _now() - timedelta(hours=2)
    await _add_tagged_attempts(
        db_session,
        topic=child,
        pattern=[False] * 5,
        base_time=base,
    )
    await _add_tagged_attempts(
        db_session,
        topic=parent,
        pattern=[True] * 10,
        base_time=base + timedelta(hours=1),
    )
    await _add_tagged_attempts(
        db_session,
        topic=sibling,
        pattern=[True] * 5,
        base_time=base + timedelta(minutes=30),
    )

    s = await trajectory_for_topic(db_session, topic_id=parent.id)
    assert s.last.n == 10
    assert s.last.correct == 10
    assert s.prior.n == 5
    assert s.prior.correct == 0
    assert s.delta == pytest.approx(1.0)
    assert s.arrow == "↑"
    assert s.scope == f"topic:{parent.id}"

    sib = await trajectory_for_topic(db_session, topic_id=sibling.id)
    assert sib.last.n == 5
    assert sib.last.correct == 5
    # Only 5 attempts on sibling → prior=0 → delta None
    assert sib.prior.n == 0
    assert sib.delta is None


async def test_topic_scope_excludes_direct_cc_only_tags(
    db_session: AsyncSession,
) -> None:
    """A question tagged only at CC granularity has no topic_id and must
    NOT count for any topic scope (§V31).
    """
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="cc_only")

    # 5 questions tagged only at CC level (no topic).
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        pattern=[True] * 5,
        base_time=_now() - timedelta(hours=1),
    )

    s = await trajectory_for_topic(db_session, topic_id=parent.id)
    assert s.last.n == 0
    assert s.prior.n == 0
    assert s.delta is None


async def test_cc_scope_includes_both_direct_and_via_topic(
    db_session: AsyncSession,
) -> None:
    """CC scope = direct-CC tags ∪ topic→CC tags (§V31, mirrors retention_for_cc)."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="cc_path")

    base = _now() - timedelta(hours=1)
    # 5 attempts tagged at topic, 5 attempts tagged directly at CC.
    await _add_tagged_attempts(db_session, topic=parent, pattern=[True] * 5, base_time=base)
    await _add_tagged_attempts(
        db_session,
        cc=cc,
        pattern=[True] * 5,
        base_time=base + timedelta(minutes=10),
    )

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    # 10 in scope → 10 in last, 0 in prior → delta None (prior < 5).
    assert s.last.n == 10
    assert s.prior.n == 0
    assert s.delta is None


async def test_multi_tag_attempt_dedupes_within_one_cc_scope(
    db_session: AsyncSession,
) -> None:
    """A question carrying both a direct-CC tag and a topic tag that
    resolves under the same CC must count its one attempt once within
    that CC's scope (DISTINCT on attempt id, §V31).
    """
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="dedupe")

    q = _make_question()
    db_session.add(q)
    await db_session.flush()
    db_session.add_all(
        [
            QuestionTag(question_id=q.id, topic_id=parent.id, confidence=0.9, source="llm"),
            QuestionTag(
                question_id=q.id,
                content_category_id=cc.id,
                confidence=0.9,
                source="llm",
            ),
        ]
    )
    db_session.add(_attempt(question_id=q.id, is_correct=True, attempted_at=_now()))
    await db_session.flush()

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    # One attempt, both tag paths reach this CC; must not double-count.
    assert s.last.n == 1
    assert s.last.correct == 1


async def test_multi_cc_attempt_counts_in_each_cc_scope(
    db_session: AsyncSession,
) -> None:
    """An attempt on a multi-CC question counts once in each CC's scope."""
    cc_a = await _first_cc(db_session)
    cc_b = await _second_cc(db_session)

    q = _make_question()
    db_session.add(q)
    await db_session.flush()
    db_session.add_all(
        [
            QuestionTag(
                question_id=q.id,
                content_category_id=cc_a.id,
                confidence=0.9,
                source="llm",
            ),
            QuestionTag(
                question_id=q.id,
                content_category_id=cc_b.id,
                confidence=0.9,
                source="llm",
            ),
        ]
    )
    db_session.add(_attempt(question_id=q.id, is_correct=True, attempted_at=_now()))
    await db_session.flush()

    a = await trajectory_for_cc(db_session, cc_code=cc_a.code)
    b = await trajectory_for_cc(db_session, cc_code=cc_b.code)
    assert a.last.n == 1 and a.last.correct == 1
    assert b.last.n == 1 and b.last.correct == 1


# --- §V36 ordering ---


async def test_ordering_strictly_by_attempted_at_desc(
    db_session: AsyncSession,
) -> None:
    """Recent attempts populate `last`; older ones populate `prior`.

    Pattern: 10 OLDER wrong + 10 NEWER correct interleaved in insertion
    order (insertion order ≠ attempted_at order). Verify the windows
    follow `attempted_at`, not insert order.
    """
    cc = await _first_cc(db_session)
    base = _now() - timedelta(hours=2)
    # Build pattern in insertion order: alternate old-wrong / new-correct.
    pattern_times: list[tuple[bool, datetime]] = []
    for i in range(10):
        pattern_times.append((False, base + timedelta(minutes=i)))  # older wrong
        pattern_times.append(
            (True, base + timedelta(hours=1, minutes=i))  # newer correct
        )

    for is_correct, when in pattern_times:
        q = _make_question()
        db_session.add(q)
        await db_session.flush()
        db_session.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc.id,
                confidence=0.9,
                source="llm",
            )
        )
        db_session.add(_attempt(question_id=q.id, is_correct=is_correct, attempted_at=when))
    await db_session.flush()

    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert s.last.n == 10 and s.last.correct == 10  # newest 10 all correct
    assert s.prior.n == 10 and s.prior.correct == 0  # older 10 all wrong


# --- §C: time_seconds must not enter the computation ---


def test_trajectory_sql_does_not_reference_time_seconds() -> None:
    """Hard-constraint guard: the trajectory module never reads
    `time_seconds` per the CLAUDE.md / §C invariant that timing data
    is not actionable.
    """
    src = Path(__file__).parent.parent / "app" / "services" / "analyzer" / "trajectory.py"
    text_ = src.read_text(encoding="utf-8")
    assert "time_seconds" not in text_


# --- shape ---


async def test_summary_shape_empty_scope(db_session: AsyncSession) -> None:
    """Empty CC scope → windows zero, delta None, valid TrajectorySummary."""
    cc = await _first_cc(db_session)
    s = await trajectory_for_cc(db_session, cc_code=cc.code)
    assert isinstance(s, TrajectorySummary)
    assert s.last.n == 0 and s.prior.n == 0
    assert s.last.accuracy is None
    assert s.delta is None
    assert s.arrow is None
