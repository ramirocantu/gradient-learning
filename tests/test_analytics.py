"""Analytics rollups + endpoints (Ticket 5.1).

Per-test outer transaction with savepoint sessions — matches the pattern used
by `test_ingest.py` and `test_categorizer_worker.py`. The AAMC outline is
session-scoped via `seeded_report`; per-test data lives inside the outer
transaction and is rolled back at teardown.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.main import app
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, Section, Topic
from app.services.analytics import (
    compute_mastery,
    wilson_lower,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def env(seeded_report, test_engine):
    """ASGI client + session factory under one rolled-back outer transaction."""
    conn = await test_engine.connect()
    await conn.begin()

    def make_session() -> AsyncSession:
        return AsyncSession(
            bind=conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    async def _override_session():
        session = make_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    app.dependency_overrides[get_session] = _override_session
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, make_session
    finally:
        app.dependency_overrides.pop(get_session, None)
        await conn.rollback()
        await conn.close()


def _new_qid() -> str:
    return f"q-{uuid.uuid4().hex[:10]}"


def _make_question(*, passage_id: int | None = None) -> Question:
    return Question(
        qid=_new_qid(),
        passage_id=passage_id,
        stem_html="<p>stem</p>",
        stem_plain="stem",
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


def _attempt(
    *,
    question_id: int,
    is_correct: bool,
    attempted_at: datetime,
    time_seconds: int | None = 30,
) -> Attempt:
    return Attempt(
        question_id=question_id,
        attempted_at=attempted_at,
        selected_choice="A" if is_correct else "B",
        is_correct=is_correct,
        time_seconds=time_seconds,
        flagged=False,
    )


async def _cc_id(session: AsyncSession, code: str) -> int:
    return (
        await session.execute(select(ContentCategory.id).where(ContentCategory.code == code))
    ).scalar_one()


async def _topic_under_cc(session: AsyncSession, cc_code: str) -> Topic:
    return (
        await session.execute(
            select(Topic)
            .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
            .where(ContentCategory.code == cc_code)
            .limit(1)
        )
    ).scalar_one()


async def _section_id_for_cc(session: AsyncSession, cc_code: str) -> int:
    """Look up Section.id by chasing CC -> FC -> Section."""
    from app.models.outline import FoundationalConcept

    return (
        await session.execute(
            select(Section.id)
            .join(FoundationalConcept, FoundationalConcept.section_id == Section.id)
            .join(
                ContentCategory,
                ContentCategory.foundational_concept_id == FoundationalConcept.id,
            )
            .where(ContentCategory.code == cc_code)
        )
    ).scalar_one()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------- #
# 1. Wilson math
# --------------------------------------------------------------------------- #


def test_wilson_lower_bound_known_values():
    # Reference: Wilson 95% (z=1.96), p=0.8, n=10 -> ~0.49.
    val = wilson_lower(8, 10)
    assert 0.47 < val < 0.51

    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(10, 10) > 0.7
    # Bounds: 0 correct out of N clamps at 0.0.
    assert wilson_lower(0, 10) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Per-CC single-tagged question
# --------------------------------------------------------------------------- #


async def test_per_cc_accuracy_single_tagged_question(env):
    _, make_session = env
    async with make_session() as s:
        cc4a = await _cc_id(s, "4A")
        base = _now()
        # 4 unique questions (3 correct, 1 wrong) → attempts=4, correct=3.
        for i, correct in enumerate([True, True, True, False]):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                QuestionTag(
                    question_id=q.id,
                    content_category_id=cc4a,
                    confidence=0.9,
                    source="llm",
                )
            )
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=correct,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    cc_rows = [r for r in report.by_content_category if r.code == "4A"]
    assert len(cc_rows) == 1
    row = cc_rows[0]
    assert row.attempts == 4
    assert row.correct == 3
    assert row.accuracy == pytest.approx(0.75)
    assert 0.3 < row.wilson_lower < 0.7


# --------------------------------------------------------------------------- #
# 3. Multi-tagged question — counts under both CCs
# --------------------------------------------------------------------------- #


async def test_per_cc_accuracy_multi_tagged_question(env):
    _, make_session = env
    async with make_session() as s:
        # Tag 2 questions each via a topic under 4A AND a direct CC tag for 5A.
        topic_4a = await _topic_under_cc(s, "4A")
        cc5a = await _cc_id(s, "5A")
        base = _now()
        for i, correct in enumerate([True, False]):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add_all(
                [
                    QuestionTag(
                        question_id=q.id,
                        topic_id=topic_4a.id,
                        confidence=0.9,
                        source="llm",
                    ),
                    QuestionTag(
                        question_id=q.id,
                        content_category_id=cc5a,
                        confidence=0.9,
                        source="llm",
                    ),
                ]
            )
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=correct,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    cc_by_code = {r.code: r for r in report.by_content_category}
    assert cc_by_code["4A"].attempts == 2
    assert cc_by_code["4A"].correct == 1
    assert cc_by_code["5A"].attempts == 2
    assert cc_by_code["5A"].correct == 1

    # Cross-CC sum exceeds total question count — intentional.
    total_cc_attempts = sum(r.attempts for r in report.by_content_category)
    assert total_cc_attempts >= 4
    assert report.total_attempts == 2


# --------------------------------------------------------------------------- #
# 4. Per-section rollup
# --------------------------------------------------------------------------- #


async def test_per_section_rollup(env):
    _, make_session = env
    async with make_session() as s:
        cp_cc = await _cc_id(s, "4A")  # CP-section
        bb_cc = await _cc_id(s, "1A")  # BB-section
        cp_section_id = await _section_id_for_cc(s, "4A")
        bb_section_id = await _section_id_for_cc(s, "1A")

        q_cp = _make_question()
        q_bb = _make_question()
        s.add_all([q_cp, q_bb])
        await s.flush()

        s.add_all(
            [
                QuestionTag(
                    question_id=q_cp.id,
                    content_category_id=cp_cc,
                    confidence=0.9,
                    source="llm",
                ),
                QuestionTag(
                    question_id=q_bb.id,
                    content_category_id=bb_cc,
                    confidence=0.9,
                    source="llm",
                ),
            ]
        )
        base = _now()
        s.add(_attempt(question_id=q_cp.id, is_correct=True, attempted_at=base))
        s.add(_attempt(question_id=q_bb.id, is_correct=False, attempted_at=base))
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    by_section = {r.target_id: r for r in report.by_section}
    assert by_section[cp_section_id].attempts == 1
    assert by_section[cp_section_id].correct == 1
    assert by_section[bb_section_id].attempts == 1
    assert by_section[bb_section_id].correct == 0


# --------------------------------------------------------------------------- #
# 5. Skill rollup
# --------------------------------------------------------------------------- #


async def test_skill_rollup_with_auto_parsed_skills(env):
    _, make_session = env
    async with make_session() as s:
        base = _now()
        for i, correct in enumerate([True, True, False]):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                QuestionTag(
                    question_id=q.id,
                    skill=2,
                    confidence=1.0,
                    source="llm",
                )
            )
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=correct,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    skill_2 = next(r for r in report.by_skill if r.code == "2")
    assert skill_2.attempts == 3
    assert skill_2.correct == 2
    assert skill_2.label == "Skill 2"


# --------------------------------------------------------------------------- #
# 6. Timing — median per question kind
# --------------------------------------------------------------------------- #


async def test_timing_median_discrete_vs_passage(env):
    _, make_session = env
    from app.models.captures import Passage

    async with make_session() as s:
        passage = Passage(
            uworld_passage_id=f"uw-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex,
            html="<p>p</p>",
            plain_text="p",
        )
        s.add(passage)
        await s.flush()

        # Discrete: times [30, 50, 70] -> median 50.
        # Passage:  times [80, 100, 120] -> median 100.
        # Plus one NULL time on each side that must be excluded.
        discrete_times = [30, 50, 70]
        passage_times = [80, 100, 120]
        base = _now()
        i = 0
        for t in discrete_times:
            q = _make_question(passage_id=None)
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                    time_seconds=t,
                )
            )
            i += 1
        for t in passage_times:
            q = _make_question(passage_id=passage.id)
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                    time_seconds=t,
                )
            )
            i += 1

        # NULL-time outliers (one of each) — must NOT shift the median.
        q_null_d = _make_question(passage_id=None)
        q_null_p = _make_question(passage_id=passage.id)
        s.add_all([q_null_d, q_null_p])
        await s.flush()
        s.add(
            _attempt(
                question_id=q_null_d.id,
                is_correct=True,
                attempted_at=base + timedelta(minutes=i),
                time_seconds=None,
            )
        )
        s.add(
            _attempt(
                question_id=q_null_p.id,
                is_correct=True,
                attempted_at=base + timedelta(minutes=i + 1),
                time_seconds=None,
            )
        )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    assert report.timing.median_seconds_discrete == pytest.approx(50.0)
    assert report.timing.median_seconds_passage_based == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# 7. Timing — over-target counts
# --------------------------------------------------------------------------- #


async def test_timing_over_target_counts(env):
    _, make_session = env
    from app.models.captures import Passage

    async with make_session() as s:
        passage = Passage(
            uworld_passage_id=f"uw-{uuid.uuid4().hex[:8]}",
            content_hash=uuid.uuid4().hex,
            html="<p>p</p>",
            plain_text="p",
        )
        s.add(passage)
        await s.flush()

        base = _now()
        # Discrete: times [40, 65, 80, 90]. 60s threshold -> 3 over (65, 80, 90).
        # Passage: times [60, 95, 120]. 95s threshold -> 1 over (120).
        # (Exactly 95 is NOT > 95.)
        for i, t in enumerate([40, 65, 80, 90]):
            q = _make_question(passage_id=None)
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                    time_seconds=t,
                )
            )
        for i, t in enumerate([60, 95, 120]):
            q = _make_question(passage_id=passage.id)
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=10 + i),
                    time_seconds=t,
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    assert report.timing.questions_over_target_discrete == 3
    assert report.timing.questions_over_target_passage == 1


# --------------------------------------------------------------------------- #
# 8. Trends — 7d windowing across 3 weeks
# --------------------------------------------------------------------------- #


async def test_trends_7d_windowing(env):
    _, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        cc = await _cc_id(s, "4A")
        s.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc,
                confidence=0.9,
                source="llm",
            )
        )

        # 3 distinct ISO weeks, sufficiently far apart that date_trunc('week')
        # produces 3 distinct buckets regardless of where "now" falls in the
        # current week.
        anchor = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)  # Monday
        week_a = anchor
        week_b = anchor + timedelta(days=7)
        week_c = anchor + timedelta(days=14)

        # Week A: 2 attempts, 1 correct -> 0.50.
        # Week B: 2 attempts, 2 correct -> 1.00.
        # Week C: 1 attempt,  0 correct -> 0.00.
        for at, correct in [
            (week_a, True),
            (week_a + timedelta(hours=1), False),
            (week_b, True),
            (week_b + timedelta(hours=1), True),
            (week_c, False),
        ]:
            s.add(_attempt(question_id=q.id, is_correct=correct, attempted_at=at))
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    assert len(report.trend_7d) == 3
    accuracies = [p.accuracy for p in report.trend_7d]
    attempts = [p.attempts for p in report.trend_7d]
    # Ascending by period_start.
    assert accuracies == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(0.0)]
    assert attempts == [2, 2, 1]


# --------------------------------------------------------------------------- #
# 9. Uncategorized count
# --------------------------------------------------------------------------- #


async def test_uncategorized_count(env):
    _, make_session = env
    async with make_session() as s:
        q_skill_only = _make_question()
        q_topic = _make_question()
        s.add_all([q_skill_only, q_topic])
        await s.flush()

        topic = await _topic_under_cc(s, "4A")
        s.add_all(
            [
                QuestionTag(
                    question_id=q_skill_only.id,
                    skill=3,
                    confidence=1.0,
                    source="llm",
                ),
                QuestionTag(
                    question_id=q_topic.id,
                    topic_id=topic.id,
                    confidence=0.9,
                    source="llm",
                ),
            ]
        )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    assert report.uncategorized_question_count == 1


# --------------------------------------------------------------------------- #
# 12. Mastery endpoint smoke
# --------------------------------------------------------------------------- #


async def test_mastery_endpoint_smoke(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        cc = await _cc_id(s, "4A")
        s.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc,
                confidence=0.9,
                source="llm",
            )
        )
        s.add(_attempt(question_id=q.id, is_correct=True, attempted_at=_now()))
        await s.commit()

    r = await client.get("/api/v1/analytics/mastery")
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    expected = {
        "by_section",
        "by_content_category",
        "by_topic",
        "by_skill",
        "timing",
        "trend_7d",
        "trend_30d",
        "uncategorized_question_count",
        "total_attempts",
        "total_questions",
    }
    assert expected <= set(body.keys())
    assert any(row["code"] == "4A" for row in body["by_content_category"])


# --------------------------------------------------------------------------- #
# 13. Weaknesses endpoint removed (6.2b) — covered by recommender tests.
# --------------------------------------------------------------------------- #


async def test_weaknesses_endpoint_removed_returns_404(env):
    client, _ = env
    r = await client.get("/api/v1/analytics/weaknesses?n=5")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 15. Mastery includes zero-attempt content categories (5.1a)
# --------------------------------------------------------------------------- #


async def test_mastery_includes_zero_attempt_content_categories(env):
    _, make_session = env
    async with make_session() as s:
        # Seed a single attempt on CC 4A only.
        q = _make_question()
        s.add(q)
        await s.flush()
        cc4a = await _cc_id(s, "4A")
        s.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc4a,
                confidence=0.9,
                source="llm",
            )
        )
        s.add(_attempt(question_id=q.id, is_correct=True, attempted_at=_now()))
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    # All CCs from the seeded outline must appear.

    async with make_session() as s:
        from sqlalchemy import func as _fn

        cc_count = (await s.execute(select(_fn.count(ContentCategory.id)))).scalar_one()

    assert len(report.by_content_category) == cc_count

    # 4A has attempts; all others have attempts=0.
    cc_by_code = {r.code: r for r in report.by_content_category}
    assert cc_by_code["4A"].attempts == 1
    assert cc_by_code["4A"].correct == 1

    zero_ccs = [r for r in report.by_content_category if r.code != "4A"]
    assert all(r.attempts == 0 for r in zero_ccs)
    assert all(r.correct == 0 for r in zero_ccs)
    assert all(r.accuracy == 0.0 for r in zero_ccs)


# --------------------------------------------------------------------------- #
# 16. Mastery includes zero-attempt topics (5.1a)
# --------------------------------------------------------------------------- #


async def test_mastery_includes_zero_attempt_topics(env):
    _, make_session = env
    async with make_session() as s:
        # Seed a single attempt via a topic tag.
        topic = await _topic_under_cc(s, "4A")
        q = _make_question()
        s.add(q)
        await s.flush()
        s.add(
            QuestionTag(
                question_id=q.id,
                topic_id=topic.id,
                confidence=0.9,
                source="llm",
            )
        )
        s.add(_attempt(question_id=q.id, is_correct=True, attempted_at=_now()))
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    # Every topic in the outline must appear.
    async with make_session() as s:
        from sqlalchemy import func as _fn

        topic_count = (await s.execute(select(_fn.count(Topic.id)))).scalar_one()

    assert len(report.by_topic) == topic_count

    # The seeded topic has 1 attempt; all others have 0.
    attempted = [t for t in report.by_topic if t.attempts > 0]
    assert len(attempted) == 1
    assert attempted[0].target_id == topic.id


# --------------------------------------------------------------------------- #
# 17. Mastery always includes all four sections (5.1a)
# --------------------------------------------------------------------------- #


async def test_mastery_always_includes_all_four_sections(env):
    _, make_session = env
    # No attempts seeded — every section should still appear.
    async with make_session() as s:
        report = await compute_mastery(s)

    assert len(report.by_section) == 4
    codes = {r.code for r in report.by_section}
    assert codes == {"CP", "CARS", "BB", "PS"}
    assert all(r.attempts == 0 for r in report.by_section)


# --------------------------------------------------------------------------- #
# 18. Mastery always includes all four skills (5.1a)
# --------------------------------------------------------------------------- #


async def test_mastery_always_includes_all_four_skills(env):
    _, make_session = env
    # No attempts seeded — every skill should still appear.
    async with make_session() as s:
        report = await compute_mastery(s)

    assert len(report.by_skill) == 4
    codes = {r.code for r in report.by_skill}
    assert codes == {"1", "2", "3", "4"}
    assert all(r.attempts == 0 for r in report.by_skill)


# --------------------------------------------------------------------------- #
# 19. Zero-attempt entries have wilson_lower == 0.0 (5.1a)
# --------------------------------------------------------------------------- #


async def test_zero_attempt_wilson_lower_is_zero(env):
    _, make_session = env
    async with make_session() as s:
        report = await compute_mastery(s)

    zero_entries = [
        r
        for r in (
            *report.by_section,
            *report.by_content_category,
            *report.by_topic,
            *report.by_skill,
        )
        if r.attempts == 0
    ]
    assert len(zero_entries) > 0, "Expected at least some zero-attempt entries"
    for entry in zero_entries:
        assert entry.wilson_lower == 0.0, f"{entry.label}: wilson_lower={entry.wilson_lower}"
        assert entry.accuracy == 0.0


# --------------------------------------------------------------------------- #
# 21. Latest-attempt semantics: single question, multiple attempts
# --------------------------------------------------------------------------- #


async def test_mastery_uses_latest_attempt_per_question(env):
    """Question attempted 3× counts as 1 unique question; latest attempt wins."""
    _, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        cc4a = await _cc_id(s, "4A")
        s.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc4a,
                confidence=0.9,
                source="llm",
            )
        )
        base = _now()
        # correct, wrong, correct — latest attempt is correct
        for i, correct in enumerate([True, False, True]):
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=correct,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    row = next(r for r in report.by_content_category if r.code == "4A")
    assert row.attempts == 1
    assert row.correct == 1
    assert row.accuracy == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 22. Latest-attempt semantics: multiple questions, mixed latest outcomes
# --------------------------------------------------------------------------- #


async def test_mastery_multiple_questions_latest_attempt(env):
    """Q_A latest=correct, Q_B latest=wrong → attempts=2, correct=1, accuracy=0.5."""
    _, make_session = env
    async with make_session() as s:
        cc4a = await _cc_id(s, "4A")
        base = _now()

        q_a = _make_question()
        s.add(q_a)
        await s.flush()
        s.add(
            QuestionTag(
                question_id=q_a.id,
                content_category_id=cc4a,
                confidence=0.9,
                source="llm",
            )
        )
        # Q_A: wrong then correct — latest=correct
        s.add(_attempt(question_id=q_a.id, is_correct=False, attempted_at=base))
        s.add(
            _attempt(
                question_id=q_a.id,
                is_correct=True,
                attempted_at=base + timedelta(minutes=1),
            )
        )

        q_b = _make_question()
        s.add(q_b)
        await s.flush()
        s.add(
            QuestionTag(
                question_id=q_b.id,
                content_category_id=cc4a,
                confidence=0.9,
                source="llm",
            )
        )
        # Q_B: correct then wrong — latest=wrong
        s.add(
            _attempt(
                question_id=q_b.id,
                is_correct=True,
                attempted_at=base + timedelta(minutes=2),
            )
        )
        s.add(
            _attempt(
                question_id=q_b.id,
                is_correct=False,
                attempted_at=base + timedelta(minutes=3),
            )
        )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    row = next(r for r in report.by_content_category if r.code == "4A")
    assert row.attempts == 2
    assert row.correct == 1
    assert row.accuracy == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 23. Timing still uses raw attempts (intentional)
# --------------------------------------------------------------------------- #


async def test_timing_still_uses_raw_attempts(env):
    """Timing reflects all attempt durations, not just the latest per question."""
    _, make_session = env
    async with make_session() as s:
        q = _make_question(passage_id=None)
        s.add(q)
        await s.flush()
        base = _now()
        # 3 attempts on same discrete question, times 30/60/90 → median 60.
        for i, t in enumerate([30, 60, 90]):
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                    time_seconds=t,
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    assert report.timing.median_seconds_discrete == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# 24. Trends still use raw attempts (intentional)
# --------------------------------------------------------------------------- #


async def test_trend_still_uses_raw_attempts(env):
    """Trends count every attempt in a period, not one per question."""
    _, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        # 3 attempts on same question in the same week → trend shows attempts=3.
        anchor = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)  # Monday
        for i in range(3):
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=anchor + timedelta(hours=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await compute_mastery(s)

    week_point = next((p for p in report.trend_7d if p.attempts == 3), None)
    assert week_point is not None, (
        f"Expected week with 3 attempts; got {[p.attempts for p in report.trend_7d]}"
    )
