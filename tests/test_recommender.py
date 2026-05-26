"""Recommender tests — Ticket 5.2.

All tests use the real Postgres test DB with the outer-transaction rollback pattern.
No mocking — pure SQL + Python math, no LLM calls.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.main import app
from app.models.captures import Attempt, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory, Topic
from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION as FEAT_VERSION
from app.services.recommender import MIN_ATTEMPTS, recommend


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def env(seeded_report, test_engine) -> AsyncIterator[tuple[AsyncClient, any]]:
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


# --------------------------------------------------------------------------- #
# Seed helpers
# --------------------------------------------------------------------------- #


def _qid() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


def _make_question() -> Question:
    return Question(
        qid=_qid(),
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
    attempted_at: datetime | None = None,
) -> Attempt:
    if attempted_at is None:
        attempted_at = datetime.now(tz=timezone.utc)
    return Attempt(
        question_id=question_id,
        attempted_at=attempted_at,
        selected_choice="A" if is_correct else "B",
        is_correct=is_correct,
        flagged=False,
    )


def _features(question_id: int, **kwargs) -> QuestionFeatures:
    defaults: dict = {
        "question_format": "discrete",
        "reasoning_type": "application",
        "requires_calculation": False,
        "calculation_steps": 0,
        "involves_graph_or_figure": False,
        "involves_data_table": False,
        "has_negative_phrasing": False,
        "distractor_difficulty": "medium",
        "trap_distractor_present": False,
        "common_misconception": None,
        "jargon_density": "medium",
        "key_concept_summary": "summary",
        "passage_length_bucket": None,
        "passage_type": None,
        "extractor_version": FEAT_VERSION,
    }
    defaults.update(kwargs)
    return QuestionFeatures(question_id=question_id, **defaults)


async def _first_topic_under(session: AsyncSession, cc_code: str) -> Topic:
    return (
        await session.execute(
            select(Topic)
            .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
            .where(ContentCategory.code == cc_code)
            .limit(1)
        )
    ).scalar_one()


async def _cc_id(session: AsyncSession, code: str) -> int:
    return (
        await session.execute(select(ContentCategory.id).where(ContentCategory.code == code))
    ).scalar_one()


async def _seed_topic_attempts(
    session: AsyncSession,
    topic: Topic,
    *,
    n_correct: int,
    n_wrong: int,
    with_features: bool = False,
    feature_kwargs: dict | None = None,
    old_cutoff_days: int | None = None,
) -> list[Question]:
    """Seed n_correct + n_wrong attempts tagged to topic, return Question list."""
    now = datetime.now(tz=timezone.utc)
    questions: list[Question] = []

    for i in range(n_correct + n_wrong):
        is_correct = i < n_correct
        q = _make_question()
        session.add(q)
        await session.flush()

        session.add(
            QuestionTag(
                question_id=q.id,
                topic_id=topic.id,
                confidence=1.0,
                source="llm",
            )
        )

        at = now
        if old_cutoff_days is not None:
            at = now - timedelta(days=old_cutoff_days + 1)

        session.add(_attempt(question_id=q.id, is_correct=is_correct, attempted_at=at))

        if with_features:
            kw = feature_kwargs or {}
            session.add(_features(q.id, **kw))

        questions.append(q)

    return questions


# --------------------------------------------------------------------------- #
# Test 1: weakest topic ranks first
# --------------------------------------------------------------------------- #


async def test_topic_weakness_ranks_clearly_weakest_first(env):
    _, make_session = env
    async with make_session() as s:
        topic_a = await _first_topic_under(s, "1A")
        topic_b = await _first_topic_under(s, "1B")

        # Topic A: 2/10 correct
        await _seed_topic_attempts(s, topic_a, n_correct=2, n_wrong=8)
        # Topic B: 9/10 correct
        await _seed_topic_attempts(s, topic_b, n_correct=9, n_wrong=1)
        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=10)

    recs = result.recommendations
    topic_recs = [r for r in recs if r.kind == "topic_weakness"]
    labels = [r.label for r in topic_recs]
    assert any("1A" in lbl for lbl in labels), f"1A topic not found: {labels}"
    assert any("1B" in lbl for lbl in labels), f"1B topic not found: {labels}"

    idx_a = next(i for i, r in enumerate(topic_recs) if r.label and "1A" in r.label)
    idx_b = next(i for i, r in enumerate(topic_recs) if r.label and "1B" in r.label)
    assert idx_a < idx_b, "Weakest topic (1A: 20%) should rank above stronger (1B: 90%)"


# --------------------------------------------------------------------------- #
# Test 2: min-attempts threshold excludes sparse topics
# --------------------------------------------------------------------------- #


async def test_min_attempts_threshold_excludes_sparse_topics(env):
    _, make_session = env
    async with make_session() as s:
        topic_sparse = await _first_topic_under(s, "2A")
        topic_ok = await _first_topic_under(s, "2B")

        # 2 attempts (below MIN_ATTEMPTS=3) — should be excluded
        await _seed_topic_attempts(s, topic_sparse, n_correct=0, n_wrong=2)
        # 5 attempts — should be included
        await _seed_topic_attempts(s, topic_ok, n_correct=1, n_wrong=4)
        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=20)

    labels = [r.label for r in result.recommendations if r.kind == "topic_weakness"]
    assert not any("2A" in lbl for lbl in labels), (
        f"Sparse topic (2A, 2 attempts) should be excluded; got: {labels}"
    )
    assert any("2B" in lbl for lbl in labels), "5-attempt topic (2B) should appear"


# --------------------------------------------------------------------------- #
# Test 3: feature_pattern recommendation appears
# --------------------------------------------------------------------------- #


async def test_feature_pattern_recommendation_appears(env):
    _, make_session = env
    async with make_session() as s:
        topic = await _first_topic_under(s, "3A")

        # 6 has_negative_phrasing=True — all wrong
        for _ in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=False))
            s.add(_features(q.id, has_negative_phrasing=True))

        # 6 has_negative_phrasing=False — all correct
        for _ in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=True))
            s.add(_features(q.id, has_negative_phrasing=False))

        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=10)

    pattern_recs = [r for r in result.recommendations if r.kind == "feature_pattern"]
    names = [r.feature_name for r in pattern_recs]
    assert "has_negative_phrasing" in names, f"Expected has_negative_phrasing finding; got: {names}"
    neg_rec = next(r for r in pattern_recs if r.feature_name == "has_negative_phrasing")
    assert neg_rec.feature_value == "True"
    assert neg_rec.accuracy_with < neg_rec.accuracy_without


# --------------------------------------------------------------------------- #
# Test 4: feature bonus elevates topic with overlapping missed qids
# --------------------------------------------------------------------------- #


async def test_feature_pattern_boost_elevates_overlapping_topic(env):
    _, make_session = env

    async with make_session() as s:
        topic_a = await _first_topic_under(s, "4A")
        topic_b = await _first_topic_under(s, "4B")

        # Both topics: 2 correct, 4 wrong — similar base scores
        # Topic A's wrong questions have has_negative_phrasing=True
        for i in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic_a.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=(i < 2)))
            s.add(_features(q.id, has_negative_phrasing=True))

        # 6 without feature (all correct) to make the finding appear
        topic_c = await _first_topic_under(s, "5A")
        for _ in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic_c.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=True))
            s.add(_features(q.id, has_negative_phrasing=False))

        # Topic B: same ratio but NO has_negative_phrasing features
        for i in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic_b.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=(i < 2)))
            s.add(_features(q.id, has_negative_phrasing=False))

        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=20)

    topic_recs = [r for r in result.recommendations if r.kind == "topic_weakness"]
    a_recs = [r for r in topic_recs if r.label and "4A" in r.label]
    b_recs = [r for r in topic_recs if r.label and "4B" in r.label]

    assert a_recs, "Topic 4A should appear in recommendations"
    assert b_recs, "Topic 4B should appear in recommendations"

    score_a = a_recs[0].priority_score
    score_b = b_recs[0].priority_score
    assert score_a > score_b, (
        f"Topic A (feature overlap) should rank higher than B: {score_a:.4f} vs {score_b:.4f}"
    )


# --------------------------------------------------------------------------- #
# Test 5: recency decline raises priority
# --------------------------------------------------------------------------- #


async def test_recency_decline_raises_priority(env):
    _, make_session = env
    now = datetime.now(tz=timezone.utc)

    async with make_session() as s:
        topic_declining = await _first_topic_under(s, "1A")
        topic_stable = await _first_topic_under(s, "1B")

        # Topic A: 5 old correct + 3 recent incorrect → overall 62.5%, recent 0%
        for i in range(5):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                QuestionTag(
                    question_id=q.id,
                    topic_id=topic_declining.id,
                    confidence=1.0,
                    source="llm",
                )
            )
            old_at = now - timedelta(days=60)
            s.add(_attempt(question_id=q.id, is_correct=True, attempted_at=old_at))

        for _ in range(3):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                QuestionTag(
                    question_id=q.id,
                    topic_id=topic_declining.id,
                    confidence=1.0,
                    source="llm",
                )
            )
            s.add(_attempt(question_id=q.id, is_correct=False, attempted_at=now))

        # Topic B: same overall accuracy (5/8 = 62.5%), no recent data
        for i in range(8):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                QuestionTag(
                    question_id=q.id,
                    topic_id=topic_stable.id,
                    confidence=1.0,
                    source="llm",
                )
            )
            old_at = now - timedelta(days=60)
            s.add(_attempt(question_id=q.id, is_correct=(i < 5), attempted_at=old_at))

        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=20)

    topic_recs = [r for r in result.recommendations if r.kind == "topic_weakness"]
    declining_recs = [r for r in topic_recs if r.label and "1A" in r.label]
    stable_recs = [r for r in topic_recs if r.label and "1B" in r.label]

    assert declining_recs, "Declining topic should appear"
    assert stable_recs, "Stable topic should appear"

    score_declining = declining_recs[0].priority_score
    score_stable = stable_recs[0].priority_score
    assert score_declining > score_stable, (
        f"Declining topic should rank higher: {score_declining:.4f} vs {score_stable:.4f}"
    )


# --------------------------------------------------------------------------- #
# Test 6: empty DB returns empty list
# --------------------------------------------------------------------------- #


async def test_empty_db_returns_empty_list(env):
    _, make_session = env
    # No data seeded — outer transaction ensures isolation.
    async with make_session() as s:
        result = await recommend(s, n=5)

    assert result.recommendations == []
    assert result.total_candidates_scored == 0


# --------------------------------------------------------------------------- #
# Test 7: n parameter limits output
# --------------------------------------------------------------------------- #


async def test_n_parameter_limits_output(env):
    _, make_session = env
    async with make_session() as s:
        # Seed 6 distinct topics with enough attempts each
        for cc_code in ("1A", "1B", "2A", "2B", "3A", "3B"):
            try:
                topic = await _first_topic_under(s, cc_code)
            except Exception:
                continue
            await _seed_topic_attempts(s, topic, n_correct=1, n_wrong=4)
        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=2)

    assert len(result.recommendations) <= 2


# --------------------------------------------------------------------------- #
# Test 8: endpoint returns 200 with correct schema
# --------------------------------------------------------------------------- #


async def test_endpoint_returns_200_with_correct_schema(env):
    client, make_session = env
    async with make_session() as s:
        topic = await _first_topic_under(s, "1A")
        await _seed_topic_attempts(s, topic, n_correct=1, n_wrong=4)
        await s.commit()

    resp = await client.get("/api/v1/recommendations/study-next?n=5")
    assert resp.status_code == 200

    data = resp.json()
    assert "recommendations" in data
    assert "total_candidates_scored" in data
    assert "min_attempts_threshold" in data
    assert data["min_attempts_threshold"] == MIN_ATTEMPTS
    assert isinstance(data["recommendations"], list)


# --------------------------------------------------------------------------- #
# Test 9: reason field is non-empty on all kinds
# --------------------------------------------------------------------------- #


async def test_reason_field_is_non_empty_on_all_kinds(env):
    _, make_session = env
    async with make_session() as s:
        topic = await _first_topic_under(s, "1A")
        topic2 = await _first_topic_under(s, "1B")

        # Topic weakness
        await _seed_topic_attempts(s, topic, n_correct=1, n_wrong=4)

        # Feature pattern: 6 negative-phrasing wrong, 6 non-negative correct
        for _ in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic2.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=False))
            s.add(_features(q.id, has_negative_phrasing=True))
        for _ in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(QuestionTag(question_id=q.id, topic_id=topic2.id, confidence=1.0, source="llm"))
            s.add(_attempt(question_id=q.id, is_correct=True))
            s.add(_features(q.id, has_negative_phrasing=False))

        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=20)

    for rec in result.recommendations:
        assert rec.reason and len(rec.reason) > 0, f"Empty reason on {rec.kind} recommendation"


# --------------------------------------------------------------------------- #
# Test 10: CARS not surfaced as topic_weakness
# --------------------------------------------------------------------------- #


async def test_cars_section_not_surfaced_as_topic_weakness(env):
    _, make_session = env
    async with make_session() as s:
        cars_cc_id = await _cc_id(s, "CARS")

        # Tag questions directly to CARS CC (no CARS topics per spec)
        for i in range(5):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                QuestionTag(
                    question_id=q.id,
                    content_category_id=cars_cc_id,
                    confidence=1.0,
                    source="llm",
                )
            )
            s.add(_attempt(question_id=q.id, is_correct=False))

        await s.commit()

    async with make_session() as s:
        result = await recommend(s, n=20)

    topic_weakness_recs = [r for r in result.recommendations if r.kind == "topic_weakness"]
    for rec in topic_weakness_recs:
        assert rec.code != "CARS", f"CARS should not appear as topic_weakness; got: {rec}"
