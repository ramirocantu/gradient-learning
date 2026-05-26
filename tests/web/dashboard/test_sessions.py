"""Tests for the Sessions dashboard view (Ticket 6.9d)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic


# --------------------------------------------------------------------------- #
# Seed helpers
# --------------------------------------------------------------------------- #


async def _seed_cc(
    session,
    *,
    code: str = "4A",
    name: str = "Test CC",
    section_code: str = "CP",
    section_name: str = "Chem/Phys",
    fc_code: str = "FC1",
    fc_name: str = "FC1",
) -> ContentCategory:
    sec = (
        await session.execute(select(Section).where(Section.code == section_code))
    ).scalar_one_or_none()
    if sec is None:
        sec = Section(code=section_code, name=section_name, position=1)
        session.add(sec)
        await session.flush()

    fc = (
        await session.execute(
            select(FoundationalConcept).where(FoundationalConcept.code == fc_code)
        )
    ).scalar_one_or_none()
    if fc is None:
        fc = FoundationalConcept(section_id=sec.id, code=fc_code, name=fc_name, position=1)
        session.add(fc)
        await session.flush()

    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == code))
    ).scalar_one_or_none()
    if cc is None:
        cc = ContentCategory(foundational_concept_id=fc.id, code=code, name=name, position=1)
        session.add(cc)
        await session.flush()
    return cc


async def _seed_topic(session, *, cc: ContentCategory, name: str, position: int = 1) -> Topic:
    t = Topic(content_category_id=cc.id, name=name, depth=0, position=position)
    session.add(t)
    await session.flush()
    return t


async def _seed_question(session, *, qid: str) -> Question:
    q = Question(
        qid=qid,
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[{"key": "A", "html": "<p>A</p>", "plain": "A"}],
        correct_choice="A",
        explanation_html="<p>e</p>",
        explanation_plain="e",
    )
    session.add(q)
    await session.flush()
    return q


async def _seed_attempt(
    session,
    *,
    question: Question,
    test_id: str | None,
    minutes_ago: int = 0,
    is_correct: bool = True,
    flagged: bool = False,
) -> Attempt:
    a = Attempt(
        question_id=question.id,
        attempted_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        selected_choice="A",
        is_correct=is_correct,
        flagged=flagged,
        uworld_test_id=test_id,
    )
    session.add(a)
    await session.flush()
    return a


async def _seed_tag(
    session,
    *,
    question: Question,
    topic: Topic | None = None,
    skill: int | None = None,
    cc: ContentCategory | None = None,
    source: str = "llm",
) -> QuestionTag:
    qt = QuestionTag(
        question_id=question.id,
        topic_id=topic.id if topic else None,
        content_category_id=cc.id if cc else None,
        skill=skill,
        source=source,
        confidence=1.0,
    )
    session.add(qt)
    await session.flush()
    return qt


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sessions_index_empty_state(client, session):
    r = await client.get("/sessions")
    assert r.status_code == 200
    assert "No sessions captured yet" in r.text


@pytest.mark.asyncio
async def test_sessions_index_renders_populated_session(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    t1 = await _seed_topic(session, cc=cc, name="ZZZ_TopicAlpha")
    t2 = await _seed_topic(session, cc=cc, name="ZZZ_TopicBeta", position=2)
    for i, topic in enumerate([t1, t1, t2]):
        q = await _seed_question(session, qid=f"q-A-{i}")
        await _seed_tag(session, question=q, topic=topic)
        await _seed_attempt(session, question=q, test_id="7392051", minutes_ago=i * 5)
    await session.commit()

    r = await client.get("/sessions")
    assert r.status_code == 200
    assert "7392051" in r.text
    # attempt count column
    assert ">3<" in r.text or ">3 " in r.text  # tolerant of formatting
    assert "ZZZ_TopicAlpha" in r.text


@pytest.mark.asyncio
async def test_sessions_index_sorts_by_latest_attempt_desc(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    t = await _seed_topic(session, cc=cc, name="ZZZ_X")
    # Session A — latest attempt 60 min ago
    q_a = await _seed_question(session, qid="q-A-1")
    await _seed_tag(session, question=q_a, topic=t)
    await _seed_attempt(session, question=q_a, test_id="A_TID111", minutes_ago=60)
    # Session B — latest attempt 10 min ago
    q_b = await _seed_question(session, qid="q-B-1")
    await _seed_tag(session, question=q_b, topic=t)
    await _seed_attempt(session, question=q_b, test_id="B_TID222", minutes_ago=10)
    await session.commit()

    r = await client.get("/sessions")
    assert r.status_code == 200
    idx_b = r.text.find("B_TID222")
    idx_a = r.text.find("A_TID111")
    assert idx_b != -1 and idx_a != -1
    assert idx_b < idx_a, "B (most recent) should render before A"


@pytest.mark.asyncio
async def test_sessions_index_unsessioned_row_appears_last(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    t = await _seed_topic(session, cc=cc, name="ZZZ_X")
    q_norm = await _seed_question(session, qid="q-NRM-1")
    await _seed_tag(session, question=q_norm, topic=t)
    await _seed_attempt(session, question=q_norm, test_id="NORMAL999", minutes_ago=30)

    q_orphan = await _seed_question(session, qid="q-ORF-1")
    await _seed_tag(session, question=q_orphan, topic=t)
    await _seed_attempt(session, question=q_orphan, test_id=None, minutes_ago=120)
    await session.commit()

    r = await client.get("/sessions")
    assert r.status_code == 200
    assert "Unsessioned" in r.text
    assert r.text.find("NORMAL999") < r.text.find("Unsessioned")


@pytest.mark.asyncio
async def test_session_detail_renders_attempts_chronologically(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    t = await _seed_topic(session, cc=cc, name="ZZZ_X")
    q1 = await _seed_question(session, qid="q-CHRON-1")
    q2 = await _seed_question(session, qid="q-CHRON-2")
    q3 = await _seed_question(session, qid="q-CHRON-3")
    for q in (q1, q2, q3):
        await _seed_tag(session, question=q, topic=t)
    # oldest → newest
    await _seed_attempt(session, question=q1, test_id="CHRONO1", minutes_ago=90)
    await _seed_attempt(session, question=q2, test_id="CHRONO1", minutes_ago=60)
    await _seed_attempt(session, question=q3, test_id="CHRONO1", minutes_ago=30)
    await session.commit()

    r = await client.get("/sessions/CHRONO1")
    assert r.status_code == 200
    i1 = r.text.find("q-CHRON-1")
    i2 = r.text.find("q-CHRON-2")
    i3 = r.text.find("q-CHRON-3")
    assert -1 < i1 < i2 < i3


@pytest.mark.asyncio
async def test_session_detail_includes_cars_chip_on_cars_attempt(client, session):
    q = await _seed_question(session, qid="q-CARS-1")
    await _seed_tag(session, question=q, skill=3)
    await _seed_attempt(session, question=q, test_id="CARSESS1", minutes_ago=15)
    await session.commit()

    r = await client.get("/sessions/CARSESS1")
    assert r.status_code == 200
    assert "CARS (skill)" in r.text


@pytest.mark.asyncio
async def test_session_detail_unsessioned_path(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    t = await _seed_topic(session, cc=cc, name="ZZZ_X")
    q1 = await _seed_question(session, qid="q-UNS-1")
    q2 = await _seed_question(session, qid="q-UNS-2")
    for q in (q1, q2):
        await _seed_tag(session, question=q, topic=t)
        await _seed_attempt(session, question=q, test_id=None, minutes_ago=90)
    await session.commit()

    r = await client.get("/sessions/unsessioned")
    assert r.status_code == 200
    assert "Unsessioned attempts" in r.text
    assert "q-UNS-1" in r.text
    assert "q-UNS-2" in r.text


@pytest.mark.asyncio
async def test_session_detail_404_for_unknown_test_id(client, session):
    r = await client.get("/sessions/9999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_sessions_index_top_topics_capped_at_three(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    topics = [await _seed_topic(session, cc=cc, name=f"ZZZ_T{i}", position=i) for i in range(1, 6)]
    for i, t in enumerate(topics):
        q = await _seed_question(session, qid=f"q-TOP-{i}")
        await _seed_tag(session, question=q, topic=t)
        await _seed_attempt(session, question=q, test_id="TOPTEST1", minutes_ago=i)
    await session.commit()

    r = await client.get("/sessions")
    assert r.status_code == 200
    rendered_topics = [t.name for t in topics if t.name in r.text]
    assert len(rendered_topics) == 3, f"expected 3 topics, saw {rendered_topics}"
