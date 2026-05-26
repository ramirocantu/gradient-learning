import pytest
from sqlalchemy import select

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Section, FoundationalConcept, ContentCategory
from datetime import datetime, timedelta, timezone


async def _get_or_create_section(session, *, code, name, position=1):
    sec = (await session.execute(select(Section).where(Section.code == code))).scalar_one_or_none()
    if sec is None:
        sec = Section(code=code, name=name, position=position)
        session.add(sec)
        await session.flush()
    return sec


async def _get_or_create_fc(session, *, section_id, code, name, position=1):
    fc = (
        await session.execute(select(FoundationalConcept).where(FoundationalConcept.code == code))
    ).scalar_one_or_none()
    if fc is None:
        fc = FoundationalConcept(section_id=section_id, code=code, name=name, position=position)
        session.add(fc)
        await session.flush()
    return fc


async def _get_or_create_cc(session, *, fc_id, code, name, position=1):
    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == code))
    ).scalar_one_or_none()
    if cc is None:
        cc = ContentCategory(foundational_concept_id=fc_id, code=code, name=name, position=position)
        session.add(cc)
        await session.flush()
    return cc


@pytest.mark.asyncio
async def test_home_renders_200(client, session):
    response = await client.get("/")
    assert response.status_code == 200
    assert "MCAT Coach" in response.text


@pytest.mark.asyncio
async def test_home_shows_total_questions_and_attempts(client, session):
    now = datetime.now(timezone.utc)
    # seed 5 questions
    questions = []
    for i in range(5):
        q = Question(
            qid=f"Q{i}",
            stem_html="test",
            stem_plain="test",
            choices=[],
            correct_choice="A",
        )
        session.add(q)
        questions.append(q)
    await session.commit()

    # seed 3 attempts
    for i in range(3):
        a = Attempt(
            question_id=questions[i].id,
            attempted_at=now,
            selected_choice="A",
            is_correct=(i < 2),
        )
        session.add(a)
    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    assert "5" in response.text
    assert "3" in response.text


@pytest.mark.asyncio
async def test_home_shows_overall_accuracy(client, session):
    now = datetime.now(timezone.utc)

    # Reuse seeded canonical AAMC outline rows where possible to avoid
    # unique-code collisions with the session-scoped ``seeded_report`` outline.
    sec = await _get_or_create_section(session, code="CP", name="Chem/Phys")
    fc = await _get_or_create_fc(session, section_id=sec.id, code="FC1", name="FC1")
    cc = await _get_or_create_cc(session, fc_id=fc.id, code="1A", name="1A")
    await session.commit()

    q1 = Question(qid="Q1", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    q2 = Question(qid="Q2", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    session.add_all([q1, q2])
    await session.commit()

    qt1 = QuestionTag(
        question_id=q1.id,
        content_category_id=cc.id,
        source="uworld_map",
        confidence=1.0,
    )
    qt2 = QuestionTag(
        question_id=q2.id,
        content_category_id=cc.id,
        source="uworld_map",
        confidence=1.0,
    )
    session.add_all([qt1, qt2])
    await session.commit()

    a1 = Attempt(
        question_id=q1.id,
        attempted_at=now - timedelta(minutes=2),
        selected_choice="A",
        is_correct=True,
    )
    a2 = Attempt(
        question_id=q2.id,
        attempted_at=now - timedelta(minutes=1),
        selected_choice="A",
        is_correct=True,
    )
    a3 = Attempt(question_id=q1.id, attempted_at=now, selected_choice="B", is_correct=False)
    session.add_all([a1, a2, a3])
    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    # Q1 latest=wrong (a3), Q2 latest=correct (a2). Unique-question accuracy = 1/2 = 50%.
    # Raw-attempt accuracy (2/3 = 66.7%) was the pre-6.2a expectation.
    assert "50.0%" in response.text


@pytest.mark.asyncio
async def test_home_streak_zero_when_no_today_attempts(client, session):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    q1 = Question(qid="Q1", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    session.add(q1)
    await session.commit()

    a1 = Attempt(question_id=q1.id, attempted_at=yesterday, selected_choice="A", is_correct=True)
    session.add(a1)
    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    # Streak should be 0 because no attempt today
    assert ">0<" in response.text


@pytest.mark.asyncio
async def test_home_streak_counts_consecutive_days(client, session):
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    q1 = Question(qid="Q1", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    session.add(q1)
    await session.commit()

    a1 = Attempt(question_id=q1.id, attempted_at=now, selected_choice="A", is_correct=True)
    a2 = Attempt(question_id=q1.id, attempted_at=yesterday, selected_choice="A", is_correct=True)
    session.add_all([a1, a2])
    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    assert ">2<" in response.text


@pytest.mark.asyncio
async def test_home_study_next_renders_recommendations(client, session):
    """Seed weak topics so recommend() produces cards; assert template renders."""
    now = datetime.now(timezone.utc)

    sec = await _get_or_create_section(session, code="CP", name="Chem/Phys")
    fc = await _get_or_create_fc(session, section_id=sec.id, code="FC1", name="FC1")
    cc = await _get_or_create_cc(session, fc_id=fc.id, code="1A", name="WeaknessArea 0")
    # Force the display name expected by this test even if a canonical seed row
    # already supplied a different name.
    cc.name = "WeaknessArea 0"
    await session.commit()

    for j in range(5):
        q = Question(
            qid=f"Q_w{j}",
            stem_html="test",
            stem_plain="test",
            choices=[],
            correct_choice="A",
        )
        session.add(q)
        await session.commit()
        session.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc.id,
                source="uworld_map",
                confidence=1.0,
            )
        )
        session.add(
            Attempt(
                question_id=q.id,
                attempted_at=now,
                selected_choice="A",
                is_correct=False,
            )
        )

    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    assert "Study Next" in response.text
    assert "WeaknessArea 0" in response.text


@pytest.mark.asyncio
async def test_home_study_next_empty_state(client, session):
    response = await client.get("/")
    assert "Not enough attempts yet to identify weak areas" in response.text


@pytest.mark.asyncio
async def test_home_recent_activity_lists_5_most_recent(client, session):
    now = datetime.now(timezone.utc)
    for i in range(8):
        q = Question(
            qid=f"Q{i}",
            stem_html="test",
            stem_plain="test",
            choices=[],
            correct_choice="A",
        )
        session.add(q)
        await session.commit()
        a = Attempt(
            question_id=q.id,
            attempted_at=now - timedelta(minutes=10 - i),
            selected_choice="A",
            is_correct=True,
        )
        session.add(a)
    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    assert response.text.count("Uncategorized") == 5


@pytest.mark.asyncio
async def test_home_recent_activity_correctness_icons(client, session):
    now = datetime.now(timezone.utc)
    q1 = Question(qid="Q1", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    q2 = Question(qid="Q2", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    session.add_all([q1, q2])
    await session.commit()

    a1 = Attempt(question_id=q1.id, attempted_at=now, selected_choice="A", is_correct=True)
    a2 = Attempt(
        question_id=q2.id,
        attempted_at=now - timedelta(minutes=1),
        selected_choice="A",
        is_correct=False,
    )
    session.add_all([a1, a2])
    await session.commit()

    response = await client.get("/")
    assert response.status_code == 200
    assert "text-green-600" in response.text or "text-green-400" in response.text
    assert "text-red-600" in response.text or "text-red-400" in response.text


@pytest.mark.asyncio
async def test_mastery_page_renders(client, session):
    response = await client.get("/mastery")
    assert response.status_code == 200
    assert "Mastery Overview" in response.text


@pytest.mark.asyncio
async def test_home_back_links_to_viewer(client, session):
    response = await client.get("/")
    assert response.status_code == 200
    assert 'href="/viewer/captures"' in response.text


# --------------------------------------------------------------------------- #
# Ticket 6.7 — Bug #16: no delete button for unknown-source tags
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_question_detail_does_not_render_delete_button_for_unknown_source(client, session):
    """Tags with source='uworld_map' render their label but no hx-delete button."""
    sec = await _get_or_create_section(session, code="CP", name="Chem/Phys")
    fc = await _get_or_create_fc(session, section_id=sec.id, code="FC1", name="FC1")
    cc = await _get_or_create_cc(session, fc_id=fc.id, code="4A", name="CC 4A")
    await session.flush()

    q = Question(
        qid="Q-NODBTN",
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[{"key": "A", "html": "<p>A</p>", "plain": "A"}],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()

    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="uworld_map",
        confidence=1.0,
    )
    session.add(qt)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text
    # Tag label renders
    assert "4A" in body
    # No delete button for this tag
    assert f'hx-delete="/tags/{qt.id}"' not in body
