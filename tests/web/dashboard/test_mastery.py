from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, FoundationalConcept, Section


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
async def test_mastery_renders_200(client, session):
    response = await client.get("/mastery")
    assert response.status_code == 200
    assert "Mastery Overview" in response.text
    assert "Back to Home" in response.text


@pytest.mark.asyncio
async def test_mastery_shows_grouped_ccs(client, session):
    now = datetime.now(timezone.utc)

    sec = await _get_or_create_section(session, code="CP", name="Chem/Phys")
    fc = await _get_or_create_fc(session, section_id=sec.id, code="FC1", name="FC1")
    cc1 = await _get_or_create_cc(session, fc_id=fc.id, code="1A", name="Category 1A")
    cc2 = await _get_or_create_cc(session, fc_id=fc.id, code="1B", name="Category 1B", position=2)
    # Force display names to match test assertions even when canonical seeds
    # already populated other names.
    sec.name = "Chem/Phys"
    cc1.name = "Category 1A"
    cc2.name = "Category 1B"
    await session.commit()

    q1 = Question(qid="Q1", stem_html="test", stem_plain="test", choices=[], correct_choice="A")
    session.add(q1)
    await session.commit()

    qt1 = QuestionTag(
        question_id=q1.id,
        content_category_id=cc1.id,
        source="uworld_map",
        confidence=1.0,
    )
    session.add(qt1)
    await session.commit()

    a1 = Attempt(question_id=q1.id, attempted_at=now, selected_choice="A", is_correct=True)
    session.add(a1)
    await session.commit()

    response = await client.get("/mastery")
    assert response.status_code == 200
    assert "Chem/Phys" in response.text
    assert "1A" in response.text
    assert "Category 1A" in response.text
    assert "1B" in response.text
    assert "Category 1B" in response.text


@pytest.mark.asyncio
async def test_mastery_side_panel_titled_study_next(client, session):
    response = await client.get("/mastery")
    assert response.status_code == 200
    assert "Study Next" in response.text


@pytest.mark.asyncio
async def test_mastery_side_panel_renders_recommendations(client, session):
    """Seed weak topics so recommend() produces topic_weakness cards.

    Logic-level ranking is covered by backend recommender tests; this only
    verifies the template renders the cards without error.
    """
    now = datetime.now(timezone.utc)

    sec = await _get_or_create_section(session, code="BB", name="Bio/Biochem", position=2)
    fc = await _get_or_create_fc(session, section_id=sec.id, code="FC2", name="FC2")

    for i in range(3):
        cc = await _get_or_create_cc(
            session, fc_id=fc.id, code=f"2{chr(65 + i)}", name=f"WeaknessArea {i}"
        )
        # Force the recommender-template-visible name to match assertions even
        # if a canonical seed already provided a different display name.
        cc.name = f"WeaknessArea {i}"
        await session.commit()

        for j in range(5):
            q = Question(
                qid=f"Q_W{i}_{j}",
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
                    attempted_at=now - timedelta(days=i),
                    selected_choice="A",
                    is_correct=False,
                )
            )

    await session.commit()

    response = await client.get("/mastery")
    assert response.status_code == 200
    assert "Topic" in response.text  # kind badge label
    assert "WeaknessArea 0" in response.text


@pytest.mark.asyncio
async def test_mastery_shows_20_recent_attempts(client, session):
    now = datetime.now(timezone.utc)

    for i in range(25):
        q = Question(
            qid=f"Q_R{i}",
            stem_html="test",
            stem_plain="test",
            choices=[],
            correct_choice="A",
        )
        session.add(q)
        await session.commit()
        a = Attempt(
            question_id=q.id,
            attempted_at=now - timedelta(minutes=i),
            selected_choice="A",
            is_correct=True,
        )
        session.add(a)
    await session.commit()

    response = await client.get("/mastery")
    assert response.status_code == 200
    assert response.text.count("Uncategorized") == 20
