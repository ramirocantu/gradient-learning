"""Tests for the per-CC drilldown page (Ticket 6.3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.captures import Attempt, Passage, Question, QuestionTag
from app.models.media import Media
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic


# --------------------------------------------------------------------------- #
# Seed helpers
# --------------------------------------------------------------------------- #


async def _seed_cc(
    session,
    *,
    code: str,
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
    t = Topic(
        content_category_id=cc.id,
        name=name,
        depth=0,
        position=position,
    )
    session.add(t)
    await session.flush()
    return t


async def _seed_question(
    session,
    *,
    qid: str,
    stem_plain: str = "What is the answer?",
    stem_html: str | None = None,
    explanation_html: str = "<p>Because.</p>",
    correct_choice: str = "A",
    choices=None,
    passage_id: int | None = None,
) -> Question:
    if choices is None:
        choices = [
            {"key": "A", "html": "<p>A</p>", "plain": "A"},
            {"key": "B", "html": "<p>B</p>", "plain": "B"},
            {"key": "C", "html": "<p>C</p>", "plain": "C"},
            {"key": "D", "html": "<p>D</p>", "plain": "D"},
        ]
    q = Question(
        qid=qid,
        stem_html=stem_html if stem_html is not None else f"<p>{stem_plain}</p>",
        stem_plain=stem_plain,
        choices=choices,
        correct_choice=correct_choice,
        explanation_html=explanation_html,
        explanation_plain="Because.",
        passage_id=passage_id,
    )
    session.add(q)
    await session.flush()
    return q


async def _seed_attempt(
    session,
    *,
    question: Question,
    is_correct: bool = True,
    selected: str = "A",
    days_ago: int = 0,
) -> Attempt:
    a = Attempt(
        question_id=question.id,
        attempted_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        selected_choice=selected,
        is_correct=is_correct,
    )
    session.add(a)
    await session.flush()
    return a


async def _seed_tag(
    session,
    *,
    question: Question,
    cc: ContentCategory | None = None,
    topic: Topic | None = None,
    skill: int | None = None,
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
async def test_drilldown_renders_200(client, session):
    cc = await _seed_cc(session, code="4A", name="Translational motion")
    q = await _seed_question(session, qid="Q1")
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await _seed_attempt(session, question=q, is_correct=True)
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    assert "4A" in r.text


@pytest.mark.asyncio
async def test_drilldown_404_unknown_cc(client, session):
    r = await client.get("/mastery/ZZ")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_drilldown_header_includes_cc_code_and_name(client, session):
    await _seed_cc(session, code="4A", name="Translational motion")
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    assert "4A" in r.text
    assert "Translational motion" in r.text


@pytest.mark.asyncio
async def test_drilldown_lists_topics_under_cc(client, session):
    cc_4a = await _seed_cc(session, code="4A", name="CC 4A")
    cc_5a = await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    t1 = await _seed_topic(session, cc=cc_4a, name="ZZZ_Force")
    t2 = await _seed_topic(session, cc=cc_4a, name="ZZZ_Energy")
    t3 = await _seed_topic(session, cc=cc_5a, name="ZZZ_Bonds")

    # Need at least one attempt per topic so they show in the table.
    for t in (t1, t2, t3):
        q = await _seed_question(session, qid=f"Q_{t.id}")
        await _seed_tag(session, question=q, topic=t, source="llm")
        await _seed_attempt(session, question=q, is_correct=False)
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    assert "ZZZ_Force" in r.text
    assert "ZZZ_Energy" in r.text
    assert "ZZZ_Bonds" not in r.text


@pytest.mark.asyncio
async def test_drilldown_orders_topics_by_wilson_lower(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    weak = await _seed_topic(session, cc=cc, name="WEAKEST", position=1)
    middle = await _seed_topic(session, cc=cc, name="MIDDLE", position=2)
    strong = await _seed_topic(session, cc=cc, name="STRONGEST", position=3)

    # weak: 0 / 5, middle: 3 / 5, strong: 5 / 5
    for topic, correct_n in ((weak, 0), (middle, 3), (strong, 5)):
        for i in range(5):
            q = await _seed_question(session, qid=f"Q_{topic.name}_{i}")
            await _seed_tag(session, question=q, topic=topic, source="llm")
            await _seed_attempt(session, question=q, is_correct=(i < correct_n))
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200

    body = r.text
    weak_pos = body.find("WEAKEST")
    middle_pos = body.find("MIDDLE")
    strong_pos = body.find("STRONGEST")
    assert weak_pos != -1 and middle_pos != -1 and strong_pos != -1
    assert weak_pos < middle_pos < strong_pos


@pytest.mark.asyncio
async def test_drilldown_question_card_shows_preview_and_attempt_status(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(
        session,
        qid="Q-CARD",
        stem_plain="A particle moves with velocity v in a field of strength E.",
        correct_choice="C",
    )
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await _seed_attempt(session, question=q, is_correct=False, selected="B")
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    body = r.text
    assert "A particle moves with velocity" in body
    assert "Q-CARD" in body
    # selected B, correct C, wrong indicator
    assert "wrong" in body
    assert (
        ">B<" in body
        or 'selected <span class="font-semibold text-gray-700 dark:text-gray-300">B</span>' in body
    )
    assert (
        ">C<" in body
        or 'correct <span class="font-semibold text-gray-700 dark:text-gray-300">C</span>' in body
    )


@pytest.mark.asyncio
async def test_drilldown_question_card_shows_existing_tags(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    topic = await _seed_topic(session, cc=cc, name="ZZ_TopicTagged")
    q = await _seed_question(session, qid="Q-TAGS")
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await _seed_tag(session, question=q, topic=topic, source="manual")
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    body = r.text
    assert "ZZ_TopicTagged" in body
    assert "manual" in body.lower()
    assert "llm" in body.lower()


@pytest.mark.asyncio
async def test_drilldown_paginates_at_20(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    for i in range(25):
        q = await _seed_question(session, qid=f"PAGE-{i:02d}")
        await _seed_tag(session, question=q, cc=cc, source="llm")
        await _seed_attempt(session, question=q, is_correct=True, days_ago=i)
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    body = r.text
    # Page 1 has the 20 most recent; PAGE-00 is most recent (days_ago=0).
    assert body.count("PAGE-00") >= 1
    assert body.count("PAGE-19") >= 1
    assert "PAGE-24" not in body
    assert "page=2" in body  # Next link

    r2 = await client.get("/mastery/4A?page=2")
    assert r2.status_code == 200
    body2 = r2.text
    assert "PAGE-24" in body2
    assert "PAGE-20" in body2
    assert "PAGE-00" not in body2


@pytest.mark.asyncio
async def test_show_full_fragment_returns_expanded_card(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    passage = Passage(
        content_hash="abc123",
        html="<p>This is the passage.</p>",
        plain_text="This is the passage.",
    )
    session.add(passage)
    await session.flush()
    q = await _seed_question(
        session,
        qid="Q-FULL",
        stem_html="<p>What is the resonance frequency?</p>",
        stem_plain="What is the resonance frequency?",
        explanation_html="<p>The frequency is f0.</p>",
        passage_id=passage.id,
    )
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await _seed_attempt(session, question=q, is_correct=True)
    await session.commit()

    r = await client.get(f"/mastery/4A/questions/{q.id}/full")
    assert r.status_code == 200
    body = r.text
    assert "What is the resonance frequency?" in body
    assert "This is the passage." in body
    assert "The frequency is f0." in body
    # Choice keys
    for letter in ("A", "B", "C", "D"):
        assert f"{letter}." in body


@pytest.mark.asyncio
async def test_show_full_fragment_rewrites_media_refs(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    media = Media(
        content_hash="HASH123",
        local_path="ab/HASH123.png",
        mime_type="image/png",
        byte_size=100,
    )
    session.add(media)
    await session.flush()
    q = await _seed_question(
        session,
        qid="Q-MEDIA",
        stem_html='<p>See: <img data-media-content-hash="HASH123" alt="diagram"/></p>',
        stem_plain="See diagram",
    )
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await session.commit()

    r = await client.get(f"/mastery/4A/questions/{q.id}/full")
    assert r.status_code == 200
    assert 'src="/media/ab/HASH123.png"' in r.text
    assert "data-media-content-hash" not in r.text


@pytest.mark.asyncio
async def test_retag_form_fragment_renders(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    q = await _seed_question(session, qid="Q-FORM")
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await session.commit()

    r = await client.get(f"/mastery/4A/questions/{q.id}/retag-form")
    assert r.status_code == 200
    body = r.text
    assert 'name="target_cc_code"' in body
    assert 'name="target_topic_id"' in body
    assert 'name="target_skill"' in body
    assert "5A" in body  # other CC option present


@pytest.mark.asyncio
async def test_topic_options_fragment_for_cc(client, session):
    cc_4a = await _seed_cc(session, code="4A", name="CC 4A")
    cc_5a = await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    await _seed_topic(session, cc=cc_4a, name="ZZ_T_4A_one")
    await _seed_topic(session, cc=cc_5a, name="ZZ_T_5A_one")
    await _seed_topic(session, cc=cc_5a, name="ZZ_T_5A_two")
    await session.commit()

    r = await client.get("/mastery/4A/topic-options?cc_code=5A")
    assert r.status_code == 200
    body = r.text
    assert "ZZ_T_5A_one" in body
    assert "ZZ_T_5A_two" in body
    assert "ZZ_T_4A_one" not in body


@pytest.mark.asyncio
async def test_retag_submit_creates_manual_tag(client, session):
    cc_4a = await _seed_cc(session, code="4A", name="CC 4A")
    cc_5a = await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    q = await _seed_question(session, qid="Q-RETAG")
    await _seed_tag(session, question=q, cc=cc_4a, source="llm")
    await session.commit()
    qid = q.id

    r = await client.post(
        f"/mastery/4A/questions/{qid}/retag",
        data={"target_cc_code": "5A"},
    )
    assert r.status_code == 200

    rows = (
        (
            await session.execute(
                select(QuestionTag).where(
                    QuestionTag.question_id == qid,
                    QuestionTag.source == "manual",
                    QuestionTag.content_category_id == cc_5a.id,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_retag_submit_handles_duplicate_409(client, session):
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(session, qid="Q-DUP")
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await session.commit()
    qid = q.id

    r1 = await client.post(
        f"/mastery/4A/questions/{qid}/retag",
        data={"target_cc_code": "4A"},
    )
    assert r1.status_code == 200

    r2 = await client.post(
        f"/mastery/4A/questions/{qid}/retag",
        data={"target_cc_code": "4A"},
    )
    # Should not 500; should return error fragment with helpful message.
    assert r2.status_code == 200
    assert "already exist" in r2.text.lower() or "Re-tag failed" in r2.text


@pytest.mark.asyncio
async def test_retag_submit_returns_refreshed_card(client, session):
    cc_4a = await _seed_cc(session, code="4A", name="CC 4A")
    await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    q = await _seed_question(session, qid="Q-REFRESH")
    await _seed_tag(session, question=q, cc=cc_4a, source="llm")
    await session.commit()
    qid = q.id

    r = await client.post(
        f"/mastery/4A/questions/{qid}/retag",
        data={"target_cc_code": "5A"},
    )
    assert r.status_code == 200
    body = r.text
    # Refreshed card should include the question id container.
    assert f"question-card-{qid}" in body
    # And the new manual tag for 5A.
    assert "5A" in body
    assert "manual" in body.lower()


@pytest.mark.asyncio
async def test_question_card_htmx_urls_contain_cc_code(client, session):
    """Bug #8 regression: question card HTMX URLs include cc_code — no double-slash."""
    cc = await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(session, qid="Q-HTMX")
    await _seed_tag(session, question=q, cc=cc, source="llm")
    await _seed_attempt(session, question=q, is_correct=True)
    await session.commit()

    r = await client.get("/mastery/4A")
    assert r.status_code == 200
    assert 'hx-get="/mastery/4A/questions/' in r.text
    assert 'hx-get="/mastery//questions/' not in r.text


# --------------------------------------------------------------------------- #
# Ticket 6.7 — Bug #14: rationale in TagSummary; Bug #16: overridden filter
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tags_summaries_include_rationale(client, session):
    """TagSummary.rationale is populated from the QuestionTag row."""
    from app.web.dashboard.services.drilldown import _tags_summaries_for

    cc = await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(session, qid="Q-RAT")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="llm",
        confidence=0.9,
        rationale="W=Fd identified in stem",
    )
    session.add(qt)
    await session.commit()

    result = await _tags_summaries_for(session, [q.id])

    summaries = result[q.id]
    assert len(summaries) == 1
    assert summaries[0].rationale == "W=Fd identified in stem"


@pytest.mark.asyncio
async def test_tags_summaries_omit_overridden(client, session):
    """Tags with is_overridden=True are excluded from TagSummary results."""
    from datetime import datetime, timezone

    from app.web.dashboard.services.drilldown import _tags_summaries_for

    cc = await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(session, qid="Q-OVR")

    qt_active = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="llm",
        confidence=0.9,
    )
    qt_overridden = QuestionTag(
        question_id=q.id,
        skill=2,
        source="llm",
        confidence=0.8,
        is_overridden=True,
        overridden_at=datetime.now(timezone.utc),
    )
    session.add(qt_active)
    session.add(qt_overridden)
    await session.commit()

    result = await _tags_summaries_for(session, [q.id])

    summaries = result[q.id]
    assert len(summaries) == 1
    assert summaries[0].kind == "content_category"


@pytest.mark.asyncio
async def test_tags_summaries_manual_tag_has_no_rationale_required(client, session):
    """Manual tags with rationale=None surface without error; detail page renders 200."""
    from app.web.dashboard.services.drilldown import _tags_summaries_for

    cc = await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(session, qid="Q-MRAT")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="manual",
        confidence=1.0,
        rationale=None,
    )
    session.add(qt)
    await session.commit()

    result = await _tags_summaries_for(session, [q.id])

    summaries = result[q.id]
    assert len(summaries) == 1
    assert summaries[0].rationale is None

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Ticket 6.7 — Bug #16: retag form disabled options
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retag_form_disables_already_tagged_ccs(client, session):
    """CC already tagged on the question renders as disabled in the retag form."""
    cc_4a = await _seed_cc(session, code="4A", name="CC 4A")
    await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    q = await _seed_question(session, qid="Q-DISA")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc_4a.id,
        source="llm",
        confidence=0.9,
    )
    session.add(qt)
    await session.commit()

    r = await client.get(f"/mastery/4A/questions/{q.id}/retag-form")
    assert r.status_code == 200
    body = r.text
    # The 4A option should be disabled with "(already tagged)"
    assert "4A" in body
    assert "(already tagged)" in body
    # Verify the disabled attribute appears near 4A option
    idx_4a = body.find('value="4A"')
    assert idx_4a != -1
    option_slice = body[idx_4a : idx_4a + 200]
    assert "disabled" in option_slice


@pytest.mark.asyncio
async def test_retag_form_disables_already_tagged_skills(client, session):
    """Skill already tagged on the question renders as disabled in the retag form."""
    await _seed_cc(session, code="4A", name="CC 4A")
    q = await _seed_question(session, qid="Q-DISKL")
    qt = QuestionTag(
        question_id=q.id,
        skill=2,
        source="llm",
        confidence=0.9,
    )
    session.add(qt)
    await session.commit()

    r = await client.get(f"/mastery/4A/questions/{q.id}/retag-form")
    assert r.status_code == 200
    body = r.text
    # Scope the search to the Skill select so we don't match topic-id "2" options
    # surfaced by the canonical seeded outline (Topic dropdown precedes Skill in
    # the rendered form).
    skill_select_start = body.find('name="target_skill"')
    assert skill_select_start != -1
    skill_slice = body[skill_select_start : skill_select_start + 2000]
    idx_sk2 = skill_slice.find('value="2"')
    assert idx_sk2 != -1
    option_slice = skill_slice[idx_sk2 : idx_sk2 + 100]
    assert "disabled" in option_slice
    assert "(already tagged)" in body


@pytest.mark.asyncio
async def test_retag_form_does_not_disable_non_tagged_ccs(client, session):
    """CCs not applied to the question are NOT disabled in the retag form."""
    cc_4a = await _seed_cc(session, code="4A", name="CC 4A")
    await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    q = await _seed_question(session, qid="Q-NODIS")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc_4a.id,
        source="llm",
        confidence=0.9,
    )
    session.add(qt)
    await session.commit()

    r = await client.get(f"/mastery/4A/questions/{q.id}/retag-form")
    assert r.status_code == 200
    body = r.text

    # 5A is not tagged, so its option should not be disabled.
    idx_5a = body.find('value="5A"')
    assert idx_5a != -1
    option_slice = body[idx_5a : idx_5a + 200]
    assert "disabled" not in option_slice


@pytest.mark.asyncio
async def test_mastery_tiles_link_to_drilldown(client, session):
    cc1 = await _seed_cc(session, code="4A", name="CC 4A")
    await _seed_cc(session, code="5A", name="CC 5A", fc_code="FC2", fc_name="FC2")
    q = await _seed_question(session, qid="Q-LINK")
    await _seed_tag(session, question=q, cc=cc1, source="llm")
    await _seed_attempt(session, question=q, is_correct=True)
    await session.commit()

    r = await client.get("/mastery")
    assert r.status_code == 200
    assert 'href="/mastery/4A"' in r.text
    assert 'href="/mastery/5A"' in r.text
