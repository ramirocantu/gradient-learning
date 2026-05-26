"""Standalone question detail routes (Ticket 6.6 + 6.9c)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Passage, Question, QuestionTag
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic
from app.services.recommender import RecommendationResult, StudyRecommendation


# --------------------------------------------------------------------------- #
# Seed helpers
# --------------------------------------------------------------------------- #


async def _seed_section_cc(session) -> ContentCategory:
    sec = (await session.execute(select(Section).where(Section.code == "CP"))).scalar_one_or_none()
    if sec is None:
        sec = Section(code="CP", name="Chem/Phys", position=1)
        session.add(sec)
        await session.flush()
    fc = (
        await session.execute(select(FoundationalConcept).where(FoundationalConcept.code == "FC1"))
    ).scalar_one_or_none()
    if fc is None:
        fc = FoundationalConcept(section_id=sec.id, code="FC1", name="FC1", position=1)
        session.add(fc)
        await session.flush()
    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one_or_none()
    if cc is None:
        cc = ContentCategory(
            foundational_concept_id=fc.id,
            code="4A",
            name="Translational motion",
            position=1,
        )
        session.add(cc)
        await session.flush()
    return cc


async def _seed_question(
    session,
    *,
    qid: str = "402391",
    correct_choice: str = "C",
    passage_id: int | None = None,
) -> Question:
    q = Question(
        qid=qid,
        stem_html="<p>What is the answer?</p>",
        stem_plain="What is the answer?",
        choices=[
            {"key": "A", "html": "<p>Alpha</p>", "plain": "Alpha"},
            {"key": "B", "html": "<p>Bravo</p>", "plain": "Bravo"},
            {"key": "C", "html": "<p>Charlie</p>", "plain": "Charlie"},
            {"key": "D", "html": "<p>Delta</p>", "plain": "Delta"},
        ],
        correct_choice=correct_choice,
        explanation_html="<p>Because.</p>",
        explanation_plain="Because.",
        passage_id=passage_id,
    )
    session.add(q)
    await session.flush()
    return q


async def _seed_attempt(session, *, question: Question, selected: str, is_correct: bool) -> Attempt:
    a = Attempt(
        question_id=question.id,
        attempted_at=datetime.now(timezone.utc),
        selected_choice=selected,
        is_correct=is_correct,
    )
    session.add(a)
    await session.flush()
    return a


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_question_detail_returns_200(client, session):
    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="402391", correct_choice="C")
    session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cc.id,
            source="llm",
            confidence=0.9,
        )
    )
    await _seed_attempt(session, question=q, selected="C", is_correct=True)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    assert "402391" in r.text


@pytest.mark.asyncio
async def test_question_detail_404_on_missing_id(client, session):
    r = await client.get("/questions/999999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_by_qid_redirects_to_integer_id(client, session):
    q = await _seed_question(session, qid="402391")
    await session.commit()

    r = await client.get("/questions/by-qid/402391", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"/questions/{q.id}"


@pytest.mark.asyncio
async def test_by_qid_404_on_missing_qid(client, session):
    r = await client.get("/questions/by-qid/notreal", follow_redirects=False)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_question_detail_shows_correct_answer_highlighted(client, session):
    q = await _seed_question(session, qid="Q-CORRECT", correct_choice="C")
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text
    assert "Charlie" in body
    assert "✓ Correct" in body


@pytest.mark.asyncio
async def test_question_detail_shows_your_answer(client, session):
    q = await _seed_question(session, qid="Q-WRONG", correct_choice="C")
    await _seed_attempt(session, question=q, selected="B", is_correct=False)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text
    assert "← Your answer" in body
    assert "✓ Correct" in body
    assert "✗ Incorrect" in body  # header badge for incorrect attempt


@pytest.mark.asyncio
async def test_question_detail_renders_passage_when_present(client, session):
    passage = Passage(
        content_hash="hash-detail-1",
        html="<p>Long passage prose.</p>",
        plain_text="The kinetic theory of gases postulates many things.",
    )
    session.add(passage)
    await session.flush()
    q = await _seed_question(session, qid="Q-PASS", passage_id=passage.id)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text
    assert "The kinetic theory of gases postulates many things." in body
    assert "Passage" in body


@pytest.mark.asyncio
async def test_recent_activity_items_are_links(client, session):
    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="Q-ACT")
    session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cc.id,
            source="llm",
            confidence=1.0,
        )
    )
    await _seed_attempt(session, question=q, selected="A", is_correct=False)
    await session.commit()

    r = await client.get("/mastery")
    assert r.status_code == 200
    assert f'href="/questions/{q.id}"' in r.text


@pytest.mark.asyncio
async def test_home_recent_activity_items_are_links(client, session):
    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="Q-HOMEACT")
    session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cc.id,
            source="llm",
            confidence=1.0,
        )
    )
    await _seed_attempt(session, question=q, selected="A", is_correct=True)
    await session.commit()

    r = await client.get("/")
    assert r.status_code == 200
    assert f'href="/questions/{q.id}"' in r.text


@pytest.mark.asyncio
async def test_recommendations_qid_chips_are_links(client, session):
    rec = StudyRecommendation(
        kind="feature_pattern",
        label=None,
        code=None,
        target_id=None,
        accuracy=None,
        wilson_lower=None,
        attempts=None,
        feature_name="involves_graph",
        feature_value="True",
        accuracy_with=0.30,
        accuracy_without=0.80,
        priority_score=0.5,
        reason="Stub.",
        representative_qids=["402391", "402392"],
    )
    result = RecommendationResult(
        recommendations=[rec],
        total_candidates_scored=1,
    )
    with patch(
        "app.web.dashboard.routes.recommendations.recommend",
        new=AsyncMock(return_value=result),
    ):
        r = await client.get("/recommendations")
    assert r.status_code == 200
    assert 'href="/questions/by-qid/402391"' in r.text
    assert 'href="/questions/by-qid/402392"' in r.text


# --------------------------------------------------------------------------- #
# Ticket 6.7 — Bug #16: dashboard DELETE /tags/{tag_id}
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_delete_manual_tag_via_dashboard_route_204(client, session):
    """DELETE /tags/{tag_id} hard-deletes a manual tag."""
    from sqlalchemy import select

    from app.models.captures import QuestionTag

    cc = await _seed_section_cc(session)
    q = await _seed_question(session)
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="manual",
        confidence=1.0,
    )
    session.add(qt)
    await session.commit()
    tag_id = qt.id

    r = await client.delete(f"/tags/{tag_id}")
    assert r.status_code == 204

    row = (
        await session.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
    ).scalar_one_or_none()
    assert row is None


@pytest.mark.asyncio
async def test_delete_llm_tag_via_dashboard_route_soft_deletes(client, session):
    """DELETE /tags/{tag_id} soft-deletes an LLM tag (is_overridden=True)."""
    from sqlalchemy import select

    from app.models.captures import QuestionTag

    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="Q-SDEL")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="llm",
        confidence=0.9,
    )
    session.add(qt)
    await session.commit()
    tag_id = qt.id

    r = await client.delete(f"/tags/{tag_id}")
    assert r.status_code == 204

    # Expire identity-map cache so the reload reflects the route's commit.
    session.expire_all()
    row = (
        await session.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
    ).scalar_one_or_none()
    assert row is not None
    assert row.is_overridden is True
    assert row.overridden_at is not None


@pytest.mark.asyncio
async def test_delete_tag_404_when_missing_on_dashboard_route(client, session):
    r = await client.delete("/tags/999999")
    assert r.status_code == 404
    assert "tag_id=999999 not found" in r.json()["detail"]


@pytest.mark.asyncio
async def test_delete_tag_403_when_source_uworld_map_on_dashboard_route(client, session):
    """Tags with source='uworld_map' return 403; row is not modified."""
    from sqlalchemy import select

    from app.models.captures import QuestionTag

    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="Q-UMAP")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="uworld_map",
        confidence=1.0,
    )
    session.add(qt)
    await session.commit()
    tag_id = qt.id

    r = await client.delete(f"/tags/{tag_id}")
    assert r.status_code == 403

    row = (
        await session.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
    ).scalar_one_or_none()
    assert row is not None
    assert row.is_overridden is False


# --------------------------------------------------------------------------- #
# Ticket 6.7 — Bug #14: rationale rendering on question detail page
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_question_detail_renders_rationale_for_llm_tag(client, session):
    """Question detail page renders the rationale text below each LLM tag."""
    from app.models.captures import QuestionTag

    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="Q-REND-RAT")
    qt = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="llm",
        confidence=0.9,
        rationale="Energy conservation applies here",
    )
    session.add(qt)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    assert "Energy conservation applies here" in r.text


@pytest.mark.asyncio
async def test_question_detail_renders_delete_button_for_removable_tags(client, session):
    """LLM and manual tags both get a delete button; only LLM gets hx-confirm."""
    from app.models.captures import QuestionTag

    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="Q-DELBTN")
    qt_llm = QuestionTag(
        question_id=q.id,
        content_category_id=cc.id,
        source="llm",
        confidence=0.9,
    )
    qt_manual = QuestionTag(
        question_id=q.id,
        skill=2,
        source="manual",
        confidence=1.0,
    )
    session.add(qt_llm)
    session.add(qt_manual)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text

    assert body.count('hx-delete="/tags/') == 2
    assert body.count("hx-confirm=") == 1


@pytest.mark.asyncio
async def test_by_qid_route_resolves_before_integer_route(client, session):
    """Regression: literal 'by-qid' must not be mistaken as an int question_id."""
    r = await client.get("/questions/by-qid/notreal", follow_redirects=False)
    # 404 from missing qid (not 422 from int parsing failure).
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Ticket 6.9a — Inline "+ Add tag" on question detail page
# --------------------------------------------------------------------------- #


async def _seed_topic(session, *, cc: ContentCategory, name: str = "Kinematics") -> Topic:
    t = Topic(content_category_id=cc.id, name=name, depth=0, position=1)
    session.add(t)
    await session.flush()
    return t


async def _seed_second_cc(session) -> ContentCategory:
    """Seed a second CC (code '4B') under the same FC as the first CC from _seed_section_cc."""
    fc = (
        await session.execute(select(FoundationalConcept).where(FoundationalConcept.code == "FC1"))
    ).scalar_one_or_none()
    assert fc is not None, "_seed_second_cc requires _seed_section_cc to have run first"
    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == "4B"))
    ).scalar_one_or_none()
    if cc is None:
        cc = ContentCategory(foundational_concept_id=fc.id, code="4B", name="Waves", position=2)
        session.add(cc)
        await session.flush()
    return cc


@pytest.mark.asyncio
async def test_add_tag_form_renders_with_cc_options(client, session):
    """GET /questions/{id}/add-tag-form returns 200 with CC options and hx-post target."""
    cc = await _seed_section_cc(session)
    q = await _seed_question(session, qid="AT-FORM-1")
    await session.commit()

    r = await client.get(f"/questions/{q.id}/add-tag-form")
    assert r.status_code == 200
    body = r.text
    assert cc.code in body
    assert f'hx-post="/questions/{q.id}/add-tag"' in body


@pytest.mark.asyncio
async def test_add_tag_form_disables_already_tagged_cc(client, session):
    """Already-tagged CC option gets disabled and (already tagged) suffix."""
    cc1 = await _seed_section_cc(session)
    cc2 = await _seed_second_cc(session)
    q = await _seed_question(session, qid="AT-FORM-2")
    session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cc1.id,
            source="manual",
            confidence=1.0,
        )
    )
    await session.commit()

    r = await client.get(f"/questions/{q.id}/add-tag-form")
    assert r.status_code == 200
    body = r.text
    # cc1 option must be disabled; cc2 must not be.
    assert "already tagged" in body
    # cc2 option present and not disabled
    assert cc2.code in body


@pytest.mark.asyncio
async def test_add_tag_form_disables_already_tagged_skill(client, session):
    """Already-tagged skill option gets disabled in the form."""
    await _seed_section_cc(session)
    q = await _seed_question(session, qid="AT-FORM-3")
    session.add(
        QuestionTag(
            question_id=q.id,
            skill=2,
            source="manual",
            confidence=1.0,
        )
    )
    await session.commit()

    r = await client.get(f"/questions/{q.id}/add-tag-form")
    assert r.status_code == 200
    body = r.text
    # Skill 2 option must carry disabled and (already tagged).
    assert "already tagged" in body
    # The skill section must still be rendered with options for 1–4.
    assert 'value="2"' in body


@pytest.mark.asyncio
async def test_add_tag_form_marks_already_tagged_topic_in_json(client, session):
    """Already-tagged topic is marked already_tagged=true in the embedded JSON map."""
    import json as _json
    import re as _re

    cc = await _seed_section_cc(session)
    topic = await _seed_topic(session, cc=cc, name="Kinematics")
    q = await _seed_question(session, qid="AT-FORM-4")
    session.add(
        QuestionTag(
            question_id=q.id,
            topic_id=topic.id,
            source="manual",
            confidence=1.0,
        )
    )
    await session.commit()

    r = await client.get(f"/questions/{q.id}/add-tag-form")
    assert r.status_code == 200
    body = r.text
    # Extract the JSON payload from the embedded <script type="application/json">.
    match = _re.search(
        rf'<script id="addtag-topics-data-{q.id}" type="application/json">(.*?)</script>',
        body,
        flags=_re.DOTALL,
    )
    assert match, "embedded topics JSON not found"
    data = _json.loads(match.group(1))
    entries = data.get(cc.code, [])
    matching = [e for e in entries if e["id"] == topic.id]
    assert matching, f"topic {topic.id} missing from JSON for cc {cc.code}"
    assert matching[0]["name"] == "Kinematics"
    assert matching[0]["already_tagged"] is True


@pytest.mark.asyncio
async def test_add_tag_submit_creates_manual_tag_with_rationale(client, session):
    """POST /questions/{id}/add-tag with skill + rationale creates the correct DB row."""
    await _seed_section_cc(session)
    q = await _seed_question(session, qid="AT-SUB-1")
    await session.commit()
    qid = q.id  # capture before expire_all

    r = await client.post(
        f"/questions/{qid}/add-tag",
        data={"tag_kind": "skill", "skill": "3", "rationale": "Mechanism question"},
    )
    assert r.status_code == 200
    # Response is the refreshed tags-section fragment.
    assert "tags-section" in r.text
    assert "Skill 3" in r.text

    session.expire_all()
    row = (
        await session.execute(
            select(QuestionTag).where(
                QuestionTag.question_id == qid,
                QuestionTag.skill == 3,
            )
        )
    ).scalar_one_or_none()
    assert row is not None
    assert row.source == "manual"
    assert float(row.confidence) == 1.0
    assert row.rationale == "Mechanism question"
    assert row.topic_id is None
    assert row.content_category_id is None


@pytest.mark.asyncio
async def test_add_tag_submit_topic_target(client, session):
    """POST with tag_kind=topic sets topic_id and leaves cc/skill NULL."""
    cc = await _seed_section_cc(session)
    topic = await _seed_topic(session, cc=cc, name="Thermodynamics")
    q = await _seed_question(session, qid="AT-SUB-2")
    await session.commit()
    qid = q.id  # capture before expire_all
    topic_id = topic.id  # capture before expire_all

    r = await client.post(
        f"/questions/{qid}/add-tag",
        data={"tag_kind": "topic", "topic_id": str(topic_id), "cc_code": cc.code},
    )
    assert r.status_code == 200

    session.expire_all()
    row = (
        await session.execute(
            select(QuestionTag).where(
                QuestionTag.question_id == qid,
                QuestionTag.topic_id == topic_id,
            )
        )
    ).scalar_one_or_none()
    assert row is not None
    assert row.topic_id == topic_id
    assert row.content_category_id is None
    assert row.skill is None


@pytest.mark.asyncio
async def test_add_tag_submit_duplicate_returns_error_partial(client, session):
    """Duplicate submission returns 200 with error indicator; no new DB row."""
    await _seed_section_cc(session)
    q = await _seed_question(session, qid="AT-DUP-1")
    session.add(
        QuestionTag(
            question_id=q.id,
            skill=2,
            source="manual",
            confidence=1.0,
        )
    )
    await session.commit()
    qid = q.id  # capture before expire_all

    r = await client.post(
        f"/questions/{qid}/add-tag",
        data={"tag_kind": "skill", "skill": "2"},
    )
    assert r.status_code == 200
    assert "add-tag-error" in r.text

    session.expire_all()
    rows = (
        (
            await session.execute(
                select(QuestionTag).where(
                    QuestionTag.question_id == qid,
                    QuestionTag.skill == 2,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# Ticket 6.9a-fix v4 — client-side topic filtering (no backend cascade)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_add_tag_form_embeds_topics_by_cc_json(client, session):
    """Form embeds a JSON map of topics keyed by CC code; CC select uses onchange JS."""
    import json as _json
    import re as _re

    cc1 = await _seed_section_cc(session)
    cc2 = await _seed_second_cc(session)
    t1 = await _seed_topic(session, cc=cc1, name="TopicAlpha")
    t2 = await _seed_topic(session, cc=cc2, name="TopicBeta")
    q = await _seed_question(session, qid="AT-JSON-1")
    await session.commit()

    r = await client.get(f"/questions/{q.id}/add-tag-form")
    assert r.status_code == 200
    body = r.text

    # CC select drives topic options via plain JS onchange, not HTMX.
    assert f'onchange="addTagBuildTopics({q.id}, this.value)"' in body
    # Confirm the HTMX cascade attrs are gone (no round trip on CC change).
    assert "/add-tag-topics" not in body
    assert 'hx-trigger="change"' not in body

    # JSON payload embedded in a <script type="application/json"> block.
    match = _re.search(
        rf'<script id="addtag-topics-data-{q.id}" type="application/json">(.*?)</script>',
        body,
        flags=_re.DOTALL,
    )
    assert match, "embedded topics JSON not found"
    data = _json.loads(match.group(1))

    # Both CCs are keys; each holds its own topics only.
    assert cc1.code in data and cc2.code in data
    names1 = [e["name"] for e in data[cc1.code]]
    names2 = [e["name"] for e in data[cc2.code]]
    assert "TopicAlpha" in names1 and "TopicBeta" not in names1
    assert "TopicBeta" in names2 and "TopicAlpha" not in names2
    # Each entry has the shape consumed by addTagBuildTopics.
    for e in data[cc1.code] + data[cc2.code]:
        assert set(e.keys()) == {"id", "name", "already_tagged"}
    # Ids are real ints (round-trip through JSON).
    assert {t1.id, t2.id}.issubset({e["id"] for e in data[cc1.code] + data[cc2.code]})


# --------------------------------------------------------------------------- #
# Ticket 6.9c — Per-attempt notes + flag-for-review
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_question_detail_renders_notes_section(client, session):
    """GET /questions/{id} includes attempt-notes-section and existing note text."""
    q = await _seed_question(session, qid="N-RENDER-1")
    a = await _seed_attempt(session, question=q, selected="A", is_correct=True)
    note = AttemptNote(
        attempt_id=a.id,
        note_text="important mechanism note",
        flag_for_review=False,
        source="user",
    )
    session.add(note)
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text
    assert "attempt-notes-section" in body
    assert "important mechanism note" in body


@pytest.mark.asyncio
async def test_question_detail_no_attempt_shows_no_notes_form(client, session):
    """GET /questions/{id} without attempt shows no-attempt message; no hx-post notes form."""
    q = await _seed_question(session, qid="N-NOATT-1")
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    body = r.text
    assert "No attempts recorded" in body
    assert f'hx-post="/questions/{q.id}/attempts/' not in body


@pytest.mark.asyncio
async def test_add_note_htmx_creates_note_and_returns_section(client, session):
    """POST /questions/{qid}/attempts/{aid}/notes creates note and returns refreshed section."""
    q = await _seed_question(session, qid="N-ADD-1")
    a = await _seed_attempt(session, question=q, selected="A", is_correct=True)
    await session.commit()
    qid = q.id
    aid = a.id

    r = await client.post(
        f"/questions/{qid}/attempts/{aid}/notes",
        data={"note_text": "focus on mechanism"},
    )
    assert r.status_code == 200
    body = r.text
    assert "attempt-notes-section" in body
    assert "focus on mechanism" in body

    session.expire_all()
    row = (
        await session.execute(select(AttemptNote).where(AttemptNote.attempt_id == aid))
    ).scalar_one_or_none()
    assert row is not None
    assert row.note_text == "focus on mechanism"


@pytest.mark.asyncio
async def test_add_note_htmx_with_flag(client, session):
    """POST with flag_for_review=true shows Flagged badge and sets DB field."""
    q = await _seed_question(session, qid="N-FLAG-1")
    a = await _seed_attempt(session, question=q, selected="A", is_correct=True)
    await session.commit()
    qid = q.id
    aid = a.id

    r = await client.post(
        f"/questions/{qid}/attempts/{aid}/notes",
        data={"note_text": "flagged item", "flag_for_review": "true"},
    )
    assert r.status_code == 200
    body = r.text
    assert "Flagged for review" in body

    session.expire_all()
    row = (
        await session.execute(select(AttemptNote).where(AttemptNote.attempt_id == aid))
    ).scalar_one_or_none()
    assert row is not None
    assert row.flag_for_review is True


@pytest.mark.asyncio
async def test_delete_note_htmx_removes_note(client, session):
    """DELETE /questions/{qid}/attempts/{aid}/notes/{nid} removes note; returns fresh section."""
    q = await _seed_question(session, qid="N-DEL-1")
    a = await _seed_attempt(session, question=q, selected="A", is_correct=True)
    note = AttemptNote(
        attempt_id=a.id,
        note_text="to be deleted",
        flag_for_review=False,
        source="user",
    )
    session.add(note)
    await session.commit()
    qid = q.id
    aid = a.id
    nid = note.id

    r = await client.delete(f"/questions/{qid}/attempts/{aid}/notes/{nid}")
    assert r.status_code == 200
    body = r.text
    assert "attempt-notes-section" in body

    session.expire_all()
    row = (
        await session.execute(select(AttemptNote).where(AttemptNote.id == nid))
    ).scalar_one_or_none()
    assert row is None


@pytest.mark.asyncio
async def test_add_note_htmx_blank_text_returns_422(client, session):
    """POST with blank note_text returns 422 and no DB row is created."""
    q = await _seed_question(session, qid="N-BLANK-1")
    a = await _seed_attempt(session, question=q, selected="A", is_correct=True)
    await session.commit()
    qid = q.id
    aid = a.id

    r = await client.post(
        f"/questions/{qid}/attempts/{aid}/notes",
        data={"note_text": "   "},
    )
    assert r.status_code == 422

    session.expire_all()
    rows = (
        (await session.execute(select(AttemptNote).where(AttemptNote.attempt_id == aid)))
        .scalars()
        .all()
    )
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_get_question_detail_includes_notes(client, session):
    """get_question_detail returns notes list newest-first for the latest attempt."""
    from datetime import datetime, timezone

    from app.web.dashboard.services.drilldown import get_question_detail as _get_detail

    q = await _seed_question(session, qid="N-DETAIL-1")
    a = await _seed_attempt(session, question=q, selected="A", is_correct=True)
    note1 = AttemptNote(
        attempt_id=a.id,
        note_text="first note",
        flag_for_review=False,
        source="user",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    note2 = AttemptNote(
        attempt_id=a.id,
        note_text="second note",
        flag_for_review=True,
        source="user",
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    session.add(note1)
    session.add(note2)
    await session.commit()

    detail = await _get_detail(session, q.id)
    assert detail is not None
    assert len(detail.notes) == 2
    assert detail.notes[0].note_text == "second note"
    assert detail.notes[1].note_text == "first note"
