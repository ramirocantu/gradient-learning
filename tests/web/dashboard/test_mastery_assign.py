"""POST /mastery/assign — the Anki state & retention widget's Assign button.

In-process (V18) create-assignment from the CC + topic mastery pages. Covers
the cc/topic scopes, the empty-scope guard, scope validation, the max_cards
slice, and the open-redirect guard on the user-controllable redirect_to.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import app.web.dashboard.routes.topics as topics_routes
from app.models.anki import AnkiAssignment, AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic

pytestmark = pytest.mark.asyncio

_CARD_BASE = 910_000


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _make_topic(session: AsyncSession, cc: ContentCategory) -> Topic:
    topic = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name="assign-route topic",
        disciplines=[],
        depth=0,
        position=900,
    )
    session.add(topic)
    await session.flush()
    return topic


async def _suspended_card_under_topic(
    session: AsyncSession, *, anki_card_id: int, topic_id: int
) -> None:
    # §V75: scope membership is note-level — seed note + aamc_topic note-tag,
    # link the card via note_id (note_id == anki_card_id for 1:1).
    session.add(AnkiNote(note_id=anki_card_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="MileDown",
        note_id=anki_card_id,
        queue=-1,
        interval_days=None,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=f"t{anki_card_id}",
            topic_id=topic_id,
            parsed_kind="aamc_topic",
            source="regex",
        )
    )
    await session.commit()


async def test_cc_scope_creates_assignment(client, session) -> None:
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)
    await _suspended_card_under_topic(session, anki_card_id=_CARD_BASE + 1, topic_id=topic.id)

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "cc",
            "scope_value": cc.code,
            "redirect_to": f"/mastery/{cc.code}",
            "unlock_date": "2026-06-01",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/mastery/{cc.code}?assigned=1"

    row = (
        await session.execute(
            select(AnkiAssignment).where(
                AnkiAssignment.scope_kind == "cc", AnkiAssignment.scope_value == cc.code
            )
        )
    ).scalar_one()
    assert row.status == "pending"
    assert row.card_ids == [_CARD_BASE + 1]


async def test_topic_scope_creates_assignment(client, session) -> None:
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)
    await _suspended_card_under_topic(session, anki_card_id=_CARD_BASE + 2, topic_id=topic.id)

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "topic",
            "scope_value": str(topic.id),
            "redirect_to": f"/mastery/{cc.code}/topics/{topic.id}",
            "unlock_date": "2026-06-01",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/mastery/{cc.code}/topics/{topic.id}?assigned=1"

    row = (
        await session.execute(
            select(AnkiAssignment).where(
                AnkiAssignment.scope_kind == "topic",
                AnkiAssignment.scope_value == str(topic.id),
            )
        )
    ).scalar_one()
    assert row.card_ids == [_CARD_BASE + 2]


async def test_max_cards_slices_snapshot(client, session) -> None:
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)
    await _suspended_card_under_topic(session, anki_card_id=_CARD_BASE + 3, topic_id=topic.id)
    await _suspended_card_under_topic(session, anki_card_id=_CARD_BASE + 4, topic_id=topic.id)

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "topic",
            "scope_value": str(topic.id),
            "redirect_to": f"/mastery/{cc.code}/topics/{topic.id}",
            "max_cards": "1",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?assigned=1")

    row = (
        await session.execute(
            select(AnkiAssignment).where(AnkiAssignment.scope_value == str(topic.id))
        )
    ).scalar_one()
    assert len(row.card_ids) == 1


async def test_empty_scope_creates_no_assignment(client, session) -> None:
    """A topic with no suspended cards → assign_none flag, no row."""
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)  # no cards tagged under it

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "topic",
            "scope_value": str(topic.id),
            "redirect_to": f"/mastery/{cc.code}/topics/{topic.id}",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/mastery/{cc.code}/topics/{topic.id}?assign_none=1"

    rows = (
        (
            await session.execute(
                select(AnkiAssignment).where(AnkiAssignment.scope_value == str(topic.id))
            )
        )
        .scalars()
        .all()
    )
    assert rows == []


async def test_invalid_scope_kind_errors(client, session) -> None:
    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "bogus",
            "scope_value": "x",
            "redirect_to": "/mastery/4A",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/mastery/4A?assign_error=scope"


async def test_open_redirect_falls_back_to_mastery(client, session) -> None:
    """External / schema-relative redirect_to is rejected; the route bounces
    back to /mastery, never the attacker host."""
    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "bogus",  # short-circuits before any DB work
            "scope_value": "x",
            "redirect_to": "https://evil.example/phish",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/mastery?assign_error=scope"


async def test_successful_assign_triggers_unlock_job(client, session, monkeypatch) -> None:
    """A committed assignment nudges the T63 unlock job so a due-now scope
    unsuspends without waiting for the next interval tick."""
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)
    await _suspended_card_under_topic(session, anki_card_id=_CARD_BASE + 5, topic_id=topic.id)

    trigger = AsyncMock()
    monkeypatch.setattr(topics_routes, "trigger_job_logic", trigger)

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "topic",
            "scope_value": str(topic.id),
            "redirect_to": f"/mastery/{cc.code}/topics/{topic.id}",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?assigned=1")
    trigger.assert_awaited_once_with("run_anki_assignment_unlock")


async def test_empty_scope_does_not_trigger_unlock_job(client, session, monkeypatch) -> None:
    """No row created → no unlock nudge."""
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)  # no suspended cards under it

    trigger = AsyncMock()
    monkeypatch.setattr(topics_routes, "trigger_job_logic", trigger)

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "topic",
            "scope_value": str(topic.id),
            "redirect_to": f"/mastery/{cc.code}/topics/{topic.id}",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?assign_none=1")
    trigger.assert_not_awaited()


async def test_assign_succeeds_when_unlock_trigger_unavailable(
    client, session, monkeypatch
) -> None:
    """Trigger is best-effort: a 503 (scheduler off) / 409 (already in-flight)
    is swallowed — the assignment stays committed and the redirect still
    reports success."""
    cc = await _first_cc(session)
    topic = await _make_topic(session, cc)
    await _suspended_card_under_topic(session, anki_card_id=_CARD_BASE + 6, topic_id=topic.id)

    trigger = AsyncMock(side_effect=HTTPException(503, detail="scheduler not running"))
    monkeypatch.setattr(topics_routes, "trigger_job_logic", trigger)

    resp = await client.post(
        "/mastery/assign",
        data={
            "scope_kind": "topic",
            "scope_value": str(topic.id),
            "redirect_to": f"/mastery/{cc.code}/topics/{topic.id}",
        },
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?assigned=1")
    trigger.assert_awaited_once_with("run_anki_assignment_unlock")

    row = (
        await session.execute(
            select(AnkiAssignment).where(AnkiAssignment.scope_value == str(topic.id))
        )
    ).scalar_one()
    assert row.status == "pending"
    assert row.card_ids == [_CARD_BASE + 6]


async def test_assign_button_renders_on_cc_page(client, session) -> None:
    """The Assign form sits inside the Anki state & retention widget on a
    real (non-CARS) CC mastery page and is scoped to that CC."""
    resp = await client.get("/mastery/4A")
    assert resp.status_code == 200
    html = resp.text
    assert 'data-assign-form="1"' in html
    assert 'name="scope_value" value="4A"' in html
    assert "Anki state &amp; retention" in html
