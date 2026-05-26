"""Tests for SPEC T67 assignment API (V51, V52).

Covers POST + GET + PATCH against the unified app client. Service-layer
exceptions (AssignmentError, AssignmentNotFoundError,
AssignmentTerminalError) must map to 422/404/409 respectively, and
X-Coach-Token enforcement on all routes per V18.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic


_AUTH = {"X-Coach-Token": "change_me_before_use"}


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _seed_topic(session: AsyncSession, cc: ContentCategory) -> Topic:
    topic = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name="T67 assign topic",
        disciplines=[],
        depth=0,
        position=970,
    )
    session.add(topic)
    await session.flush()
    return topic


async def _seed_suspended_card(
    session: AsyncSession, *, anki_card_id: int, topic_id: int
) -> AnkiCard:
    # §V75: scope membership is note-level — seed the note + its aamc_topic
    # tag, link the card via note_id (note_id == anki_card_id for 1:1).
    session.add(AnkiNote(note_id=anki_card_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="MileDown",
        note_id=anki_card_id,
        queue=-1,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=f"t::{anki_card_id}",
            topic_id=topic_id,
            parsed_kind="aamc_topic",
            source="regex",
        )
    )
    await session.flush()
    return card


def _later(days: int = 1) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


# --- auth ---


async def test_assignments_routes_require_coach_token(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    r = await client.post("/api/v1/anki/assignments", json={})
    assert r.status_code == 401
    r = await client.get("/api/v1/anki/assignments")
    assert r.status_code == 401
    r = await client.patch("/api/v1/anki/assignments/1", json={"status": "skipped"})
    assert r.status_code == 401


# --- POST ---


async def test_post_assignment_snapshots_and_returns(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    cc = await _first_cc(db_session)
    topic = await _seed_topic(db_session, cc)
    for n in (1, 2):
        await _seed_suspended_card(db_session, anki_card_id=1_750_000_000 + n, topic_id=topic.id)

    payload = {
        "scope_kind": "topic",
        "scope_value": str(topic.id),
        "scheduled_unlock_at": _later(2),
        "max_cards": 5,
    }
    r = await client.post("/api/v1/anki/assignments", json=payload, headers=_AUTH)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "pending"
    assert body["scope_kind"] == "topic"
    assert body["scope_value"] == str(topic.id)
    assert set(body["card_ids"]) == {1_750_000_001, 1_750_000_002}
    assert body["priority"] == "most_specific_first"


async def test_post_assignment_invalid_scope_kind_422(
    client: AsyncClient,
) -> None:
    # Pydantic catches this at the schema layer.
    r = await client.post(
        "/api/v1/anki/assignments",
        headers=_AUTH,
        json={
            "scope_kind": "section",
            "scope_value": "CP",
            "scheduled_unlock_at": _later(),
        },
    )
    assert r.status_code == 422


async def test_post_assignment_topic_non_int_scope_422(
    client: AsyncClient,
) -> None:
    """scope_value='not-a-number' under topic scope passes schema but
    AssignmentError → 422 at the service boundary."""
    r = await client.post(
        "/api/v1/anki/assignments",
        headers=_AUTH,
        json={
            "scope_kind": "topic",
            "scope_value": "not-a-number",
            "scheduled_unlock_at": _later(),
        },
    )
    assert r.status_code == 422


# --- GET ---


async def test_list_assignments_filters(client: AsyncClient, db_session: AsyncSession) -> None:
    now = datetime.now(timezone.utc)
    db_session.add_all(
        [
            AnkiAssignment(
                scope_kind="cc",
                scope_value="4C",
                scheduled_unlock_at=now + timedelta(days=1),
                card_ids=[1],
                status="pending",
            ),
            AnkiAssignment(
                scope_kind="cc",
                scope_value="4C",
                scheduled_unlock_at=now + timedelta(days=10),
                card_ids=[2],
                status="pending",
            ),
            AnkiAssignment(
                scope_kind="cc",
                scope_value="4C",
                scheduled_unlock_at=now + timedelta(days=2),
                card_ids=[3],
                status="completed",
            ),
        ]
    )
    await db_session.flush()

    # No filter → all 3.
    r = await client.get("/api/v1/anki/assignments", headers=_AUTH)
    assert r.status_code == 200
    assert len(r.json()) == 3

    # status filter.
    r = await client.get("/api/v1/anki/assignments?status=pending", headers=_AUTH)
    assert r.status_code == 200
    statuses = {row["status"] for row in r.json()}
    assert statuses == {"pending"}

    # window filter — only the 1d + 2d rows fall in [now, now+5d].
    r = await client.get("/api/v1/anki/assignments?window_days=5", headers=_AUTH)
    assert r.status_code == 200
    assert len(r.json()) == 2


# --- PATCH ---


async def test_patch_assignment_skip(client: AsyncClient, db_session: AsyncSession) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=datetime.now(timezone.utc) + timedelta(days=1),
        card_ids=[1],
        status="pending",
    )
    db_session.add(a)
    await db_session.flush()

    r = await client.patch(
        f"/api/v1/anki/assignments/{a.id}",
        headers=_AUTH,
        json={"status": "skipped"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "skipped"


async def test_patch_assignment_complete(client: AsyncClient, db_session: AsyncSession) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=datetime.now(timezone.utc) + timedelta(days=1),
        card_ids=[1],
        status="unlocked",
        actual_unlock_at=datetime.now(timezone.utc),
    )
    db_session.add(a)
    await db_session.flush()

    r = await client.patch(
        f"/api/v1/anki/assignments/{a.id}",
        headers=_AUTH,
        json={"status": "completed"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


async def test_patch_assignment_not_found_404(client: AsyncClient) -> None:
    r = await client.patch(
        "/api/v1/anki/assignments/999999",
        headers=_AUTH,
        json={"status": "skipped"},
    )
    assert r.status_code == 404


@pytest.mark.parametrize("terminal", ["completed", "skipped", "failed"])
async def test_patch_assignment_terminal_409(
    client: AsyncClient, db_session: AsyncSession, terminal: str
) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=datetime.now(timezone.utc) + timedelta(days=1),
        card_ids=[1],
        status=terminal,
    )
    db_session.add(a)
    await db_session.flush()

    r = await client.patch(
        f"/api/v1/anki/assignments/{a.id}",
        headers=_AUTH,
        json={"status": "skipped"},
    )
    assert r.status_code == 409


async def test_patch_assignment_invalid_status_422(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=datetime.now(timezone.utc) + timedelta(days=1),
        card_ids=[1],
        status="pending",
    )
    db_session.add(a)
    await db_session.flush()

    # PATCH back to pending or unlocked is not exposed via this endpoint.
    r = await client.patch(
        f"/api/v1/anki/assignments/{a.id}",
        headers=_AUTH,
        json={"status": "unlocked"},
    )
    assert r.status_code == 422
