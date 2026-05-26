"""End-to-end P11 integration: sync HTTP route → DB writes → read HTTP route.

Per-task tests (T1–T6) each cover one layer in isolation. This test
proves the layers compose correctly: posting to /api/v1/anki/sync with a
MockTransport-backed AnkiConnect yields rows that the read endpoints
return when queried afterwards.

§V16 spirit: no real network — AnkiConnect is fully stubbed via
httpx.MockTransport.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.anki import _anki_client
from app.main import app
from app.models.outline import ContentCategory, Topic
from app.services.anki.client import AnkiConnectClient


_AUTH = {"X-Coach-Token": "change_me_before_use"}


def _ok(result: Any) -> bytes:
    return json.dumps({"result": result, "error": None}).encode()


async def _ensure_topic(session: AsyncSession, name: str) -> Topic:
    cc = (await session.execute(select(ContentCategory).limit(1))).scalar_one()
    topic = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=name,
        disciplines=[],
        depth=0,
        position=999,
    )
    session.add(topic)
    await session.flush()
    return topic


@pytest.mark.asyncio
async def test_sync_then_read_round_trip(client: AsyncClient, db_session: AsyncSession) -> None:
    """Full P11 chain: POST /anki/sync stubs the AnkiConnect actions, lands
    rows in `anki_cards`/`anki_notes`/`anki_note_tags` (§V75), and GET
    /anki/cards/by-qid/{qid} surfaces the synced uworld qid tag."""
    topic = await _ensure_topic(db_session, name="T7 integration topic")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        action = body.get("action")
        if action == "findCards":
            return httpx.Response(200, content=_ok([8001]))
        if action == "cardsInfo":
            return httpx.Response(
                200,
                content=_ok(
                    [
                        {
                            "cardId": 8001,
                            "note": 9001,
                            "modelName": "MileDown",
                            "fields": {"Front": {"value": "f", "order": 0}},
                            "queue": 2,
                            "interval": 21,
                            "factor": 2500,
                            "lapses": 0,
                            "due": 1,
                        }
                    ]
                ),
            )
        if action == "notesInfo":
            return httpx.Response(
                200,
                content=_ok(
                    [
                        {
                            "noteId": 9001,
                            "tags": ["#AK_MCAT_v2::#UWorld::777777", "LegacyTag::unparsed"],
                        }
                    ]
                ),
            )
        if action == "cardReviews":
            return httpx.Response(200, content=_ok([]))
        return httpx.Response(200, content=_ok(None))

    def _factory() -> AnkiConnectClient:
        return AnkiConnectClient("http://localhost:8765", transport=httpx.MockTransport(handler))

    app.dependency_overrides[_anki_client] = _factory
    try:
        # Sync writes through the route's session (the per-test connection +
        # outer transaction from conftest.client).
        sync_resp = await client.post("/api/v1/anki/sync", headers=_AUTH)
        assert sync_resp.status_code == 200
        summary = sync_resp.json()
        assert summary == {
            "synced_cards": 1,
            "linked_qids": 1,
            "reviews_synced": 0,
            "error": None,
        }

        # Read back via the dedicated read endpoint — same connection sees the
        # uncommitted-but-flushed savepoint state from the sync.
        by_qid = await client.get("/api/v1/anki/cards/by-qid/777777", headers=_AUTH)
        assert by_qid.status_code == 200
        cards = by_qid.json()
        assert len(cards) == 1
        assert cards[0]["anki_card_id"] == 8001
        kinds = {t["parsed_kind"] for t in cards[0]["tags"]}
        assert "uworld_qid" in kinds
        assert "unparsed" in kinds
    finally:
        app.dependency_overrides.pop(_anki_client, None)
        _ = topic  # silence unused-name lint; the topic seed is part of fixture scaffolding


@pytest.mark.asyncio
async def test_sync_anki_down_keeps_db_clean(client: AsyncClient, db_session: AsyncSession) -> None:
    """§V4 end-to-end: when AnkiConnect is unreachable the route returns the
    error envelope and writes zero rows."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    from sqlalchemy import func

    from app.models.anki import AnkiCard

    def _factory() -> AnkiConnectClient:
        return AnkiConnectClient("http://localhost:8765", transport=httpx.MockTransport(handler))

    app.dependency_overrides[_anki_client] = _factory
    try:
        before = (await db_session.execute(select(func.count()).select_from(AnkiCard))).scalar_one()

        r = await client.post("/api/v1/anki/sync", headers=_AUTH)
        assert r.status_code == 200
        assert r.json()["error"] == "anki_not_running"

        after = (await db_session.execute(select(func.count()).select_from(AnkiCard))).scalar_one()
        assert after == before
    finally:
        app.dependency_overrides.pop(_anki_client, None)
