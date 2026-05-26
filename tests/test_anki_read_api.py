"""SPEC §T5 read-endpoint tests.

Each test seeds a small set of AnkiCard + AnkiNote + AnkiNoteTag rows (§V75)
via the `db_session` fixture (which the unified `client` fixture binds to the
per-test connection) and hits the live FastAPI app via httpx.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic


_AUTH = {"X-Coach-Token": "change_me_before_use"}


async def _seed_topic(session: AsyncSession, name: str = "T5 topic") -> Topic:
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


async def _seed_card_with_tags(
    session: AsyncSession,
    *,
    anki_card_id: int,
    deck_name: str = "MileDown",
    due_date: date | None = None,
    tags: list[tuple[str, str, int | None, str | None]] = (),
) -> AnkiCard:
    """Seed one card + its tags.

    `tags` is a list of (tag_raw, parsed_kind, topic_id, question_qid) tuples.
    """
    # §V75: a card's tags live on its note. Seed note (note_id == anki_card_id)
    # before the FK, link the card, attach tags to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name=deck_name))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name=deck_name,
        note_id=anki_card_id,
        due_date=due_date,
        interval_days=14,
        ease=2500,
        lapses=0,
        queue=2,
    )
    session.add(card)
    await session.flush()
    for tag_raw, parsed_kind, topic_id, qid in tags:
        session.add(
            AnkiNoteTag(
                note_id=anki_card_id,
                tag_raw=tag_raw,
                parsed_kind=parsed_kind,
                topic_id=topic_id,
                question_qid=qid,
            )
        )
    await session.flush()
    return card


# --- /cards?topic_id ---


@pytest.mark.asyncio
async def test_cards_by_topic_requires_auth(client: AsyncClient) -> None:
    r = await client.get("/api/v1/anki/cards", params={"topic_id": 1})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_cards_by_topic_returns_matching_cards(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    topic = await _seed_topic(db_session, name="T5 cards-by-topic")
    other_topic = await _seed_topic(db_session, name="T5 other-topic")
    await _seed_card_with_tags(
        db_session,
        anki_card_id=7001,
        tags=[("aamc::CP::4A::Topic", "aamc_topic", topic.id, None)],
    )
    await _seed_card_with_tags(
        db_session,
        anki_card_id=7002,
        tags=[("aamc::CP::4A::Other", "aamc_topic", other_topic.id, None)],
    )

    r = await client.get("/api/v1/anki/cards", params={"topic_id": topic.id}, headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["anki_card_id"] == 7001
    assert body[0]["tags"][0]["topic_id"] == topic.id


@pytest.mark.asyncio
async def test_cards_by_topic_limit_clamped(client: AsyncClient, db_session: AsyncSession) -> None:
    topic = await _seed_topic(db_session, name="T5 limit-clamp")
    for i in range(5):
        await _seed_card_with_tags(
            db_session,
            anki_card_id=7100 + i,
            tags=[("aamc::CP::4A::T", "aamc_topic", topic.id, None)],
        )
    r = await client.get(
        "/api/v1/anki/cards",
        params={"topic_id": topic.id, "limit": 3},
        headers=_AUTH,
    )
    assert r.status_code == 200
    assert len(r.json()) == 3


# --- /review-queue ---


@pytest.mark.asyncio
async def test_review_queue_orders_by_due_date_asc(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    today = date.today()
    await _seed_card_with_tags(db_session, anki_card_id=7201, due_date=today + timedelta(days=5))
    await _seed_card_with_tags(db_session, anki_card_id=7202, due_date=today - timedelta(days=2))
    await _seed_card_with_tags(db_session, anki_card_id=7203, due_date=today + timedelta(days=1))
    r = await client.get("/api/v1/anki/review-queue", headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    ids = [c["anki_card_id"] for c in body if c["anki_card_id"] in (7201, 7202, 7203)]
    # 7202 (overdue) → 7203 (+1 day) → 7201 (+5 days)
    assert ids == [7202, 7203, 7201]


@pytest.mark.asyncio
async def test_review_queue_excludes_null_due_date(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_card_with_tags(db_session, anki_card_id=7301, due_date=None)
    await _seed_card_with_tags(db_session, anki_card_id=7302, due_date=date.today())
    r = await client.get("/api/v1/anki/review-queue", headers=_AUTH)
    assert r.status_code == 200
    ids = {c["anki_card_id"] for c in r.json()}
    assert 7301 not in ids
    assert 7302 in ids


# --- /cards/by-qid ---


@pytest.mark.asyncio
async def test_cards_by_qid_matches_uworld_qid_tag(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    card = await _seed_card_with_tags(
        db_session,
        anki_card_id=7401,
        tags=[("uworld::qid::402391", "uworld_qid", None, "402391")],
    )
    r = await client.get("/api/v1/anki/cards/by-qid/402391", headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["id"] == card.id
    assert body[0]["tags"][0]["question_qid"] == "402391"


@pytest.mark.asyncio
async def test_cards_by_qid_unseen_returns_empty(client: AsyncClient) -> None:
    r = await client.get("/api/v1/anki/cards/by-qid/000000000", headers=_AUTH)
    assert r.status_code == 200
    assert r.json() == []
