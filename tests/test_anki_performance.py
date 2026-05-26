"""Tests for SPEC §T39 — GET /api/v1/anki/performance.

Covers:
- §V37 — data exposure only; raw state + retention windows.
- Mutex on cc_code vs topic_id (422 on both / neither).
- window_days param: omit → default 3 windows; supplied → single window.
- Auth required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic

_AUTH = {"X-Coach-Token": "change_me_before_use"}

_CARD_BASE = 950_000
_REVIEW_BASE = 1_950_000_000_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _make_topic(session: AsyncSession, cc: ContentCategory, *, name: str) -> Topic:
    t = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=name,
        disciplines=[],
        depth=0,
        position=950,
    )
    session.add(t)
    await session.flush()
    return t


async def _make_card(
    session: AsyncSession,
    *,
    anki_card_id: int,
    queue: int | None = None,
    interval_days: int | None = None,
) -> AnkiCard:
    # §V75: seed the note (note_id == anki_card_id) before the FK; tags attach
    # to the note, the card carries SRS state + the note_id link.
    session.add(AnkiNote(note_id=anki_card_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="MileDown",
        note_id=anki_card_id,
        queue=queue,
        interval_days=interval_days,
    )
    session.add(card)
    await session.flush()
    return card


def _tag(
    *,
    note_id: int,
    parsed_kind: str,
    topic_id: int | None = None,
    cc_id: int | None = None,
) -> AnkiNoteTag:
    return AnkiNoteTag(
        note_id=note_id,
        tag_raw=f"t::perf::{note_id}",
        topic_id=topic_id,
        content_category_id=cc_id,
        parsed_kind=parsed_kind,
        source="regex",
    )


async def _add_review(
    session: AsyncSession,
    *,
    card_pk: int,
    review_id: int,
    ease: int,
    type_: str = "review",
    age_days: float = 1.0,
) -> None:
    session.add(
        AnkiCardReview(
            review_id=review_id,
            card_id=card_pk,
            reviewed_at=_now() - timedelta(days=age_days),
            ease=ease,
            type=type_,
        )
    )
    await session.flush()


# --- auth + mutex ---


@pytest.mark.asyncio
async def test_requires_auth(client: AsyncClient) -> None:
    r = await client.get("/api/v1/anki/performance", params={"cc_code": "4C"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_mutex_both_params_422(client: AsyncClient) -> None:
    r = await client.get(
        "/api/v1/anki/performance",
        params={"cc_code": "4C", "topic_id": 1},
        headers=_AUTH,
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_mutex_neither_param_422(client: AsyncClient) -> None:
    r = await client.get("/api/v1/anki/performance", headers=_AUTH)
    assert r.status_code == 422


# --- happy path: cc scope ---


@pytest.mark.asyncio
async def test_cc_scope_returns_state_and_default_windows(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    cc = await _first_cc(db_session)
    topic = await _make_topic(db_session, cc, name="perf cc happy")
    # 1 mature (q=2, ivl=30) + 1 suspended (q=-1).
    c_mature = await _make_card(db_session, anki_card_id=_CARD_BASE + 1, queue=2, interval_days=30)
    c_susp = await _make_card(db_session, anki_card_id=_CARD_BASE + 2, queue=-1)
    db_session.add_all(
        [
            _tag(note_id=c_mature.note_id, parsed_kind="aamc_topic", topic_id=topic.id),
            _tag(note_id=c_susp.note_id, parsed_kind="aamc_topic", topic_id=topic.id),
        ]
    )
    await db_session.flush()
    # One pass review in the 7d window.
    await _add_review(
        db_session,
        card_pk=c_mature.id,
        review_id=_REVIEW_BASE + 1,
        ease=3,
        age_days=1,
    )

    r = await client.get(
        "/api/v1/anki/performance",
        params={"cc_code": cc.code},
        headers=_AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scope"] == f"cc:{cc.code}"
    s = body["state"]
    assert s["scope"] == f"cc:{cc.code}"
    assert s["total_cards"] == 2
    assert s["assigned"] == 1
    assert s["suspended"] == 1
    assert s["mature"] == 1
    assert s["unlock_pct"] == pytest.approx(0.5)

    ret = body["retention"]
    assert ret["scope"] == f"cc:{cc.code}"
    windows = {w["window_days"]: w for w in ret["windows"]}
    assert set(windows.keys()) == {7, 30, 0}
    assert windows[7]["pass_count"] == 1
    assert windows[7]["fail_count"] == 0
    assert windows[7]["total"] == 1
    assert windows[7]["retention"] == pytest.approx(1.0)


# --- happy path: topic scope ---


@pytest.mark.asyncio
async def test_topic_scope_returns_state_and_windows(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    cc = await _first_cc(db_session)
    topic = await _make_topic(db_session, cc, name="perf topic happy")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 10, queue=2, interval_days=14)
    db_session.add(_tag(note_id=card.note_id, parsed_kind="aamc_topic", topic_id=topic.id))
    await db_session.flush()
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 10, ease=1, age_days=2)

    r = await client.get(
        "/api/v1/anki/performance",
        params={"topic_id": topic.id},
        headers=_AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scope"] == f"topic:{topic.id}"
    assert body["state"]["total_cards"] == 1
    assert body["state"]["young"] == 1
    ret = body["retention"]
    w7 = next(w for w in ret["windows"] if w["window_days"] == 7)
    assert w7["fail_count"] == 1
    assert w7["pass_count"] == 0
    assert w7["retention"] == pytest.approx(0.0)


# --- window_days param ---


@pytest.mark.asyncio
async def test_window_days_single_window(client: AsyncClient, db_session: AsyncSession) -> None:
    """When window_days supplied → exactly one entry returned."""
    cc = await _first_cc(db_session)
    topic = await _make_topic(db_session, cc, name="perf single window")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 20, queue=2, interval_days=14)
    db_session.add(_tag(note_id=card.note_id, parsed_kind="aamc_topic", topic_id=topic.id))
    await db_session.flush()
    # Two pass reviews — one within 7d, one within 30d but outside 7d.
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 20, ease=3, age_days=1)
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 21, ease=3, age_days=20)

    r = await client.get(
        "/api/v1/anki/performance",
        params={"topic_id": topic.id, "window_days": 30},
        headers=_AUTH,
    )
    assert r.status_code == 200
    windows = r.json()["retention"]["windows"]
    assert len(windows) == 1
    assert windows[0]["window_days"] == 30
    assert windows[0]["pass_count"] == 2


@pytest.mark.asyncio
async def test_window_days_zero_all_time(client: AsyncClient, db_session: AsyncSession) -> None:
    """window_days=0 → all-time (no date filter)."""
    cc = await _first_cc(db_session)
    topic = await _make_topic(db_session, cc, name="perf all time")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 30, queue=2, interval_days=21)
    db_session.add(_tag(note_id=card.note_id, parsed_kind="aamc_topic", topic_id=topic.id))
    await db_session.flush()
    # 60 days ago — would drop out of 7d/30d windows but in all-time.
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 30, ease=3, age_days=60)

    r = await client.get(
        "/api/v1/anki/performance",
        params={"topic_id": topic.id, "window_days": 0},
        headers=_AUTH,
    )
    assert r.status_code == 200
    windows = r.json()["retention"]["windows"]
    assert len(windows) == 1
    assert windows[0]["window_days"] == 0
    assert windows[0]["pass_count"] == 1


# --- empty scope ---


@pytest.mark.asyncio
async def test_empty_scope_returns_zeros_and_null_unlock_pct(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Topic with no cards/reviews → all-zero counts + unlock_pct=null."""
    cc = await _first_cc(db_session)
    topic = await _make_topic(db_session, cc, name="perf empty")
    r = await client.get(
        "/api/v1/anki/performance",
        params={"topic_id": topic.id},
        headers=_AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["total_cards"] == 0
    assert body["state"]["assigned"] == 0
    assert body["state"]["unlock_pct"] is None
    for w in body["retention"]["windows"]:
        assert w["pass_count"] == 0
        assert w["fail_count"] == 0
        assert w["total"] == 0
        assert w["retention"] is None
