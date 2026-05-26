"""Sync-service tests for SPEC §T3.

AnkiConnect is stubbed with `httpx.MockTransport` per the established
T1 pattern (no respx dep on backend/). Each test boots a fresh
AnkiConnectClient bound to a hand-rolled handler that returns the
shaped action payloads `sync_deck` needs.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Callable

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.services.anki.client import AnkiConnectClient
from app.services.anki.sync import sync_deck
from app.services.categorizer.outline_lookup import OutlineLookup


_URL = "http://localhost:8765"


def _ok(result: Any) -> bytes:
    return json.dumps({"result": result, "error": None}).encode()


def _make_handler(
    *,
    card_ids: list[int],
    cards: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    reviews: list[list[int]] | None = None,
    record_calls: list[dict[str, Any]] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Returns a handler that responds to the AnkiConnect actions
    sync_deck issues. If `record_calls` is provided, every request is
    appended to it so tests can assert call shape (e.g. T36 asserts the
    `startID` param sent to `cardReviews`)."""
    expected = {
        "findCards": card_ids,
        "cardsInfo": cards,
        "notesInfo": notes,
        "cardReviews": reviews if reviews is not None else [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        action = body.get("action")
        if record_calls is not None:
            record_calls.append(body)
        if action not in expected:
            return httpx.Response(
                200,
                content=json.dumps(
                    {"result": None, "error": f"unexpected action {action}"}
                ).encode(),
            )
        return httpx.Response(200, content=_ok(expected[action]))

    return handler


def _card_data(
    *,
    card_id: int,
    note_id: int,
    queue: int = 2,
    interval: int = 21,
    factor: int = 2500,
    lapses: int = 0,
    fields: dict[str, dict[str, Any]] | None = None,
    model_name: str = "MileDown Premed",
) -> dict[str, Any]:
    return {
        "cardId": card_id,
        "note": note_id,
        "modelName": model_name,
        "fields": fields
        or {"Front": {"value": "f", "order": 0}, "Back": {"value": "b", "order": 1}},
        "queue": queue,
        "interval": interval,
        "factor": factor,
        "lapses": lapses,
        "due": 100,
    }


def _note_data(*, note_id: int, tags: list[str]) -> dict[str, Any]:
    return {"noteId": note_id, "tags": tags, "fields": {}}


@pytest.fixture
async def lookup(db_session: AsyncSession) -> OutlineLookup:
    return await OutlineLookup.load(db_session)


@pytest.fixture
def make_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], AnkiConnectClient]:
    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> AnkiConnectClient:
        return AnkiConnectClient(_URL, transport=httpx.MockTransport(handler))

    return factory


# --- V1: idempotent upsert ---


async def test_sync_inserts_new_cards(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    handler = _make_handler(
        card_ids=[1001, 1002],
        cards=[
            _card_data(card_id=1001, note_id=2001, queue=2, interval=14),
            _card_data(card_id=1002, note_id=2002, queue=0),
        ],
        notes=[
            _note_data(note_id=2001, tags=["#AK_MCAT_v2::#UWorld::402391"]),
            _note_data(note_id=2002, tags=["aamc::CP::4A::Translational_Motion"]),
        ],
    )
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    assert summary.synced_cards == 2
    assert summary.linked_qids == 1
    assert summary.error is None
    count = (
        await db_session.execute(
            select(func.count()).select_from(AnkiCard).where(AnkiCard.deck_name == "MileDown")
        )
    ).scalar_one()
    assert count == 2


async def test_sync_is_idempotent_on_unchanged_card(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V1: re-syncing the same Anki state does not duplicate rows and does
    not churn the Anki-mirrored data columns."""
    handler = _make_handler(
        card_ids=[1010],
        cards=[_card_data(card_id=1010, note_id=2010, queue=2, interval=21)],
        notes=[_note_data(note_id=2010, tags=["#AK_MCAT_v2::#UWorld::99999"])],
    )

    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    cards = (
        (
            await db_session.execute(
                select(AnkiCard).where(
                    AnkiCard.deck_name == "MileDown", AnkiCard.anki_card_id == 1010
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(cards) == 1
    assert cards[0].interval_days == 21
    assert cards[0].queue == 2

    tag_count = (
        await db_session.execute(
            select(func.count()).select_from(AnkiNoteTag).where(AnkiNoteTag.note_id == 2010)
        )
    ).scalar_one()
    assert tag_count == 1
    assert summary.synced_cards == 1


async def test_sync_updates_changed_card(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V1 upsert path: when Anki returns new field values the row updates."""
    handler_v1 = _make_handler(
        card_ids=[1020],
        cards=[_card_data(card_id=1020, note_id=2020, queue=2, interval=10, lapses=0)],
        notes=[_note_data(note_id=2020, tags=[])],
    )
    handler_v2 = _make_handler(
        card_ids=[1020],
        cards=[_card_data(card_id=1020, note_id=2020, queue=2, interval=30, lapses=2)],
        notes=[_note_data(note_id=2020, tags=[])],
    )

    async with make_client(handler_v1) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    async with make_client(handler_v2) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    row = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 1020))
    ).scalar_one()
    assert row.interval_days == 30
    assert row.lapses == 2


async def test_sync_replaces_tags_on_change(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V1/§V3: tag set follows Anki — removed tags drop, new ones land."""
    handler_v1 = _make_handler(
        card_ids=[1030],
        cards=[_card_data(card_id=1030, note_id=2030, queue=2, interval=14)],
        notes=[
            _note_data(note_id=2030, tags=["#AK_MCAT_v2::#UWorld::1", "#AK_MCAT_v2::#UWorld::2"])
        ],
    )
    handler_v2 = _make_handler(
        card_ids=[1030],
        cards=[_card_data(card_id=1030, note_id=2030, queue=2, interval=14)],
        notes=[
            _note_data(note_id=2030, tags=["#AK_MCAT_v2::#UWorld::2", "#AK_MCAT_v2::#UWorld::3"])
        ],
    )

    async with make_client(handler_v1) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    async with make_client(handler_v2) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    tag_rows = (
        (await db_session.execute(select(AnkiNoteTag).where(AnkiNoteTag.note_id == 2030)))
        .scalars()
        .all()
    )
    qids = {t.question_qid for t in tag_rows}
    assert qids == {"2", "3"}
    assert summary.linked_qids == 2


# --- V3: tag parsing classes ---


async def test_sync_unparsed_tag_persists(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    handler = _make_handler(
        card_ids=[1040],
        cards=[_card_data(card_id=1040, note_id=2040)],
        notes=[_note_data(note_id=2040, tags=["LegacyTag::Weird"])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    tag = (
        await db_session.execute(
            select(AnkiNoteTag).where(AnkiNoteTag.tag_raw == "LegacyTag::Weird")
        )
    ).scalar_one()
    assert tag.parsed_kind == "unparsed"
    assert tag.topic_id is None
    assert tag.question_qid is None


async def test_sync_aamc_skill_resolved_tag_writes_skill_number(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V3 amended (T34): AnKing Skill tags persist with skill_number set."""
    skill_tag = "#AK_MCAT_v2::#AAMC::Skills::Skill_4-Data_and_Statistics"
    handler = _make_handler(
        card_ids=[1051],
        cards=[_card_data(card_id=1051, note_id=2051)],
        notes=[_note_data(note_id=2051, tags=[skill_tag])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    tag = (
        await db_session.execute(select(AnkiNoteTag).where(AnkiNoteTag.tag_raw == skill_tag))
    ).scalar_one()
    assert tag.parsed_kind == "aamc_skill"
    assert tag.skill_number == 4
    assert tag.content_category_id is None
    assert tag.topic_id is None


async def test_sync_aamc_cc_resolved_tag_links_cc(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V3 amended (T31): AnKing AAMC tags resolve at CC granularity."""
    aamc_tag = (
        "#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::"
        "4E-Atoms_Nuclear_Decay_Electronic_Structure_and_Behavior"
    )
    handler = _make_handler(
        card_ids=[1050],
        cards=[_card_data(card_id=1050, note_id=2050)],
        notes=[_note_data(note_id=2050, tags=[aamc_tag])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    tag = (
        await db_session.execute(select(AnkiNoteTag).where(AnkiNoteTag.tag_raw == aamc_tag))
    ).scalar_one()
    assert tag.parsed_kind == "aamc_cc"
    assert tag.content_category_id is not None
    assert tag.topic_id is None


# --- V4: AnkiConnect unreachable ---


async def test_sync_returns_anki_not_running_on_connect_error(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V4: unreachable AnkiConnect surfaces as the error envelope, never raises."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    assert summary.synced_cards == 0
    assert summary.linked_qids == 0
    assert summary.error == "anki_not_running"


async def test_sync_returns_deck_empty_envelope_on_empty_findcards(
    db_session: AsyncSession, lookup: OutlineLookup, make_client, caplog
) -> None:
    """§V20: empty findCards → loud-fail envelope + WARN log w/ deckNames."""
    import logging

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        action = body.get("action")
        if action == "findCards":
            return httpx.Response(200, content=_ok([]))
        if action == "deckNames":
            return httpx.Response(200, content=_ok(["AnKing MCAT Deck", "Other"]))
        return httpx.Response(200, content=_ok(None))

    with caplog.at_level(logging.WARNING, logger="app.services.anki.sync"):
        async with make_client(handler) as client:
            summary = await sync_deck(
                db_session, client, deck_name="MileDown", outline_lookup=lookup
            )

    assert summary.synced_cards == 0
    assert summary.linked_qids == 0
    assert summary.error == "deck_empty_or_misspelled"
    # WARN log mentions the configured deck and the available list
    log_msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "'MileDown'" in log_msgs
    assert "AnKing MCAT Deck" in log_msgs


async def test_sync_logs_deck_name_on_entry(
    db_session: AsyncSession, lookup: OutlineLookup, make_client, caplog
) -> None:
    """§V20: deck_name logged @ INFO on every sync entry."""
    import logging

    handler = _make_handler(card_ids=[], cards=[], notes=[])
    with caplog.at_level(logging.INFO, logger="app.services.anki.sync"):
        async with make_client(handler) as client:
            await sync_deck(db_session, client, deck_name="ProbeDeck", outline_lookup=lookup)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("starting deck=" in m and "ProbeDeck" in m for m in msgs)


# --- V2: review-state derivation ---


async def test_sync_due_date_review_card(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    handler = _make_handler(
        card_ids=[1060],
        cards=[_card_data(card_id=1060, note_id=2060, queue=2, interval=14)],
        notes=[_note_data(note_id=2060, tags=[])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    row = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 1060))
    ).scalar_one()
    assert row.due_date == date.today() + timedelta(days=14)


async def test_sync_due_date_learning_card(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    handler = _make_handler(
        card_ids=[1061],
        cards=[_card_data(card_id=1061, note_id=2061, queue=1, interval=0)],
        notes=[_note_data(note_id=2061, tags=[])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    row = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 1061))
    ).scalar_one()
    assert row.due_date == date.today()


async def test_sync_due_date_new_card_null(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    handler = _make_handler(
        card_ids=[1062],
        cards=[_card_data(card_id=1062, note_id=2062, queue=0, interval=0)],
        notes=[_note_data(note_id=2062, tags=[])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)
    row = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 1062))
    ).scalar_one()
    assert row.due_date is None


# --- T36 / V26: append-only review log ---


def _review_row(
    *,
    review_id: int,
    card_id: int,
    button: int = 3,
    new_iv: int = 21,
    prev_iv: int = 7,
    duration_ms: int = 5500,
    rtype: int = 1,
) -> list[int]:
    """Shape AnkiConnect's cardReviews returns:
    [reviewTime_ms, cardId, usn, button, newInterval, prevInterval,
     newFactor, reviewDuration_ms, reviewType]."""
    return [review_id, card_id, 0, button, new_iv, prev_iv, 2500, duration_ms, rtype]


async def test_sync_review_first_run_uses_startid_zero(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V26 first-run backfill: startID=0 when no rows in anki_card_reviews."""
    from app.models.anki import AnkiCardReview

    calls: list[dict[str, Any]] = []
    handler = _make_handler(
        card_ids=[2100],
        cards=[_card_data(card_id=2100, note_id=3100, queue=2, interval=21)],
        notes=[_note_data(note_id=3100, tags=[])],
        reviews=[_review_row(review_id=1_700_000_000_000, card_id=2100)],
        record_calls=calls,
    )
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    review_calls = [c for c in calls if c["action"] == "cardReviews"]
    assert len(review_calls) == 1
    assert review_calls[0]["params"]["startID"] == 0
    assert review_calls[0]["params"]["deck"] == "MileDown"
    assert summary.reviews_synced == 1

    rev = (
        await db_session.execute(
            select(AnkiCardReview).where(AnkiCardReview.review_id == 1_700_000_000_000)
        )
    ).scalar_one()
    assert rev.ease == 3
    assert rev.type == "review"
    assert rev.interval_before == 7
    assert rev.interval_after == 21
    assert rev.time_ms == 5500


async def test_sync_review_field_mapping(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V27 reviewType int → enum mapping; ease ∈ 1..4 honored."""
    from app.models.anki import AnkiCardReview

    handler = _make_handler(
        card_ids=[2110],
        cards=[_card_data(card_id=2110, note_id=3110, queue=2)],
        notes=[_note_data(note_id=3110, tags=[])],
        reviews=[
            _review_row(review_id=1_700_000_000_001, card_id=2110, button=1, rtype=0),
            _review_row(review_id=1_700_000_000_002, card_id=2110, button=2, rtype=1),
            _review_row(review_id=1_700_000_000_003, card_id=2110, button=3, rtype=2),
            _review_row(review_id=1_700_000_000_004, card_id=2110, button=4, rtype=3),
        ],
    )
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    assert summary.reviews_synced == 4
    rows = (
        (
            await db_session.execute(
                select(AnkiCardReview)
                .where(AnkiCardReview.review_id.between(1_700_000_000_001, 1_700_000_000_004))
                .order_by(AnkiCardReview.review_id)
            )
        )
        .scalars()
        .all()
    )
    assert [r.ease for r in rows] == [1, 2, 3, 4]
    assert [r.type for r in rows] == ["learn", "review", "relearn", "cram"]


async def test_sync_review_incremental_uses_max_plus_one(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V26: subsequent sync issues startID = MAX(review_id) + 1."""
    handler_v1 = _make_handler(
        card_ids=[2120],
        cards=[_card_data(card_id=2120, note_id=3120, queue=2)],
        notes=[_note_data(note_id=3120, tags=[])],
        reviews=[
            _review_row(review_id=1_700_000_000_100, card_id=2120),
            _review_row(review_id=1_700_000_000_101, card_id=2120),
        ],
    )
    async with make_client(handler_v1) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    calls_v2: list[dict[str, Any]] = []
    handler_v2 = _make_handler(
        card_ids=[2120],
        cards=[_card_data(card_id=2120, note_id=3120, queue=2)],
        notes=[_note_data(note_id=3120, tags=[])],
        reviews=[_review_row(review_id=1_700_000_000_102, card_id=2120)],
        record_calls=calls_v2,
    )
    async with make_client(handler_v2) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    review_calls = [c for c in calls_v2 if c["action"] == "cardReviews"]
    assert review_calls[0]["params"]["startID"] == 1_700_000_000_102
    assert summary.reviews_synced == 1


async def test_sync_review_reinsert_is_idempotent(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V26 append-only PK on review_id — overlapping ids no-op via ON CONFLICT."""
    from app.models.anki import AnkiCardReview

    reviews = [
        _review_row(review_id=1_700_000_000_200, card_id=2130),
        _review_row(review_id=1_700_000_000_201, card_id=2130),
    ]
    handler = _make_handler(
        card_ids=[2130],
        cards=[_card_data(card_id=2130, note_id=3130, queue=2)],
        notes=[_note_data(note_id=3130, tags=[])],
        reviews=reviews,
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    # Force the second run to re-fetch the same rows (handler still answers
    # cardReviews with the same list regardless of startID).
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    assert summary.reviews_synced == 0
    row_count = (
        await db_session.execute(
            select(func.count())
            .select_from(AnkiCardReview)
            .where(AnkiCardReview.review_id.between(1_700_000_000_200, 1_700_000_000_201))
        )
    ).scalar_one()
    assert row_count == 2


async def test_sync_review_skips_unknown_card_id(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """Defensive: revlog row referencing a card not in this sync drops, no FK violation."""
    from app.models.anki import AnkiCardReview

    handler = _make_handler(
        card_ids=[2140],
        cards=[_card_data(card_id=2140, note_id=3140, queue=2)],
        notes=[_note_data(note_id=3140, tags=[])],
        reviews=[
            _review_row(review_id=1_700_000_000_300, card_id=2140),  # known
            _review_row(review_id=1_700_000_000_301, card_id=999_999),  # unknown
        ],
    )
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    assert summary.reviews_synced == 1
    persisted = (
        (
            await db_session.execute(
                select(AnkiCardReview.review_id).where(
                    AnkiCardReview.review_id.between(1_700_000_000_300, 1_700_000_000_301)
                )
            )
        )
        .scalars()
        .all()
    )
    assert set(persisted) == {1_700_000_000_300}


# --- V43 (B9): sync must not delete non-regex tags ---
# (T20: removed `test_sync_preserves_llm_aamc_topic_rows` — it seeded
#  `Topic` + `ContentCategory` directly; both models are dropped. Re-coverage
#  on the OutlineNode shape lands when T22/T26 ports the LLM aamc_topic
#  writer onto `AnkiNoteTag.node_id`.)


async def test_sync_preserves_manual_override_rows(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V43 — `source='manual'` rows survive sync the same way LLM rows do."""
    handler = _make_handler(
        card_ids=[5201],
        cards=[_card_data(card_id=5201, note_id=6201, queue=2, interval=21)],
        notes=[_note_data(note_id=6201, tags=[])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    db_session.add(
        AnkiNoteTag(
            note_id=6201,
            tag_raw="__manual__::override::foo",
            parsed_kind="unparsed",
            source="manual",
        )
    )
    await db_session.flush()

    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    surviving = (
        (
            await db_session.execute(
                select(AnkiNoteTag).where(
                    AnkiNoteTag.note_id == 6201,
                    AnkiNoteTag.source == "manual",
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(surviving) == 1


# --- T94 / V75: note-as-unit sync ---


async def test_sync_note_with_multiple_cards_one_tag_set(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V75: a note with N cards yields ONE tag set on anki_note_tags (no
    per-card fan-out) backing all N cards; linked_qids counts the note once."""
    handler = _make_handler(
        card_ids=[7001, 7002, 7003],
        cards=[
            _card_data(card_id=7001, note_id=8001, queue=2, interval=14),
            _card_data(card_id=7002, note_id=8001, queue=0),  # sibling of 7001
            _card_data(card_id=7003, note_id=8001, queue=1),  # sibling of 7001
        ],
        notes=[_note_data(note_id=8001, tags=["#AK_MCAT_v2::#UWorld::402391"])],
    )
    async with make_client(handler) as client:
        summary = await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    assert summary.synced_cards == 3
    note_count = (
        await db_session.execute(
            select(func.count()).select_from(AnkiNote).where(AnkiNote.note_id == 8001)
        )
    ).scalar_one()
    assert note_count == 1
    card_count = (
        await db_session.execute(
            select(func.count()).select_from(AnkiCard).where(AnkiCard.note_id == 8001)
        )
    ).scalar_one()
    assert card_count == 3
    tag_rows = (
        (await db_session.execute(select(AnkiNoteTag).where(AnkiNoteTag.note_id == 8001)))
        .scalars()
        .all()
    )
    assert len(tag_rows) == 1, "note tags must not fan out across the note's 3 cards"
    assert tag_rows[0].question_qid == "402391"
    assert summary.linked_qids == 1


async def test_sync_populates_note_content(
    db_session: AsyncSession, lookup: OutlineLookup, make_client
) -> None:
    """§V75: model_name + fields_json land on anki_notes (note-level, sourced
    from cardsInfo); the card carries SRS state + the note_id FK."""
    fields = {"Text": {"value": "enzyme kinetics", "order": 0}}
    handler = _make_handler(
        card_ids=[7100],
        cards=[
            _card_data(
                card_id=7100,
                note_id=8100,
                queue=2,
                interval=21,
                fields=fields,
                model_name="AnKingOverhaul",
            )
        ],
        notes=[_note_data(note_id=8100, tags=[])],
    )
    async with make_client(handler) as client:
        await sync_deck(db_session, client, deck_name="MileDown", outline_lookup=lookup)

    note = (await db_session.execute(select(AnkiNote).where(AnkiNote.note_id == 8100))).scalar_one()
    assert note.model_name == "AnKingOverhaul"
    assert note.fields_json == fields
    assert note.deck_name == "MileDown"
    card = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 7100))
    ).scalar_one()
    assert card.note_id == 8100
    assert card.queue == 2
    assert card.interval_days == 21
