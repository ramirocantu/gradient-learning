"""T13 smoke — outline-free anki query helpers still work; stubs return empty."""

import os
from datetime import date, datetime, timedelta, timezone

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import Base
from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import Course, OutlineNode
from app.services.anki.queries import (
    due_count_for_subtree,
    get_anki_card_total,
    get_tag_parse_stats,
    list_cards_for_cc,
    list_cards_for_qid,
    list_cards_for_topic,
    list_review_queue,
    list_review_queue_for_cc,
    list_review_queue_for_topic_subtree,
)

_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_DB_URL = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"

_TABLES = [
    Course.__table__,
    OutlineNode.__table__,
    AnkiNote.__table__,
    AnkiCard.__table__,
    AnkiNoteTag.__table__,
]


@pytest.fixture
async def engine():
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute("CREATE DATABASE gradient_test")
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

    eng = create_async_engine(_DB_URL)
    async with eng.begin() as c:
        await c.execute(text("DROP SCHEMA public CASCADE"))
        await c.execute(text("CREATE SCHEMA public"))
        await c.run_sync(Base.metadata.create_all, tables=_TABLES)
    yield eng
    await eng.dispose()


async def _seed(eng) -> tuple[int, int, str]:
    async with AsyncSession(eng) as s:
        note = AnkiNote(note_id=42)
        s.add(note)
        await s.flush()
        card = AnkiCard(
            anki_card_id=1001,
            deck_name="AnKing MCAT Deck",
            note_id=42,
            due_date=date.today() + timedelta(days=1),
        )
        s.add(card)
        await s.flush()
        s.add(
            AnkiNoteTag(
                note_id=42,
                tag_raw="#AK::resolved",
                node_id=None,
                question_qid="q-anki-1",
                parsed_kind="resolved",
                source="schema_map",
            )
        )
        await s.flush()
        ids = (note.note_id, card.id, "q-anki-1")
        await s.commit()
    return ids


# ── outline-free helpers (preserved) ────────────────────────────────────────


async def test_list_review_queue_returns_scheduled_cards(engine):
    await _seed(engine)
    async with AsyncSession(engine) as s:
        cards = await list_review_queue(s, limit=10)
    assert len(cards) == 1
    assert cards[0].anki_card_id == 1001


async def test_list_cards_for_qid_finds_by_question_qid(engine):
    await _seed(engine)
    async with AsyncSession(engine) as s:
        cards = await list_cards_for_qid(s, qid="q-anki-1")
    assert len(cards) == 1


async def test_get_anki_card_total_counts(engine):
    await _seed(engine)
    async with AsyncSession(engine) as s:
        assert await get_anki_card_total(s) == 1


async def test_get_tag_parse_stats_groups_by_parsed_kind(engine):
    await _seed(engine)
    async with AsyncSession(engine) as s:
        stats = await get_tag_parse_stats(s)
    assert stats == {"resolved": 1}


# ── stubbed subtree helpers — return empty pending T14 ──────────────────────


async def test_subtree_helpers_return_empty(engine):
    await _seed(engine)
    due = datetime.now(tz=timezone.utc) + timedelta(days=1)
    async with AsyncSession(engine) as s:
        assert await list_cards_for_topic(s, topic_id=1) == []
        assert await list_cards_for_cc(s, cc_code="5E") == []
        assert await list_review_queue_for_cc(s, cc_code="5E", due_before=due) == []
        assert await list_review_queue_for_topic_subtree(s, topic_id=1, due_before=due) == []
        assert await due_count_for_subtree(s, topic_id=1, due_before=due) == 0
