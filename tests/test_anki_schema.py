"""Schema tests for SPEC T2 — anki_cards.

Tests run against the shared test Postgres DB via the conftest `db_session`
fixture. Tables come from `Base.metadata.create_all` at session start, which
mirrors the migration in `2026_05_19_ticket_t2_anki_cards.py`. The migration
itself is verified by running `alembic upgrade head` in the build verification
step.

§V75 (note-as-unit): the per-tag schema now lives on `anki_note_tags` —
its coverage is in `test_anki_note_schema.py`. This file keeps the
`anki_cards` identity / review-state / index assertions only.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard


def _card_kwargs(**overrides) -> dict:
    base = dict(
        anki_card_id=1_000_000_000_001,
        deck_name="MileDown",
        # T93: anki_cards.note_id now FKs anki_notes — these V1 identity tests
        # key on (deck_name, anki_card_id), not the note, so leave it NULL.
        note_id=None,
        model_name="MileDown Premed",
        fields_json={"Front": "What is enzyme?", "Back": "Protein catalyst"},
        due_date=date(2026, 6, 1),
        interval_days=21,
        ease=2500,
        lapses=1,
        queue=2,
    )
    base.update(overrides)
    return base


# --- V1: upsert identity ---


async def test_anki_cards_unique_constraint_on_deck_card(db_session: AsyncSession) -> None:
    """V1: (deck_name, anki_card_id) is the upsert key — duplicate inserts fail."""
    db_session.add(AnkiCard(**_card_kwargs(anki_card_id=42, deck_name="MileDown")))
    await db_session.flush()

    db_session.add(AnkiCard(**_card_kwargs(anki_card_id=42, deck_name="MileDown")))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_anki_cards_same_card_id_different_deck_allowed(db_session: AsyncSession) -> None:
    """V1: same anki_card_id across different decks does not collide."""
    db_session.add(AnkiCard(**_card_kwargs(anki_card_id=42, deck_name="MileDown")))
    db_session.add(AnkiCard(**_card_kwargs(anki_card_id=42, deck_name="OtherDeck")))
    await db_session.flush()


# --- V2: review state ---


async def test_anki_cards_review_state_columns_nullable(db_session: AsyncSession) -> None:
    """V2: review-state columns are nullable (new card may have no scheduling state yet)."""
    db_session.add(
        AnkiCard(
            anki_card_id=99,
            deck_name="MileDown",
            due_date=None,
            interval_days=None,
            ease=None,
            lapses=None,
            queue=None,
        )
    )
    await db_session.flush()
    row = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 99))
    ).scalar_one()
    assert row.due_date is None
    assert row.interval_days is None


async def test_anki_cards_sync_at_defaults_to_now(db_session: AsyncSession) -> None:
    """V2: sync_at server-default = now() so callers omitting the field still get a timestamp."""
    db_session.add(AnkiCard(anki_card_id=101, deck_name="MileDown"))
    await db_session.flush()
    row = (
        await db_session.execute(select(AnkiCard).where(AnkiCard.anki_card_id == 101))
    ).scalar_one()
    assert isinstance(row.sync_at, datetime)
    assert row.sync_at.tzinfo is not None
    assert (datetime.now(timezone.utc) - row.sync_at).total_seconds() < 60


async def test_due_date_index_exists(db_session: AsyncSession) -> None:
    """V2: due_date is indexed so the review-queue endpoint sorts efficiently."""

    def _check(sync_conn) -> list[dict]:
        return inspect(sync_conn).get_indexes("anki_cards")

    conn = await db_session.connection()
    indexes = await conn.run_sync(_check)
    names = {ix["name"] for ix in indexes}
    assert "ix_anki_cards_due_date" in names
