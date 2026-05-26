"""Schema tests for SPEC T35 — anki_card_reviews append-only review log.

Substrate for T37 retention.py. V26 (append-only idempotency via review_id
PK) + V27 (ease/type stored so retention math can window pass = ease ∈
{2,3,4} and exclude type='learn').
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiCardReview


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _make_card(session: AsyncSession, anki_card_id: int = 700_000) -> AnkiCard:
    card = AnkiCard(anki_card_id=anki_card_id, deck_name="MileDown")
    session.add(card)
    await session.flush()
    return card


# --- V26: append-only idempotency ---


async def test_review_id_pk_rejects_duplicate_insert(db_session: AsyncSession) -> None:
    """V26: review_id PK = Anki revlog id; re-sync of same review is a no-op insert."""
    card = await _make_card(db_session, anki_card_id=700_001)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_000_001,
            card_id=card.id,
            reviewed_at=_now(),
            ease=3,
            type="review",
        )
    )
    await db_session.flush()

    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_000_001,
            card_id=card.id,
            reviewed_at=_now(),
            ease=3,
            type="review",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_card_id_fk_cascade_on_card_delete(db_session: AsyncSession) -> None:
    """V26 ergonomics: dropping a card cascades its review history (no orphans)."""
    card = await _make_card(db_session, anki_card_id=700_002)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_000_002,
            card_id=card.id,
            reviewed_at=_now(),
            ease=2,
            type="review",
        )
    )
    await db_session.flush()
    await db_session.execute(text("DELETE FROM anki_cards WHERE id = :id"), {"id": card.id})
    await db_session.flush()
    rows = (
        (await db_session.execute(select(AnkiCardReview).where(AnkiCardReview.card_id == card.id)))
        .scalars()
        .all()
    )
    assert rows == []


# --- V27: retention math substrate ---


@pytest.mark.parametrize("bad_ease", [0, 5, -1])
async def test_ease_check_constraint_rejects_out_of_range(
    db_session: AsyncSession, bad_ease: int
) -> None:
    """V27: ease ∈ {1,2,3,4} (Again/Hard/Good/Easy). Out-of-range corrupts
    pass/fail bucketing in retention.py."""
    card = await _make_card(db_session, anki_card_id=700_010 + bad_ease)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_010_000 + bad_ease,
            card_id=card.id,
            reviewed_at=_now(),
            ease=bad_ease,
            type="review",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.parametrize("good_ease", [1, 2, 3, 4])
async def test_ease_check_accepts_valid_range(db_session: AsyncSession, good_ease: int) -> None:
    card = await _make_card(db_session, anki_card_id=700_100 + good_ease)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_100_000 + good_ease,
            card_id=card.id,
            reviewed_at=_now(),
            ease=good_ease,
            type="review",
        )
    )
    await db_session.flush()


async def test_type_check_constraint_rejects_unknown(db_session: AsyncSession) -> None:
    """V27: type ∈ {learn, review, relearn, cram} mirrors Anki revlog enum.
    Unknown value would break the type='learn' exclusion filter."""
    card = await _make_card(db_session, anki_card_id=700_200)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_200_001,
            card_id=card.id,
            reviewed_at=_now(),
            ease=3,
            type="bogus",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.parametrize("good_type", ["learn", "review", "relearn", "cram"])
async def test_type_check_accepts_known_values(db_session: AsyncSession, good_type: str) -> None:
    card = await _make_card(db_session, anki_card_id=700_300 + hash(good_type) % 100)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_300_000 + hash(good_type) % 1000,
            card_id=card.id,
            reviewed_at=_now(),
            ease=3,
            type=good_type,
        )
    )
    await db_session.flush()


async def test_optional_columns_nullable(db_session: AsyncSession) -> None:
    """V27: interval_before/after + time_ms nullable — Anki may omit on early reviews."""
    card = await _make_card(db_session, anki_card_id=700_400)
    db_session.add(
        AnkiCardReview(
            review_id=1_700_000_400_001,
            card_id=card.id,
            reviewed_at=_now(),
            ease=3,
            type="learn",
            interval_before=None,
            interval_after=None,
            time_ms=None,
        )
    )
    await db_session.flush()
    row = (
        await db_session.execute(
            select(AnkiCardReview).where(AnkiCardReview.review_id == 1_700_000_400_001)
        )
    ).scalar_one()
    assert row.interval_before is None
    assert row.interval_after is None
    assert row.time_ms is None


# --- V26: incremental-sync substrate ---


async def test_max_review_id_supports_incremental_startid(db_session: AsyncSession) -> None:
    """V26: T36 sync uses `startID = MAX(review_id) + 1`. Verify MAX() returns the
    sentinel needed; first-run scenario (table empty) returns None so caller falls
    back to startID=0."""
    card = await _make_card(db_session, anki_card_id=700_500)

    # First-run: empty table → MAX is NULL → caller backfills from startID=0.
    max_empty = (
        await db_session.execute(select(text("MAX(review_id) FROM anki_card_reviews")))
    ).scalar()
    assert max_empty is None

    db_session.add_all(
        [
            AnkiCardReview(
                review_id=1_700_000_500_001,
                card_id=card.id,
                reviewed_at=_now(),
                ease=3,
                type="review",
            ),
            AnkiCardReview(
                review_id=1_700_000_500_005,
                card_id=card.id,
                reviewed_at=_now(),
                ease=2,
                type="review",
            ),
        ]
    )
    await db_session.flush()

    max_filled = (
        await db_session.execute(select(text("MAX(review_id) FROM anki_card_reviews")))
    ).scalar()
    assert max_filled == 1_700_000_500_005


# --- index ---


async def test_card_reviewed_index_exists(db_session: AsyncSession) -> None:
    """V27: composite index (card_id, reviewed_at) supports the per-card windowed
    scan retention.py runs (`WHERE card_id = ? AND reviewed_at >= window_start`)."""

    def _check(sync_conn) -> list[dict]:
        return inspect(sync_conn).get_indexes("anki_card_reviews")

    conn = await db_session.connection()
    indexes = await conn.run_sync(_check)
    by_name = {ix["name"]: ix for ix in indexes}
    assert "ix_anki_card_reviews_card_reviewed" in by_name
    assert by_name["ix_anki_card_reviews_card_reviewed"]["column_names"] == [
        "card_id",
        "reviewed_at",
    ]


# --- relationship ---


async def test_card_reviews_relationship_loads(db_session: AsyncSession) -> None:
    """ORM relationship: AnkiCard.reviews collects its AnkiCardReview rows."""
    card = await _make_card(db_session, anki_card_id=700_600)
    db_session.add_all(
        [
            AnkiCardReview(
                review_id=1_700_000_600_001,
                card_id=card.id,
                reviewed_at=_now(),
                ease=3,
                type="learn",
            ),
            AnkiCardReview(
                review_id=1_700_000_600_002,
                card_id=card.id,
                reviewed_at=_now(),
                ease=2,
                type="review",
            ),
        ]
    )
    await db_session.flush()
    await db_session.refresh(card, attribute_names=["reviews"])
    assert len(card.reviews) == 2
    assert {r.type for r in card.reviews} == {"learn", "review"}
