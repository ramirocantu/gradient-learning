"""Schema tests for SPEC T76 — anki_reviews table (V53 amended).

V53 amended 2026-05-23: reviews are standalone, no UNIQUE on
(review_date, *); deck name derives from row PK in the service layer
(`<ANKI_DECK_PREFIX>::review::{id}`). This file covers the storage-
layer invariants that DID survive: CHECK on status enum + status/
failure_count defaults.

T79 / V61 dropped the `study_plan_item_id` FK when Phase 7 was cut.

Disambiguation note: `test_anki_reviews_schema.py` already exists and
covers the unrelated `anki_card_reviews` table (T35 revlog) — this
file's singular `review_schema` keeps the namespaces separate.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiReview


# --- V53 amended: NO UNIQUE constraint ---


async def test_no_unique_on_review_date(db_session: AsyncSession) -> None:
    """V53 amended: dup reviews on the same date are allowed (tags-as-log
    accepts accumulation; idempotency is UI-debounce concern, ⊥ DB)."""
    db_session.add_all(
        [
            AnkiReview(
                review_date=date(2026, 5, 22),
                card_ids=[1, 2],
                deck_name="mcat-coach::review::placeholder-1",
            ),
            AnkiReview(
                review_date=date(2026, 5, 22),
                card_ids=[3, 4],
                deck_name="mcat-coach::review::placeholder-2",
            ),
        ]
    )
    await db_session.flush()
    rows = (await db_session.execute(select(AnkiReview))).scalars().all()
    assert len(list(rows)) == 2


# --- V53: status enum ---


@pytest.mark.parametrize("good", ["pending", "pushed", "failed"])
async def test_status_check_accepts_v53(db_session: AsyncSession, good: str) -> None:
    db_session.add(
        AnkiReview(
            review_date=date(2026, 5, 23),
            card_ids=[1],
            deck_name=f"mcat-coach::review::ok-{good}",
            status=good,
        )
    )
    await db_session.flush()


async def test_status_check_rejects_unknown(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiReview(
            review_date=date(2026, 5, 24),
            card_ids=[1],
            deck_name="mcat-coach::review::bad",
            status="completed",  # belongs to assignments, not reviews
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- defaults ---


async def test_defaults_status_pending(db_session: AsyncSession) -> None:
    """deck_name is required (NOT NULL); status defaults to pending."""
    db_session.add(
        AnkiReview(
            review_date=date(2026, 5, 26),
            card_ids=[10],
            deck_name="mcat-coach::review::defaults",
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiReview))).scalar_one()
    assert row.status == "pending"
    assert row.failure_count == 0
    assert row.pushed_at is None
