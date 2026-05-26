"""Schema tests for SPEC T61 — anki_load_config singleton (V59).

Validates V59 invariants at storage layer: CHECK (id=1) enforces a
single row; positive budgets on both columns; daily_minutes_budget
stored as NUMERIC so the load evaluator (T66) can carry sub-minute
precision without floating-point drift.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiLoadConfig


# --- V59: singleton enforced via CHECK (id=1) ---


async def test_singleton_id_must_be_one(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiLoadConfig(
            id=2,
            daily_card_review_budget=200,
            daily_minutes_budget=Decimal("60"),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_id_one_accepted(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=200,
            daily_minutes_budget=Decimal("60"),
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiLoadConfig))).scalar_one()
    assert row.id == 1


# --- V59: budgets must be > 0 ---


@pytest.mark.parametrize("bad", [0, -1, -100])
async def test_card_budget_non_positive_rejected(db_session: AsyncSession, bad: int) -> None:
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=bad,
            daily_minutes_budget=Decimal("60"),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.parametrize("bad_minutes", [Decimal("0"), Decimal("-1"), Decimal("-15.5")])
async def test_minutes_budget_non_positive_rejected(
    db_session: AsyncSession, bad_minutes: Decimal
) -> None:
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=200,
            daily_minutes_budget=bad_minutes,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_minutes_budget_numeric_precision(
    db_session: AsyncSession,
) -> None:
    """Numeric (not Float) so the evaluator can store fractional-minute
    budgets without binary-float drift."""
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=180,
            daily_minutes_budget=Decimal("45.25"),
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiLoadConfig))).scalar_one()
    assert row.daily_minutes_budget == Decimal("45.25")


async def test_updated_at_defaults_to_now(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=100,
            daily_minutes_budget=Decimal("30"),
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiLoadConfig))).scalar_one()
    assert row.updated_at is not None
