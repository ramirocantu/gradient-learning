"""Tests for SPEC T66 — deterministic plan-adherence (V54, V60).

Pure computation: same inputs always produce the same output, no LLM.
Covers the V54 status thresholds + the V60 contract (no
`recommended_changes` field in the payload).
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import (
    AnkiAssignment,
    AnkiCard,
    AnkiCardReview,
    AnkiLoadConfig,
)
from app.services.anki.load_adherence import (
    AnkiLoadAdherence,
    compute_load_adherence,
)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _seed_config(
    session: AsyncSession,
    *,
    card_budget: int = 200,
    minutes_budget: float = 60.0,
) -> None:
    session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=card_budget,
            daily_minutes_budget=Decimal(str(minutes_budget)),
        )
    )
    await session.flush()


async def _seed_card(session: AsyncSession, *, native_id: int) -> AnkiCard:
    card = AnkiCard(
        anki_card_id=native_id,
        deck_name="MileDown",
        queue=2,
    )
    session.add(card)
    await session.flush()
    return card


async def _seed_reviews(
    session: AsyncSession,
    *,
    card: AnkiCard,
    count: int,
    seconds_each: int = 10,
    review_type: str = "review",
    base_review_id: int = 1_900_000_000_000,
    spread_over_days: int = 14,
    now: datetime | None = None,
) -> None:
    """Spread `count` reviews evenly across the most recent
    `spread_over_days` so they land inside the 14d rolling window."""
    now = now or _now()
    for i in range(count):
        delta_days = (i * spread_over_days) / max(count, 1)
        session.add(
            AnkiCardReview(
                review_id=base_review_id + i,
                card_id=card.id,
                reviewed_at=now - timedelta(days=delta_days, minutes=1),
                ease=3,
                type=review_type,
                time_ms=seconds_each * 1000,
            )
        )
    await session.flush()


async def _seed_pending_assignment(
    session: AsyncSession,
    *,
    card_ids: list[int],
    unlock_at: datetime,
    scope_value: str,
) -> AnkiAssignment:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value=scope_value,
        scheduled_unlock_at=_utc(unlock_at),
        card_ids=card_ids,
        status="pending",
    )
    session.add(a)
    await session.flush()
    return a


# --- V54 shape + V60 contract -------------------------------------------- #


async def test_payload_shape_matches_v54_and_v60(
    db_session: AsyncSession,
) -> None:
    """V60: ⊥ `recommended_changes` field. V54 lists exactly these
    keys."""
    await _seed_config(db_session)
    result = await compute_load_adherence(db_session)
    assert isinstance(result, AnkiLoadAdherence)

    field_names = {f.name for f in fields(AnkiLoadAdherence)}
    assert field_names == {
        "window_days",
        "projected_daily_load",
        "projected_daily_minutes",
        "daily_card_review_budget",
        "daily_minutes_budget",
        "headroom_card_review_pct",
        "headroom_minutes_pct",
        "status_label",
    }
    # V60: no recommended_changes anywhere in the payload.
    assert "recommended_changes" not in field_names


async def test_window_days_input_reflected_in_output(
    db_session: AsyncSession,
) -> None:
    await _seed_config(db_session)
    r7 = await compute_load_adherence(db_session, window_days=7)
    r30 = await compute_load_adherence(db_session, window_days=30)
    assert r7.window_days == 7
    assert r30.window_days == 30


async def test_window_days_must_be_positive(db_session: AsyncSession) -> None:
    await _seed_config(db_session)
    with pytest.raises(ValueError):
        await compute_load_adherence(db_session, window_days=0)
    with pytest.raises(ValueError):
        await compute_load_adherence(db_session, window_days=-1)


# --- empty state + budget echo ------------------------------------------- #


async def test_empty_state_projects_zero_and_is_feasible(
    db_session: AsyncSession,
) -> None:
    await _seed_config(db_session, card_budget=200, minutes_budget=60.0)
    result = await compute_load_adherence(db_session)
    assert result.projected_daily_load == 0
    assert result.projected_daily_minutes == pytest.approx(0.0)
    assert result.headroom_card_review_pct == pytest.approx(100.0)
    assert result.headroom_minutes_pct == pytest.approx(100.0)
    assert result.status_label == "feasible"
    # Budget echoed verbatim.
    assert result.daily_card_review_budget == 200
    assert result.daily_minutes_budget == pytest.approx(60.0)


async def test_falls_back_to_seed_defaults_when_config_missing(
    db_session: AsyncSession,
) -> None:
    """T61 migration seeds (200, 60); but the test DB uses
    Base.metadata.create_all (no seed). Service must remain deterministic
    when the row is absent so dashboard chips still render."""
    # No _seed_config call.
    result = await compute_load_adherence(db_session)
    assert result.daily_card_review_budget == 200
    assert result.daily_minutes_budget == pytest.approx(60.0)


# --- review history feeds projection ------------------------------------- #


async def test_reviews_in_window_drive_base_load(
    db_session: AsyncSession,
) -> None:
    """140 reviews across the 14d window → 10 reviews/day base."""
    await _seed_config(db_session)
    card = await _seed_card(db_session, native_id=1_900_000_001)
    await _seed_reviews(db_session, card=card, count=140, seconds_each=12)
    result = await compute_load_adherence(db_session)
    assert result.projected_daily_load == 10
    assert result.projected_daily_minutes == pytest.approx(10 * 12 / 60.0, rel=1e-3)
    assert result.status_label == "feasible"


async def test_learn_reviews_excluded_from_projection(
    db_session: AsyncSession,
) -> None:
    """V54 mirrors retention.py's "true review" cohort — `type='learn'`
    must not count toward the base load."""
    await _seed_config(db_session)
    card = await _seed_card(db_session, native_id=1_900_000_002)
    await _seed_reviews(db_session, card=card, count=140, review_type="learn")
    result = await compute_load_adherence(db_session)
    assert result.projected_daily_load == 0
    assert result.status_label == "feasible"


async def test_upcoming_assignments_inflate_projection(
    db_session: AsyncSession,
) -> None:
    """30 cards unlocked across a 30d window adds 1 review/day to the
    projection."""
    await _seed_config(db_session)
    # No prior reviews -> base=0; only the upcoming assignment contributes.
    await _seed_pending_assignment(
        db_session,
        card_ids=list(range(1_900_000_100, 1_900_000_130)),  # 30 cards
        unlock_at=_now() + timedelta(days=2),
        scope_value="upcoming",
    )
    result = await compute_load_adherence(db_session, window_days=30)
    assert result.projected_daily_load == 1  # 30 / 30
    # 1 review/day at default 10s/review → ~0.167 min/day
    assert result.projected_daily_minutes == pytest.approx(10 / 60.0, rel=1e-3)


async def test_assignment_outside_window_not_counted(
    db_session: AsyncSession,
) -> None:
    """Pending assignment scheduled after `now + window_days` must not
    contribute to projected_daily_load."""
    await _seed_config(db_session)
    await _seed_pending_assignment(
        db_session,
        card_ids=list(range(1_900_000_200, 1_900_000_400)),  # 200 cards
        unlock_at=_now() + timedelta(days=90),
        scope_value="far-future",
    )
    result = await compute_load_adherence(db_session, window_days=30)
    assert result.projected_daily_load == 0
    assert result.status_label == "feasible"


async def test_non_pending_assignments_not_counted(
    db_session: AsyncSession,
) -> None:
    """Only `pending` assignments contribute; `unlocked` rows are already
    in the review history via T36 sync, `completed`/`skipped`/`failed`
    are inert."""
    await _seed_config(db_session)
    for status in ("unlocked", "completed", "skipped", "failed"):
        a = AnkiAssignment(
            scope_kind="cc",
            scope_value=f"non-pending-{status}",
            scheduled_unlock_at=_now() + timedelta(days=5),
            actual_unlock_at=_now() - timedelta(days=1) if status != "skipped" else None,
            card_ids=list(range(1_900_000_500, 1_900_000_600)),  # 100 cards
            status=status,
        )
        db_session.add(a)
    await db_session.flush()
    result = await compute_load_adherence(db_session, window_days=30)
    assert result.projected_daily_load == 0


# --- thresholds ---------------------------------------------------------- #


async def test_feasible_when_well_under_budget(db_session: AsyncSession) -> None:
    await _seed_config(db_session, card_budget=500, minutes_budget=120.0)
    card = await _seed_card(db_session, native_id=1_900_001_001)
    await _seed_reviews(db_session, card=card, count=140, seconds_each=10)
    result = await compute_load_adherence(db_session)
    assert result.status_label == "feasible"
    assert result.headroom_card_review_pct > 0
    assert result.headroom_minutes_pct > 0


async def test_tight_when_card_headroom_in_minus_15_zero_band(
    db_session: AsyncSession,
) -> None:
    """Card axis 10% over budget → headroom -10% → tight (≥ -15)."""
    await _seed_config(db_session, card_budget=100, minutes_budget=1000.0)
    card = await _seed_card(db_session, native_id=1_900_002_001)
    # 1540 reviews / 14d = 110/day vs budget 100 → -10% headroom.
    await _seed_reviews(db_session, card=card, count=1540, seconds_each=2)
    result = await compute_load_adherence(db_session)
    assert -15.0 <= result.headroom_card_review_pct < 0
    assert result.headroom_minutes_pct >= 0
    assert result.status_label == "tight"


async def test_tight_when_minutes_headroom_in_minus_15_zero_band(
    db_session: AsyncSession,
) -> None:
    """Minutes axis 10% over budget while card axis is fine."""
    await _seed_config(db_session, card_budget=1000, minutes_budget=10.0)
    card = await _seed_card(db_session, native_id=1_900_003_001)
    # 70 reviews / 14d = 5/day; 5 × 132s / 60 = 11.0 min/day vs budget 10 → -10%.
    await _seed_reviews(db_session, card=card, count=70, seconds_each=132)
    result = await compute_load_adherence(db_session)
    assert result.headroom_card_review_pct >= 0
    assert -15.0 <= result.headroom_minutes_pct < 0
    assert result.status_label == "tight"


async def test_overload_when_card_headroom_below_minus_15(
    db_session: AsyncSession,
) -> None:
    """Card axis 30% over budget → overload regardless of minutes."""
    await _seed_config(db_session, card_budget=100, minutes_budget=1000.0)
    card = await _seed_card(db_session, native_id=1_900_004_001)
    # 1820 / 14d = 130/day → headroom -30%.
    await _seed_reviews(db_session, card=card, count=1820, seconds_each=2)
    result = await compute_load_adherence(db_session)
    assert result.headroom_card_review_pct < -15.0
    assert result.status_label == "overload"


async def test_overload_when_minutes_headroom_below_minus_15(
    db_session: AsyncSession,
) -> None:
    """Minutes axis 30% over budget while card axis stays fine."""
    await _seed_config(db_session, card_budget=1000, minutes_budget=10.0)
    card = await _seed_card(db_session, native_id=1_900_005_001)
    # 70 reviews / 14d = 5/day; 5 × 156s / 60 = 13.0 min/day vs budget 10 → -30%.
    await _seed_reviews(db_session, card=card, count=70, seconds_each=156)
    result = await compute_load_adherence(db_session)
    assert result.headroom_minutes_pct < -15.0
    assert result.status_label == "overload"


async def test_boundary_zero_headroom_is_feasible(
    db_session: AsyncSession,
) -> None:
    """V54 thresholds are `≥ 0 → feasible`; exact 0 must not slip to tight."""
    await _seed_config(db_session, card_budget=10, minutes_budget=1000.0)
    card = await _seed_card(db_session, native_id=1_900_006_001)
    # 140 reviews / 14d = exactly 10/day = budget. headroom = 0.
    await _seed_reviews(db_session, card=card, count=140, seconds_each=2)
    result = await compute_load_adherence(db_session)
    assert result.headroom_card_review_pct == pytest.approx(0.0, abs=1e-6)
    assert result.status_label == "feasible"


async def test_boundary_minus_15_is_tight_not_overload(
    db_session: AsyncSession,
) -> None:
    """V54: overload requires `< -15.0` (strict). Exactly -15 is tight."""
    await _seed_config(db_session, card_budget=100, minutes_budget=1000.0)
    card = await _seed_card(db_session, native_id=1_900_007_001)
    # 1610 / 14d = 115/day vs 100 → -15.0%.
    await _seed_reviews(db_session, card=card, count=1610, seconds_each=2)
    result = await compute_load_adherence(db_session)
    assert result.headroom_card_review_pct == pytest.approx(-15.0, abs=1e-6)
    assert result.status_label == "tight"


async def test_classify_uses_worse_axis(db_session: AsyncSession) -> None:
    """When the two axes disagree, the worse-axis governs the label."""
    await _seed_config(db_session, card_budget=100, minutes_budget=1000.0)
    card = await _seed_card(db_session, native_id=1_900_008_001)
    # Card axis well over budget (overload); minutes axis trivially feasible.
    await _seed_reviews(db_session, card=card, count=1820, seconds_each=1)
    result = await compute_load_adherence(db_session)
    assert result.headroom_card_review_pct < -15.0
    assert result.headroom_minutes_pct > 0
    assert result.status_label == "overload"


# --- now override + window math ------------------------------------------ #


async def test_now_override_relocates_review_window(
    db_session: AsyncSession,
) -> None:
    """Reviews dated relative to `now` — passing an explicit `now` lets
    tests + the API freeze the projection at a known moment."""
    await _seed_config(db_session)
    card = await _seed_card(db_session, native_id=1_900_009_001)
    # Seed 140 reviews in the 14d window ending at `frozen_now`.
    frozen_now = _now() - timedelta(days=60)
    await _seed_reviews(db_session, card=card, count=140, seconds_each=10, now=frozen_now)
    # Querying at frozen_now → projection sees the reviews.
    fresh = await compute_load_adherence(db_session, now=frozen_now)
    assert fresh.projected_daily_load == 10
    # Querying now (60 days later) → reviews now sit outside the window.
    stale = await compute_load_adherence(db_session)
    assert stale.projected_daily_load == 0
