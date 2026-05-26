"""Anki plan-adherence — deterministic projected-load metric (SPEC T66).

V54 + V60: pure-computation service. Reads recent review cadence,
upcoming unsuspend pressure, and the user's daily budget; returns a
numeric headroom + a threshold-derived status label. ⊥ LLM, ⊥ SQLite
cache, ⊥ `evaluator_version`, ⊥ token cost. ⊥ a `recommended_changes`
field in the payload — advisory lives in the MCP host chat reading
this tool (V60).

The projection model is intentionally simple so the user can audit
the math from the dashboard chip:

  base_daily_load    = 14d rolling mean reviews/day
  upcoming_per_day   = sum(len(card_ids) for pending assignments
                            scheduled to unlock in
                            [now, now + window_days]) / window_days
  projected_load     = base_daily_load + upcoming_per_day
  projected_minutes  = projected_load × mean_seconds_per_review / 60

Headroom is the slack against the user's budget on each axis:

  headroom_pct = (budget - projected) / budget × 100

Status thresholds (module constants below) translate the worse-axis
headroom into a chip:

  feasible : both headrooms ≥ 0
  overload : either headroom < _OVERLOAD_HEADROOM (default -15)
  tight    : otherwise
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiCardReview, AnkiLoadConfig


# V54 threshold band on the worse-axis headroom percentage.
_FEASIBLE_HEADROOM: float = 0.0
_OVERLOAD_HEADROOM: float = -15.0

# Fallback used when no review history exists yet so the minutes axis
# does not silently degenerate to 0. Anki's median per-card review time
# across mature decks sits around 8-12 seconds; 10s keeps the projection
# honest before the user has logged enough reviews to compute a real mean.
_DEFAULT_SECONDS_PER_REVIEW: float = 10.0

_REVIEWS_WINDOW_DAYS: int = 14


@dataclass(frozen=True, slots=True)
class AnkiLoadAdherence:
    """V54 payload shape — note the absence of `recommended_changes`
    per V60."""

    window_days: int
    projected_daily_load: int
    projected_daily_minutes: float
    daily_card_review_budget: int
    daily_minutes_budget: float
    headroom_card_review_pct: float
    headroom_minutes_pct: float
    status_label: str  # 'feasible' | 'tight' | 'overload'


def _classify(headroom_card_pct: float, headroom_minutes_pct: float) -> str:
    """V54 thresholds. The worse-axis governs the label."""
    worse = min(headroom_card_pct, headroom_minutes_pct)
    if worse >= _FEASIBLE_HEADROOM:
        return "feasible"
    if worse < _OVERLOAD_HEADROOM:
        return "overload"
    return "tight"


def _headroom_pct(budget: float, projected: float) -> float:
    """`(budget - projected) / budget * 100`. Returns +inf when budget=0
    and projected=0 (vacuous slack); returns -inf when budget=0 and
    projected>0 (any load over a zero budget is overload). Budget=0 is
    forbidden by the CHECK constraint on `anki_load_config`, so this
    branch is defensive only."""
    if budget <= 0:
        return float("-inf") if projected > 0 else float("inf")
    return (budget - projected) / budget * 100.0


async def _load_config(session: AsyncSession) -> tuple[int, float]:
    """Read the singleton `anki_load_config`. The T61 migration seeds
    row id=1 in production; tests that touch this service must insert
    a row before calling (the schema's CHECK (id=1) keeps it unique)."""
    row = (
        await session.execute(select(AnkiLoadConfig).where(AnkiLoadConfig.id == 1))
    ).scalar_one_or_none()
    if row is None:
        # Conservative fallback matching the migration seed defaults.
        # Service stays deterministic — caller does not need to know
        # whether the seed has run.
        return 200, 60.0
    return int(row.daily_card_review_budget), float(row.daily_minutes_budget)


async def _recent_review_stats(session: AsyncSession, *, now: datetime) -> tuple[float, float]:
    """14d rolling: (mean reviews/day, mean seconds/review). Excludes
    `type='learn'` to mirror retention.py's "true review" cohort — learn
    steps are short and noisy, so including them inflates the load
    projection."""
    window_start = now - timedelta(days=_REVIEWS_WINDOW_DAYS)
    stmt = select(
        func.count(),
        func.coalesce(
            func.avg(AnkiCardReview.time_ms),
            None,
        ),
    ).where(
        AnkiCardReview.reviewed_at >= window_start,
        AnkiCardReview.reviewed_at <= now,
        AnkiCardReview.type != "learn",
    )
    row = (await session.execute(stmt)).one()
    review_count = int(row[0] or 0)
    avg_time_ms = row[1]
    mean_per_day = review_count / float(_REVIEWS_WINDOW_DAYS)
    mean_seconds = (
        float(avg_time_ms) / 1000.0 if avg_time_ms is not None else _DEFAULT_SECONDS_PER_REVIEW
    )
    return mean_per_day, mean_seconds


async def _upcoming_unlock_pressure(
    session: AsyncSession, *, now: datetime, window_days: int
) -> float:
    """Sum of card-ids scheduled to unsuspend inside [now, now+window],
    divided by window_days. Pending assignments only — `unlocked` rows
    already contribute to the review history via T36 sync."""
    window_end = now + timedelta(days=window_days)
    rows = (
        await session.execute(
            select(AnkiAssignment.card_ids)
            .where(AnkiAssignment.status == "pending")
            .where(AnkiAssignment.scheduled_unlock_at >= now)
            .where(AnkiAssignment.scheduled_unlock_at <= window_end)
        )
    ).all()
    total_cards = sum(len(row[0] or []) for row in rows)
    return total_cards / float(window_days) if window_days > 0 else 0.0


async def compute_load_adherence(
    session: AsyncSession,
    *,
    window_days: int = 30,
    now: Optional[datetime] = None,
) -> AnkiLoadAdherence:
    """V54 main entry. Pure computation — no side effects, no LLM."""
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    now = now or datetime.now(timezone.utc)

    base_per_day, mean_seconds = await _recent_review_stats(session, now=now)
    upcoming_per_day = await _upcoming_unlock_pressure(session, now=now, window_days=window_days)
    daily_card_review_budget, daily_minutes_budget = await _load_config(session)

    projected_daily_load_f = base_per_day + upcoming_per_day
    projected_daily_minutes = projected_daily_load_f * mean_seconds / 60.0
    projected_daily_load = int(round(projected_daily_load_f))

    headroom_card_pct = _headroom_pct(float(daily_card_review_budget), projected_daily_load_f)
    headroom_minutes_pct = _headroom_pct(daily_minutes_budget, projected_daily_minutes)

    return AnkiLoadAdherence(
        window_days=window_days,
        projected_daily_load=projected_daily_load,
        projected_daily_minutes=projected_daily_minutes,
        daily_card_review_budget=daily_card_review_budget,
        daily_minutes_budget=daily_minutes_budget,
        headroom_card_review_pct=headroom_card_pct,
        headroom_minutes_pct=headroom_minutes_pct,
        status_label=_classify(headroom_card_pct, headroom_minutes_pct),
    )


__all__ = [
    "AnkiLoadAdherence",
    "compute_load_adherence",
]
