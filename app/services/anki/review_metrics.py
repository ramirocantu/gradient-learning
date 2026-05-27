"""Per-card review metrics for the anki review surface (T43, desktop ¶T6).

Read-only derived numbers over the canonical `anki_cards` / `anki_card_reviews`
tables — V13 (⊥ mutate Anki scheduling) holds trivially since nothing here
writes back to AnkiConnect. No legacy `topic_id` / `cc_code` joins (V-RB2): the
metrics are node-blind, computed per card id.

Two quantities feed the review-queue payload:

  retention  — the card's lifetime "true retention": fraction of its non-learn
               reviews that passed (pass = ease ∈ {2,3,4}, mirroring
               retention.py's cohort, §V26/§V27). None when the card has no
               qualifying reviews yet.

  retrievability — current estimated recall probability from the forgetting
               curve. Anki schedules each interval to land on a target
               retention (default 0.9) at the due date, so the curve
               R(t) = 0.9 ** (elapsed / interval) gives R = 0.9 the day a card
               comes due and < 0.9 once overdue. `elapsed` is derived from the
               stored due_date + interval (last review ≈ due_date − interval),
               so no extra revlog read is needed. None when interval is unknown.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiCardReview

# Anki's default desired retention — the recall probability each scheduled
# interval is built to hit on the due date.
_TARGET_RETENTION_AT_DUE: float = 0.9

# "True review" pass cohort, carried from retention.py (§V26/§V27).
_PASS_EASES: tuple[int, ...] = (2, 3, 4)


async def retention_by_card(
    session: AsyncSession, *, card_ids: list[int]
) -> dict[int, float | None]:
    """Map each card id → lifetime true-retention (pass/total over non-learn
    reviews), or None when the card has no qualifying reviews. One grouped
    query, ⊥ N+1."""
    if not card_ids:
        return {}
    pass_expr = func.count().filter(AnkiCardReview.ease.in_(_PASS_EASES))
    total_expr = func.count()
    stmt = (
        select(AnkiCardReview.card_id, pass_expr, total_expr)
        .where(AnkiCardReview.card_id.in_(card_ids))
        .where(AnkiCardReview.type != "learn")
        .group_by(AnkiCardReview.card_id)
    )
    out: dict[int, float | None] = {cid: None for cid in card_ids}
    for card_id, passed, total in (await session.execute(stmt)).all():
        out[card_id] = (int(passed) / int(total)) if total else None
    return out


def retrievability(card: AnkiCard, *, today: date | None = None) -> float | None:
    """Estimated current recall probability for `card` via the forgetting
    curve R = 0.9 ** (elapsed / interval), clamped to [0, 1].

    Returns None when `interval_days` is missing or non-positive (new /
    unscheduled cards have no curve). Result is 0.9 on the due date, higher
    before, lower once overdue.
    """
    interval = card.interval_days
    if interval is None or interval <= 0 or card.due_date is None:
        return None
    today = today or date.today()
    days_to_due = (card.due_date - today).days
    elapsed = interval - days_to_due  # last review ≈ due_date − interval
    r = _TARGET_RETENTION_AT_DUE ** (elapsed / interval)
    return max(0.0, min(1.0, r))


__all__ = ["retention_by_card", "retrievability"]
