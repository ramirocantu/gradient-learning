"""Per-card review metrics on the anki review surface (T43, desktop ¶T6).

retention_by_card + retrievability are read-only derived numbers (V13: no
AnkiConnect mutation) and node-blind (V-RB2: no topic_id/cc_code). The
review-queue route carries them per card.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.anki import AnkiCard, AnkiCardReview
from app.services.anki.review_metrics import retention_by_card, retrievability

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _card(
    db: AsyncSession,
    *,
    native_id: int,
    due: date | None = None,
    interval: int | None = None,
) -> AnkiCard:
    c = AnkiCard(
        anki_card_id=native_id,
        deck_name="MileDown",
        queue=2,
        due_date=due,
        interval_days=interval,
    )
    db.add(c)
    await db.flush()
    return c


async def _review(
    db: AsyncSession, *, review_id: int, card_id: int, ease: int, rtype: str = "review"
) -> None:
    db.add(
        AnkiCardReview(
            review_id=review_id,
            card_id=card_id,
            reviewed_at=datetime.now(timezone.utc) - timedelta(days=1),
            ease=ease,
            type=rtype,
        )
    )
    await db.flush()


# ---------- retention_by_card (§V26/§V27 cohort) ----------


@pytest.mark.asyncio
async def test_retention_pass_over_total_excludes_learn(db_session: AsyncSession) -> None:
    c = await _card(db_session, native_id=9_001)
    # 3 passes (ease 2/3/4), 1 fail (ease 1) → 3/4. One learn row ignored.
    await _review(db_session, review_id=1, card_id=c.id, ease=2)
    await _review(db_session, review_id=2, card_id=c.id, ease=3)
    await _review(db_session, review_id=3, card_id=c.id, ease=4)
    await _review(db_session, review_id=4, card_id=c.id, ease=1)
    await _review(db_session, review_id=5, card_id=c.id, ease=1, rtype="learn")

    out = await retention_by_card(db_session, card_ids=[c.id])
    assert out[c.id] == pytest.approx(3 / 4)


@pytest.mark.asyncio
async def test_retention_none_when_no_qualifying_reviews(db_session: AsyncSession) -> None:
    c = await _card(db_session, native_id=9_002)
    await _review(db_session, review_id=10, card_id=c.id, ease=3, rtype="learn")
    out = await retention_by_card(db_session, card_ids=[c.id])
    assert out[c.id] is None


@pytest.mark.asyncio
async def test_retention_empty_card_ids(db_session: AsyncSession) -> None:
    assert await retention_by_card(db_session, card_ids=[]) == {}


# ---------- retrievability (forgetting curve) ----------


def test_retrievability_is_target_on_due_date() -> None:
    c = AnkiCard(anki_card_id=1, deck_name="d", interval_days=10, due_date=date(2026, 5, 27))
    assert retrievability(c, today=date(2026, 5, 27)) == pytest.approx(0.9)


def test_retrievability_higher_before_due_lower_when_overdue() -> None:
    c = AnkiCard(anki_card_id=1, deck_name="d", interval_days=10, due_date=date(2026, 5, 27))
    before = retrievability(c, today=date(2026, 5, 22))  # 5d early
    overdue = retrievability(c, today=date(2026, 6, 6))  # 10d late
    assert before > 0.9
    assert overdue < 0.9
    assert 0.0 <= overdue <= 1.0


def test_retrievability_none_without_interval() -> None:
    c = AnkiCard(anki_card_id=1, deck_name="d", interval_days=None, due_date=date(2026, 5, 27))
    assert retrievability(c) is None


# ---------- route ----------


@pytest.mark.asyncio
async def test_review_queue_route_carries_metrics(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    c = await _card(db_session, native_id=9_100, due=date.today(), interval=10)
    await _review(db_session, review_id=100, card_id=c.id, ease=3)
    await _review(db_session, review_id=101, card_id=c.id, ease=1)
    await db_session.commit()

    r = await client.get("/api/v1/anki/review-queue", headers=_AUTH)
    assert r.status_code == 200, r.text
    [row] = [x for x in r.json() if x["anki_card_id"] == 9_100]
    assert row["retention"] == pytest.approx(0.5)
    assert row["retrievability"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_review_queue_metrics_none_when_unscheduled_card_filtered(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # due_date NULL → excluded from queue entirely (no metrics row).
    await _card(db_session, native_id=9_200, due=None, interval=None)
    await db_session.commit()
    r = await client.get("/api/v1/anki/review-queue", headers=_AUTH)
    assert r.status_code == 200
    assert not [x for x in r.json() if x["anki_card_id"] == 9_200]
