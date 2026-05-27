"""Tests for SPEC T64 — `run_complete_unlocked` auto-completion (V51).

Universal-quantifier check: every AnKing-native card_id in an unlocked
assignment must have at least one `anki_card_reviews` row with
`reviewed_at > actual_unlock_at` for the assignment to flip to
`completed`. The join goes through `anki_cards` to bridge native ids
(stored on assignments per V52/B11) and local SERIAL ids (used in the
reviews FK).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiCard, AnkiCardReview
from app.services.anki.assignment import run_complete_unlocked


_NATIVE_BASE = 1_700_000_000_001


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _make_anki_card(session: AsyncSession, *, native_id: int) -> AnkiCard:
    card = AnkiCard(
        anki_card_id=native_id,
        deck_name="MileDown",
        queue=0,  # already unsuspended by an unlock
    )
    session.add(card)
    await session.flush()
    return card


async def _make_review(
    session: AsyncSession,
    *,
    review_id: int,
    card: AnkiCard,
    reviewed_at: datetime,
    ease: int = 3,
    type: str = "review",
) -> AnkiCardReview:
    r = AnkiCardReview(
        review_id=review_id,
        card_id=card.id,
        reviewed_at=_utc(reviewed_at),
        ease=ease,
        type=type,
    )
    session.add(r)
    await session.flush()
    return r


async def _make_unlocked_assignment(
    session: AsyncSession,
    *,
    native_card_ids: list[int],
    actual_unlock_at: datetime,
    scope_value: str = "4C",
) -> AnkiAssignment:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value=scope_value,
        scheduled_unlock_at=actual_unlock_at - timedelta(hours=1),
        actual_unlock_at=_utc(actual_unlock_at),
        card_ids=native_card_ids,
        status="unlocked",
    )
    session.add(a)
    await session.flush()
    return a


# --- happy path ---


async def test_complete_flips_when_all_cards_reviewed_after_unlock(
    db_session: AsyncSession,
) -> None:
    unlock_at = _now() - timedelta(days=3)
    n1, n2 = _NATIVE_BASE, _NATIVE_BASE + 1
    c1 = await _make_anki_card(db_session, native_id=n1)
    c2 = await _make_anki_card(db_session, native_id=n2)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n1, n2],
        actual_unlock_at=unlock_at,
    )
    # Both cards reviewed AFTER actual_unlock_at.
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 100,
        card=c1,
        reviewed_at=unlock_at + timedelta(hours=2),
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 101,
        card=c2,
        reviewed_at=unlock_at + timedelta(days=1),
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.processed == 1
    assert summary.completed == 1
    assert summary.still_unlocked == 0

    await db_session.refresh(assignment)
    assert assignment.status == "completed"


# --- partial / edge cases ---


async def test_complete_stays_unlocked_when_one_card_not_reviewed(
    db_session: AsyncSession,
) -> None:
    unlock_at = _now() - timedelta(days=1)
    n1, n2 = _NATIVE_BASE + 10, _NATIVE_BASE + 11
    c1 = await _make_anki_card(db_session, native_id=n1)
    _c2 = await _make_anki_card(db_session, native_id=n2)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n1, n2],
        actual_unlock_at=unlock_at,
    )
    # Only c1 has a qualifying review.
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 200,
        card=c1,
        reviewed_at=unlock_at + timedelta(hours=4),
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.processed == 1
    assert summary.completed == 0
    assert summary.still_unlocked == 1

    await db_session.refresh(assignment)
    assert assignment.status == "unlocked"


async def test_review_at_or_before_unlock_does_not_satisfy(
    db_session: AsyncSession,
) -> None:
    """V51 requires strict `reviewed_at > actual_unlock_at`. Reviews at or
    before the unlock time were done during the old (pre-unlock) suspended
    lifecycle and must not count."""
    unlock_at = _now() - timedelta(hours=12)
    n1 = _NATIVE_BASE + 20
    c1 = await _make_anki_card(db_session, native_id=n1)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n1],
        actual_unlock_at=unlock_at,
    )
    # Review BEFORE unlock and another EXACTLY at unlock.
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 300,
        card=c1,
        reviewed_at=unlock_at - timedelta(hours=1),
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 301,
        card=c1,
        reviewed_at=unlock_at,
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.completed == 0
    assert summary.still_unlocked == 1
    await db_session.refresh(assignment)
    assert assignment.status == "unlocked"


async def test_empty_card_ids_vacuously_completes(
    db_session: AsyncSession,
) -> None:
    """V51: ∀-quantifier over an empty set is vacuously true."""
    unlock_at = _now() - timedelta(hours=1)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[],
        actual_unlock_at=unlock_at,
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.completed == 1
    await db_session.refresh(assignment)
    assert assignment.status == "completed"


# --- non-unlocked statuses are skipped ---


@pytest.mark.parametrize("status", ["pending", "completed", "skipped", "failed"])
async def test_skip_non_unlocked_statuses(db_session: AsyncSession, status: str) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=_now() - timedelta(hours=2),
        actual_unlock_at=_now() - timedelta(hours=1) if status != "pending" else None,
        card_ids=[_NATIVE_BASE + 30],
        status=status,
    )
    db_session.add(a)
    await db_session.flush()

    summary = await run_complete_unlocked(db_session)

    assert summary.processed == 0
    await db_session.refresh(a)
    assert a.status == status


# --- idempotency ---


async def test_re_run_on_already_completed_is_noop(
    db_session: AsyncSession,
) -> None:
    unlock_at = _now() - timedelta(days=2)
    n1 = _NATIVE_BASE + 40
    c1 = await _make_anki_card(db_session, native_id=n1)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n1],
        actual_unlock_at=unlock_at,
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 400,
        card=c1,
        reviewed_at=unlock_at + timedelta(hours=2),
    )

    first = await run_complete_unlocked(db_session)
    assert first.completed == 1

    second = await run_complete_unlocked(db_session)
    # On re-run there are no unlocked rows left → processed=0.
    assert second.processed == 0
    assert second.completed == 0

    await db_session.refresh(assignment)
    assert assignment.status == "completed"


# --- duplicate card_ids in snapshot do not over-count ---


async def test_duplicate_card_ids_in_snapshot_dedup_correctly(
    db_session: AsyncSession,
) -> None:
    """A single review on a duplicated id must still satisfy the quantifier
    (uniqueness is via DISTINCT in the SQL)."""
    unlock_at = _now() - timedelta(hours=6)
    n1, n2 = _NATIVE_BASE + 50, _NATIVE_BASE + 51
    c1 = await _make_anki_card(db_session, native_id=n1)
    c2 = await _make_anki_card(db_session, native_id=n2)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n1, n1, n2],  # duplicate of n1
        actual_unlock_at=unlock_at,
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 500,
        card=c1,
        reviewed_at=unlock_at + timedelta(minutes=30),
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 501,
        card=c2,
        reviewed_at=unlock_at + timedelta(minutes=45),
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.completed == 1
    await db_session.refresh(assignment)
    assert assignment.status == "completed"


# --- mixed batch ---


async def test_mixed_batch_completes_only_fully_reviewed(
    db_session: AsyncSession,
) -> None:
    unlock_at = _now() - timedelta(hours=4)
    n_done = _NATIVE_BASE + 60
    n_partial_a, n_partial_b = _NATIVE_BASE + 61, _NATIVE_BASE + 62
    c_done = await _make_anki_card(db_session, native_id=n_done)
    c_pa = await _make_anki_card(db_session, native_id=n_partial_a)
    _c_pb = await _make_anki_card(db_session, native_id=n_partial_b)

    fully = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n_done],
        actual_unlock_at=unlock_at,
        scope_value="fully",
    )
    partial = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[n_partial_a, n_partial_b],
        actual_unlock_at=unlock_at,
        scope_value="partial",
    )

    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 600,
        card=c_done,
        reviewed_at=unlock_at + timedelta(hours=1),
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 601,
        card=c_pa,
        reviewed_at=unlock_at + timedelta(hours=1),
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.processed == 2
    assert summary.completed == 1
    assert summary.still_unlocked == 1

    await db_session.refresh(fully)
    await db_session.refresh(partial)
    assert fully.status == "completed"
    assert partial.status == "unlocked"


# --- review on unsynced anki_cards row stalls completion ---


async def test_native_id_without_anki_cards_row_stalls(
    db_session: AsyncSession,
) -> None:
    """If the snapshot references a native id that has not been synced
    into `anki_cards` yet, the join finds zero matching reviews and the
    assignment stays unlocked. Sync (T36) eventually populates the row
    and the next run completes it."""
    unlock_at = _now() - timedelta(hours=3)
    synced_native = _NATIVE_BASE + 70
    unsynced_native = _NATIVE_BASE + 71  # No anki_cards row created.
    c_synced = await _make_anki_card(db_session, native_id=synced_native)
    assignment = await _make_unlocked_assignment(
        db_session,
        native_card_ids=[synced_native, unsynced_native],
        actual_unlock_at=unlock_at,
    )
    await _make_review(
        db_session,
        review_id=_NATIVE_BASE + 700,
        card=c_synced,
        reviewed_at=unlock_at + timedelta(minutes=15),
    )

    summary = await run_complete_unlocked(db_session)

    assert summary.completed == 0
    assert summary.still_unlocked == 1
    await db_session.refresh(assignment)
    assert assignment.status == "unlocked"


# --- admin / scheduler wiring ---


def test_run_anki_assignment_complete_in_valid_jobs() -> None:
    from app.api.v1.admin import _VALID_JOBS

    assert "run_anki_assignment_complete" in _VALID_JOBS
