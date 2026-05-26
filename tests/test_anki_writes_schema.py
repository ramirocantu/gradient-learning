"""Schema tests for SPEC T59 — anki_writes append-only audit (V50, V55).

Validates the audit-log shape at storage layer: CHECK on status enum +
source enum, both FKs (assignment_id, review_id — renamed per T76)
SET NULL on parent delete so audit history outlives its referent, and
the occurred_at index needed by /admin's recent-writes scan.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiReview, AnkiWrite


def _later() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=1)


async def _make_assignment(session: AsyncSession) -> AnkiAssignment:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=_later(),
        card_ids=[1001, 1002],
    )
    session.add(a)
    await session.flush()
    return a


async def _make_review(session: AsyncSession) -> AnkiReview:
    r = AnkiReview(
        review_date=date(2026, 5, 27),
        card_ids=[1],
        deck_name="mcat-coach::review::audit-link",
    )
    session.add(r)
    await session.flush()
    return r


# --- V55: status enum ---


@pytest.mark.parametrize("good", ["succeeded", "failed"])
async def test_status_check_accepts(db_session: AsyncSession, good: str) -> None:
    db_session.add(
        AnkiWrite(
            action="unsuspend",
            payload_hash="deadbeef",
            status=good,
            source="scheduler",
        )
    )
    await db_session.flush()


async def test_status_check_rejects_unknown(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiWrite(
            action="unsuspend",
            payload_hash="x",
            status="pending",
            source="scheduler",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- V50: source enum ---


@pytest.mark.parametrize("good", ["mcp", "scheduler", "manual", "test"])
async def test_source_check_accepts(db_session: AsyncSession, good: str) -> None:
    db_session.add(
        AnkiWrite(
            action="addTags",
            payload_hash=f"hash-{good}",
            status="succeeded",
            source=good,
        )
    )
    await db_session.flush()


async def test_source_check_rejects_unknown(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiWrite(
            action="addTags",
            payload_hash="x",
            status="succeeded",
            source="cli",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- FK behavior: SET NULL preserves audit row when parent is deleted ---


async def test_assignment_fk_set_null_on_delete(db_session: AsyncSession) -> None:
    a = await _make_assignment(db_session)
    db_session.add(
        AnkiWrite(
            action="unsuspend",
            payload_hash="audit-1",
            status="succeeded",
            source="scheduler",
            assignment_id=a.id,
        )
    )
    await db_session.flush()
    await db_session.execute(text("DELETE FROM anki_assignments WHERE id = :id"), {"id": a.id})
    await db_session.flush()
    row = (await db_session.execute(select(AnkiWrite))).scalar_one()
    assert row.assignment_id is None
    # audit row itself MUST survive parent removal
    assert row.action == "unsuspend"


async def test_review_fk_set_null_on_delete(db_session: AsyncSession) -> None:
    r = await _make_review(db_session)
    db_session.add(
        AnkiWrite(
            action="createFilteredDeck",
            payload_hash="audit-2",
            status="succeeded",
            source="scheduler",
            review_id=r.id,
        )
    )
    await db_session.flush()
    await db_session.execute(text("DELETE FROM anki_reviews WHERE id = :id"), {"id": r.id})
    await db_session.flush()
    row = (await db_session.execute(select(AnkiWrite))).scalar_one()
    assert row.review_id is None
    assert row.action == "createFilteredDeck"


# --- JSONB response storage ---


async def test_response_json_roundtrip(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiWrite(
            action="addTags",
            payload_hash="resp",
            status="succeeded",
            source="manual",
            response_json={"result": None, "error": None, "echo": [1, 2, 3]},
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiWrite))).scalar_one()
    assert row.response_json == {"result": None, "error": None, "echo": [1, 2, 3]}


async def test_response_json_nullable_on_unreachable(
    db_session: AsyncSession,
) -> None:
    """AnkiUnreachableError path: no body to record; response_json stays NULL,
    error_text carries the message instead."""
    db_session.add(
        AnkiWrite(
            action="unsuspend",
            payload_hash="x",
            status="failed",
            source="scheduler",
            error_text="anki_not_running",
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiWrite))).scalar_one()
    assert row.response_json is None
    assert row.error_text == "anki_not_running"


# --- defaults + index ---


async def test_occurred_at_default_now(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiWrite(
            action="unsuspend",
            payload_hash="d",
            status="succeeded",
            source="test",
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiWrite))).scalar_one()
    assert row.occurred_at is not None
    assert (datetime.now(timezone.utc) - row.occurred_at) < timedelta(seconds=30)


async def test_occurred_at_index_exists(db_session: AsyncSession) -> None:
    def _check(sync_conn) -> list[dict]:
        return inspect(sync_conn).get_indexes("anki_writes")

    conn = await db_session.connection()
    indexes = await conn.run_sync(_check)
    names = {ix["name"] for ix in indexes}
    assert "ix_anki_writes_occurred_at" in names
