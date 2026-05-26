"""Schema tests for SPEC T58 — anki_assignments table (V51).

Validates the V51 state machine + V52 card-id snapshot at the storage
layer: CHECK on status enum + scope_kind enum + max_cards positivity,
and the two indexes that back the scheduler scans
(status+scheduled, actual_unlock).

T79 / V61 dropped the `study_plan_item_id` FK + its index when the
Phase 7 study plan layer was cut.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment


def _later() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=1)


# --- V51: status state machine encoded as CHECK ---


@pytest.mark.parametrize(
    "good_status",
    ["pending", "unlocked", "completed", "skipped", "failed"],
)
async def test_status_check_accepts_v51_states(db_session: AsyncSession, good_status: str) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="cc",
            scope_value="4C",
            scheduled_unlock_at=_later(),
            card_ids=[1001, 1002],
            status=good_status,
        )
    )
    await db_session.flush()


async def test_status_check_rejects_unknown(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="cc",
            scope_value="4C",
            scheduled_unlock_at=_later(),
            card_ids=[1001],
            status="bogus",
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- V52: scope_kind enum ---


@pytest.mark.parametrize("good_scope", ["cc", "topic"])
async def test_scope_kind_accepts_cc_or_topic(db_session: AsyncSession, good_scope: str) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind=good_scope,
            scope_value="x",
            scheduled_unlock_at=_later(),
            card_ids=[1],
        )
    )
    await db_session.flush()


async def test_scope_kind_rejects_other(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="section",
            scope_value="CP",
            scheduled_unlock_at=_later(),
            card_ids=[1],
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- V52: max_cards positivity (NULL allowed = unbounded) ---


async def test_max_cards_null_allowed(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="cc",
            scope_value="4C",
            scheduled_unlock_at=_later(),
            card_ids=[1],
            max_cards=None,
        )
    )
    await db_session.flush()


@pytest.mark.parametrize("bad", [0, -1, -100])
async def test_max_cards_non_positive_rejected(db_session: AsyncSession, bad: int) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="cc",
            scope_value="4C",
            scheduled_unlock_at=_later(),
            card_ids=[1],
            max_cards=bad,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- defaults + array column ---


async def test_defaults_status_pending_priority_most_specific(
    db_session: AsyncSession,
) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="cc",
            scope_value="4C",
            scheduled_unlock_at=_later(),
            card_ids=[1, 2, 3],
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiAssignment))).scalar_one()
    assert row.status == "pending"
    assert row.priority == "most_specific_first"
    assert row.failure_count == 0
    assert row.actual_unlock_at is None


async def test_card_ids_array_roundtrip(db_session: AsyncSession) -> None:
    db_session.add(
        AnkiAssignment(
            scope_kind="cc",
            scope_value="4C",
            scheduled_unlock_at=_later(),
            card_ids=[42, 7, 1735689600000],  # mixed-magnitude ids
        )
    )
    await db_session.flush()
    row = (await db_session.execute(select(AnkiAssignment))).scalar_one()
    assert row.card_ids == [42, 7, 1735689600000]


# --- indexes ---


async def test_required_indexes_present(db_session: AsyncSession) -> None:
    def _check(sync_conn) -> list[dict]:
        return inspect(sync_conn).get_indexes("anki_assignments")

    conn = await db_session.connection()
    indexes = await conn.run_sync(_check)
    names = {ix["name"]: ix["column_names"] for ix in indexes}
    assert names.get("ix_anki_assignments_status_scheduled") == [
        "status",
        "scheduled_unlock_at",
    ]
    assert names.get("ix_anki_assignments_actual_unlock") == ["actual_unlock_at"]
    assert "ix_anki_assignments_study_plan_item" not in names, (
        "T79 / V61 dropped this index when Phase 7 was cut"
    )
