"""Tests for SPEC T63 — `run_unlock_due` unlock scheduler core (V51, V55).

The scheduler wrapper (`run_anki_assignment_unlock_job`) is a thin
TaskRun-recording shell over `run_unlock_due`; we cover the wrapper
indirectly by exercising the inner function with a stub AnkiConnect
client. Stub matches the real client's exception types so the V55
failure → retry contract is enforced from the same code path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiWrite
from app.services.anki.assignment import run_unlock_due
from app.services.anki.client import (
    AnkiConnectError,
    AnkiUnreachableError,
    AnkiWriteFailed,
)


class _StubAnkiConnectClient:
    """Captures `unsuspend_cards` + `add_tags` (T75) calls and replays
    scripted side-effects.

    `responses` lists scripted `unsuspend_cards` outcomes — entries are
    either `True`/`False` (success) or `BaseException` (raise). Each call
    pops the next entry; running dry raises a loud `RuntimeError`.

    `add_tags_responses` is the same shape but for the addTags chain
    (T75). Default = always succeed silently — most tests don't care
    about the chain failure path. Tests that DO care script explicit
    entries.
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        add_tags_responses: list[Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._add_tags_responses = list(add_tags_responses or [])
        self.calls: list[list[int]] = []
        self.add_tags_calls: list[tuple[list[int], list[str]]] = []

    async def unsuspend_cards(self, card_ids: list[int]) -> bool:
        self.calls.append(list(card_ids))
        if not self._responses:
            raise RuntimeError("stub AnkiConnect ran out of scripted responses")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, BaseException):
            raise next_resp
        return bool(next_resp)

    async def add_tags(self, note_ids: list[int], tags: list[str]) -> None:
        self.add_tags_calls.append((list(note_ids), list(tags)))
        if not self._add_tags_responses:
            return  # default: silent success
        next_resp = self._add_tags_responses.pop(0)
        if isinstance(next_resp, BaseException):
            raise next_resp
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _make_pending(
    session: AsyncSession,
    *,
    scope_value: str = "4C",
    card_ids: list[int] | None = None,
    note_ids: list[int] | None = None,
    scheduled_unlock_at: datetime | None = None,
    failure_count: int = 0,
    status: str = "pending",
) -> AnkiAssignment:
    # §V75: the addTags chain targets `note_ids`. A real resolved assignment
    # snapshots both; here we mirror card_ids onto note_ids by default so the
    # chain has notes to tag (these are hand-built rows, no real cards/notes).
    resolved_card_ids = card_ids or [1735689600001, 1735689600002]
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value=scope_value,
        scheduled_unlock_at=scheduled_unlock_at or (_now() - timedelta(minutes=5)),
        card_ids=resolved_card_ids,
        note_ids=note_ids if note_ids is not None else resolved_card_ids,
        failure_count=failure_count,
        status=status,
    )
    session.add(a)
    await session.flush()
    return a


# --- happy path ---


async def test_unlock_due_flips_pending_to_unlocked_and_audits(
    db_session: AsyncSession,
) -> None:
    """T75: success path now writes TWO audit rows per assignment —
    one `unsuspend` + one `addTags` chained per V50."""
    assignment = await _make_pending(db_session)
    stub = _StubAnkiConnectClient(responses=[True])

    summary = await run_unlock_due(db_session, stub)

    assert summary.processed == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.terminal_failed == 0
    assert stub.calls == [[1735689600001, 1735689600002]]
    # T75: addTags chain fires with `coach::assignment:{id}` tag value.
    assert stub.add_tags_calls == [
        ([1735689600001, 1735689600002], [f"coach::assignment:{assignment.id}"])
    ]

    await db_session.refresh(assignment)
    assert assignment.status == "unlocked"
    assert assignment.actual_unlock_at is not None

    audits = list(
        (await db_session.execute(select(AnkiWrite).order_by(AnkiWrite.id.asc()))).scalars().all()
    )
    assert len(audits) == 2
    unsuspend_audit, addtags_audit = audits
    assert unsuspend_audit.action == "unsuspend"
    assert unsuspend_audit.status == "succeeded"
    assert unsuspend_audit.source == "scheduler"
    assert unsuspend_audit.assignment_id == assignment.id
    assert unsuspend_audit.response_json == {"result": True}
    assert addtags_audit.action == "addTags"
    assert addtags_audit.status == "succeeded"
    assert addtags_audit.source == "scheduler"
    assert addtags_audit.assignment_id == assignment.id
    assert addtags_audit.response_json == {"tags": [f"coach::assignment:{assignment.id}"]}


async def test_unlock_due_addtags_failure_does_not_revert_status(
    db_session: AsyncSession,
) -> None:
    """T75: addTags failure ⊥ revert `status='unlocked'` and ⊥ bump
    `failure_count`. The tag is an audit-only write-only marker per V50;
    unsuspend already succeeded so the assignment lifecycle holds.
    Audit rows: 1 succeeded `unsuspend` + 1 failed `addTags`."""
    assignment = await _make_pending(db_session)
    stub = _StubAnkiConnectClient(
        responses=[True],
        add_tags_responses=[AnkiUnreachableError("addTags transient")],
    )

    summary = await run_unlock_due(db_session, stub)

    # Unlock counts as succeeded — addTags failure is non-load-bearing.
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.terminal_failed == 0

    await db_session.refresh(assignment)
    assert assignment.status == "unlocked"
    assert assignment.actual_unlock_at is not None
    assert assignment.failure_count == 0
    assert assignment.error_text is None

    audits = list(
        (await db_session.execute(select(AnkiWrite).order_by(AnkiWrite.id.asc()))).scalars().all()
    )
    assert len(audits) == 2
    unsuspend_audit, addtags_audit = audits
    assert unsuspend_audit.action == "unsuspend"
    assert unsuspend_audit.status == "succeeded"
    assert addtags_audit.action == "addTags"
    assert addtags_audit.status == "failed"
    assert addtags_audit.error_text is not None
    assert "addTags transient" in addtags_audit.error_text


async def test_unlock_path_orthogonal_to_create_filtered_deck(
    db_session: AsyncSession,
) -> None:
    """V53 amended / T78: unlock op is `unsuspend` + `addTags` only —
    ⊥ `createFilteredDeck` under any path (filtered decks are the
    orthogonal review op, T76). This stub instruments every method
    the unlock service might call; the test asserts createFilteredDeck
    is never invoked, even on the success path that DOES chain addTags."""

    class _CFDInstrumented:
        def __init__(self) -> None:
            self.unsuspend_calls: list[list[int]] = []
            self.add_tags_calls: list[tuple[list[int], list[str]]] = []
            self.create_filtered_deck_calls: list[tuple[str, list[int]]] = []

        async def unsuspend_cards(self, card_ids: list[int]) -> bool:
            self.unsuspend_calls.append(list(card_ids))
            return True

        async def add_tags(self, note_ids: list[int], tags: list[str]) -> None:
            self.add_tags_calls.append((list(note_ids), list(tags)))

        async def create_filtered_deck(self, name: str, card_ids: list[int]) -> int:
            self.create_filtered_deck_calls.append((name, list(card_ids)))
            return 1

    assignment = await _make_pending(db_session)
    stub = _CFDInstrumented()
    summary = await run_unlock_due(db_session, stub)

    assert summary.succeeded == 1
    assert stub.unsuspend_calls == [list(assignment.card_ids or [])]
    # §V75: addTags chain targets the note_ids snapshot.
    assert stub.add_tags_calls == [
        (list(assignment.note_ids or []), [f"coach::assignment:{assignment.id}"])
    ]
    # V53 amend: unlock path is orthogonal to filtered-deck creation.
    assert stub.create_filtered_deck_calls == []


async def test_unlock_due_no_addtags_call_on_unsuspend_failure(
    db_session: AsyncSession,
) -> None:
    """T75: addTags chain only fires after a successful unsuspend. On
    unsuspend failure the chain is skipped — no audit row for addTags,
    no AnkiConnect call."""
    await _make_pending(db_session)
    stub = _StubAnkiConnectClient(responses=[AnkiUnreachableError("transient")])

    await run_unlock_due(db_session, stub)

    assert stub.add_tags_calls == []
    audits = list((await db_session.execute(select(AnkiWrite))).scalars().all())
    assert len(audits) == 1
    assert audits[0].action == "unsuspend"
    assert audits[0].status == "failed"


# --- skips ---


async def test_unlock_due_skips_future_pending(db_session: AsyncSession) -> None:
    future = _now() + timedelta(hours=2)
    assignment = await _make_pending(db_session, scheduled_unlock_at=future)
    stub = _StubAnkiConnectClient(responses=[])  # would fail if called

    summary = await run_unlock_due(db_session, stub)

    assert summary.processed == 0
    assert stub.calls == []
    await db_session.refresh(assignment)
    assert assignment.status == "pending"
    assert assignment.actual_unlock_at is None


@pytest.mark.parametrize("status", ["unlocked", "completed", "skipped", "failed"])
async def test_unlock_due_skips_non_pending(db_session: AsyncSession, status: str) -> None:
    assignment = await _make_pending(db_session, status=status)
    stub = _StubAnkiConnectClient(responses=[])

    summary = await run_unlock_due(db_session, stub)

    assert summary.processed == 0
    assert stub.calls == []
    await db_session.refresh(assignment)
    assert assignment.status == status


# --- failure semantics (V55) ---


async def test_unlock_due_single_failure_keeps_pending_and_increments_count(
    db_session: AsyncSession,
) -> None:
    assignment = await _make_pending(db_session)
    stub = _StubAnkiConnectClient(responses=[AnkiUnreachableError("Connection refused")])

    summary = await run_unlock_due(db_session, stub)

    assert summary.processed == 1
    assert summary.succeeded == 0
    assert summary.failed == 1
    assert summary.terminal_failed == 0

    await db_session.refresh(assignment)
    assert assignment.status == "pending"  # V55: status untouched on failure
    assert assignment.failure_count == 1
    assert assignment.actual_unlock_at is None
    assert assignment.error_text is None  # only set on terminal

    audit = (await db_session.execute(select(AnkiWrite))).scalar_one()
    assert audit.status == "failed"
    assert audit.source == "scheduler"
    assert audit.assignment_id == assignment.id
    assert audit.error_text is not None and "Connection refused" in audit.error_text
    assert audit.response_json is None


async def test_unlock_due_third_failure_marks_terminal_failed(
    db_session: AsyncSession,
) -> None:
    """V55: failure_count ≥ 3 → terminal status='failed'. Test starts at
    failure_count=2 so the next failure flips the state."""
    assignment = await _make_pending(db_session, failure_count=2)
    stub = _StubAnkiConnectClient(responses=[AnkiWriteFailed("deck not found")])

    summary = await run_unlock_due(db_session, stub)

    assert summary.failed == 1
    assert summary.terminal_failed == 1

    await db_session.refresh(assignment)
    assert assignment.status == "failed"
    assert assignment.failure_count == 3
    assert assignment.error_text is not None
    assert "deck not found" in assignment.error_text


@pytest.mark.parametrize(
    "exc",
    [
        AnkiUnreachableError("unreachable"),
        AnkiWriteFailed("anki returned error field"),
        AnkiConnectError("malformed body"),
    ],
    ids=["AnkiUnreachableError", "AnkiWriteFailed", "AnkiConnectError"],
)
async def test_unlock_due_maps_each_v55_failure_type(
    db_session: AsyncSession, exc: BaseException
) -> None:
    """Each of the three V55 failure exception types funnels to the same
    failure path — pending status preserved, count incremented, audit
    row written."""
    assignment = await _make_pending(db_session, scope_value=f"exc-{type(exc).__name__}")
    stub = _StubAnkiConnectClient(responses=[exc])
    summary = await run_unlock_due(db_session, stub)
    assert summary.failed == 1
    await db_session.refresh(assignment)
    assert assignment.status == "pending"
    assert assignment.failure_count == 1


# --- mixed batch ---


async def test_unlock_due_mixed_batch_processes_each_independently(
    db_session: AsyncSession,
) -> None:
    """A mid-batch failure must not roll back earlier-in-loop success."""
    early_success = await _make_pending(
        db_session,
        scope_value="early",
        scheduled_unlock_at=_now() - timedelta(minutes=10),
    )
    later_failure = await _make_pending(
        db_session,
        scope_value="later",
        scheduled_unlock_at=_now() - timedelta(minutes=5),
    )
    even_later_success = await _make_pending(
        db_session,
        scope_value="even-later",
        scheduled_unlock_at=_now() - timedelta(minutes=1),
    )

    stub = _StubAnkiConnectClient(
        responses=[
            True,
            AnkiUnreachableError("transient"),
            True,
        ]
    )

    summary = await run_unlock_due(db_session, stub)

    assert summary.processed == 3
    assert summary.succeeded == 2
    assert summary.failed == 1

    await db_session.refresh(early_success)
    await db_session.refresh(later_failure)
    await db_session.refresh(even_later_success)
    assert early_success.status == "unlocked"
    assert later_failure.status == "pending"
    assert later_failure.failure_count == 1
    assert even_later_success.status == "unlocked"

    audits = list((await db_session.execute(select(AnkiWrite))).scalars().all())
    # T75: each success now produces 2 audit rows (unsuspend + addTags);
    # the single failure still produces 1 audit row (no addTags chain on
    # unsuspend failure).
    assert len(audits) == 5
    by_assignment: dict[int, list[AnkiWrite]] = {}
    for a in audits:
        by_assignment.setdefault(a.assignment_id, []).append(a)
    # Two audit rows per successful assignment — one unsuspend, one addTags.
    for aid in (early_success.id, even_later_success.id):
        rows = by_assignment[aid]
        assert len(rows) == 2
        assert {r.action for r in rows} == {"unsuspend", "addTags"}
        assert {r.status for r in rows} == {"succeeded"}
    # Failed unsuspend has 1 row, no addTags chain.
    failed_rows = by_assignment[later_failure.id]
    assert len(failed_rows) == 1
    assert failed_rows[0].action == "unsuspend"
    assert failed_rows[0].status == "failed"


async def test_unlock_due_now_override_changes_due_filter(
    db_session: AsyncSession,
) -> None:
    """Caller may supply `now=` to control the eligibility cutoff (used
    by tests for deterministic time + by the scheduler if it ever needs
    to back-process)."""
    future_assignment = await _make_pending(
        db_session, scheduled_unlock_at=_now() + timedelta(hours=1)
    )
    stub = _StubAnkiConnectClient(responses=[True])
    # Override `now` to a moment after the scheduled time → assignment becomes due.
    summary = await run_unlock_due(db_session, stub, now=_now() + timedelta(hours=2))
    assert summary.succeeded == 1
    await db_session.refresh(future_assignment)
    assert future_assignment.status == "unlocked"


# --- admin / scheduler wiring ---


def test_run_anki_assignment_unlock_in_valid_jobs() -> None:
    """Admin `/jobs/{name}/trigger` plumbing relies on the job id appearing
    in the API `_VALID_JOBS` set."""
    from app.api.v1.admin import _VALID_JOBS

    assert "run_anki_assignment_unlock" in _VALID_JOBS
