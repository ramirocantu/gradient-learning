"""Tests for SPEC T76 — review service + scheduler core (V53 amended,
V55, V50 addTags chain).

`create_review` covers the standalone create + deck-name-from-PK
contract; `run_review_due` covers the V53/V55 scheduler loop plus the
T76 V50 addTags chain (parallels T75's unlock chain). Stub AnkiConnect
client matches the real client's exception types so V55 retry semantics
flow through the same code path as T63/T65.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiReview, AnkiWrite
from app.services.anki.client import (
    AnkiConnectError,
    AnkiUnreachableError,
    AnkiWriteFailed,
)
from app.services.anki.review import (
    create_review,
    run_review_due,
)


_PREFIX = "mcat-coach"


async def _seed_notes_for_cards(session: AsyncSession, native_ids: list[int]) -> None:
    """§V75: `create_review` derives `note_ids` from the anki_cards lookup, and
    the push-time addTags audit write targets those notes. Seed a note + a
    card (note_id == anki_card_id for 1:1) for each native id so the lookup
    resolves a note to tag. Cards not seeded simply contribute no note."""
    for nid in native_ids:
        session.add(AnkiNote(note_id=nid, deck_name="MileDown"))
        await session.flush()
        session.add(AnkiCard(anki_card_id=nid, deck_name="MileDown", note_id=nid, queue=-1))
    await session.flush()


class _StubAnkiConnectClient:
    """Captures `create_filtered_deck` + `add_tags` calls and replays
    scripted side-effects per V50 + V53.

    `responses` scripts create_filtered_deck outcomes (int = deck_id;
    BaseException = raise). `add_tags_responses` scripts addTags; default
    silent-success matches T76's audit-only chain contract."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        add_tags_responses: list[Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._add_tags_responses = list(add_tags_responses or [])
        self.calls: list[tuple[str, list[int]]] = []
        self.add_tags_calls: list[tuple[list[int], list[str]]] = []

    async def create_filtered_deck(self, name: str, card_ids: list[int]) -> int:
        self.calls.append((name, list(card_ids)))
        if not self._responses:
            raise RuntimeError("stub AnkiConnect ran out of scripted responses")
        next_resp = self._responses.pop(0)
        if isinstance(next_resp, BaseException):
            raise next_resp
        return int(next_resp)

    async def add_tags(self, note_ids: list[int], tags: list[str]) -> None:
        self.add_tags_calls.append((list(note_ids), list(tags)))
        if not self._add_tags_responses:
            return
        next_resp = self._add_tags_responses.pop(0)
        if isinstance(next_resp, BaseException):
            raise next_resp
        return None


# --------------------------- create_review (V53 amended) --------------------------- #


async def test_create_review_deck_name_uses_review_id(db_session: AsyncSession) -> None:
    """V53 amended: deck name = `<prefix>::review::{review.id}`. The
    two-flush dance assigns the PK first, then sets deck_name."""
    review = await create_review(
        db_session,
        card_ids=[1_700_000_500_001, 1_700_000_500_002],
        review_date=date(2026, 5, 24),
        write_deck_prefix=_PREFIX,
    )
    assert review.id is not None
    assert review.deck_name == f"{_PREFIX}::review::{review.id}"
    assert review.review_date == date(2026, 5, 24)
    assert review.status == "pending"
    assert review.card_ids == [1_700_000_500_001, 1_700_000_500_002]


async def test_create_review_dup_same_day_same_cards_allowed(
    db_session: AsyncSession,
) -> None:
    """V53 amended: no UNIQUE constraint on (review_date, *). Two
    consecutive create_review calls w/ identical args yield two distinct
    rows w/ distinct deck names. Tags-as-log accepts the dup; idempotency
    lives in UI debounce."""
    a = await create_review(
        db_session,
        card_ids=[1_700_000_600_001],
        review_date=date(2026, 5, 25),
        write_deck_prefix=_PREFIX,
    )
    b = await create_review(
        db_session,
        card_ids=[1_700_000_600_001],
        review_date=date(2026, 5, 25),
        write_deck_prefix=_PREFIX,
    )
    assert a.id != b.id
    assert a.deck_name != b.deck_name
    assert a.deck_name.endswith(f"::{a.id}")
    assert b.deck_name.endswith(f"::{b.id}")


# ------------------------ run_review_due (V53, V55, V50) ------------------------ #


async def test_run_review_due_creates_filtered_deck_and_audits(
    db_session: AsyncSession,
) -> None:
    """T76: success path writes TWO audit rows — createFilteredDeck +
    addTags — both keyed on review_id with tag value `coach::review:{id}`.
    §V75: addTags targets the notes backing the review's cards."""
    await _seed_notes_for_cards(db_session, [1_700_000_700_001, 1_700_000_700_002])
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_001, 1_700_000_700_002],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )
    stub = _StubAnkiConnectClient(responses=[1_700_000_800_000])

    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))

    assert summary.processed == 1
    assert summary.pushed == 1
    assert summary.failed == 0
    assert stub.calls == [(review.deck_name, [1_700_000_700_001, 1_700_000_700_002])]
    # §V75: note_ids == card_ids here (1:1 note↔card seeding above).
    assert stub.add_tags_calls == [(list(review.note_ids), [f"coach::review:{review.id}"])]
    assert set(review.note_ids) == {1_700_000_700_001, 1_700_000_700_002}

    await db_session.refresh(review)
    assert review.status == "pushed"
    assert review.pushed_at is not None

    audits = list(
        (await db_session.execute(select(AnkiWrite).order_by(AnkiWrite.id.asc()))).scalars().all()
    )
    assert len(audits) == 2
    cf_audit, tag_audit = audits
    assert cf_audit.action == "createFilteredDeck"
    assert cf_audit.status == "succeeded"
    assert cf_audit.review_id == review.id
    assert cf_audit.response_json == {"deck_id": 1_700_000_800_000}
    assert tag_audit.action == "addTags"
    assert tag_audit.status == "succeeded"
    assert tag_audit.review_id == review.id
    assert tag_audit.response_json == {"tags": [f"coach::review:{review.id}"]}


async def test_run_review_due_addtags_failure_does_not_revert_status(
    db_session: AsyncSession,
) -> None:
    """T76 (parallels T75): addTags failure ⊥ revert `status='pushed'`
    and ⊥ bump `failure_count`. Audit rows: 1 succeeded createFilteredDeck
    + 1 failed addTags."""
    await _seed_notes_for_cards(db_session, [1_700_000_700_010])
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_010],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )
    stub = _StubAnkiConnectClient(
        responses=[1_700_000_800_010],
        add_tags_responses=[AnkiUnreachableError("addTags transient")],
    )

    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))
    assert summary.pushed == 1
    assert summary.failed == 0

    await db_session.refresh(review)
    assert review.status == "pushed"
    assert review.failure_count == 0
    assert review.error_text is None

    audits = list(
        (await db_session.execute(select(AnkiWrite).order_by(AnkiWrite.id.asc()))).scalars().all()
    )
    assert [(a.action, a.status) for a in audits] == [
        ("createFilteredDeck", "succeeded"),
        ("addTags", "failed"),
    ]
    assert audits[1].error_text is not None and "addTags transient" in audits[1].error_text


async def test_review_path_orthogonal_to_unsuspend(
    db_session: AsyncSession,
) -> None:
    """V53 amended / T78: review op is `createFilteredDeck` + `addTags`
    only — ⊥ `unsuspend_cards` under any path (unsuspend is the
    orthogonal unlock op, T75). Stub instruments every method the
    review service might call; the test asserts unsuspend_cards is
    never invoked, even on the success path that DOES chain addTags."""

    class _UnsuspendInstrumented:
        def __init__(self) -> None:
            self.create_filtered_deck_calls: list[tuple[str, list[int]]] = []
            self.add_tags_calls: list[tuple[list[int], list[str]]] = []
            self.unsuspend_calls: list[list[int]] = []

        async def create_filtered_deck(self, name: str, card_ids: list[int]) -> int:
            self.create_filtered_deck_calls.append((name, list(card_ids)))
            return 1_700_000_900_000

        async def add_tags(self, note_ids: list[int], tags: list[str]) -> None:
            self.add_tags_calls.append((list(note_ids), list(tags)))

        async def unsuspend_cards(self, card_ids: list[int]) -> bool:
            self.unsuspend_calls.append(list(card_ids))
            return True

    await _seed_notes_for_cards(db_session, [1_700_000_700_500])
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_500],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )
    stub = _UnsuspendInstrumented()
    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))

    assert summary.pushed == 1
    assert stub.create_filtered_deck_calls == [(review.deck_name, [1_700_000_700_500])]
    # §V75: addTags targets the note backing the card (note_id == card id 1:1).
    assert stub.add_tags_calls == [(list(review.note_ids), [f"coach::review:{review.id}"])]
    assert review.note_ids == [1_700_000_700_500]
    # V53 amend: review path is orthogonal to unsuspend.
    assert stub.unsuspend_calls == []


async def test_run_review_due_no_addtags_on_createfiltered_deck_failure(
    db_session: AsyncSession,
) -> None:
    """T76: addTags chain only fires after successful createFilteredDeck.
    On createFilteredDeck failure the chain is skipped — no addTags
    audit row, no AnkiConnect call."""
    await create_review(
        db_session,
        card_ids=[1_700_000_700_020],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )
    stub = _StubAnkiConnectClient(responses=[AnkiUnreachableError("transient")])

    await run_review_due(db_session, stub, today=date(2026, 5, 22))

    assert stub.add_tags_calls == []
    audits = list((await db_session.execute(select(AnkiWrite))).scalars().all())
    assert len(audits) == 1
    assert audits[0].action == "createFilteredDeck"
    assert audits[0].status == "failed"


async def test_run_review_due_skips_future_review_date(
    db_session: AsyncSession,
) -> None:
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_030],
        review_date=date(2026, 6, 1),
        write_deck_prefix=_PREFIX,
    )
    stub = _StubAnkiConnectClient(responses=[])  # would fail if called

    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))
    assert summary.processed == 0
    assert stub.calls == []
    await db_session.refresh(review)
    assert review.status == "pending"


@pytest.mark.parametrize("status", ["pushed", "failed"])
async def test_run_review_due_skips_non_pending(db_session: AsyncSession, status: str) -> None:
    review = AnkiReview(
        review_date=date(2026, 5, 20),
        card_ids=[1_700_000_700_040],
        deck_name=f"{_PREFIX}::review::nonpending-{status}",
        status=status,
    )
    db_session.add(review)
    await db_session.flush()
    stub = _StubAnkiConnectClient(responses=[])

    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))
    assert summary.processed == 0
    assert stub.calls == []


async def test_run_review_due_third_failure_marks_terminal_failed(
    db_session: AsyncSession,
) -> None:
    """V55: failure_count ≥ 3 → terminal status='failed'. Failures here
    are createFilteredDeck failures (load-bearing); addTags chain ⊥
    contributes."""
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_050],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )
    # Set failure_count=2 so the next failure flips terminal.
    review.failure_count = 2
    await db_session.flush()

    stub = _StubAnkiConnectClient(responses=[AnkiWriteFailed("deck busy")])
    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))

    assert summary.failed == 1
    assert summary.terminal_failed == 1

    await db_session.refresh(review)
    assert review.status == "failed"
    assert review.failure_count == 3
    assert review.error_text is not None and "deck busy" in review.error_text


@pytest.mark.parametrize(
    "exc",
    [
        AnkiUnreachableError("unreachable"),
        AnkiWriteFailed("anki error field"),
        AnkiConnectError("malformed body"),
    ],
    ids=["AnkiUnreachableError", "AnkiWriteFailed", "AnkiConnectError"],
)
async def test_run_review_due_maps_each_v55_failure_type(
    db_session: AsyncSession, exc: BaseException
) -> None:
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_060],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )
    stub = _StubAnkiConnectClient(responses=[exc])
    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))
    assert summary.failed == 1
    await db_session.refresh(review)
    assert review.status == "pending"
    assert review.failure_count == 1


async def test_run_review_due_mixed_batch_independent(
    db_session: AsyncSession,
) -> None:
    """Mid-batch failure must not roll back earlier-in-loop success;
    each success produces 2 audit rows (createFilteredDeck + addTags)."""
    await _seed_notes_for_cards(db_session, [1, 2, 3])
    early = await create_review(
        db_session,
        card_ids=[1],
        review_date=date(2026, 5, 20),
        write_deck_prefix=_PREFIX,
    )
    middle = await create_review(
        db_session,
        card_ids=[2],
        review_date=date(2026, 5, 21),
        write_deck_prefix=_PREFIX,
    )
    later = await create_review(
        db_session,
        card_ids=[3],
        review_date=date(2026, 5, 22),
        write_deck_prefix=_PREFIX,
    )

    stub = _StubAnkiConnectClient(
        responses=[1_700_000_900_001, AnkiUnreachableError("transient"), 1_700_000_900_003]
    )
    summary = await run_review_due(db_session, stub, today=date(2026, 5, 22))

    assert summary.processed == 3
    assert summary.pushed == 2
    assert summary.failed == 1

    await db_session.refresh(early)
    await db_session.refresh(middle)
    await db_session.refresh(later)
    assert early.status == "pushed"
    assert middle.status == "pending"  # V55 — single failure keeps pending
    assert middle.failure_count == 1
    assert later.status == "pushed"

    audits = list((await db_session.execute(select(AnkiWrite))).scalars().all())
    # 2 successes × 2 audit rows (createFilteredDeck + addTags) + 1 failure × 1 row = 5.
    assert len(audits) == 5


async def test_run_review_due_today_defaults_to_wall_clock(
    db_session: AsyncSession,
) -> None:
    """`today` arg defaults to UTC wall-clock when omitted — the caller
    pattern from the scheduler wrapper."""
    yesterday = date.today() - timedelta(days=1)
    review = await create_review(
        db_session,
        card_ids=[1_700_000_700_080],
        review_date=yesterday,
        write_deck_prefix=_PREFIX,
    )
    stub = _StubAnkiConnectClient(responses=[1_700_000_910_000])
    summary = await run_review_due(db_session, stub)
    assert summary.pushed == 1
    await db_session.refresh(review)
    assert review.status == "pushed"


# --- admin / scheduler wiring ---


def test_run_anki_review_in_valid_jobs() -> None:
    """Admin `/jobs/{name}/trigger` plumbing relies on the job id appearing
    in the API `_VALID_JOBS` set."""
    from app.api.v1.admin import _VALID_JOBS

    assert "run_anki_review" in _VALID_JOBS
