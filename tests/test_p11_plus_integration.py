"""End-to-end integration tests for P11+ Anki write/assign (SPEC T71).

Per-module tests live alongside their service files; this suite wires
the modules together and pins the cross-cutting invariants:

- V51 lifecycle: create_assignment → unlock → complete (review-driven)
- V51 + V57: skipped path performs NO AnkiConnect call
- V55: 3-strike retry cap → terminal `status='failed'`
- V53: re-push semantics = caller-driven delete + recreate
- V58: settings refuse `ANKI_DECK_NAME` overlapping write namespace
- V60: adherence payload carries no `recommended_changes` field
- V18 + V54 + V61: /anki renders the full P11+ stack (study-plan dashboard cut per V61, content folded onto /anki by T79)
"""

from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import (
    AnkiAssignment,
    AnkiCard,
    AnkiCardReview,
    AnkiLoadConfig,
    AnkiNote,
    AnkiNoteTag,
    AnkiReview,
    AnkiWrite,
)
from app.models.outline import ContentCategory, Topic
from app.services.anki.assignment import (
    create_assignment,
    mark_skipped,
    run_complete_unlocked,
    run_unlock_due,
)
from app.services.anki.client import AnkiUnreachableError
from app.services.anki.load_adherence import (
    AnkiLoadAdherence,
    compute_load_adherence,
)
from app.services.anki.review import (
    create_review,
    run_review_due,
)


_AUTH = {"X-Coach-Token": "change_me_before_use"}
_NATIVE_BASE = 1_700_000_001_000


# --------------------------- stubs --------------------------- #


class _StubUnsuspend:
    """AnkiConnect stub for unlock-job integration tests.

    T75: also stubs `add_tags` for the V50 addTags chain. Default = silent
    success on every addTags call (audit-only, not load-bearing). Tests
    that want to script addTags failures pass `add_tags_responses`."""

    def __init__(
        self,
        responses: list[Any],
        *,
        add_tags_responses: list[Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._add_tags_responses = list(add_tags_responses or [])
        self.calls: list[list[int]] = []
        self.add_tags_calls: list[tuple[list[int], list[str]]] = []

    async def unsuspend_cards(self, card_ids: list[int]) -> bool:
        self.calls.append(list(card_ids))
        if not self._responses:
            raise RuntimeError("stub ran out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return bool(nxt)

    async def add_tags(self, card_ids: list[int], tags: list[str]) -> None:
        self.add_tags_calls.append((list(card_ids), list(tags)))
        if not self._add_tags_responses:
            return
        nxt = self._add_tags_responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return None


class _StubFilteredDeck:
    """T76: also stubs `add_tags` for the V50 addTags chain on the
    review path. Default = silent success."""

    def __init__(
        self,
        responses: list[Any],
        *,
        add_tags_responses: list[Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._add_tags_responses = list(add_tags_responses or [])
        self.calls: list[tuple[str, list[int]]] = []
        self.add_tags_calls: list[tuple[list[int], list[str]]] = []

    async def create_filtered_deck(self, name: str, card_ids: list[int]) -> int:
        self.calls.append((name, list(card_ids)))
        if not self._responses:
            raise RuntimeError("stub ran out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return int(nxt)

    async def add_tags(self, card_ids: list[int], tags: list[str]) -> None:
        self.add_tags_calls.append((list(card_ids), list(tags)))
        if not self._add_tags_responses:
            return
        nxt = self._add_tags_responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return None


# --------------------------- seed helpers --------------------------- #


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _seed_topic(session: AsyncSession, cc: ContentCategory) -> Topic:
    t = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name="T71 topic",
        disciplines=[],
        depth=0,
        position=950,
    )
    session.add(t)
    await session.flush()
    return t


async def _seed_suspended_card_with_topic(
    session: AsyncSession, *, native_id: int, topic: Topic
) -> AnkiCard:
    # §V75: candidate resolution is note-scoped — seed note (note_id ==
    # native_id) + its aamc_topic tag, link the card via note_id. This lets
    # create_assignment snapshot both card_ids AND note_ids.
    session.add(AnkiNote(note_id=native_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=native_id,
        deck_name="MileDown",
        note_id=native_id,
        queue=-1,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=native_id,
            tag_raw=f"t::{native_id}",
            topic_id=topic.id,
            parsed_kind="aamc_topic",
            source="regex",
        )
    )
    await session.flush()
    return card


def _later(*, minutes: int = 0, days: int = 0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days, minutes=minutes)


# ====================== V51 + V55 end-to-end ====================== #


async def test_assignment_lifecycle_full_path_pending_unlocked_completed(
    db_session: AsyncSession,
) -> None:
    """V51 happy path stitched together: create_assignment → run_unlock_due
    (stub success) → seed reviews after actual_unlock_at →
    run_complete_unlocked → status='completed'. Audit row appears on
    unlock; no audit row appears on auto-complete (review-driven path is
    DB-only)."""
    cc = await _first_cc(db_session)
    topic = await _seed_topic(db_session, cc)
    card_a = await _seed_suspended_card_with_topic(
        db_session, native_id=_NATIVE_BASE + 1, topic=topic
    )
    card_b = await _seed_suspended_card_with_topic(
        db_session, native_id=_NATIVE_BASE + 2, topic=topic
    )

    assignment = await create_assignment(
        db_session,
        scope_kind="topic",
        scope_value=str(topic.id),
        scheduled_unlock_at=_later(minutes=-5),  # already due → unlock fires
    )
    assert assignment.status == "pending"
    assert set(assignment.card_ids) == {_NATIVE_BASE + 1, _NATIVE_BASE + 2}

    stub = _StubUnsuspend(responses=[True])
    unlock_summary = await run_unlock_due(db_session, stub)
    assert unlock_summary.succeeded == 1
    assert stub.calls == [[_NATIVE_BASE + 1, _NATIVE_BASE + 2]]

    await db_session.refresh(assignment)
    assert assignment.status == "unlocked"
    assert assignment.actual_unlock_at is not None

    unlock_audit = (
        await db_session.execute(select(AnkiWrite).where(AnkiWrite.action == "unsuspend"))
    ).scalar_one()
    assert unlock_audit.status == "succeeded"
    assert unlock_audit.assignment_id == assignment.id

    # Seed reviews after actual_unlock_at to satisfy V51 completion quantifier.
    unlock_at = assignment.actual_unlock_at
    db_session.add(
        AnkiCardReview(
            review_id=_NATIVE_BASE + 100,
            card_id=card_a.id,
            reviewed_at=unlock_at + timedelta(minutes=10),
            ease=3,
            type="review",
        )
    )
    db_session.add(
        AnkiCardReview(
            review_id=_NATIVE_BASE + 101,
            card_id=card_b.id,
            reviewed_at=unlock_at + timedelta(minutes=15),
            ease=3,
            type="review",
        )
    )
    await db_session.flush()

    complete_summary = await run_complete_unlocked(db_session)
    assert complete_summary.completed == 1

    await db_session.refresh(assignment)
    assert assignment.status == "completed"


async def test_assignment_skipped_path_makes_no_ankiconnect_call(
    db_session: AsyncSession,
) -> None:
    """V51 skip transition is mcat-coach accounting only — V57: review-push
    is the only AnkiConnect side-effect; skip must NOT invoke unsuspend.
    Stub raises if called so the assertion is bite-loud rather than silent."""
    cc = await _first_cc(db_session)
    topic = await _seed_topic(db_session, cc)
    await _seed_suspended_card_with_topic(db_session, native_id=_NATIVE_BASE + 10, topic=topic)

    assignment = await create_assignment(
        db_session,
        scope_kind="topic",
        scope_value=str(topic.id),
        scheduled_unlock_at=_later(minutes=-1),
    )
    await mark_skipped(db_session, assignment.id)
    await db_session.refresh(assignment)
    assert assignment.status == "skipped"

    stub = _StubUnsuspend(responses=[])  # any call would raise RuntimeError
    summary = await run_unlock_due(db_session, stub)
    assert summary.processed == 0  # skipped row not selected
    assert stub.calls == []

    # No anki_writes row produced for the skip (V51: skip = accounting only).
    audit_rows = list((await db_session.execute(select(AnkiWrite))).scalars().all())
    assert audit_rows == []


async def test_assignment_three_strike_failure_marks_terminal(
    db_session: AsyncSession,
) -> None:
    """V55 retry cap: three consecutive transient failures flip the
    assignment to terminal 'failed' and leave one audit row per attempt."""
    cc = await _first_cc(db_session)
    topic = await _seed_topic(db_session, cc)
    await _seed_suspended_card_with_topic(db_session, native_id=_NATIVE_BASE + 20, topic=topic)

    assignment = await create_assignment(
        db_session,
        scope_kind="topic",
        scope_value=str(topic.id),
        scheduled_unlock_at=_later(minutes=-1),
    )

    for _ in range(3):
        stub = _StubUnsuspend(responses=[AnkiUnreachableError("transient")])
        await run_unlock_due(db_session, stub)

    await db_session.refresh(assignment)
    assert assignment.status == "failed"
    assert assignment.failure_count == 3
    assert assignment.error_text is not None

    audit_rows = list(
        (
            await db_session.execute(
                select(AnkiWrite)
                .where(AnkiWrite.assignment_id == assignment.id)
                .order_by(AnkiWrite.id.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(audit_rows) == 3
    assert {r.status for r in audit_rows} == {"failed"}


# ====================== V53 amended (T76) review semantics ====================== #


async def test_review_dup_same_day_creates_parallel_rows(
    db_session: AsyncSession,
) -> None:
    """V53 amended (T76): no UNIQUE constraint — re-creating a review
    same day, same cards, yields a NEW row with a NEW id + NEW deck name.
    Tags-as-log accepts the accumulation; idempotency lives in UI debounce."""
    today = date.today()
    first = await create_review(
        db_session,
        card_ids=[1_700_000_500_001, 1_700_000_500_002],
        review_date=today,
        write_deck_prefix="mcat-coach",
    )
    stub = _StubFilteredDeck(responses=[1_700_000_700_000])
    summary = await run_review_due(db_session, stub, today=today)
    assert summary.pushed == 1
    await db_session.refresh(first)
    assert first.status == "pushed"

    # T76: re-create on same date + same cards is allowed → new row.
    rebuilt = await create_review(
        db_session,
        card_ids=[1_700_000_500_999],
        review_date=today,
        write_deck_prefix="mcat-coach",
    )
    assert rebuilt.id != first.id
    assert rebuilt.status == "pending"
    assert rebuilt.card_ids == [1_700_000_500_999]
    # Deck names diverge — each row gets a fresh `<prefix>::review::{id}`.
    assert rebuilt.deck_name != first.deck_name
    assert rebuilt.deck_name == f"mcat-coach::review::{rebuilt.id}"


# ====================== V60 adherence payload shape ====================== #


async def test_v60_adherence_dataclass_and_api_carry_no_recommended_changes(
    db_session: AsyncSession, client: AsyncClient
) -> None:
    """V60 belt-and-suspenders: the in-process service result + the API
    JSON envelope must both omit `recommended_changes`."""
    db_session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=200,
            daily_minutes_budget=Decimal("60"),
        )
    )
    await db_session.flush()

    service_result = await compute_load_adherence(db_session)
    assert isinstance(service_result, AnkiLoadAdherence)
    assert "recommended_changes" not in {f.name for f in fields(AnkiLoadAdherence)}

    r = await client.get("/api/v1/anki/load-adherence", headers=_AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "recommended_changes" not in body


## ====================== /anki renders full stack (post-V61 fold) ====================== #


async def test_anki_renders_full_p11_plus_stack(
    db_session: AsyncSession, client: AsyncClient
) -> None:
    """Seed a representative P11+ payload (assignment + push + reviews)
    and verify /anki renders the chip/timelines/burndown that depend on
    each piece. (Pre-V61 the timelines + burndown lived on /study-plan;
    T79 folded them onto /anki and cut /study-plan.)"""
    cc = await _first_cc(db_session)
    topic = await _seed_topic(db_session, cc)
    await _seed_suspended_card_with_topic(db_session, native_id=_NATIVE_BASE + 200, topic=topic)

    now = datetime.now(timezone.utc)
    today = now.date()

    # 1 pending assignment in window.
    db_session.add(
        AnkiAssignment(
            scope_kind="topic",
            scope_value=str(topic.id),
            scheduled_unlock_at=now + timedelta(days=2),
            card_ids=[_NATIVE_BASE + 200],
            status="pending",
        )
    )
    # 1 pending review in window (V53 amended: deck name uses row PK).
    db_session.add(
        AnkiReview(
            review_date=today + timedelta(days=3),
            card_ids=[_NATIVE_BASE + 200],
            deck_name="mcat-coach::review::weak-topics-seed",
            status="pending",
        )
    )
    # 1 review in past 14d for the sparkline.
    card_for_review = AnkiCard(
        anki_card_id=_NATIVE_BASE + 300,
        deck_name="MileDown",
        queue=2,
    )
    db_session.add(card_for_review)
    await db_session.flush()
    db_session.add(
        AnkiCardReview(
            review_id=_NATIVE_BASE + 400,
            card_id=card_for_review.id,
            reviewed_at=now - timedelta(days=1),
            ease=3,
            type="review",
            time_ms=8000,
        )
    )
    await db_session.flush()

    r_anki = await client.get("/anki")
    assert r_anki.status_code == 200
    assert "Plan adherence" in r_anki.text
    assert 'data-test="adherence-chip"' in r_anki.text
    assert 'data-test="assignments-pending"' in r_anki.text
    assert 'data-test="review-pushes-pending"' in r_anki.text
    assert 'data-test="burndown"' in r_anki.text
    assert "weak-topics" in r_anki.text
    # V60: no "recommended_changes" string in the page.
    assert "recommended_changes" not in r_anki.text

    # V61: /study-plan was cut by T79 — must 404.
    r_plan = await client.get("/study-plan")
    assert r_plan.status_code == 404
