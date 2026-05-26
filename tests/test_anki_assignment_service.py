"""Tests for SPEC T62 — assignment service (V51 lifecycle + V52 resolve).

Exercises both halves of the new module:
  * resolve_card_ids — subtree membership, queue=-1 filter, confidence
    threshold, and the four V52 priority orderings.
  * create_assignment / mark_skipped / mark_completed_manual — V51
    pending→unlocked→(completed|skipped|failed) state machine with
    terminal-state refusal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic
from app.services.anki.assignment import (
    AssignmentError,
    AssignmentNotFoundError,
    AssignmentTerminalError,
    create_assignment,
    mark_completed_manual,
    mark_skipped,
    resolve_card_ids,
)


_CARD_BASE = 800_000


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _second_cc(session: AsyncSession) -> ContentCategory:
    rows = (await session.execute(select(ContentCategory).limit(2))).scalars().all()
    return rows[1]


async def _make_topic_tree(
    session: AsyncSession, cc: ContentCategory, *, label: str
) -> tuple[Topic, Topic, Topic]:
    """parent (depth=0) → child (depth=1); independent sibling at root."""
    parent = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T62 parent {label}",
        disciplines=[],
        depth=0,
        position=800,
    )
    sibling = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T62 sibling {label}",
        disciplines=[],
        depth=0,
        position=801,
    )
    session.add_all([parent, sibling])
    await session.flush()
    child = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent.id,
        name=f"T62 child {label}",
        disciplines=[],
        depth=1,
        position=802,
    )
    session.add(child)
    await session.flush()
    return parent, child, sibling


async def _make_card(
    session: AsyncSession,
    *,
    anki_card_id: int,
    queue: Optional[int] = -1,
    interval_days: Optional[int] = None,
) -> AnkiCard:
    # §V75: candidates are note-scoped (a card matches iff its NOTE carries an
    # in-scope tag). Seed the note (note_id == anki_card_id) before the FK.
    session.add(AnkiNote(note_id=anki_card_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="MileDown",
        note_id=anki_card_id,
        queue=queue,
        interval_days=interval_days,
    )
    session.add(card)
    await session.flush()
    return card


def _topic_tag(
    *,
    note_id: int,
    topic_id: int,
    tag_raw: str,
    confidence: Optional[float] = None,
    source: str = "regex",
) -> AnkiNoteTag:
    return AnkiNoteTag(
        note_id=note_id,
        tag_raw=tag_raw,
        topic_id=topic_id,
        parsed_kind="aamc_topic",
        source=source,
        confidence=confidence,
    )


def _cc_tag(
    *,
    note_id: int,
    cc_id: int,
    tag_raw: str,
) -> AnkiNoteTag:
    return AnkiNoteTag(
        note_id=note_id,
        tag_raw=tag_raw,
        content_category_id=cc_id,
        parsed_kind="aamc_cc",
        source="regex",
    )


def _later(days: int = 1) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days)


# --------------------------- V52 resolve_card_ids --------------------------- #


async def test_topic_scope_returns_subtree_aamc_topic_rows(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, child, sibling = await _make_topic_tree(db_session, cc, label="subtree")

    # one card under parent, one under child (both in subtree), one under
    # sibling (out of subtree). All suspended.
    c1 = await _make_card(db_session, anki_card_id=_CARD_BASE + 1, queue=-1)
    c2 = await _make_card(db_session, anki_card_id=_CARD_BASE + 2, queue=-1)
    c3 = await _make_card(db_session, anki_card_id=_CARD_BASE + 3, queue=-1)
    db_session.add_all(
        [
            _topic_tag(note_id=c1.note_id, topic_id=parent.id, tag_raw="t1"),
            _topic_tag(note_id=c2.note_id, topic_id=child.id, tag_raw="t2"),
            _topic_tag(note_id=c3.note_id, topic_id=sibling.id, tag_raw="t3"),
        ]
    )
    await db_session.flush()

    ids = await resolve_card_ids(db_session, scope_kind="topic", scope_value=str(parent.id))
    assert set(ids) == {_CARD_BASE + 1, _CARD_BASE + 2}
    assert _CARD_BASE + 3 not in ids


async def test_resolve_dedups_card_matched_via_multiple_tags(
    db_session: AsyncSession,
) -> None:
    """V64 (§B13): a card tagged under multiple topics in the same subtree
    produces multiple candidate rows (the SQL `SELECT DISTINCT` spans the
    full row incl confidence/depth, so it does NOT collapse same-card rows).
    resolve_card_ids must return the native card_id exactly once — duplicate
    cids blow AnkiConnect's `search_cids` UNIQUE(cid) at unsuspend time."""
    cc = await _first_cc(db_session)
    parent, child, _sibling = await _make_topic_tree(db_session, cc, label="v64-dedup")

    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 64, queue=-1)
    # one card, two aamc_topic tags (parent + child) — both in subtree(parent)
    db_session.add_all(
        [
            _topic_tag(note_id=card.note_id, topic_id=parent.id, tag_raw="v64a"),
            _topic_tag(note_id=card.note_id, topic_id=child.id, tag_raw="v64b"),
        ]
    )
    await db_session.flush()

    ids = await resolve_card_ids(db_session, scope_kind="topic", scope_value=str(parent.id))
    assert ids == [_CARD_BASE + 64]
    assert len(ids) == len(set(ids))


async def test_topic_scope_ignores_cc_tags(db_session: AsyncSession) -> None:
    """V52 topic scope filters parsed_kind='aamc_topic' only — aamc_cc rows
    on cards under that subtree are not pulled in."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="cc-ignore")

    c = await _make_card(db_session, anki_card_id=_CARD_BASE + 10, queue=-1)
    db_session.add(_cc_tag(note_id=c.note_id, cc_id=cc.id, tag_raw="cc-only"))
    await db_session.flush()

    ids = await resolve_card_ids(db_session, scope_kind="topic", scope_value=str(parent.id))
    assert ids == []


async def test_cc_scope_includes_direct_and_via_topic(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, child, _sib = await _make_topic_tree(db_session, cc, label="cc-mixed")
    other_cc = await _second_cc(db_session)

    c_direct = await _make_card(db_session, anki_card_id=_CARD_BASE + 20, queue=-1)
    c_via_topic = await _make_card(db_session, anki_card_id=_CARD_BASE + 21, queue=-1)
    c_other_cc = await _make_card(db_session, anki_card_id=_CARD_BASE + 22, queue=-1)

    db_session.add_all(
        [
            _cc_tag(note_id=c_direct.note_id, cc_id=cc.id, tag_raw="d"),
            _topic_tag(note_id=c_via_topic.note_id, topic_id=child.id, tag_raw="vt"),
            _cc_tag(note_id=c_other_cc.note_id, cc_id=other_cc.id, tag_raw="o"),
        ]
    )
    await db_session.flush()

    ids = await resolve_card_ids(db_session, scope_kind="cc", scope_value=cc.code)
    assert set(ids) == {_CARD_BASE + 20, _CARD_BASE + 21}
    assert _CARD_BASE + 22 not in ids


async def test_queue_filter_excludes_already_unsuspended(
    db_session: AsyncSession,
) -> None:
    """V52: only queue=-1 (suspended) cards are candidates — already
    unsuspended cards (queue >= 0) reflect the 'unlock' semantics; we do
    not re-unlock them."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="queue")

    suspended = await _make_card(db_session, anki_card_id=_CARD_BASE + 30, queue=-1)
    new_card = await _make_card(db_session, anki_card_id=_CARD_BASE + 31, queue=0)
    review = await _make_card(db_session, anki_card_id=_CARD_BASE + 32, queue=2, interval_days=10)
    db_session.add_all(
        [
            _topic_tag(note_id=suspended.note_id, topic_id=parent.id, tag_raw="s"),
            _topic_tag(note_id=new_card.note_id, topic_id=parent.id, tag_raw="n"),
            _topic_tag(note_id=review.note_id, topic_id=parent.id, tag_raw="r"),
        ]
    )
    await db_session.flush()

    ids = await resolve_card_ids(db_session, scope_kind="topic", scope_value=str(parent.id))
    assert ids == [_CARD_BASE + 30]


async def test_confidence_threshold_excludes_low_confidence(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="conf")
    high = await _make_card(db_session, anki_card_id=_CARD_BASE + 40)
    low = await _make_card(db_session, anki_card_id=_CARD_BASE + 41)
    null = await _make_card(db_session, anki_card_id=_CARD_BASE + 42)
    db_session.add_all(
        [
            _topic_tag(
                note_id=high.note_id,
                topic_id=parent.id,
                tag_raw="h",
                confidence=0.9,
                source="llm",
            ),
            _topic_tag(
                note_id=low.note_id,
                topic_id=parent.id,
                tag_raw="l",
                confidence=0.3,
                source="llm",
            ),
            _topic_tag(
                note_id=null.note_id,
                topic_id=parent.id,
                tag_raw="n",
                confidence=None,
                source="regex",
            ),
        ]
    )
    await db_session.flush()

    ids = await resolve_card_ids(db_session, scope_kind="topic", scope_value=str(parent.id))
    assert set(ids) == {_CARD_BASE + 40, _CARD_BASE + 42}
    assert _CARD_BASE + 41 not in ids


# --- priority orderings ---


async def test_most_specific_first_orders_by_confidence_then_depth(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, child, _sib = await _make_topic_tree(db_session, cc, label="msf")

    # card A: parent topic, conf NULL (regex)
    # card B: child topic (deeper), conf 0.6 (llm)
    # card C: child topic (deeper), conf 0.95 (llm)  <- highest confidence
    a = await _make_card(db_session, anki_card_id=_CARD_BASE + 50)
    b = await _make_card(db_session, anki_card_id=_CARD_BASE + 51)
    c = await _make_card(db_session, anki_card_id=_CARD_BASE + 52)
    db_session.add_all(
        [
            _topic_tag(note_id=a.note_id, topic_id=parent.id, tag_raw="a", confidence=None),
            _topic_tag(
                note_id=b.note_id,
                topic_id=child.id,
                tag_raw="b",
                confidence=0.6,
                source="llm",
            ),
            _topic_tag(
                note_id=c.note_id,
                topic_id=child.id,
                tag_raw="c",
                confidence=0.95,
                source="llm",
            ),
        ]
    )
    await db_session.flush()

    ids = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="most_specific_first",
    )
    # confidence 0.95 → 0.6 → NULL (NULL sorts last);
    # 0.95 and 0.6 are both at child depth 1 vs NULL at parent depth 0.
    assert ids == [_CARD_BASE + 52, _CARD_BASE + 51, _CARD_BASE + 50]


async def test_random_priority_is_deterministic_per_seed(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="rand")
    cards = []
    for i in range(8):
        c = await _make_card(db_session, anki_card_id=_CARD_BASE + 60 + i)
        cards.append(c)
        db_session.add(_topic_tag(note_id=c.note_id, topic_id=parent.id, tag_raw=f"r{i}"))
    await db_session.flush()

    first = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="random",
        random_seed=42,
    )
    second = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="random",
        random_seed=42,
    )
    third = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="random",
        random_seed=7,
    )
    assert first == second
    assert first != third


async def test_random_priority_requires_seed(db_session: AsyncSession) -> None:
    with pytest.raises(AssignmentError, match="random_seed"):
        await resolve_card_ids(
            db_session,
            scope_kind="cc",
            scope_value="anything",
            priority="random",
        )


async def test_mature_first_and_young_first_order_by_interval(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="ivl")

    # interval_days stored on the suspended row from the last review cycle.
    intervals = [(70, 5), (71, 30), (72, 100), (73, None)]
    for offset, ivl in intervals:
        card = await _make_card(
            db_session,
            anki_card_id=_CARD_BASE + offset,
            interval_days=ivl,
        )
        db_session.add(_topic_tag(note_id=card.note_id, topic_id=parent.id, tag_raw=str(offset)))
    await db_session.flush()

    mature = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="mature_first",
    )
    # 100 > 30 > 5 > None (treated as 0)
    assert mature == [
        _CARD_BASE + 72,
        _CARD_BASE + 71,
        _CARD_BASE + 70,
        _CARD_BASE + 73,
    ]

    young = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="young_first",
    )
    # None (0) → 5 → 30 → 100
    assert young == [
        _CARD_BASE + 73,
        _CARD_BASE + 70,
        _CARD_BASE + 71,
        _CARD_BASE + 72,
    ]


async def test_max_cards_slices(db_session: AsyncSession) -> None:
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="max")
    for i in range(5):
        card = await _make_card(db_session, anki_card_id=_CARD_BASE + 80 + i)
        db_session.add(_topic_tag(note_id=card.note_id, topic_id=parent.id, tag_raw=f"m{i}"))
    await db_session.flush()

    ids = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        max_cards=3,
    )
    assert len(ids) == 3


# --------------------------- V51 lifecycle --------------------------- #


async def test_create_assignment_snapshots_card_ids(
    db_session: AsyncSession,
) -> None:
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="create")
    for i in range(3):
        card = await _make_card(db_session, anki_card_id=_CARD_BASE + 90 + i)
        db_session.add(_topic_tag(note_id=card.note_id, topic_id=parent.id, tag_raw=f"c{i}"))
    await db_session.flush()

    a = await create_assignment(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        scheduled_unlock_at=_later(),
        max_cards=10,
    )
    assert a.id is not None
    assert a.status == "pending"
    assert set(a.card_ids) == {_CARD_BASE + 90, _CARD_BASE + 91, _CARD_BASE + 92}
    assert a.priority == "most_specific_first"

    # Persisted shape matches the in-memory object.
    fetched = (
        await db_session.execute(select(AnkiAssignment).where(AnkiAssignment.id == a.id))
    ).scalar_one()
    assert set(fetched.card_ids) == set(a.card_ids)


async def test_create_assignment_random_uses_assignment_id_seed(
    db_session: AsyncSession,
) -> None:
    """priority='random' must work end-to-end without an external seed —
    create_assignment supplies its own id post-flush so the snapshot is
    reproducible from the assignment row alone."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="rand-asg")
    for i in range(5):
        card = await _make_card(db_session, anki_card_id=_CARD_BASE + 100 + i)
        db_session.add(_topic_tag(note_id=card.note_id, topic_id=parent.id, tag_raw=f"x{i}"))
    await db_session.flush()

    a = await create_assignment(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        scheduled_unlock_at=_later(),
        priority="random",
    )
    assert len(a.card_ids) == 5
    # Replay the same scope with the assignment_id as seed → same order.
    replay = await resolve_card_ids(
        db_session,
        scope_kind="topic",
        scope_value=str(parent.id),
        priority="random",
        random_seed=a.id,
    )
    assert a.card_ids == replay


@pytest.mark.parametrize("from_status", ["pending", "unlocked"])
async def test_mark_skipped_from_active_states(db_session: AsyncSession, from_status: str) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=_later(),
        card_ids=[1, 2, 3],
        status=from_status,
    )
    db_session.add(a)
    await db_session.flush()

    updated = await mark_skipped(db_session, a.id)
    assert updated.status == "skipped"


@pytest.mark.parametrize("from_status", ["pending", "unlocked"])
async def test_mark_completed_manual_from_active_states(
    db_session: AsyncSession, from_status: str
) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=_later(),
        card_ids=[1, 2, 3],
        status=from_status,
    )
    db_session.add(a)
    await db_session.flush()

    updated = await mark_completed_manual(db_session, a.id)
    assert updated.status == "completed"


@pytest.mark.parametrize("terminal", ["completed", "skipped", "failed"])
async def test_mark_skipped_refuses_terminal(db_session: AsyncSession, terminal: str) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=_later(),
        card_ids=[1],
        status=terminal,
    )
    db_session.add(a)
    await db_session.flush()
    with pytest.raises(AssignmentTerminalError):
        await mark_skipped(db_session, a.id)


@pytest.mark.parametrize("terminal", ["completed", "skipped", "failed"])
async def test_mark_completed_manual_refuses_terminal(
    db_session: AsyncSession, terminal: str
) -> None:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value="4C",
        scheduled_unlock_at=_later(),
        card_ids=[1],
        status=terminal,
    )
    db_session.add(a)
    await db_session.flush()
    with pytest.raises(AssignmentTerminalError):
        await mark_completed_manual(db_session, a.id)


async def test_mark_skipped_not_found(db_session: AsyncSession) -> None:
    with pytest.raises(AssignmentNotFoundError):
        await mark_skipped(db_session, 999_999)


async def test_mark_completed_manual_not_found(db_session: AsyncSession) -> None:
    with pytest.raises(AssignmentNotFoundError):
        await mark_completed_manual(db_session, 999_999)


async def test_invalid_scope_kind_raises(db_session: AsyncSession) -> None:
    with pytest.raises(AssignmentError, match="scope_kind"):
        await resolve_card_ids(db_session, scope_kind="section", scope_value="CP")


async def test_topic_scope_with_non_int_value_raises(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(AssignmentError, match="topic"):
        await resolve_card_ids(db_session, scope_kind="topic", scope_value="not-a-number")
