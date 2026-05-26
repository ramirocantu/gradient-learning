"""Tests for SPEC §T38 — subtree Anki state counts + unlock%.

Covers:
- §V28 — bucket definitions mirror Anki (q=-1 susp, q=0 new, q=1 learning,
  q=2 review further split by interval into young/mature, q=3 day-learn
  folds into learning); assigned = queue >= 0; unlock_pct = assigned/total.
- §V31 — subtree-membership rollup; card with multiple in-scope tags
  counted once.
"""

from __future__ import annotations

from typing import Optional

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic
from app.services.anki.state import (
    StateCounts,
    state_for_cc,
    state_for_topic,
)


_CARD_BASE = 900_000


async def _make_card(
    session: AsyncSession,
    *,
    anki_card_id: int,
    queue: Optional[int] = None,
    interval_days: Optional[int] = None,
    deck_name: str = "MileDown",
) -> AnkiCard:
    # §V75: a card's tags live on its note. Use note_id == anki_card_id for
    # 1:1 simplicity; the note must exist before the FK on anki_cards.note_id.
    session.add(AnkiNote(note_id=anki_card_id, deck_name=deck_name))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name=deck_name,
        note_id=anki_card_id,
        queue=queue,
        interval_days=interval_days,
    )
    session.add(card)
    await session.flush()
    return card


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _second_cc(session: AsyncSession) -> ContentCategory:
    rows = (await session.execute(select(ContentCategory).limit(2))).scalars().all()
    return rows[1]


async def _make_topic_tree(
    session: AsyncSession, cc: ContentCategory, *, label: str
) -> tuple[Topic, Topic, Topic]:
    """Create parent → child + an unrelated sibling under the given CC."""
    parent = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T38 parent {label}",
        disciplines=[],
        depth=0,
        position=900,
    )
    sibling = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T38 sibling {label}",
        disciplines=[],
        depth=0,
        position=901,
    )
    session.add_all([parent, sibling])
    await session.flush()
    child = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent.id,
        name=f"T38 child {label}",
        disciplines=[],
        depth=1,
        position=902,
    )
    session.add(child)
    await session.flush()
    return parent, child, sibling


def _tag(
    *,
    note_id: int,
    tag_raw: str,
    parsed_kind: str,
    topic_id: int | None = None,
    cc_id: int | None = None,
) -> AnkiNoteTag:
    return AnkiNoteTag(
        note_id=note_id,
        tag_raw=tag_raw,
        topic_id=topic_id,
        content_category_id=cc_id,
        parsed_kind=parsed_kind,
        source="regex",
    )


# --- §V28: queue → bucket mapping ---


async def test_queue_value_to_bucket(db_session: AsyncSession) -> None:
    """One card per queue value lands in exactly the right bucket."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="buckets")

    queue_to_offset = {
        -1: 1,  # suspended
        0: 2,  # new
        1: 3,  # learning
        2: 4,  # review → young (interval 5)
        3: 5,  # day-learn → learning
    }
    for q, off in queue_to_offset.items():
        card = await _make_card(
            db_session,
            anki_card_id=_CARD_BASE + off,
            queue=q,
            interval_days=5 if q == 2 else None,
        )
        db_session.add(
            _tag(
                note_id=card.note_id,
                tag_raw=f"t::bucket::{q}",
                parsed_kind="aamc_topic",
                topic_id=parent.id,
            )
        )
    await db_session.flush()

    s = await state_for_topic(db_session, topic_id=parent.id)
    assert s.total_cards == 5
    assert s.suspended == 1
    assert s.new == 1
    assert s.learning == 2  # q=1 ∪ q=3
    assert s.young == 1
    assert s.mature == 0
    assert s.assigned == 4  # all non-suspended


async def test_young_mature_interval_boundary(db_session: AsyncSession) -> None:
    """q=2 split at interval_days >= 21 (mature) vs < 21 (young)."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="ivl")
    cases = [
        (1, 20, "young"),  # < 21
        (2, 21, "mature"),  # >= 21
        (3, 365, "mature"),
        (4, 0, "young"),
    ]
    for off, ivl, _label in cases:
        card = await _make_card(
            db_session,
            anki_card_id=_CARD_BASE + 100 + off,
            queue=2,
            interval_days=ivl,
        )
        db_session.add(
            _tag(
                note_id=card.note_id,
                tag_raw=f"t::ivl::{ivl}",
                parsed_kind="aamc_topic",
                topic_id=parent.id,
            )
        )
    await db_session.flush()

    s = await state_for_topic(db_session, topic_id=parent.id)
    assert s.young == 2
    assert s.mature == 2
    assert s.suspended == 0
    assert s.new == 0
    assert s.learning == 0


async def test_null_interval_on_review_card_treated_as_young(
    db_session: AsyncSession,
) -> None:
    """Defensive: q=2 card with NULL interval_days → young (COALESCE→0)."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="nullivl")
    card = await _make_card(
        db_session,
        anki_card_id=_CARD_BASE + 200,
        queue=2,
        interval_days=None,
    )
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="t::nullivl",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()

    s = await state_for_topic(db_session, topic_id=parent.id)
    assert s.young == 1
    assert s.mature == 0


# --- §V28: unlock% ---


async def test_unlock_pct(db_session: AsyncSession) -> None:
    """assigned / total_cards. Suspended in denom, not numerator."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="unlock")
    # 3 unsuspended (q=0,1,2) + 1 suspended (q=-1) → assigned=3, total=4
    queues = [-1, 0, 1, 2]
    for off, q in enumerate(queues):
        card = await _make_card(
            db_session,
            anki_card_id=_CARD_BASE + 300 + off,
            queue=q,
            interval_days=10 if q == 2 else None,
        )
        db_session.add(
            _tag(
                note_id=card.note_id,
                tag_raw=f"t::unlock::{q}",
                parsed_kind="aamc_topic",
                topic_id=parent.id,
            )
        )
    await db_session.flush()

    s = await state_for_topic(db_session, topic_id=parent.id)
    assert s.total_cards == 4
    assert s.assigned == 3
    assert s.unlock_pct == pytest.approx(3 / 4)


async def test_empty_scope_unlock_pct_none(db_session: AsyncSession) -> None:
    """Scope with no in-scope cards → unlock_pct is None, not div-by-zero."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="empty")
    s = await state_for_topic(db_session, topic_id=parent.id)
    assert s.total_cards == 0
    assert s.assigned == 0
    assert s.unlock_pct is None


# --- §V31: subtree rollup ---


async def test_subtree_rollup_aggregates_descendants(
    db_session: AsyncSession,
) -> None:
    """Parent's counts = own + descendants'. Sibling out of scope."""
    cc = await _first_cc(db_session)
    parent, child, sibling = await _make_topic_tree(db_session, cc, label="rollup")

    parent_card = await _make_card(
        db_session, anki_card_id=_CARD_BASE + 400, queue=2, interval_days=30
    )
    child_card = await _make_card(db_session, anki_card_id=_CARD_BASE + 401, queue=0)
    sibling_card = await _make_card(
        db_session, anki_card_id=_CARD_BASE + 402, queue=2, interval_days=30
    )
    db_session.add_all(
        [
            _tag(
                note_id=parent_card.note_id,
                tag_raw="t::rollup::parent",
                parsed_kind="aamc_topic",
                topic_id=parent.id,
            ),
            _tag(
                note_id=child_card.note_id,
                tag_raw="t::rollup::child",
                parsed_kind="aamc_topic",
                topic_id=child.id,
            ),
            _tag(
                note_id=sibling_card.note_id,
                tag_raw="t::rollup::sib",
                parsed_kind="aamc_topic",
                topic_id=sibling.id,
            ),
        ]
    )
    await db_session.flush()

    parent_s = await state_for_topic(db_session, topic_id=parent.id)
    child_s = await state_for_topic(db_session, topic_id=child.id)
    sib_s = await state_for_topic(db_session, topic_id=sibling.id)

    assert parent_s.total_cards == 2  # parent + child
    assert parent_s.mature == 1
    assert parent_s.new == 1
    assert child_s.total_cards == 1
    assert sib_s.total_cards == 1
    assert sib_s.mature == 1


async def test_card_with_multiple_in_scope_tags_counted_once(
    db_session: AsyncSession,
) -> None:
    """§V31 dedupe: a card holding both an aamc_topic tag and an aamc_cc tag
    that both fall in scope must count exactly once."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="dedupe")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 500, queue=2, interval_days=30)
    db_session.add_all(
        [
            _tag(
                note_id=card.note_id,
                tag_raw="t::dedupe::topic",
                parsed_kind="aamc_topic",
                topic_id=parent.id,
            ),
            _tag(
                note_id=card.note_id,
                tag_raw="t::dedupe::cc",
                parsed_kind="aamc_cc",
                cc_id=cc.id,
            ),
        ]
    )
    await db_session.flush()

    s = await state_for_cc(db_session, cc_code=cc.code)
    assert s.total_cards == 1  # not 2
    assert s.mature == 1


# --- CC scope: aamc_cc + aamc_topic paths ---


async def test_cc_scope_includes_direct_aamc_cc_tag(db_session: AsyncSession) -> None:
    """A card tagged at CC granularity (aamc_cc, topic_id=NULL) lands in CC scope."""
    cc = await _first_cc(db_session)
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 600, queue=2, interval_days=30)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="cc::direct",
            parsed_kind="aamc_cc",
            cc_id=cc.id,
        )
    )
    await db_session.flush()

    s = await state_for_cc(db_session, cc_code=cc.code)
    assert s.total_cards == 1
    assert s.mature == 1
    assert s.scope == f"cc:{cc.code}"


async def test_cc_scope_includes_topic_path(db_session: AsyncSession) -> None:
    """A card tagged at topic granularity rolls up to its CC via topic→CC join."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="topic_path")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 601, queue=0)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="cc::via_topic",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()

    s = await state_for_cc(db_session, cc_code=cc.code)
    assert s.total_cards == 1
    assert s.new == 1


async def test_cc_scope_excludes_other_cc(db_session: AsyncSession) -> None:
    """Cards in CC-B don't leak into CC-A counts."""
    cc_a = await _first_cc(db_session)
    cc_b = await _second_cc(db_session)
    card_b = await _make_card(db_session, anki_card_id=_CARD_BASE + 602, queue=2, interval_days=30)
    db_session.add(
        _tag(
            note_id=card_b.note_id,
            tag_raw="cc::other",
            parsed_kind="aamc_cc",
            cc_id=cc_b.id,
        )
    )
    await db_session.flush()

    a = await state_for_cc(db_session, cc_code=cc_a.code)
    b = await state_for_cc(db_session, cc_code=cc_b.code)
    assert a.total_cards == 0
    assert b.total_cards == 1


# --- shape ---


async def test_summary_shape(db_session: AsyncSession) -> None:
    cc = await _first_cc(db_session)
    s = await state_for_cc(db_session, cc_code=cc.code)
    assert isinstance(s, StateCounts)
    assert s.scope == f"cc:{cc.code}"
