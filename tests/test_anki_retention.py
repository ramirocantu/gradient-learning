"""Tests for SPEC §T37 — windowed Anki true retention.

Covers:
- §V27 — pass = ease ∈ {2,3,4}; type='learn' excluded from both num+denom;
  windows 7d / 30d / all-time.
- §V31 — subtree-membership rollup; each review counted once even when
  a card carries multiple tag rows in scope.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic
from app.services.anki.retention import (
    RetentionSummary,
    retention_for_cc,
    retention_for_topic,
)


# Unique-ish bases so parallel-running tests in the same DB session don't
# step on each other when the test transaction reuses anki_card_id values.
_CARD_BASE = 800_000
_REVIEW_BASE = 1_800_000_000_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _make_card(
    session: AsyncSession, *, anki_card_id: int, deck_name: str = "MileDown"
) -> AnkiCard:
    # §V75: note-as-unit — seed the note (note_id == anki_card_id) before the
    # FK on anki_cards.note_id, so tags can attach to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name=deck_name))
    await session.flush()
    card = AnkiCard(anki_card_id=anki_card_id, deck_name=deck_name, note_id=anki_card_id)
    session.add(card)
    await session.flush()
    return card


async def _add_review(
    session: AsyncSession,
    *,
    card_pk: int,
    review_id: int,
    ease: int,
    type_: str = "review",
    age_days: float = 1.0,
) -> AnkiCardReview:
    r = AnkiCardReview(
        review_id=review_id,
        card_id=card_pk,
        reviewed_at=_now() - timedelta(days=age_days),
        ease=ease,
        type=type_,
    )
    session.add(r)
    await session.flush()
    return r


async def _first_cc(session: AsyncSession) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _second_cc(session: AsyncSession) -> ContentCategory:
    rows = (await session.execute(select(ContentCategory).limit(2))).scalars().all()
    return rows[1]


async def _make_topic_tree(
    session: AsyncSession, cc: ContentCategory, *, label: str
) -> tuple[Topic, Topic, Topic]:
    """Create parent → child + an unrelated sibling under the given CC.

    Returns (parent, child, sibling). Used to exercise subtree-membership
    rollup vs sibling-isolation per §V31.
    """
    parent = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T37 parent {label}",
        disciplines=[],
        depth=0,
        position=900,
    )
    sibling = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=f"T37 sibling {label}",
        disciplines=[],
        depth=0,
        position=901,
    )
    session.add_all([parent, sibling])
    await session.flush()
    child = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent.id,
        name=f"T37 child {label}",
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


# --- §V27: pass/fail bucketing + learn exclusion ---


async def test_pass_fail_bucketing_by_ease(db_session: AsyncSession) -> None:
    """ease ∈ {2,3,4} count as pass; ease=1 as fail. Type='review'."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="bucket")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 1)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="t::bucket",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()

    for i, ease in enumerate([1, 2, 3, 4, 4]):
        await _add_review(
            db_session,
            card_pk=card.id,
            review_id=_REVIEW_BASE + 1_000 + i,
            ease=ease,
            type_="review",
            age_days=1,
        )

    summary = await retention_for_topic(db_session, topic_id=parent.id)
    w = summary.windows[7]
    assert w.pass_count == 4  # eases 2,3,4,4
    assert w.fail_count == 1  # ease=1
    assert w.total == 5
    assert w.retention == pytest.approx(4 / 5)


async def test_learn_type_excluded_from_both_num_and_denom(
    db_session: AsyncSession,
) -> None:
    """§V27: type='learn' rows neither pass nor fail — they drop out entirely."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="learn_excl")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 2)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="t::learn_excl",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()

    # Two real reviews (1 pass, 1 fail) + two 'learn' rows that must be ignored.
    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 2_001, ease=3, type_="review"
    )
    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 2_002, ease=1, type_="review"
    )
    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 2_003, ease=3, type_="learn"
    )
    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 2_004, ease=1, type_="learn"
    )

    w = (await retention_for_topic(db_session, topic_id=parent.id)).windows[7]
    assert w.pass_count == 1
    assert w.fail_count == 1
    assert w.total == 2  # 4 raw rows, 2 after learn-exclusion


@pytest.mark.parametrize("kept_type", ["review", "relearn", "cram"])
async def test_non_learn_types_counted(db_session: AsyncSession, kept_type: str) -> None:
    """§V27 spec says exclude *only* type='learn'; relearn + cram count."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label=f"kept_{kept_type}")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 100 + hash(kept_type) % 50)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw=f"t::kept_{kept_type}",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()
    await _add_review(
        db_session,
        card_pk=card.id,
        review_id=_REVIEW_BASE + 3_000 + hash(kept_type) % 500,
        ease=3,
        type_=kept_type,
    )

    w = (await retention_for_topic(db_session, topic_id=parent.id)).windows[0]
    assert w.pass_count == 1
    assert w.fail_count == 0
    assert w.retention == 1.0


# --- §V27: windowing ---


async def test_window_boundaries_7_30_alltime(db_session: AsyncSession) -> None:
    """A review at -5d hits 7d+30d+all; at -15d hits 30d+all; at -100d hits all only."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="windowing")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 3)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="t::windowing",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()

    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 4_001, ease=3, age_days=5
    )
    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 4_002, ease=3, age_days=15
    )
    await _add_review(
        db_session, card_pk=card.id, review_id=_REVIEW_BASE + 4_003, ease=3, age_days=100
    )

    summary = await retention_for_topic(db_session, topic_id=parent.id)
    assert summary.windows[7].total == 1
    assert summary.windows[30].total == 2
    assert summary.windows[0].total == 3


async def test_empty_returns_none_retention(db_session: AsyncSession) -> None:
    """Topic with no in-scope reviews → retention is None, not div-by-zero."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="empty")

    summary = await retention_for_topic(db_session, topic_id=parent.id)
    w = summary.windows[7]
    assert w.pass_count == 0
    assert w.fail_count == 0
    assert w.total == 0
    assert w.retention is None


# --- §V31: subtree rollup ---


async def test_subtree_rollup_aggregates_child_reviews(
    db_session: AsyncSession,
) -> None:
    """Parent topic retention = reviews of cards tagged to any descendant + the
    parent itself (§V31 subtree membership). Sibling stays out."""
    cc = await _first_cc(db_session)
    parent, child, sibling = await _make_topic_tree(db_session, cc, label="rollup")

    parent_card = await _make_card(db_session, anki_card_id=_CARD_BASE + 10)
    child_card = await _make_card(db_session, anki_card_id=_CARD_BASE + 11)
    sibling_card = await _make_card(db_session, anki_card_id=_CARD_BASE + 12)
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
                tag_raw="t::rollup::sibling",
                parsed_kind="aamc_topic",
                topic_id=sibling.id,
            ),
        ]
    )
    await db_session.flush()

    # Parent: 1 pass; Child: 1 pass; Sibling: 1 pass (must be excluded from parent).
    await _add_review(db_session, card_pk=parent_card.id, review_id=_REVIEW_BASE + 5_001, ease=3)
    await _add_review(db_session, card_pk=child_card.id, review_id=_REVIEW_BASE + 5_002, ease=4)
    await _add_review(db_session, card_pk=sibling_card.id, review_id=_REVIEW_BASE + 5_003, ease=3)

    parent_summary = await retention_for_topic(db_session, topic_id=parent.id)
    child_summary = await retention_for_topic(db_session, topic_id=child.id)
    sibling_summary = await retention_for_topic(db_session, topic_id=sibling.id)

    assert parent_summary.windows[0].total == 2  # parent + child reviews
    assert child_summary.windows[0].total == 1
    assert sibling_summary.windows[0].total == 1
    assert parent_summary.windows[0].pass_count == 2


async def test_card_with_multiple_in_scope_tags_counted_once(
    db_session: AsyncSession,
) -> None:
    """§V31 dedupe: a card holding both an aamc_topic tag and an aamc_cc tag
    that both fall in scope must count each review exactly once."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="dedupe")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 20)
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
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 6_001, ease=3)

    cc_summary = await retention_for_cc(db_session, cc_code=cc.code)
    assert cc_summary.windows[0].total == 1  # not 2 — dedupe across tags


# --- CC scope: aamc_cc + aamc_topic paths ---


async def test_cc_scope_includes_direct_aamc_cc_tag(db_session: AsyncSession) -> None:
    """A card tagged at CC granularity (aamc_cc, topic_id=NULL) lands in CC scope."""
    cc = await _first_cc(db_session)
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 30)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="cc::direct",
            parsed_kind="aamc_cc",
            cc_id=cc.id,
        )
    )
    await db_session.flush()
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 7_001, ease=3)

    summary = await retention_for_cc(db_session, cc_code=cc.code)
    assert summary.windows[0].total == 1
    assert summary.scope == f"cc:{cc.code}"


async def test_cc_scope_includes_topic_path(db_session: AsyncSession) -> None:
    """A card tagged at topic granularity rolls up to its CC via topic→CC join."""
    cc = await _first_cc(db_session)
    parent, _child, _sib = await _make_topic_tree(db_session, cc, label="topic_path")
    card = await _make_card(db_session, anki_card_id=_CARD_BASE + 31)
    db_session.add(
        _tag(
            note_id=card.note_id,
            tag_raw="cc::via_topic",
            parsed_kind="aamc_topic",
            topic_id=parent.id,
        )
    )
    await db_session.flush()
    await _add_review(db_session, card_pk=card.id, review_id=_REVIEW_BASE + 7_002, ease=3)

    summary = await retention_for_cc(db_session, cc_code=cc.code)
    assert summary.windows[0].total == 1


async def test_cc_scope_excludes_other_cc(db_session: AsyncSession) -> None:
    """Reviews of cards in CC-B don't leak into CC-A retention."""
    cc_a = await _first_cc(db_session)
    cc_b = await _second_cc(db_session)
    card_b = await _make_card(db_session, anki_card_id=_CARD_BASE + 32)
    db_session.add(
        _tag(
            note_id=card_b.note_id,
            tag_raw="cc::other",
            parsed_kind="aamc_cc",
            cc_id=cc_b.id,
        )
    )
    await db_session.flush()
    await _add_review(db_session, card_pk=card_b.id, review_id=_REVIEW_BASE + 7_003, ease=3)

    a = await retention_for_cc(db_session, cc_code=cc_a.code)
    b = await retention_for_cc(db_session, cc_code=cc_b.code)
    assert a.windows[0].total == 0
    assert b.windows[0].total == 1


# --- shape ---


async def test_summary_shape(db_session: AsyncSession) -> None:
    """Returns RetentionSummary; default windows = (7, 30, 0)."""
    cc = await _first_cc(db_session)
    summary = await retention_for_cc(db_session, cc_code=cc.code)
    assert isinstance(summary, RetentionSummary)
    assert set(summary.windows.keys()) == {7, 30, 0}


async def test_custom_windows_param(db_session: AsyncSession) -> None:
    """Caller-supplied windows tuple is honored."""
    cc = await _first_cc(db_session)
    summary = await retention_for_cc(db_session, cc_code=cc.code, windows=(1, 90))
    assert set(summary.windows.keys()) == {1, 90}
