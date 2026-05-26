"""SPEC §T46 — mastery dashboard rebuild acceptance pack.

One focused test per axis of the §T35–§T45 work; together they form
the "did the rebuild ship correctly?" gate. Other test files cover
each area in depth — this file is intentionally narrow but
end-to-end so a single `pytest tests/test_mastery_acceptance.py`
verifies the whole feature lane.

Axes covered (one test each):
- §V26 — anki_card_reviews append-only PK keeps re-syncs idempotent
- §V27 — true-retention math matches Anki desktop's reference rule
  (pass = ease ∈ {2,3,4}; learn excluded from both num + denom)
- §V31 — subtree rollups: parent topic aggregates descendant items
- §V29 — heatmap tile encodes Wilson + unlock-bar + arrow + retention
  badge + (non-)ghost
- §V33 — topic page renders six sections in order
- §V37 — `get_anki_performance` MCP envelope shape (data-only, no
  heuristics): `{scope, state{...}, retention{scope, windows[...]}}`
- §V32 — parent-chain validation on the topic drilldown URL → 404
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, Topic
from app.services.analyzer.trajectory import trajectory_for_topic
from app.services.anki.retention import retention_for_topic
from app.services.anki.state import state_for_topic


_AUTH = {"X-Coach-Token": "change_me_before_use"}
_CARD_BASE = 9_990_000
_REVIEW_BASE = 1_999_000_000_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_question(qid: str) -> Question:
    return Question(
        qid=qid,
        passage_id=None,
        stem_html="<p>s</p>",
        stem_plain="s",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_content_hashes": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_content_hashes": []},
        ],
        correct_choice="A",
        explanation_html=None,
        explanation_plain="why",
        uworld_aamc_tags=[],
        needs_categorization=False,
    )


async def _cc_by_code(session: AsyncSession, code: str) -> ContentCategory:
    return (
        await session.execute(select(ContentCategory).where(ContentCategory.code == code))
    ).scalar_one()


async def _make_topic(
    session: AsyncSession,
    *,
    cc: ContentCategory,
    name: str,
    parent_id: int | None = None,
    depth: int = 0,
) -> Topic:
    t = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent_id,
        name=name,
        disciplines=[],
        depth=depth,
        position=999,
    )
    session.add(t)
    await session.flush()
    return t


# --- §V26: append-only revlog idempotency ---


@pytest.mark.asyncio
async def test_sync_log_idempotent_on_reinsert(db_session: AsyncSession) -> None:
    """Re-inserting the same review_ids must not duplicate rows (PK + ON CONFLICT)."""
    card = AnkiCard(anki_card_id=_CARD_BASE + 1, deck_name="AnKing", queue=2)
    db_session.add(card)
    await db_session.flush()

    rows = [
        {
            "review_id": _REVIEW_BASE + 1,
            "card_id": card.id,
            "reviewed_at": _now() - timedelta(days=2),
            "ease": 3,
            "type": "review",
        },
        {
            "review_id": _REVIEW_BASE + 2,
            "card_id": card.id,
            "reviewed_at": _now() - timedelta(days=1),
            "ease": 2,
            "type": "review",
        },
    ]

    stmt = (
        pg_insert(AnkiCardReview)
        .values(rows)
        .on_conflict_do_nothing(index_elements=[AnkiCardReview.review_id])
    )
    await db_session.execute(stmt)
    await db_session.flush()
    # Re-insert the exact same rows — must no-op.
    await db_session.execute(stmt)
    await db_session.flush()

    count = (
        await db_session.execute(
            select(func.count())
            .select_from(AnkiCardReview)
            .where(AnkiCardReview.card_id == card.id)
        )
    ).scalar_one()
    assert count == 2


# --- §V27: retention math matches Anki desktop reference ---


@pytest.mark.asyncio
async def test_retention_math_matches_anki_desktop_reference(
    db_session: AsyncSession,
) -> None:
    """Reference set: 3 pass (ease 2/3/4) + 2 fail (ease 1) + 1 'learn'.

    Anki desktop's "true retention" drops type='learn' from BOTH
    sides → 3 / (3+2) = 60%.
    """
    cc = await _cc_by_code(db_session, "1A")
    topic = await _make_topic(db_session, cc=cc, name="T46 ret reference")

    db_session.add(AnkiNote(note_id=_CARD_BASE + 10, deck_name="AnKing"))
    await db_session.flush()
    card = AnkiCard(
        anki_card_id=_CARD_BASE + 10,
        deck_name="AnKing",
        note_id=_CARD_BASE + 10,
        queue=2,
        interval_days=30,
    )
    db_session.add(card)
    await db_session.flush()
    db_session.add(
        AnkiNoteTag(
            note_id=card.note_id,
            tag_raw="t::ret_ref",
            topic_id=topic.id,
            parsed_kind="aamc_topic",
            source="regex",
        )
    )
    await db_session.flush()

    reviews = [
        (1, 2, "review"),  # pass
        (2, 3, "review"),  # pass
        (3, 4, "review"),  # pass
        (4, 1, "review"),  # fail
        (5, 1, "review"),  # fail
        (6, 3, "learn"),  # excluded
    ]
    for offset, ease, type_ in reviews:
        db_session.add(
            AnkiCardReview(
                review_id=_REVIEW_BASE + 100 + offset,
                card_id=card.id,
                reviewed_at=_now() - timedelta(days=2),
                ease=ease,
                type=type_,
            )
        )
    await db_session.flush()

    summary = await retention_for_topic(db_session, topic_id=topic.id, windows=(7,))
    w = summary.windows[7]
    assert w.pass_count == 3
    assert w.fail_count == 2
    assert w.total == 5
    assert w.retention == pytest.approx(0.6)


# --- §V31: subtree rollup correctness ---


@pytest.mark.asyncio
async def test_subtree_rollup_parent_aggregates_descendants(
    db_session: AsyncSession,
) -> None:
    """Parent topic's Anki + trajectory rollups = parent items ∪ child items."""
    cc = await _cc_by_code(db_session, "1A")
    parent = await _make_topic(db_session, cc=cc, name="T46 parent")
    child = await _make_topic(db_session, cc=cc, name="T46 child", parent_id=parent.id, depth=1)
    sibling = await _make_topic(db_session, cc=cc, name="T46 sibling")

    # Parent: 1 mature card; child: 1 mature card; sibling: 1 mature card.
    for label, t, anki_off in (
        ("parent", parent, 20),
        ("child", child, 21),
        ("sibling", sibling, 22),
    ):
        note_id = _CARD_BASE + anki_off
        db_session.add(AnkiNote(note_id=note_id, deck_name="AnKing"))
        await db_session.flush()
        c = AnkiCard(
            anki_card_id=note_id,
            deck_name="AnKing",
            note_id=note_id,
            queue=2,
            interval_days=30,
        )
        db_session.add(c)
        await db_session.flush()
        db_session.add(
            AnkiNoteTag(
                note_id=note_id,
                tag_raw=f"t::sub_{label}",
                topic_id=t.id,
                parsed_kind="aamc_topic",
                source="regex",
            )
        )
    await db_session.flush()

    parent_state = await state_for_topic(db_session, topic_id=parent.id)
    child_state = await state_for_topic(db_session, topic_id=child.id)
    sibling_state = await state_for_topic(db_session, topic_id=sibling.id)

    # Parent subtree = parent + child = 2 mature cards. Sibling stays out.
    assert parent_state.mature == 2
    assert child_state.mature == 1
    assert sibling_state.mature == 1

    # Trajectory subtree-membership: seed 15 attempts split parent/child;
    # parent's trajectory should see all 15.
    spacing = timedelta(minutes=1)
    base = _now() - timedelta(hours=2)
    for i in range(8):
        q = _make_question(f"Q_T46_sub_p_{i}")
        db_session.add(q)
        await db_session.flush()
        db_session.add(
            QuestionTag(question_id=q.id, topic_id=parent.id, confidence=0.9, source="llm")
        )
        db_session.add(
            Attempt(
                question_id=q.id,
                attempted_at=base + spacing * i,
                selected_choice="A",
                is_correct=True,
            )
        )
    for i in range(7):
        q = _make_question(f"Q_T46_sub_c_{i}")
        db_session.add(q)
        await db_session.flush()
        db_session.add(
            QuestionTag(question_id=q.id, topic_id=child.id, confidence=0.9, source="llm")
        )
        db_session.add(
            Attempt(
                question_id=q.id,
                attempted_at=base + spacing * (i + 8),
                selected_choice="B",
                is_correct=False,
            )
        )
    await db_session.flush()

    traj = await trajectory_for_topic(db_session, topic_id=parent.id)
    # 15 in subtree → last 10 = newest (7 child wrong + 3 parent correct = 3/10)
    # prior 5 = oldest 5 parent correct = 5/5.
    assert traj.last.n == 10
    assert traj.prior.n == 5
    assert traj.last.correct == 3
    assert traj.prior.correct == 5


# --- §V29: heatmap tile encoding ---


@pytest.mark.asyncio
async def test_heatmap_tile_encoding_kitchen_sink(client, session) -> None:
    """One CC tile carries every encoding axis: color, unlock bar,
    trajectory arrow, retention badge, non-ghost."""
    cc = await _cc_by_code(session, "1A")
    # 15 UWorld attempts: oldest 5 correct (prior window), newest 10 mixed (last).
    base = _now() - timedelta(hours=3)
    for i in range(5):
        q = _make_question(f"Q_T46_HS_OLD_{i}")
        session.add(q)
        await session.flush()
        session.add(
            QuestionTag(question_id=q.id, content_category_id=cc.id, confidence=0.9, source="llm")
        )
        session.add(
            Attempt(
                question_id=q.id,
                attempted_at=base + timedelta(minutes=i),
                selected_choice="A",
                is_correct=True,
            )
        )
    for i in range(10):
        q = _make_question(f"Q_T46_HS_NEW_{i}")
        session.add(q)
        await session.flush()
        session.add(
            QuestionTag(question_id=q.id, content_category_id=cc.id, confidence=0.9, source="llm")
        )
        session.add(
            Attempt(
                question_id=q.id,
                attempted_at=base + timedelta(hours=1) + timedelta(minutes=i),
                selected_choice="A" if i < 8 else "B",
                is_correct=i < 8,
            )
        )
    # Anki card + review for unlock-bar + retention badge.
    session.add(AnkiNote(note_id=_CARD_BASE + 50, deck_name="AnKing"))
    await session.flush()
    anki_card = AnkiCard(
        anki_card_id=_CARD_BASE + 50,
        deck_name="AnKing",
        note_id=_CARD_BASE + 50,
        queue=2,
        interval_days=30,
    )
    session.add(anki_card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card.note_id,
            tag_raw="t::heat",
            content_category_id=cc.id,
            parsed_kind="aamc_cc",
            source="regex",
        )
    )
    session.add(
        AnkiCardReview(
            review_id=_REVIEW_BASE + 500,
            card_id=anki_card.id,
            reviewed_at=_now() - timedelta(days=3),
            ease=3,
            type="review",
        )
    )
    await session.commit()

    resp = await client.get("/mastery")
    assert resp.status_code == 200
    text = resp.text
    # Find the 1A anchor and inspect its block.
    import re

    open_re = re.compile(
        r'<a\b[^>]*data-cell-code="1A"[^>]*>',
        re.IGNORECASE,
    )
    m = open_re.search(text)
    assert m is not None
    end = text.find("</a>", m.end())
    block = text[m.start() : end]
    # Color bucket present + not "empty" + not low-signal.
    assert re.search(r'data-color-bucket="(red|yellow|green)"', block)
    assert 'data-low-signal="0"' in block
    # Anki unlock bar.
    assert 'data-unlock-bar="1"' in block
    assert "data-unlock-pct=" in block
    # Trajectory arrow rendered (15 attempts → 10/5 windows satisfy V36 minimum).
    assert re.search(r'data-trajectory-arrow="[↑↓→]"', block)
    # Retention badge.
    assert "data-retention-30d=" in block


# --- §V33: topic page six sections ---


@pytest.mark.asyncio
async def test_topic_page_renders_all_six_sections(client, session) -> None:
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T46 6sec root")
    await _make_topic(session, cc=cc, name="T46 6sec child", parent_id=root.id, depth=1)
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    assert resp.status_code == 200
    html = resp.text
    sections = (
        "breadcrumb",
        "header",
        "state-breakdown",
        "anki-review-queue",
        "child-topics-tree",
        "questions-grid",
    )
    positions = []
    for name in sections:
        marker = f'data-section="{name}"'
        idx = html.find(marker)
        assert idx != -1, f"missing data-section={name!r}"
        positions.append(idx)
    assert positions == sorted(positions), "V33 sections out of order"


# --- §V37: MCP envelope shape ---


@pytest.mark.asyncio
async def test_get_anki_performance_envelope_shape(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Envelope: `{scope, state, retention{scope, windows[..]}}` — data only."""
    cc = await _cc_by_code(db_session, "1A")
    topic = await _make_topic(db_session, cc=cc, name="T46 envelope")
    db_session.add(AnkiNote(note_id=_CARD_BASE + 80, deck_name="AnKing"))
    await db_session.flush()
    card = AnkiCard(
        anki_card_id=_CARD_BASE + 80,
        deck_name="AnKing",
        note_id=_CARD_BASE + 80,
        queue=2,
        interval_days=30,
    )
    db_session.add(card)
    await db_session.flush()
    db_session.add(
        AnkiNoteTag(
            note_id=card.note_id,
            tag_raw="t::envelope",
            topic_id=topic.id,
            parsed_kind="aamc_topic",
            source="regex",
        )
    )
    db_session.add(
        AnkiCardReview(
            review_id=_REVIEW_BASE + 800,
            card_id=card.id,
            reviewed_at=_now() - timedelta(days=2),
            ease=3,
            type="review",
        )
    )
    await db_session.commit()

    r = await client.get(
        "/api/v1/anki/performance",
        params={"topic_id": topic.id},
        headers=_AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["scope"] == f"topic:{topic.id}"
    # state subobject.
    state = body["state"]
    for key in (
        "scope",
        "total_cards",
        "assigned",
        "suspended",
        "new",
        "learning",
        "young",
        "mature",
        "unlock_pct",
    ):
        assert key in state, f"state missing key {key!r}"
    # retention subobject + windows[].
    ret = body["retention"]
    assert ret["scope"] == f"topic:{topic.id}"
    assert isinstance(ret["windows"], list)
    for w in ret["windows"]:
        for key in ("window_days", "pass_count", "fail_count", "total", "retention"):
            assert key in w
    # Heuristic-free: no advisory fields snuck in (V37 = data only).
    forbidden = {"verdict", "good", "recommendation", "advice", "rating"}
    leaked = forbidden & body.keys()
    assert not leaked, f"V37 violation — heuristic fields in envelope: {leaked}"


# --- §V32: parent-chain 404 ---


@pytest.mark.asyncio
async def test_parent_chain_invalid_returns_404(client, session) -> None:
    cc = await _cc_by_code(session, "1A")
    a = await _make_topic(session, cc=cc, name="T46 parent_chain_A")
    b = await _make_topic(session, cc=cc, name="T46 parent_chain_B")  # also root, NOT child of a
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{a.id}/{b.id}")
    assert resp.status_code == 404
