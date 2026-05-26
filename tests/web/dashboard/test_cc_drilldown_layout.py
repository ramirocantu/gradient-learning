"""Tests for SPEC §T44 — /mastery/{cc} section reorder + hierarchical topics tree.

Covers:
- §V34 — section order: state-breakdown → anki-review-queue → topics-tree
  → questions-grid.
- §V35 — topics table is a flat tree w/ indent; parent rows bold, child
  rows not; eight columns; rows clickable to V32 drilldown URL.
- §V27 — retention 7d / 30d / all surfaced in the state breakdown.
- §V28 — state buckets {assigned, suspended, new, learning, young, mature}
  surfaced.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.captures import Question
from app.models.outline import ContentCategory, Topic


_CARD_BASE = 970_000
_REVIEW_BASE = 1_970_000_000_000


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


async def _cc_by_code(session, code: str) -> ContentCategory:
    return (
        await session.execute(select(ContentCategory).where(ContentCategory.code == code))
    ).scalar_one()


async def _make_topic(
    session, *, cc: ContentCategory, name: str, parent_id: int | None = None, depth: int = 0
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


async def _seed_anki_card_under_topic(
    session, *, topic: Topic, label: str, due_offset_days: int | None = None
) -> AnkiCard:
    """Mature card linked at topic level + one passing review."""
    anki_card_id = _CARD_BASE + abs(hash(label)) % 50_000
    # §V75: tags live on the note. Seed note (note_id == anki_card_id) before
    # the FK, link the card, attach the aamc_topic tag to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name="AnKing"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="AnKing",
        note_id=anki_card_id,
        queue=2,
        interval_days=30,
        due_date=_now() + timedelta(days=due_offset_days) if due_offset_days is not None else None,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=f"t::{label}",
            topic_id=topic.id,
            parsed_kind="aamc_topic",
            source="regex",
        )
    )
    session.add(
        AnkiCardReview(
            review_id=_REVIEW_BASE + abs(hash(label)) % 200_000,
            card_id=card.id,
            reviewed_at=_now() - timedelta(days=5),
            ease=3,
            type="review",
        )
    )
    await session.flush()
    return card


# --- §V34 section order ---


@pytest.mark.asyncio
async def test_section_order_state_review_topics_questions(client, session):
    """state-breakdown → anki-review-queue → topics-tree → questions-grid."""
    resp = await client.get("/mastery/1A")
    assert resp.status_code == 200
    html = resp.text
    order_markers = [
        ("state-breakdown", html.find('data-section="state-breakdown"')),
        ("anki-review-queue", html.find('data-section="anki-review-queue"')),
        ("topics-tree", html.find('data-section="topics-tree"')),
        ("questions-grid", html.find('data-section="questions-grid"')),
    ]
    # All sections rendered.
    for name, idx in order_markers:
        assert idx != -1, f"missing section data-section={name!r}"
    positions = [idx for _name, idx in order_markers]
    assert positions == sorted(positions), f"sections out of V34 order: {order_markers}"


# --- §V27 / §V28 state breakdown ---


@pytest.mark.asyncio
async def test_state_breakdown_renders_all_buckets(client, session):
    resp = await client.get("/mastery/1A")
    html = resp.text
    for bucket in ("assigned", "new", "learning", "young", "mature", "suspended"):
        assert f'data-bucket="{bucket}"' in html


@pytest.mark.asyncio
async def test_state_breakdown_surfaces_retention_7_30_all(client, session):
    cc = await _cc_by_code(session, "1A")
    topic = await _make_topic(session, cc=cc, name="T44 ret topic")
    await _seed_anki_card_under_topic(session, topic=topic, label="ret_7_30")
    await session.commit()
    resp = await client.get("/mastery/1A")
    html = resp.text
    assert "data-retention-7d=" in html
    assert "data-retention-30d=" in html
    assert "data-retention-all=" in html


# --- §V34 review queue ---


@pytest.mark.asyncio
async def test_review_queue_renders_due_cards_scoped_to_cc(client, session):
    cc = await _cc_by_code(session, "1A")
    topic = await _make_topic(session, cc=cc, name="T44 due topic")
    card = await _seed_anki_card_under_topic(
        session, topic=topic, label="due_today", due_offset_days=0
    )
    await session.commit()
    resp = await client.get("/mastery/1A")
    assert resp.status_code == 200
    assert f'data-queue-card-id="{card.anki_card_id}"' in resp.text


@pytest.mark.asyncio
async def test_review_queue_empty_state_when_nothing_due(client, session):
    resp = await client.get("/mastery/1A")
    assert 'data-queue-empty="1"' in resp.text


# --- §V35 hierarchical topics tree ---


def _row_by_topic_id(html: str, topic_id: int) -> str:
    """Extract the `<tr ...>...</tr>` block matching `data-topic-id`."""
    m = re.search(
        r'<tr[^>]*data-topic-id="' + str(topic_id) + r'"[^>]*>.*?</tr>',
        html,
        re.DOTALL,
    )
    assert m is not None, f"row for topic_id={topic_id} not found"
    return m.group(0)


@pytest.mark.asyncio
async def test_topics_tree_renders_parent_and_child(client, session):
    cc = await _cc_by_code(session, "1A")
    parent = await _make_topic(session, cc=cc, name="T44_PARENT")
    child = await _make_topic(session, cc=cc, name="T44_CHILD", parent_id=parent.id, depth=1)
    await session.commit()
    resp = await client.get("/mastery/1A")
    html = resp.text
    # Both names rendered.
    assert "T44_PARENT" in html
    assert "T44_CHILD" in html
    parent_row = _row_by_topic_id(html, parent.id)
    child_row = _row_by_topic_id(html, child.id)
    assert 'data-topic-has-children="1"' in parent_row
    assert 'data-topic-has-children="0"' in child_row
    # Parent row carries font-bold; child does not (§V35 bold-parent).
    assert "font-bold" in parent_row
    assert "font-bold" not in child_row


@pytest.mark.asyncio
async def test_topics_tree_indents_by_depth(client, session):
    cc = await _cc_by_code(session, "1A")
    parent = await _make_topic(session, cc=cc, name="T44_INDENT_P")
    await _make_topic(session, cc=cc, name="T44_INDENT_C", parent_id=parent.id, depth=1)
    await session.commit()
    resp = await client.get("/mastery/1A")
    html = resp.text
    # Child row uses left padding > 0 (depth=1 → 16px).
    child_row = re.search(
        r'data-topic-depth="1"[^>]*>.*?T44_INDENT_C',
        html,
        re.DOTALL,
    )
    assert child_row is not None
    indent_match = re.search(
        r'padding-left:\s*16px;[^"]*"[^>]*data-topic-link="1">\s*T44_INDENT_C',
        html,
        re.DOTALL,
    )
    assert indent_match is not None


@pytest.mark.asyncio
async def test_topics_tree_row_links_to_v32_drilldown_url(client, session):
    cc = await _cc_by_code(session, "1A")
    parent = await _make_topic(session, cc=cc, name="T44_URL_P")
    child = await _make_topic(session, cc=cc, name="T44_URL_C", parent_id=parent.id, depth=1)
    await session.commit()
    resp = await client.get("/mastery/1A")
    html = resp.text
    # Parent link: /mastery/1A/topics/{parent_id}
    assert f"/mastery/1A/topics/{parent.id}" in html
    # Child link: full id-path /mastery/1A/topics/{parent_id}/{child_id}
    assert f"/mastery/1A/topics/{parent.id}/{child.id}" in html


@pytest.mark.asyncio
async def test_topics_tree_has_eight_columns(client, session):
    """Header row of topics-tree carries 8 <th> cells per V35."""
    resp = await client.get("/mastery/1A")
    html = resp.text
    # Pull the topics-tree section.
    section = re.search(
        r'<section\s+data-section="topics-tree".*?</section>',
        html,
        re.DOTALL,
    )
    assert section is not None
    # Header may not render when topics_tree is empty — seed one.
    if "<thead" not in section.group(0):
        cc = await _cc_by_code(session, "1A")
        await _make_topic(session, cc=cc, name="T44_COL_PROBE")
        await session.commit()
        resp = await client.get("/mastery/1A")
        section = re.search(
            r'<section\s+data-section="topics-tree".*?</section>',
            resp.text,
            re.DOTALL,
        )
        assert section is not None
    thead = re.search(r"<thead.*?</thead>", section.group(0), re.DOTALL)
    assert thead is not None
    th_count = thead.group(0).count("</th>")
    assert th_count == 8, f"expected 8 column headers in V35 topics table, got {th_count}"


@pytest.mark.asyncio
async def test_topics_tree_row_exposes_due_count(client, session):
    cc = await _cc_by_code(session, "1A")
    topic = await _make_topic(session, cc=cc, name="T44_DUE_PROBE")
    await _seed_anki_card_under_topic(session, topic=topic, label="due_probe", due_offset_days=0)
    await session.commit()
    resp = await client.get("/mastery/1A")
    html = resp.text
    row = re.search(
        r'<tr[^>]*data-topic-id="' + str(topic.id) + r'"[^>]*>.*?</tr>',
        html,
        re.DOTALL,
    )
    assert row is not None
    due_match = re.search(r'data-due-count="(\d+)"', row.group(0))
    assert due_match is not None
    assert int(due_match.group(1)) >= 1


# --- CARS skip Anki ---


@pytest.mark.asyncio
async def test_cars_drilldown_omits_anki_sections(client, session):
    resp = await client.get("/mastery/CARS")
    assert resp.status_code == 200
    html = resp.text
    assert 'data-section="state-breakdown"' not in html
    assert 'data-section="anki-review-queue"' not in html
    # Topics-tree section MAY render (CARS has no topics → empty state hint).
    # Questions-grid section must still appear.
    assert 'data-section="questions-grid"' in html
