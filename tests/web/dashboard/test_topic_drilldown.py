"""Tests for SPEC §T45 — `GET /mastery/{cc}/topics/{id1}/.../{leaf_id}`.

Covers:
- §V32 — parent-chain validation: every id exists; ids[0] is root in the CC;
  ids[k].parent_topic_id == ids[k-1] for k > 0; any violation → 404.
- §V33 — six-section page (breadcrumb → header → state-breakdown →
  anki-review-queue → child-topics-tree (omit if leaf) → questions-grid).
- §V30 — header is two side-by-side numbers, subtree-scoped.
- §V27 / §V28 — Anki state + windowed retention surfaced in section 3.
- §V35 — child-topics-tree is flat-tree-with-indent (parent bold, child not).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, Topic


_CARD_BASE = 990_000
_REVIEW_BASE = 1_990_000_000_000


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
    anki_card_id = _CARD_BASE + abs(hash(label)) % 80_000
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


async def _seed_question_under_topic(
    session, *, topic: Topic, qid: str, is_correct: bool = True
) -> Question:
    q = _make_question(qid)
    session.add(q)
    await session.flush()
    session.add(
        QuestionTag(
            question_id=q.id,
            topic_id=topic.id,
            confidence=0.9,
            source="llm",
        )
    )
    session.add(
        Attempt(
            question_id=q.id,
            attempted_at=_now(),
            selected_choice="A" if is_correct else "B",
            is_correct=is_correct,
        )
    )
    await session.flush()
    return q


# --- §V32 chain validation ---


@pytest.mark.asyncio
async def test_valid_root_topic_chain_renders_200(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_ROOT")
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    assert resp.status_code == 200
    assert "T45_ROOT" in resp.text


@pytest.mark.asyncio
async def test_valid_two_level_chain_renders_200(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_R1")
    child = await _make_topic(session, cc=cc, name="T45_C1", parent_id=root.id, depth=1)
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}/{child.id}")
    assert resp.status_code == 200
    assert "T45_C1" in resp.text


@pytest.mark.asyncio
async def test_unknown_topic_id_returns_404(client, session):
    resp = await client.get("/mastery/1A/topics/99999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_broken_chain_id2_not_child_of_id1_returns_404(client, session):
    cc = await _cc_by_code(session, "1A")
    root_a = await _make_topic(session, cc=cc, name="T45_BROKEN_A")
    unrelated = await _make_topic(session, cc=cc, name="T45_BROKEN_B")
    await session.commit()
    # unrelated.parent_topic_id is NULL → not a child of root_a → 404.
    resp = await client.get(f"/mastery/1A/topics/{root_a.id}/{unrelated.id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_first_id_not_root_returns_404(client, session):
    """ids[0] must have parent_topic_id IS NULL."""
    cc = await _cc_by_code(session, "1A")
    parent = await _make_topic(session, cc=cc, name="T45_FROOT_P")
    child = await _make_topic(session, cc=cc, name="T45_FROOT_C", parent_id=parent.id, depth=1)
    await session.commit()
    # Starting at child (which has a parent) is invalid per V32.
    resp = await client.get(f"/mastery/1A/topics/{child.id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_first_id_wrong_cc_returns_404(client, session):
    """ids[0].content_category_id must match the CC referenced by cc_code."""
    cc_b = await _cc_by_code(session, "1B")
    root_b = await _make_topic(session, cc=cc_b, name="T45_WRONGCC")
    await session.commit()
    # The id refers to a 1B topic but the URL claims 1A → 404.
    resp = await client.get(f"/mastery/1A/topics/{root_b.id}")
    assert resp.status_code == 404
    # But the same id under 1B is fine.
    resp_ok = await client.get(f"/mastery/1B/topics/{root_b.id}")
    assert resp_ok.status_code == 200


# --- §V33 sections + ordering ---


@pytest.mark.asyncio
async def test_section_order_six_sections_in_v33_order(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_ORDER_R")
    child = await _make_topic(session, cc=cc, name="T45_ORDER_C", parent_id=root.id, depth=1)
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    assert resp.status_code == 200
    html = resp.text
    order_markers = [
        ("breadcrumb", html.find('data-section="breadcrumb"')),
        ("header", html.find('data-section="header"')),
        ("state-breakdown", html.find('data-section="state-breakdown"')),
        ("anki-review-queue", html.find('data-section="anki-review-queue"')),
        ("child-topics-tree", html.find('data-section="child-topics-tree"')),
        ("questions-grid", html.find('data-section="questions-grid"')),
    ]
    for name, idx in order_markers:
        assert idx != -1, f"missing section data-section={name!r}"
    positions = [idx for _name, idx in order_markers]
    assert positions == sorted(positions), f"sections out of V33 order: {order_markers}"
    # child reference for build-time symmetry; ensures the child still exists
    assert str(child.id) in html


@pytest.mark.asyncio
async def test_leaf_topic_omits_child_topics_section(client, session):
    cc = await _cc_by_code(session, "1A")
    leaf = await _make_topic(session, cc=cc, name="T45_LEAF")
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{leaf.id}")
    assert resp.status_code == 200
    assert 'data-section="child-topics-tree"' not in resp.text


@pytest.mark.asyncio
async def test_breadcrumb_renders_cc_root_and_leaf(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_BC_R")
    child = await _make_topic(session, cc=cc, name="T45_BC_C", parent_id=root.id, depth=1)
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}/{child.id}")
    html = resp.text
    # CC link with code present, root crumb anchor, leaf crumb marked last.
    assert 'href="/mastery/1A"' in html
    assert f'href="/mastery/1A/topics/{root.id}"' in html
    assert 'data-crumb-last="1"' in html
    assert "T45_BC_C" in html


# --- §V30 header subtree-scoped ---


@pytest.mark.asyncio
async def test_header_uworld_block_counts_subtree_attempts(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_HDR_R")
    child = await _make_topic(session, cc=cc, name="T45_HDR_C", parent_id=root.id, depth=1)
    # Attach attempts under the child topic — subtree rollup should pick them up.
    for i in range(8):
        await _seed_question_under_topic(
            session, topic=child, qid=f"Q_T45_HDR_{i}", is_correct=True
        )
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    html = resp.text
    assert 'data-uworld-n="8"' in html
    m = re.search(r'data-wilson-pct="([^"]+)"', html)
    assert m and float(m.group(1)) > 0


@pytest.mark.asyncio
async def test_header_anki_block_reflects_subtree_coverage(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_ANKI_R")
    child = await _make_topic(session, cc=cc, name="T45_ANKI_C", parent_id=root.id, depth=1)
    await _seed_anki_card_under_topic(session, topic=child, label="t45_anki")
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    html = resp.text
    m_em = re.search(r'data-effective-mastery="([^"]+)"', html)
    assert m_em is not None
    assert m_em.group(1) != ""


# --- §V27 / §V28 state breakdown ---


@pytest.mark.asyncio
async def test_state_breakdown_renders_all_buckets(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_STATE_R")
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    html = resp.text
    for bucket in ("assigned", "new", "learning", "young", "mature", "suspended"):
        assert f'data-bucket="{bucket}"' in html


@pytest.mark.asyncio
async def test_state_breakdown_surfaces_three_retention_windows(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_RET_R")
    await _seed_anki_card_under_topic(session, topic=root, label="t45_ret")
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    html = resp.text
    assert "data-retention-7d=" in html
    assert "data-retention-30d=" in html
    assert "data-retention-all=" in html


# --- §V33 sec 4 review queue ---


@pytest.mark.asyncio
async def test_review_queue_lists_due_card_in_subtree(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_QUEUE_R")
    child = await _make_topic(session, cc=cc, name="T45_QUEUE_C", parent_id=root.id, depth=1)
    card = await _seed_anki_card_under_topic(
        session, topic=child, label="t45_due", due_offset_days=0
    )
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    assert f'data-queue-card-id="{card.anki_card_id}"' in resp.text


# --- §V33 sec 6 q-grid ---


@pytest.mark.asyncio
async def test_questions_grid_counts_subtree_questions(client, session):
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_QG_R")
    child = await _make_topic(session, cc=cc, name="T45_QG_C", parent_id=root.id, depth=1)
    await _seed_question_under_topic(session, topic=child, qid="Q_T45_QG_1")
    await _seed_question_under_topic(session, topic=root, qid="Q_T45_QG_2")
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    html = resp.text
    # The "Questions in this subtree (2)" header should appear.
    assert "Questions in this subtree" in html
    assert "(2)" in html


# --- §V35 child-topics tree shape ---


@pytest.mark.asyncio
async def test_child_topics_tree_excludes_root_topic_itself(client, session):
    """Header carries the root's metric; the children table starts at the
    root's immediate children (depth=0 relative to root)."""
    cc = await _cc_by_code(session, "1A")
    root = await _make_topic(session, cc=cc, name="T45_NO_SELF_R")
    child = await _make_topic(session, cc=cc, name="T45_NO_SELF_C", parent_id=root.id, depth=1)
    await session.commit()
    resp = await client.get(f"/mastery/1A/topics/{root.id}")
    html = resp.text
    # Root topic should NOT appear as a row inside the child-topics-tree.
    section = re.search(
        r'<section\s+data-section="child-topics-tree".*?</section>',
        html,
        re.DOTALL,
    )
    assert section is not None
    block = section.group(0)
    assert f'data-topic-id="{root.id}"' not in block
    assert f'data-topic-id="{child.id}"' in block
