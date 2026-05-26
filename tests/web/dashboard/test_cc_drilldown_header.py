"""Tests for SPEC §T43 — /mastery/{cc} header rebuild per §V30.

Header surfaces two numbers side-by-side, never blended:
- UWorld block: Wilson lower bound + N
- Anki block: effective mastery = retention_30d * unlock_pct

CARS branch shows no Anki number (no AAMC tags by construction).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory


_CARD_BASE = 950_000
_REVIEW_BASE = 1_950_000_000_000


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


async def _seed_uworld_for_cc(
    session, *, cc: ContentCategory, n_correct: int, n_wrong: int
) -> None:
    base = _now() - timedelta(hours=2)
    spacing = timedelta(minutes=1)
    for i, is_correct in enumerate([True] * n_correct + [False] * n_wrong):
        q = _make_question(f"Q_T43_{cc.code}_{i}")
        session.add(q)
        await session.flush()
        session.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cc.id,
                confidence=0.9,
                source="llm",
            )
        )
        session.add(
            Attempt(
                question_id=q.id,
                attempted_at=base + spacing * i,
                selected_choice="A" if is_correct else "B",
                is_correct=is_correct,
            )
        )
    await session.flush()


async def _seed_anki_for_cc(
    session, *, cc: ContentCategory, label: str, queue: int = 2, interval: int = 30
) -> AnkiCard:
    """One Anki card linked to the CC + one passing review."""
    anki_card_id = _CARD_BASE + abs(hash(label)) % 50_000
    # §V75: tags live on the note. Seed note (note_id == anki_card_id) before
    # the FK, link the card, attach the aamc_cc tag to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name="AnKing"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="AnKing",
        note_id=anki_card_id,
        queue=queue,
        interval_days=interval,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=f"t::{label}",
            content_category_id=cc.id,
            parsed_kind="aamc_cc",
            source="regex",
        )
    )
    session.add(
        AnkiCardReview(
            review_id=_REVIEW_BASE + abs(hash(label)) % 100_000,
            card_id=card.id,
            reviewed_at=_now() - timedelta(days=5),
            ease=3,
            type="review",
        )
    )
    await session.flush()
    return card


def _header_block(html: str) -> str:
    """Return the smallest HTML region spanning both header blocks.

    Bounding by `</div>` is unreliable — both blocks contain nested
    divs. Take from the UWorld marker to the end of the page; assertions
    look at the full slice (Anki block always renders after UWorld).
    """
    start = html.find("data-uworld-block")
    assert start != -1, "UWorld header block not rendered"
    return html[start:]


# --- §V30: two blocks side-by-side ---


@pytest.mark.asyncio
async def test_header_renders_both_uworld_and_anki_blocks(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_uworld_for_cc(session, cc=cc, n_correct=8, n_wrong=2)
    await _seed_anki_for_cc(session, cc=cc, label="hdr_1A")
    await session.commit()
    resp = await client.get("/mastery/1A")
    assert resp.status_code == 200
    block = _header_block(resp.text)
    assert "data-uworld-block" in block
    assert "data-anki-block" in block
    # UWorld block precedes Anki block in DOM (side-by-side ordering).
    assert block.index("data-uworld-block") < block.index("data-anki-block")


@pytest.mark.asyncio
async def test_header_uworld_block_renders_wilson_and_n(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_uworld_for_cc(session, cc=cc, n_correct=8, n_wrong=2)
    await session.commit()
    resp = await client.get("/mastery/1A")
    block = _header_block(resp.text)
    # Wilson lower bound percentage attribute populated (non-empty).
    m = re.search(r'data-wilson-pct="([^"]+)"', block)
    assert m is not None
    assert m.group(1) != ""
    assert float(m.group(1)) > 0  # 8/10 → wilson > 0
    # N matches the seeded total.
    assert 'data-uworld-n="10"' in block


@pytest.mark.asyncio
async def test_header_uworld_block_dash_when_zero_attempts(client, session):
    # No attempts seeded under 1A.
    resp = await client.get("/mastery/1A")
    block = _header_block(resp.text)
    assert 'data-wilson-pct=""' in block
    assert 'data-uworld-n="0"' in block
    assert "No attempts yet" in block


# --- §V30: Anki block effective mastery ---


@pytest.mark.asyncio
async def test_header_anki_block_renders_effective_mastery(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_anki_for_cc(session, cc=cc, label="em_1A")
    await session.commit()
    resp = await client.get("/mastery/1A")
    block = _header_block(resp.text)
    m_em = re.search(r'data-effective-mastery="([^"]+)"', block)
    m_ret = re.search(r'data-retention-30d="([^"]+)"', block)
    m_un = re.search(r'data-unlock-pct="([^"]+)"', block)
    assert m_em and m_ret and m_un
    em = float(m_em.group(1))
    ret = float(m_ret.group(1))
    unlock = float(m_un.group(1))
    # effective_mastery = retention × unlock (rounded — allow small drift).
    expected = round(ret * unlock / 100, 1)
    assert abs(em - expected) < 0.2


@pytest.mark.asyncio
async def test_header_anki_block_dash_when_no_anki(client, session):
    cc = await _cc_by_code(session, "1A")
    # Seed UWorld but no Anki — Anki block should show "—".
    await _seed_uworld_for_cc(session, cc=cc, n_correct=1, n_wrong=0)
    await session.commit()
    resp = await client.get("/mastery/1A")
    block = _header_block(resp.text)
    assert 'data-effective-mastery=""' in block
    assert "No AnKing coverage" in block


# --- §V30 / §V29 CARS: no Anki number ---


@pytest.mark.asyncio
async def test_header_cars_anki_block_marked_and_dashed(client, session):
    resp = await client.get("/mastery/CARS")
    assert resp.status_code == 200
    block = _header_block(resp.text)
    assert 'data-anki-cars="1"' in block
    assert 'data-effective-mastery=""' in block
    assert "CARS has no AnKing AAMC tags" in block
