"""Tests for SPEC §T42 — /mastery heatmap re-encoding per §V29.

Each encoding piece (Wilson color bucket / trajectory arrow / Anki
unlock-% bar / retention-30d badge / N<3 ghost fade) is asserted on
the rendered HTML via data-* attributes the template emits for
exactly this purpose. The CARS branch is checked in isolation
because §V29 requires a single-cell section block with no Anki ring
and no retention badge.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory


_CARD_BASE = 900_000
_REVIEW_BASE = 1_900_000_000_000


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


async def _seed_cc_attempts(
    session, *, cc: ContentCategory, n_correct: int, n_wrong: int, base_time: datetime
) -> None:
    """Seed `n_correct + n_wrong` UWorld attempts tagged at the CC."""
    spacing = timedelta(minutes=1)
    i = 0
    for is_correct in [True] * n_correct + [False] * n_wrong:
        q = _make_question(f"Q_T42_{cc.code}_{i}")
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
                attempted_at=base_time + spacing * i,
                selected_choice="A" if is_correct else "B",
                is_correct=is_correct,
            )
        )
        i += 1
    await session.flush()


def _find_cell(html: str, code: str) -> str:
    """Extract the single <a> tag attributes block for a given cell code.

    Returns the substring matching the opening tag (no inner content).
    The template emits `data-cell-code="<code>"` on the anchor; we
    capture from `<a ` up to the first `>` containing that marker.
    """
    pattern = re.compile(
        r'<a\b[^>]*data-cell-code="' + re.escape(code) + r'"[^>]*>',
        re.IGNORECASE,
    )
    m = pattern.search(html)
    assert m is not None, f"cell {code!r} not found in rendered HTML"
    return m.group(0)


def _cell_block(html: str, code: str) -> str:
    """Return the inner-block HTML for a cell (anchor open → closing </a>)."""
    open_re = re.compile(
        r'<a\b[^>]*data-cell-code="' + re.escape(code) + r'"[^>]*>',
        re.IGNORECASE,
    )
    m = open_re.search(html)
    assert m is not None
    start = m.start()
    end = html.find("</a>", m.end())
    assert end != -1
    return html[start:end]


# --- §V29: Wilson color buckets ---


@pytest.mark.asyncio
async def test_high_wilson_cc_renders_green_bucket(client, session):
    cc = await _cc_by_code(session, "1A")
    # 30 correct, 0 wrong → wilson_lower well above 0.70.
    await _seed_cc_attempts(
        session,
        cc=cc,
        n_correct=30,
        n_wrong=0,
        base_time=_now() - timedelta(hours=2),
    )
    await session.commit()
    resp = await client.get("/mastery")
    assert resp.status_code == 200
    tag = _find_cell(resp.text, "1A")
    assert 'data-color-bucket="green"' in tag


@pytest.mark.asyncio
async def test_low_wilson_cc_renders_red_bucket(client, session):
    cc = await _cc_by_code(session, "1A")
    # 1 correct, 29 wrong → ~3% accuracy → wilson well below 0.50.
    await _seed_cc_attempts(
        session,
        cc=cc,
        n_correct=1,
        n_wrong=29,
        base_time=_now() - timedelta(hours=2),
    )
    await session.commit()
    resp = await client.get("/mastery")
    tag = _find_cell(resp.text, "1A")
    assert 'data-color-bucket="red"' in tag


@pytest.mark.asyncio
async def test_zero_attempts_cc_renders_empty_bucket(client, session):
    # No attempts seeded — 1A defaults to attempts=0.
    resp = await client.get("/mastery")
    tag = _find_cell(resp.text, "1A")
    assert 'data-color-bucket="empty"' in tag


# --- §V29: N<3 ghost fade ---


@pytest.mark.asyncio
async def test_low_signal_cc_marked_low_signal_and_faded(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_cc_attempts(
        session,
        cc=cc,
        n_correct=2,
        n_wrong=0,
        base_time=_now() - timedelta(hours=2),
    )
    await session.commit()
    resp = await client.get("/mastery")
    tag = _find_cell(resp.text, "1A")
    assert 'data-low-signal="1"' in tag
    assert "opacity-50" in tag


@pytest.mark.asyncio
async def test_high_signal_cc_not_low_signal(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_cc_attempts(
        session,
        cc=cc,
        n_correct=10,
        n_wrong=0,
        base_time=_now() - timedelta(hours=2),
    )
    await session.commit()
    resp = await client.get("/mastery")
    tag = _find_cell(resp.text, "1A")
    assert 'data-low-signal="0"' in tag


# --- §V36 trajectory arrow ---


@pytest.mark.asyncio
async def test_trajectory_arrow_rendered_when_present(client, session):
    """15+ attempts (last=10 wrong, prior=5 correct) → strong ↓ arrow."""
    cc = await _cc_by_code(session, "1A")
    base = _now() - timedelta(hours=3)
    spacing = timedelta(minutes=1)
    # Older 5 correct (prior window).
    for i in range(5):
        q = _make_question(f"Q_T42_TRAJ_OLD_{i}")
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
                selected_choice="A",
                is_correct=True,
            )
        )
    # Newer 10 wrong (last window).
    for i in range(10):
        q = _make_question(f"Q_T42_TRAJ_NEW_{i}")
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
                attempted_at=base + timedelta(hours=1) + spacing * i,
                selected_choice="B",
                is_correct=False,
            )
        )
    await session.commit()
    resp = await client.get("/mastery")
    block = _cell_block(resp.text, "1A")
    assert 'data-trajectory-arrow="↓"' in block


# --- §V27 / §V28 — Anki unlock bar + retention badge ---


async def _seed_anki_for_cc(session, *, cc: ContentCategory, label: str) -> AnkiCard:
    """Make one mature card linked to the CC + one passing review."""
    anki_card_id = _CARD_BASE + abs(hash(label)) % 50_000
    # §V75: tags live on the note. Seed note (note_id == anki_card_id) before
    # the FK, link the card, attach the aamc_cc tag to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name="AnKing"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="AnKing",
        note_id=anki_card_id,
        queue=2,
        interval_days=30,
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


@pytest.mark.asyncio
async def test_unlock_bar_renders_when_anki_cards_linked(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_anki_for_cc(session, cc=cc, label="unlock_1A")
    await session.commit()
    resp = await client.get("/mastery")
    block = _cell_block(resp.text, "1A")
    assert 'data-unlock-bar="1"' in block
    assert "data-unlock-pct=" in block


@pytest.mark.asyncio
async def test_retention_badge_renders_when_reviews_present(client, session):
    cc = await _cc_by_code(session, "1A")
    await _seed_anki_for_cc(session, cc=cc, label="ret_1A")
    await session.commit()
    resp = await client.get("/mastery")
    block = _cell_block(resp.text, "1A")
    # retention 30d == 100% (one ease=3 review in window)
    assert "data-retention-30d=" in block
    assert "Anki ret" in block


@pytest.mark.asyncio
async def test_no_anki_no_badge_no_bar(client, session):
    # CC w/ no Anki cards.
    resp = await client.get("/mastery")
    block = _cell_block(resp.text, "1A")
    assert "data-unlock-bar=" not in block
    assert "data-retention-30d=" not in block


# --- §V29 CARS single-cell section ---


@pytest.mark.asyncio
async def test_cars_section_is_single_cell_no_anki(client, session):
    """CARS section has one cell (`CARS`) and emits no unlock bar
    nor retention badge — even if AnKing somehow tagged a card."""
    resp = await client.get("/mastery")
    assert resp.status_code == 200
    html = resp.text
    # CARS section data-cars marker.
    cars_section_match = re.search(
        r'<section\s+data-section-name="([^"]+)"\s+data-cars="1"[^>]*>(.*?)</section>',
        html,
        re.DOTALL,
    )
    assert cars_section_match is not None, "CARS section block missing"
    cars_block = cars_section_match.group(2)
    # Exactly one cell anchor inside.
    cells = re.findall(r'data-cell-code="([^"]+)"', cars_block)
    assert len(cells) == 1
    assert cells[0] == "CARS"
    # No Anki bar, no retention badge per §V29.
    assert "data-unlock-bar" not in cars_block
    assert "data-retention-30d" not in cars_block
