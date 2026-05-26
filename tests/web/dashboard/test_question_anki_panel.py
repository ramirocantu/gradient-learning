"""SPEC §T26 — Anki cards panel on /questions/{id} detail page."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.captures import Question


async def _seed_question(session, *, qid: str = "402391") -> Question:
    q = Question(
        qid=qid,
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a"},
            {"key": "B", "html": "<p>b</p>", "plain": "b"},
        ],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()
    return q


async def _seed_anki_card_for_qid(
    session,
    *,
    anki_card_id: int,
    qid: str,
    deck_name: str = "MileDown",
) -> AnkiCard:
    # §V75: a card's tags live on its note. Seed note (note_id == anki_card_id)
    # before the FK, link the card, attach the uworld_qid tag to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name=deck_name))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name=deck_name,
        note_id=anki_card_id,
        due_date=date.today() + timedelta(days=7),
        interval_days=7,
        queue=2,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=f"#AK_MCAT_v2::#UWorld::{qid}",
            parsed_kind="uworld_qid",
            question_qid=qid,
        )
    )
    await session.flush()
    return card


@pytest.mark.asyncio
async def test_question_detail_renders_empty_anki_panel(client, session) -> None:
    """No Anki cards for this qid → panel renders empty-state copy."""
    q = await _seed_question(session, qid="402391")
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    assert "Anki cards for this qid" in r.text
    assert "No Anki cards reference this qid" in r.text


@pytest.mark.asyncio
async def test_question_detail_renders_anki_card_panel(client, session) -> None:
    """One Anki card tagged with `uworld::qid::N` → renders the card row."""
    q = await _seed_question(session, qid="555555")
    await _seed_anki_card_for_qid(session, anki_card_id=30001, qid="555555", deck_name="MileDown")
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    assert "Anki cards for this qid" in r.text
    assert "#30001" in r.text
    assert "MileDown" in r.text
    assert "qid 555555" in r.text


@pytest.mark.asyncio
async def test_question_detail_anki_panel_filters_by_qid(client, session) -> None:
    """Panel only shows cards for THIS question's qid, not all Anki cards."""
    q = await _seed_question(session, qid="111111")
    await _seed_anki_card_for_qid(session, anki_card_id=30010, qid="111111")
    # Different qid — should NOT appear on /questions/{q.id}
    await _seed_anki_card_for_qid(session, anki_card_id=30011, qid="222222")
    await session.commit()

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    assert "#30010" in r.text
    assert "#30011" not in r.text


@pytest.mark.asyncio
async def test_question_detail_anki_panel_uses_service_helper(client, session, monkeypatch) -> None:
    """V18: route calls `app.services.anki.queries.list_cards_for_qid` in-process."""
    from app.web.dashboard.routes import questions as questions_module

    q = await _seed_question(session, qid="999999")
    await session.commit()

    called_with: list[str] = []

    async def _fake(_session, *, qid: str):
        called_with.append(qid)
        return []

    monkeypatch.setattr(questions_module, "list_cards_for_qid", _fake)

    r = await client.get(f"/questions/{q.id}")
    assert r.status_code == 200
    assert called_with == ["999999"]


def test_question_detail_route_module_no_http_self_call() -> None:
    """V18 regression guard: questions.py module ⊥ import httpx."""
    src = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "web"
        / "dashboard"
        / "routes"
        / "questions.py"
    ).read_text()
    assert "httpx" not in src
