"""SPEC §T27 — Anki cards panel on /mastery/{cc_code} CC drilldown page."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic


async def _first_cc(session) -> ContentCategory:
    return (await session.execute(select(ContentCategory).limit(1))).scalar_one()


async def _seed_topic(session, *, cc: ContentCategory, name: str) -> Topic:
    t = Topic(
        content_category_id=cc.id,
        parent_topic_id=None,
        name=name,
        disciplines=[],
        depth=0,
        position=999,
    )
    session.add(t)
    await session.flush()
    return t


async def _seed_anki_card_for_topic(
    session,
    *,
    anki_card_id: int,
    topic: Topic,
    deck_name: str = "MileDown",
) -> AnkiCard:
    """Topic-level tag. Reserved for future topic-granular sources (e.g. T32
    LLM-derived 'aamc_topic_llm' rows). AnKing itself does NOT emit topic-level
    tags — see _seed_anki_card_for_cc for the AnKing-style direct CC link."""
    session.add(AnkiNote(note_id=anki_card_id, deck_name=deck_name))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name=deck_name,
        note_id=anki_card_id,
        due_date=date.today() + timedelta(days=3),
        interval_days=3,
        queue=2,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=f"topic-tag::{topic.name}",
            parsed_kind="aamc_topic",
            topic_id=topic.id,
        )
    )
    await session.flush()
    return card


async def _seed_anki_card_for_cc(
    session,
    *,
    anki_card_id: int,
    cc: ContentCategory,
    deck_name: str = "AnKing MCAT Deck",
) -> AnkiCard:
    """Direct CC tag (AnKing-style — parsed_kind='aamc_cc', content_category_id set)."""
    session.add(AnkiNote(note_id=anki_card_id, deck_name=deck_name))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name=deck_name,
        note_id=anki_card_id,
        due_date=date.today() + timedelta(days=3),
        interval_days=3,
        queue=2,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=(
                f"#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_01::"
                f"{cc.code}-Some_Topic_Text"
            ),
            parsed_kind="aamc_cc",
            content_category_id=cc.id,
        )
    )
    await session.flush()
    return card


@pytest.mark.asyncio
async def test_cc_drilldown_renders_empty_anki_panel(client, session) -> None:
    """No Anki cards for any topic in this CC → empty-state copy."""
    cc = await _first_cc(session)
    await session.commit()

    r = await client.get(f"/mastery/{cc.code}")
    assert r.status_code == 200
    assert "Anki cards for this content category" in r.text
    assert "No Anki cards tagged under topics in this content category" in r.text


@pytest.mark.asyncio
async def test_cc_drilldown_renders_anki_cards_for_topics_under_cc(client, session) -> None:
    cc = await _first_cc(session)
    topic = await _seed_topic(session, cc=cc, name="T27 Topic A")
    await _seed_anki_card_for_topic(session, anki_card_id=40001, topic=topic)
    await session.commit()

    r = await client.get(f"/mastery/{cc.code}")
    assert r.status_code == 200
    assert "Anki cards for this content category" in r.text
    assert "#40001" in r.text
    assert "MileDown" in r.text
    assert "T27 Topic A" in r.text


@pytest.mark.asyncio
async def test_cc_drilldown_anki_panel_only_includes_topics_in_this_cc(client, session) -> None:
    """Card tagged under a topic in a DIFFERENT CC must not appear on this CC."""
    ccs = (await session.execute(select(ContentCategory).limit(2))).scalars().all()
    assert len(ccs) >= 2
    cc_a, cc_b = ccs[0], ccs[1]

    topic_a = await _seed_topic(session, cc=cc_a, name="T27 A in CC_A")
    topic_b = await _seed_topic(session, cc=cc_b, name="T27 B in CC_B")
    await _seed_anki_card_for_topic(session, anki_card_id=40010, topic=topic_a)
    await _seed_anki_card_for_topic(session, anki_card_id=40011, topic=topic_b)
    await session.commit()

    r = await client.get(f"/mastery/{cc_a.code}")
    assert r.status_code == 200
    assert "#40010" in r.text
    assert "#40011" not in r.text


@pytest.mark.asyncio
async def test_cc_drilldown_anki_panel_surfaces_direct_cc_link(client, session) -> None:
    """T31 § list_cards_for_cc union: direct `aamc_cc` tag rows (AnKing-shape)
    surface alongside topic→CC links, since AnKing tops out at CC granularity."""
    cc = await _first_cc(session)
    await _seed_anki_card_for_cc(session, anki_card_id=40020, cc=cc)
    await session.commit()

    r = await client.get(f"/mastery/{cc.code}")
    assert r.status_code == 200
    assert "#40020" in r.text


@pytest.mark.asyncio
async def test_cc_drilldown_anki_panel_uses_service_helper(client, session, monkeypatch) -> None:
    """V18: route calls `app.services.anki.queries.list_cards_for_cc` in-process."""
    from app.web.dashboard.routes import topics as topics_module

    cc = await _first_cc(session)
    await session.commit()

    called_with: list[str] = []

    async def _fake(_session, *, cc_code: str, limit: int):
        called_with.append(cc_code)
        return []

    monkeypatch.setattr(topics_module, "list_cards_for_cc", _fake)

    r = await client.get(f"/mastery/{cc.code}")
    assert r.status_code == 200
    assert called_with == [cc.code]


def test_cc_drilldown_route_module_no_http_self_call() -> None:
    """V18 regression guard: topics.py module ⊥ import httpx."""
    src = (
        Path(__file__).resolve().parents[3] / "app" / "web" / "dashboard" / "routes" / "topics.py"
    ).read_text()
    assert "httpx" not in src
