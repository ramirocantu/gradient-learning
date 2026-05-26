"""SPEC T93 — note-as-unit schema (§V75).

Covers the note-level schema the note-as-unit refactor introduced:
  * source preservation — regex/llm/manual rows carry their §V24 orthogonal
    columns intact (§V43),
  * FK integrity — anki_cards.note_id and anki_note_tags.note_id reject
    orphan note ids,
  * UNIQUE(note_id, tag_raw) + parsed_kind/source CHECK constraints mirror
    the retired anki_card_tags shape (§V3/§V24),
  * ORM relationships (AnkiNote.cards / AnkiNote.tags / AnkiCard.note).

Tests run against the create_all schema (conftest), which builds the FK on
`anki_cards.note_id` up front.

§V75 (T95) dropped `anki_card_tags`; the one-time fan-out-collapse backfill it
fed (cards-tags → note-tags) is a migration concern verified by
`alembic upgrade head`, not a create_all-schema unit test — the source table no
longer exists in the ORM metadata, so those replay tests were removed.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag


async def _make_note(session: AsyncSession, note_id: int, **overrides) -> AnkiNote:
    note = AnkiNote(
        note_id=note_id,
        deck_name=overrides.get("deck_name", "MileDown"),
        model_name=overrides.get("model_name", "AnKingOverhaul"),
        fields_json=overrides.get("fields_json", {"Text": {"value": "x"}}),
    )
    session.add(note)
    await session.flush()
    return note


async def _make_card(session: AsyncSession, anki_card_id: int, note_id: int | None) -> AnkiCard:
    card = AnkiCard(anki_card_id=anki_card_id, deck_name="MileDown", note_id=note_id, queue=-1)
    session.add(card)
    await session.flush()
    return card


async def _note_tags(session: AsyncSession, note_id: int) -> list[AnkiNoteTag]:
    return list(
        (
            await session.execute(
                select(AnkiNoteTag)
                .where(AnkiNoteTag.note_id == note_id)
                .order_by(AnkiNoteTag.tag_raw)
            )
        )
        .scalars()
        .all()
    )


# --- source + §V24 column preservation (§V43) ---


async def test_note_tags_preserve_all_sources(db_session: AsyncSession) -> None:
    """regex + llm + manual rows on a note all carry confidence/rationale/
    extractor_version intact (§V43/§V24)."""
    note = await _make_note(db_session, 5_000_000_000_002)
    db_session.add_all(
        [
            AnkiNoteTag(
                note_id=note.note_id,
                tag_raw="#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_4::4A-Motion",
                parsed_kind="aamc_cc",
                source="regex",
            ),
            AnkiNoteTag(
                note_id=note.note_id,
                tag_raw="__llm_topic__::anki-v9::4A >> Translational Motion",
                parsed_kind="aamc_topic",
                source="llm",
                confidence=0.82,
                rationale="stem describes constant-velocity motion",
                extractor_version="anki-v9",
            ),
            AnkiNoteTag(
                note_id=note.note_id,
                tag_raw="__manual__::user-override",
                parsed_kind="aamc_topic",
                source="manual",
            ),
        ]
    )
    await db_session.flush()

    rows = await _note_tags(db_session, note.note_id)
    by_source = {r.source: r for r in rows}
    assert set(by_source) == {"regex", "llm", "manual"}
    llm = by_source["llm"]
    assert float(llm.confidence) == pytest.approx(0.82)
    assert llm.rationale == "stem describes constant-velocity motion"
    assert llm.extractor_version == "anki-v9"
    assert by_source["regex"].confidence is None


# --- FK integrity (§V75) ---


async def test_anki_card_note_fk_rejects_orphan(db_session: AsyncSession) -> None:
    """anki_cards.note_id must reference an existing note."""
    db_session.add(AnkiCard(anki_card_id=7001, deck_name="MileDown", note_id=9_999_999_999_999))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_anki_note_tag_fk_rejects_orphan(db_session: AsyncSession) -> None:
    """anki_note_tags.note_id must reference an existing note."""
    db_session.add(
        AnkiNoteTag(note_id=9_999_999_999_998, tag_raw="x", parsed_kind="unparsed", source="regex")
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_anki_note_tag_unique_per_note_tag_raw(db_session: AsyncSession) -> None:
    """§V75 UNIQUE(note_id, tag_raw): the collapse target rejects dup rows."""
    note = await _make_note(db_session, 5_000_000_000_004)
    db_session.add(
        AnkiNoteTag(note_id=note.note_id, tag_raw="dup", parsed_kind="unparsed", source="regex")
    )
    await db_session.flush()
    db_session.add(
        AnkiNoteTag(note_id=note.note_id, tag_raw="dup", parsed_kind="unparsed", source="regex")
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_anki_note_tag_check_constraints_mirror_card_tag(db_session: AsyncSession) -> None:
    """§V3/§V24: parsed_kind + source closed sets mirror the retired anki_card_tags."""
    note = await _make_note(db_session, 5_000_000_000_005)
    db_session.add(
        AnkiNoteTag(
            note_id=note.note_id, tag_raw="bad-kind", parsed_kind="nonsense", source="regex"
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()

    note = await _make_note(db_session, 5_000_000_000_006)
    db_session.add(
        AnkiNoteTag(
            note_id=note.note_id, tag_raw="bad-source", parsed_kind="unparsed", source="bogus"
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


# --- note backfill shape (cards -> notes), via the canonical SELECT DISTINCT ---


async def test_note_distinct_select_collapses_sibling_cards(db_session: AsyncSession) -> None:
    """§V75: N sibling cards share one note. A DISTINCT ON (note_id) over
    anki_cards yields one row per note — the shape the T93 note backfill used."""
    n1, n2 = 6_000_000_000_001, 6_000_000_000_002
    await _make_note(
        db_session, n1, model_name="AnKingOverhaul", fields_json={"Text": {"value": "motion"}}
    )
    await _make_note(db_session, n2, model_name="Cloze", fields_json={"Text": {"value": "enzyme"}})
    await _make_card(db_session, 4001, n1)
    await _make_card(db_session, 4002, n1)  # sibling of 4001
    await _make_card(db_session, 4003, n2)
    await db_session.flush()

    distinct_notes = (
        await db_session.execute(
            select(func.count(func.distinct(AnkiCard.note_id))).where(
                AnkiCard.note_id.in_([n1, n2])
            )
        )
    ).scalar_one()
    assert distinct_notes == 2
    note_count = (
        await db_session.execute(
            select(func.count()).select_from(AnkiNote).where(AnkiNote.note_id.in_([n1, n2]))
        )
    ).scalar_one()
    assert note_count == 2


# --- ORM relationship ---


async def test_anki_note_relationships_load(db_session: AsyncSession) -> None:
    """AnkiNote.cards / AnkiNote.tags and AnkiCard.note resolve."""
    note = await _make_note(db_session, 5_000_000_000_007)
    card = await _make_card(db_session, 8001, note.note_id)
    db_session.add(
        AnkiNoteTag(note_id=note.note_id, tag_raw="t::x", parsed_kind="unparsed", source="regex")
    )
    await db_session.flush()
    await db_session.refresh(note, attribute_names=["cards", "tags"])
    await db_session.refresh(card, attribute_names=["note"])
    assert [c.id for c in note.cards] == [card.id]
    assert len(note.tags) == 1
    assert card.note is not None and card.note.note_id == note.note_id


async def test_anki_card_tags_view_resolves_to_note_tags(db_session: AsyncSession) -> None:
    """§V75: AnkiCard.tags is a viewonly relationship onto the note's tags via
    the shared note_id — so existing readers (AnkiCardOut.tags) keep working."""
    note = await _make_note(db_session, 5_000_000_000_008)
    card = await _make_card(db_session, 8002, note.note_id)
    db_session.add_all(
        [
            AnkiNoteTag(
                note_id=note.note_id,
                tag_raw="aamc::CP::4A::a",
                parsed_kind="aamc_topic",
                source="regex",
            ),
            AnkiNoteTag(
                note_id=note.note_id,
                tag_raw="uworld::qid::123",
                question_qid="123",
                parsed_kind="uworld_qid",
                source="regex",
            ),
        ]
    )
    await db_session.flush()
    await db_session.refresh(card, attribute_names=["tags"])
    assert len(card.tags) == 2
    assert {t.parsed_kind for t in card.tags} == {"aamc_topic", "uworld_qid"}
