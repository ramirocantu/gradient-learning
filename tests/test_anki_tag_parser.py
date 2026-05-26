"""Unit tests for SPEC §T31 tag parser (AnKing shape, CC-level resolution)."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import ContentCategory
from app.services.anki.tag_parser import parse_tag
from app.services.categorizer.outline_lookup import OutlineLookup


@pytest.fixture
async def lookup(db_session: AsyncSession) -> OutlineLookup:
    return await OutlineLookup.load(db_session)


async def _cc_id(session: AsyncSession, code: str) -> int:
    return (
        await session.execute(select(ContentCategory.id).where(ContentCategory.code == code))
    ).scalar_one()


# --- uworld qid ---


def test_uworld_qid_extracts_digits(lookup: OutlineLookup) -> None:
    parsed = parse_tag("#AK_MCAT_v2::#UWorld::402391", outline_lookup=lookup)
    assert parsed.parsed_kind == "uworld_qid"
    assert parsed.question_qid == "402391"
    assert parsed.topic_id is None
    assert parsed.content_category_id is None


def test_uworld_qid_non_numeric_rejected(lookup: OutlineLookup) -> None:
    parsed = parse_tag("#AK_MCAT_v2::#UWorld::abc", outline_lookup=lookup)
    assert parsed.parsed_kind == "unparsed"
    assert parsed.question_qid is None


def test_legacy_uworld_qid_shape_unparsed(lookup: OutlineLookup) -> None:
    """Pre-AnKing `uworld::qid::N` (MileDown conjecture) → unparsed under new regex."""
    parsed = parse_tag("uworld::qid::402391", outline_lookup=lookup)
    assert parsed.parsed_kind == "unparsed"


# --- aamc cc ---


async def test_aamc_cc_resolves_to_content_category(
    db_session: AsyncSession, lookup: OutlineLookup
) -> None:
    cc_4e_id = await _cc_id(db_session, "4E")
    parsed = parse_tag(
        "#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4E-Atoms_Nuclear_Decay_Electronic_Structure_and_Behavior",
        outline_lookup=lookup,
    )
    assert parsed.parsed_kind == "aamc_cc"
    assert parsed.content_category_id == cc_4e_id
    assert parsed.topic_id is None
    assert parsed.question_qid is None


@pytest.mark.parametrize(
    "section_slash, cc_code",
    [
        ("C/P", "5D"),
        ("B/B", "1A"),
        ("P/S", "7A"),
    ],
)
async def test_aamc_cc_section_slash_normalize(
    db_session: AsyncSession,
    lookup: OutlineLookup,
    section_slash: str,
    cc_code: str,
) -> None:
    cc_id = await _cc_id(db_session, cc_code)
    tag = f"#AK_MCAT_v2::#AAMC::Concepts::{section_slash}::Foundational_Concept_03::{cc_code}-Some_Topic_Name_Goes_Here"
    parsed = parse_tag(tag, outline_lookup=lookup)
    assert parsed.parsed_kind == "aamc_cc"
    assert parsed.content_category_id == cc_id


def test_aamc_cc_unknown_cc_demoted_to_unparsed(lookup: OutlineLookup) -> None:
    parsed = parse_tag(
        "#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_99::ZZ-NonExistent_CC",
        outline_lookup=lookup,
    )
    assert parsed.parsed_kind == "unparsed"
    assert parsed.content_category_id is None


def test_aamc_cc_bad_section_unparsed(lookup: OutlineLookup) -> None:
    """Regex only allows (C/P|CARS|B/B|P/S) — bogus section → unparsed."""
    parsed = parse_tag(
        "#AK_MCAT_v2::#AAMC::Concepts::X/Y::Foundational_Concept_01::1A-Foo",
        outline_lookup=lookup,
    )
    assert parsed.parsed_kind == "unparsed"


# --- noise tags from real AnKing notes (per probe 2026-05-19) ---


def test_kaplan_tag_unparsed_until_p13(lookup: OutlineLookup) -> None:
    """Kaplan chapter tags are book TOC refs — out of scope until P13."""
    parsed = parse_tag(
        "#AK_MCAT_v2::#Kaplan::General_Chemistry::Ch-01-Atomic-Structure",
        outline_lookup=lookup,
    )
    assert parsed.parsed_kind == "unparsed"


def test_ankihub_subdeck_tag_unparsed(lookup: OutlineLookup) -> None:
    parsed = parse_tag(
        "AnkiHub_Subdeck::AnKing-MCAT::General-Chemistry",
        outline_lookup=lookup,
    )
    assert parsed.parsed_kind == "unparsed"


def test_unrecognised_tag_unparsed(lookup: OutlineLookup) -> None:
    parsed = parse_tag("LegacyTag::SomeWeirdThing", outline_lookup=lookup)
    assert parsed.parsed_kind == "unparsed"
    assert parsed.topic_id is None
    assert parsed.content_category_id is None
    assert parsed.skill_number is None
    assert parsed.question_qid is None


# --- aamc skill (T34) ---


@pytest.mark.parametrize("skill_num", [1, 2, 3, 4])
def test_aamc_skill_extracts_number(lookup: OutlineLookup, skill_num: int) -> None:
    parsed = parse_tag(
        f"#AK_MCAT_v2::#AAMC::Skills::Skill_{skill_num}-Data_and_Statistics",
        outline_lookup=lookup,
    )
    assert parsed.parsed_kind == "aamc_skill"
    assert parsed.skill_number == skill_num
    assert parsed.topic_id is None
    assert parsed.content_category_id is None
    assert parsed.question_qid is None


def test_aamc_skill_out_of_range_unparsed(lookup: OutlineLookup) -> None:
    """AAMC publishes 4 skills; Skill_5 etc → unparsed (regex constrained)."""
    parsed = parse_tag(
        "#AK_MCAT_v2::#AAMC::Skills::Skill_5-Fictitious_Skill", outline_lookup=lookup
    )
    assert parsed.parsed_kind == "unparsed"
    assert parsed.skill_number is None
