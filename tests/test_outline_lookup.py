"""Tests for OutlineLookup.topic_id_by_path — Ticket 6.8."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import ContentCategory, Topic
from app.services.categorizer.outline_lookup import OutlineLookup


async def _lookup(session: AsyncSession) -> OutlineLookup:
    return await OutlineLookup.load(session)


async def _topic_id_direct(session: AsyncSession, name: str, cc_code: str) -> int:
    return (
        await session.execute(
            select(Topic.id)
            .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
            .where(
                Topic.name == name,
                ContentCategory.code == cc_code,
                Topic.parent_topic_id.is_(None),
            )
        )
    ).scalar_one()


async def test_topic_id_by_path_depth_0(seeded_report, test_engine):
    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        result = lookup.topic_id_by_path("5A >> Solubility")
        expected = await _topic_id_direct(s, "Solubility", "5A")
        assert result == expected
        assert result is not None


async def test_topic_id_by_path_depth_1(seeded_report, test_engine):
    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        child_path = (
            "5A >> Solubility >> Solubility product constant; the equilibrium expression Ksp"
        )
        child_id = lookup.topic_id_by_path(child_path)
        assert child_id is not None

        parent_id = lookup.topic_id_by_path("5A >> Solubility")
        assert parent_id is not None

        # The child topic's parent_topic_id must point to "Solubility"
        child_row = lookup._topics_by_id[child_id]
        assert child_row.parent_topic_id == parent_id


async def test_topic_id_by_path_unknown_cc(seeded_report, test_engine, caplog):
    import logging

    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        with caplog.at_level(logging.WARNING, logger="app.services.categorizer.outline_lookup"):
            result = lookup.topic_id_by_path("9Z >> anything")
        assert result is None
        assert any("9Z" in r.message for r in caplog.records)


async def test_topic_id_by_path_unknown_segment(seeded_report, test_engine, caplog):
    import logging

    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        with caplog.at_level(logging.WARNING, logger="app.services.categorizer.outline_lookup"):
            result = lookup.topic_id_by_path("5A >> Solubility >> not-a-real-child")
        assert result is None
        assert any("not-a-real-child" in r.message for r in caplog.records)


def test_topic_id_by_path_ambiguous_intermediate_returns_none():
    # The topics table has a UNIQUE constraint on (content_category_id, parent_topic_id, name),
    # so ambiguous topic names cannot exist in production data. The test would require
    # raw SQL to bypass the constraint; per spec, we skip it when the constraint exists.
    pytest.skip(
        "UNIQUE constraint uq_topic_cc_parent_name prevents inserting duplicate "
        "topic names — ambiguity is structurally impossible in this schema."
    )


# --------------------------------------------------------------------------- #
# Ticket 6.8b — Unicode typographic apostrophe normalization
# --------------------------------------------------------------------------- #


async def test_topic_id_by_path_resolves_straight_apostrophe_against_curly_stored_name(
    seeded_report, test_engine
):
    """LLM echoes straight ' (U+0027); DB stores curly ' (U+2019). Must resolve."""
    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        # Use straight ASCII apostrophe in input (as LLM would emit)
        straight_path = (
            "6B >> Cognition >> Cognitive development >> Piaget's stages of cognitive development"
        )
        assert "'" in straight_path  # sanity: path has straight apostrophe
        result = lookup.topic_id_by_path(straight_path)
        assert result is not None, (
            "topic_id_by_path must resolve straight-apostrophe path against curly DB row"
        )
        # Verify it's the correct topic (integer ID matches a real DB row)
        topic_row = (await s.execute(select(Topic).where(Topic.id == result))).scalar_one()
        assert "Piaget" in topic_row.name


# --------------------------------------------------------------------------- #
# Fail-loud guard — option 3, complements app.startup.ensure_outline_seeded
# --------------------------------------------------------------------------- #


async def test_outline_lookup_raises_on_empty_tables(test_engine):
    """If a future entrypoint forgets to seed, OutlineLookup must fail loud."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.services.categorizer.outline_lookup import OutlineNotSeededError
    from app.startup import ensure_outline_seeded

    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    # TRUNCATE ... CASCADE bulldozes any FK-attached user data inserted by
    # earlier tests in the session. The final ensure_outline_seeded call
    # re-seeds so subsequent tests see populated outline tables.
    async with factory() as session:
        await session.execute(
            text(
                "TRUNCATE TABLE topics, content_categories, "
                "foundational_concepts, sections RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()

    async with factory() as session:
        with pytest.raises(OutlineNotSeededError):
            await OutlineLookup.load(session)

    await ensure_outline_seeded(session_factory=factory)


async def test_topic_id_by_path_resolves_curly_apostrophe_input(seeded_report, test_engine):
    """Curly-apostrophe path input resolves to same ID as straight-apostrophe path."""
    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        straight_path = (
            "6B >> Cognition >> Cognitive development >> Piaget's stages of cognitive development"
        )
        curly_path = (
            "6B >> Cognition >> Cognitive development >> Piaget’s stages of cognitive development"
        )
        id_straight = lookup.topic_id_by_path(straight_path)
        id_curly = lookup.topic_id_by_path(curly_path)
        assert id_straight is not None
        assert id_curly is not None
        assert id_straight == id_curly, "Both apostrophe variants must resolve to the same topic ID"


# --------------------------------------------------------------------------- #
# §V40 / §B5 — slash-in-name regression guard
# --------------------------------------------------------------------------- #


async def test_topic_id_by_path_resolves_topic_name_containing_slash(seeded_report, test_engine):
    """§B5 regression: topic `Resistivity: ρ = R•A / L` (CC 4C) contains ` / `
    in its name. Under the old ` / ` delimiter the parser mis-split the name
    into two segments and the resolver silently failed. The new ` >> ` delimiter
    (§V40) must round-trip this name cleanly.
    """
    async with AsyncSession(bind=test_engine) as s:
        lookup = await _lookup(s)
        # Top-level parent under 4C is "Circuit Elements"; the offending leaf
        # is a grandchild "Resistivity: ρ = R•A / L" under "Resistance".
        path = "4C >> Circuit Elements >> Resistance >> Resistivity: ρ = R•A / L"
        result = lookup.topic_id_by_path(path)
        assert result is not None, (
            f"topic_id_by_path must resolve a topic name containing literal "
            f"` / ` under the ` >> ` delimiter (§V40); got None for {path!r}"
        )
        # Verify it's the correct topic row.
        topic_row = (await s.execute(select(Topic).where(Topic.id == result))).scalar_one()
        assert "Resistivity" in topic_row.name and "/" in topic_row.name
