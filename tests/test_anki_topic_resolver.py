"""Unit tests for SPEC §T32 topic resolver + worker.

Mocks the Anthropic SDK at the boundary (per §V16). Cache is fresh per test
(temp file) so we exercise both miss + hit paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory, Topic
from app.services.anki.topic_resolver import (
    EXTRACTOR_VERSION,
    resolve_topic,
)
from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache
from app.services.anki.topic_resolver_worker import (
    _candidate_notes,
    _filter_anking_tags,
    run,
)


# --------------------------------------------------------------------------- #
# Anthropic stub
# --------------------------------------------------------------------------- #


@dataclass
class _ToolBlock:
    input: dict[str, Any]

    @property
    def __class__(self):  # pragma: no cover
        from anthropic.types import ToolUseBlock

        return ToolUseBlock


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0


@dataclass
class _Message:
    content: list[Any]
    usage: _Usage


class _FakeMessages:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        from anthropic.types import ToolUseBlock

        block = ToolUseBlock.model_construct(
            id="toolu_x",
            name="submit_anki_topic",
            type="tool_use",
            input=self._payload,
        )
        return _Message(content=[block], usage=_Usage())


class _FakeAnthropic:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.messages = _FakeMessages(payload)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _cc_with_topic(session: AsyncSession) -> tuple[ContentCategory, Topic, str]:
    """Pick a real CC + topic from the seed and return the canonical path."""
    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    topic = (
        (
            await session.execute(
                select(Topic).where(
                    Topic.content_category_id == cc.id, Topic.parent_topic_id.is_(None)
                )
            )
        )
        .scalars()
        .first()
    )
    assert topic is not None, "seed 4A must have at least one top-level topic"
    return cc, topic, f"{cc.code} >> {topic.name}"


def _temp_cache(tmp_path: Path) -> AnkiTopicResolverCache:
    return AnkiTopicResolverCache(tmp_path / "topic-cache.db")


# --------------------------------------------------------------------------- #
# resolve_topic — single-card path
# --------------------------------------------------------------------------- #


async def test_resolve_topic_returns_picks_under_cc(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    _, _, topic_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {
                    "topic_path": topic_path,
                    "confidence": 0.82,
                    "rationale": "tag set matches",
                },
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
            extractor_version=EXTRACTOR_VERSION,
        )
    finally:
        cache.close()

    assert len(result.picks) == 1
    assert result.picks[0].topic_path == topic_path
    assert result.picks[0].confidence == pytest.approx(0.82)
    assert result.cache_hit is False
    sent = fake.messages.calls[0]
    assert sent["tool_choice"]["name"] == "submit_anki_topic"


async def test_resolve_topic_attaches_cache_control_to_tool_block(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """§V38: tool def carrying the per-CC topic enum must be cacheable.

    Without `cache_control` on the tool block, Anthropic re-bills the full
    enum (up to ~3.4k tokens for large CCs) on every call (§B4).
    """
    _, _, topic_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.9, "rationale": "ok"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    sent = fake.messages.calls[0]
    tool = sent["tools"][0]
    assert tool.get("cache_control") == {"type": "ephemeral"}, (
        "tool block missing cache_control — topic enum will be re-billed every call"
    )
    # System block must remain cacheable too (existing behavior).
    system = sent["system"][0]
    assert system.get("cache_control") == {"type": "ephemeral"}


async def test_resolve_topic_declined_returns_empty_picks(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    fake = _FakeAnthropic({"decline": True, "topic_picks": []})
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()
    assert result.picks == []


async def test_resolve_topic_rejects_path_not_in_enum(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """Belt-and-suspenders: if the LLM ignores the enum and returns a path
    not in the CC's candidate set, we drop that pick (others kept)."""
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {
                    "topic_path": "9Z >> Bogus_Topic",
                    "confidence": 0.9,
                    "rationale": "off-list",
                },
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()
    assert result.picks == []


async def test_resolve_topic_cache_hit_skips_llm(db_session: AsyncSession, tmp_path: Path) -> None:
    _, _, topic_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.9, "rationale": "match"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    tags = ["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"]
    try:
        # Miss → puts row in cache
        first = await resolve_topic(
            filtered_tags=tags,
            card_text="same text",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
        # Hit → 0 SDK calls past the first
        second = await resolve_topic(
            filtered_tags=tags,
            card_text="same text",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(fake.messages.calls) == 1


async def test_resolve_topic_no_candidates_skips(db_session: AsyncSession, tmp_path: Path) -> None:
    """CARS CC has no topic paths → empty picks without any LLM call."""
    fake = _FakeAnthropic({})
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::CARS::x"],
            card_text="some CARS-tagged card content for resolver",
            cc_code="CARS",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()
    assert result.picks == []
    assert fake.messages.calls == []


async def test_resolve_topic_multi_pick_returns_all_in_enum(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """§V25: a card may legitimately cover multiple topics in one CC. The
    resolver returns one pick per topic, all enum-constrained.
    """
    cc = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    topics = (
        (
            await db_session.execute(
                select(Topic).where(
                    Topic.content_category_id == cc.id, Topic.parent_topic_id.is_(None)
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(topics) >= 2, "seed 4A must have at least two top-level topics for this test"
    path_a = f"{cc.code} >> {topics[0].name}"
    path_b = f"{cc.code} >> {topics[1].name}"
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": path_a, "confidence": 0.9, "rationale": "primary"},
                {"topic_path": path_b, "confidence": 0.7, "rationale": "secondary"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()
    assert [p.topic_path for p in result.picks] == [path_a, path_b]
    assert [p.confidence for p in result.picks] == pytest.approx([0.9, 0.7])


async def test_resolve_topic_tool_enum_uses_integer_ids(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v9: tool-schema `topic_id` enum is integer IDs `[1..N]` instead of
    full-string paths. The canonical natural-language list lives in the
    system block as the model's reasoning surface; server maps id → path.
    This recovers v5 leaf quality without the ~3.4k-token string-enum.
    """
    _, _, topic_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.9, "rationale": "ok"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    sent = fake.messages.calls[0]
    tool = sent["tools"][0]
    pick_item = tool["input_schema"]["properties"]["topic_picks"]["items"]
    assert "topic_id" in pick_item["properties"], (
        "v9 schema must expose `topic_id` integer field, not `topic_path`"
    )
    topic_id_schema = pick_item["properties"]["topic_id"]
    assert topic_id_schema["type"] == "integer"
    enum = topic_id_schema["enum"]
    assert enum, "tool def must ship a non-empty enum for a CC with topics"
    assert enum == list(range(1, len(enum) + 1)), "v9 enum must be 1-indexed contiguous integer IDs"
    assert all(isinstance(v, int) for v in enum)


async def test_resolve_topic_system_block_includes_numbered_canonical_list(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v9: system block carries the canonical topic list as a numbered list
    (`1. {rel_path}`...). The list is the model's reasoning surface; without
    it, v8 measurements showed parent-path drift + wrong-branch errors.
    """
    cc, _, _ = await _cc_with_topic(db_session)
    fake = _FakeAnthropic({"decline": True, "topic_picks": []})
    cache = _temp_cache(tmp_path)
    try:
        await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code=cc.code,
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    sent = fake.messages.calls[0]
    system_text = sent["system"][0]["text"]
    # v10 header is "Topics under {cc} (pick by ID):" — v9's longer
    # "CANONICAL TOPIC PATHS UNDER ..." was the same idea pre-trim.
    assert "Topics under 4A" in system_text, (
        "system block must include the canonical topic list header for the CC"
    )
    # First two IDs must appear as numbered list items.
    assert "\n1. " in system_text
    assert "\n2. " in system_text
    # Rel-path form: prefix factored out, lines should NOT start w/ `4A >> `.
    assert "\n1. 4A >> " not in system_text, (
        "numbered list should be REL paths (CC prefix factored to header)"
    )


async def test_resolve_topic_tool_def_is_strict_and_schema_subset_compliant(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v8: tool def opts into grammar-constrained sampling via `strict: true`
    and satisfies the structured-outputs JSON-schema subset:
      - `additionalProperties: false` on every object schema
      - No `minimum`/`maximum`/`maxItems` (unsupported under strict)

    https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
    """
    _, _, topic_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.9, "rationale": "ok"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    sent = fake.messages.calls[0]
    tool = sent["tools"][0]
    assert tool.get("strict") is True, "v8 tool def must opt into strict mode"

    schema = tool["input_schema"]
    assert schema.get("additionalProperties") is False, (
        "strict mode requires additionalProperties=false on the top-level input_schema object"
    )
    pick_item = schema["properties"]["topic_picks"]["items"]
    assert pick_item.get("additionalProperties") is False, (
        "strict mode requires additionalProperties=false on the pick item object"
    )

    # Forbidden keywords under strict — these were stripped in v8.
    confidence = pick_item["properties"]["confidence"]
    assert "minimum" not in confidence
    assert "maximum" not in confidence
    assert "maxItems" not in schema["properties"]["topic_picks"]


async def test_resolve_topic_dedupes_repeated_topic_ids(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v9: model sometimes emits the same `topic_id` twice in `topic_picks`
    (observed in v8 sample: parent paths repeated). Server dedupes to keep
    one TopicPick per distinct id; first occurrence wins.
    """
    cc, topic, full_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": full_path, "confidence": 0.85, "rationale": "first"},
                {"topic_path": full_path, "confidence": 0.95, "rationale": "dup"},
                {"topic_path": full_path, "confidence": 0.60, "rationale": "dup2"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code=cc.code,
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    # Three picks in, exactly one out (first-occurrence kept).
    assert len(result.picks) == 1
    assert result.picks[0].topic_path == full_path
    assert result.picks[0].rationale == "first"


async def test_resolve_topic_accepts_topic_id_integer_from_model(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v9 happy path: model emits `topic_id` integer; server maps it back to
    the canonical full path via the deterministic position index."""
    cc, topic, full_path = await _cc_with_topic(db_session)
    # Compute the canonical ID for this topic — same deterministic mapping
    # the resolver uses.
    from app.services.anki.topic_resolver import _topic_paths_for_cc

    paths = _topic_paths_for_cc(cc.code)
    topic_id = paths.index(full_path) + 1

    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_id": topic_id, "confidence": 0.87, "rationale": "id form"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code=cc.code,
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    assert len(result.picks) == 1
    assert result.picks[0].topic_path == full_path
    assert result.picks[0].confidence == pytest.approx(0.87)


async def test_resolve_topic_rejects_out_of_range_topic_id(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v9 defense in depth: a `topic_id` outside `[1, N]` is dropped."""
    cc, _, _ = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_id": 99999, "confidence": 0.9, "rationale": "bogus"},
                {"topic_id": 0, "confidence": 0.9, "rationale": "also bogus"},
                {"topic_id": -1, "confidence": 0.9, "rationale": "neg"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code=cc.code,
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()
    assert result.picks == []


async def test_resolve_topic_accepts_relative_path_from_model(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """v6: when the model emits a relative `topic_path` (no `{cc} >> `
    prefix, matching the schema enum verbatim), the server reconstructs the
    canonical full path before validation + persistence."""
    cc, topic, full_path = await _cc_with_topic(db_session)
    rel = topic.name  # under CC, this is the topic name w/o any prefix
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": rel, "confidence": 0.88, "rationale": "rel form"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        result = await resolve_topic(
            filtered_tags=["#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4A-x"],
            card_text="probe card content for resolver test",
            cc_code=cc.code,
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    assert len(result.picks) == 1
    # Server normalizes back to canonical full-path form for downstream code
    # (worker lookup → topic_id by full path).
    assert result.picks[0].topic_path == full_path


async def test_resolve_topic_user_message_includes_both_tags_and_card_text(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """§V25 (hybrid): user message must surface BOTH the filtered tag list
    and the stripped card text. Empirically the tag list disambiguates the
    broad area while the text disambiguates the specific leaf topic (§B8).
    """
    _, _, topic_path = await _cc_with_topic(db_session)
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.9, "rationale": "ok"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    unique_tag = "#AK_MCAT_v2::#FirstAid::Unique_Hybrid_Token"
    unique_text_marker = "HYBRID_TEXT_MARKER_for_pytest"
    try:
        await resolve_topic(
            filtered_tags=[unique_tag],
            card_text=f"some card content that contains {unique_text_marker}",
            cc_code="4A",
            anthropic_client=fake,
            cache=cache,
        )
    finally:
        cache.close()

    sent = fake.messages.calls[0]
    user_msg = sent["messages"][0]["content"]
    assert unique_tag in user_msg, "tag list missing from user message"
    assert unique_text_marker in user_msg, "card text missing from user message"


# --------------------------------------------------------------------------- #
# Tag-filter helper (§V25)
# --------------------------------------------------------------------------- #


def _bare_tag(raw: str) -> AnkiNoteTag:
    """Construct an AnkiNoteTag with just enough fields to drive the filter
    (the filter only reads `tag_raw`). §V75: the filter is note-level."""
    return AnkiNoteTag(
        note_id=0,
        tag_raw=raw,
        parsed_kind="unparsed",
        source="regex",
    )


def test_filter_anking_tags_keeps_taxonomy_drops_noise() -> None:
    note = AnkiNote(note_id=1, deck_name="AnKing MCAT Deck")
    note.tags = [
        _bare_tag("#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4E-Atoms"),
        _bare_tag("#AK_MCAT_v2::#FirstAid::Pulmonology::Embryology"),
        _bare_tag("#AK_MCAT_v2::#Bootcamp::Biochemistry::Amino_Acids"),
        _bare_tag("#AK_MCAT_v2::#UWorld::401649"),  # qid-only, DROP
        _bare_tag("marked"),  # anki internal, DROP
        _bare_tag("leech"),  # anki internal, DROP
        _bare_tag("AnKing MCAT Deck"),  # deck name, DROP
        _bare_tag(""),  # empty, DROP
    ]
    kept = _filter_anking_tags(note)
    assert kept == [
        "#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4E-Atoms",
        "#AK_MCAT_v2::#Bootcamp::Biochemistry::Amino_Acids",
        "#AK_MCAT_v2::#FirstAid::Pulmonology::Embryology",
    ]


def test_filter_anking_tags_sorts_for_stable_cache_key() -> None:
    """Cache key depends on the rendered payload order; the filter sorts
    case-insensitively by tag_raw so the ORM's unordered tag load can't
    produce different cache rows for equivalent tag sets."""
    note = AnkiNote(note_id=1, deck_name="X")
    note.tags = [
        _bare_tag("#AK_MCAT_v2::#Bootcamp::B"),
        _bare_tag("#AK_MCAT_v2::#AAMC::Concepts::C/P::FC04::4A-x"),
        _bare_tag("#AK_MCAT_v2::#FirstAid::F"),
    ]
    kept = _filter_anking_tags(note)
    assert kept[0].endswith("4A-x")
    assert kept[1].endswith("Bootcamp::B")
    assert kept[2].endswith("FirstAid::F")


def test_filter_anking_tags_empty_when_only_noise() -> None:
    note = AnkiNote(note_id=1, deck_name="AnKing MCAT Deck")
    note.tags = [
        _bare_tag("#AK_MCAT_v2::#UWorld::123"),
        _bare_tag("marked"),
        _bare_tag("AnKing MCAT Deck"),
    ]
    assert _filter_anking_tags(note) == []


# --------------------------------------------------------------------------- #
# Worker / orchestrator
# --------------------------------------------------------------------------- #


async def _seed_card_with_cc_tag(
    session: AsyncSession, *, anki_card_id: int, cc: ContentCategory, text: str
) -> AnkiCard:
    """§V75: seed a NOTE carrying the content + the aamc_cc tag, plus a card
    linked to it. The resolver is note-scoped (`_candidate_notes`), reading
    content from `note.fields_json` and persisting onto the note."""
    note = AnkiNote(
        note_id=anki_card_id,
        deck_name="AnKing MCAT Deck",
        fields_json={"Text": {"value": text, "order": 0}},
    )
    session.add(note)
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="AnKing MCAT Deck",
        note_id=note.note_id,
        fields_json={"Text": {"value": text, "order": 0}},
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=note.note_id,
            tag_raw=f"#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::{cc.code}-x",
            parsed_kind="aamc_cc",
            content_category_id=cc.id,
            source="regex",
        )
    )
    await session.flush()
    return card


async def test_worker_persists_topic_tag_with_llm_source(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    cc, _, topic_path = await _cc_with_topic(db_session)
    card = await _seed_card_with_cc_tag(
        db_session,
        anki_card_id=42,
        cc=cc,
        text="Card content about " + topic_path,
    )
    await db_session.commit()

    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.85, "rationale": "matches"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(db_session, anthropic_client=fake, cache=cache, max_cost_usd=10.0)
    finally:
        cache.close()

    assert summary.processed == 1
    assert summary.persisted == 1
    assert summary.declined_by_llm == 0
    assert summary.skipped_low_confidence == 0

    tag = (
        await db_session.execute(
            select(AnkiNoteTag).where(
                AnkiNoteTag.note_id == card.note_id,
                AnkiNoteTag.source == "llm",
            )
        )
    ).scalar_one()
    assert tag.parsed_kind == "aamc_topic"
    assert tag.topic_id is not None
    assert tag.content_category_id == cc.id
    assert float(tag.confidence) == pytest.approx(0.85)
    assert tag.extractor_version == EXTRACTOR_VERSION


async def test_worker_skips_low_confidence(db_session: AsyncSession, tmp_path: Path) -> None:
    cc, _, topic_path = await _cc_with_topic(db_session)
    await _seed_card_with_cc_tag(db_session, anki_card_id=43, cc=cc, text="some text" * 5)
    await db_session.commit()
    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                # Below default 0.5 threshold → dropped server-side.
                {"topic_path": topic_path, "confidence": 0.2, "rationale": "weak match"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(db_session, anthropic_client=fake, cache=cache, max_cost_usd=10.0)
    finally:
        cache.close()
    assert summary.persisted == 0
    assert summary.skipped_low_confidence == 1


async def test_worker_skips_card_with_empty_filtered_tags(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """§V25 skip rule: a card whose only AnKing tag is the UWorld qid (no
    taxonomy signal) must be skipped without an LLM call.
    """
    cc = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    # §V75: candidates are note-scoped. A note carrying ONLY a UWorld qid
    # (no aamc_cc) is never picked up by `_candidate_notes` at all, so the
    # empty-signal skip never even reaches an LLM call.
    note = AnkiNote(note_id=44, deck_name="AnKing MCAT Deck")
    db_session.add(note)
    await db_session.flush()
    card = AnkiCard(anki_card_id=44, deck_name="AnKing MCAT Deck", note_id=note.note_id)
    db_session.add(card)
    await db_session.flush()
    db_session.add(
        AnkiNoteTag(
            note_id=note.note_id,
            tag_raw="#AK_MCAT_v2::#UWorld::401649",
            parsed_kind="uworld_qid",
            content_category_id=None,
            source="regex",
        )
    )
    await db_session.commit()

    fake = _FakeAnthropic({})
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(db_session, anthropic_client=fake, cache=cache, max_cost_usd=10.0)
    finally:
        cache.close()
    # Card has no aamc_cc tag → candidate query excludes it → no LLM call,
    # no DB write. The unused `cc` reference is kept for fixture parity.
    assert summary.processed == 0
    assert fake.messages.calls == []
    _ = cc  # noqa: F841


async def test_worker_persists_multiple_picks_per_card(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """§V25/§V75: a note whose LLM returns 2 topic_picks under the same CC ends
    up with 2 anki_note_tags rows (parsed_kind='aamc_topic', source='llm').
    """
    cc = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    topics = (
        (
            await db_session.execute(
                select(Topic).where(
                    Topic.content_category_id == cc.id, Topic.parent_topic_id.is_(None)
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(topics) >= 2
    path_a = f"{cc.code} >> {topics[0].name}"
    path_b = f"{cc.code} >> {topics[1].name}"

    card = await _seed_card_with_cc_tag(db_session, anki_card_id=50, cc=cc, text="multi-topic card")
    await db_session.commit()

    fake = _FakeAnthropic(
        {
            "decline": False,
            "topic_picks": [
                {"topic_path": path_a, "confidence": 0.9, "rationale": "primary"},
                {"topic_path": path_b, "confidence": 0.7, "rationale": "secondary"},
            ],
        }
    )
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(db_session, anthropic_client=fake, cache=cache, max_cost_usd=10.0)
    finally:
        cache.close()

    assert summary.processed == 1
    assert summary.persisted == 2

    rows = (
        (
            await db_session.execute(
                select(AnkiNoteTag)
                .where(
                    AnkiNoteTag.note_id == card.note_id,
                    AnkiNoteTag.source == "llm",
                    AnkiNoteTag.parsed_kind == "aamc_topic",
                )
                .order_by(AnkiNoteTag.confidence.desc())
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert rows[0].topic_id is not None
    assert rows[1].topic_id is not None
    assert rows[0].topic_id != rows[1].topic_id


async def test_worker_excludes_already_resolved_cards(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """Cards already carrying source='llm' + parsed_kind='aamc_topic' for the
    same CC should not be reprocessed (idempotent run)."""
    cc, topic, _ = await _cc_with_topic(db_session)
    card = await _seed_card_with_cc_tag(
        db_session, anki_card_id=45, cc=cc, text="long enough text here right now"
    )
    db_session.add(
        AnkiNoteTag(
            note_id=card.note_id,
            tag_raw="__llm_topic__::existing",
            topic_id=topic.id,
            content_category_id=cc.id,
            parsed_kind="aamc_topic",
            source="llm",
            confidence=0.9,
            rationale="prior",
            extractor_version=EXTRACTOR_VERSION,
        )
    )
    await db_session.commit()

    fake = _FakeAnthropic({})
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(db_session, anthropic_client=fake, cache=cache, max_cost_usd=10.0)
    finally:
        cache.close()
    assert summary.processed == 0
    assert fake.messages.calls == []


async def test_worker_respects_cost_cap(db_session: AsyncSession, tmp_path: Path) -> None:
    """Budget=0 → resolver should not enter the loop body for any card."""
    cc, _, topic_path = await _cc_with_topic(db_session)
    await _seed_card_with_cc_tag(db_session, anki_card_id=46, cc=cc, text="enough text" * 5)
    await _seed_card_with_cc_tag(db_session, anki_card_id=47, cc=cc, text="enough text two" * 5)
    await db_session.commit()
    fake = _FakeAnthropic({})
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(
            db_session,
            anthropic_client=fake,
            cache=cache,
            max_cost_usd=0.0,
        )
    finally:
        cache.close()
    assert summary.processed == 0
    assert fake.messages.calls == []


# --------------------------------------------------------------------------- #
# §V41 — transient API error resilience (graceful partial-failure)
# --------------------------------------------------------------------------- #


class _FlakyAnthropic:
    """Anthropic stub that succeeds for the first `succeed_first_n` calls and
    then raises APIError on the next one. Mimics SDK behavior after its own
    retries have been exhausted on a sustained 529 overloaded.
    """

    def __init__(self, ok_payload: dict[str, Any], succeed_first_n: int) -> None:
        self._ok_payload = ok_payload
        self._remaining_ok = succeed_first_n
        self.messages = self  # so `client.messages.create(...)` works
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._remaining_ok > 0:
            self._remaining_ok -= 1
            from anthropic.types import ToolUseBlock

            block = ToolUseBlock.model_construct(
                id="toolu_x",
                name="submit_anki_topic",
                type="tool_use",
                input=self._ok_payload,
            )
            return _Message(content=[block], usage=_Usage())
        # SDK-exhausted retry path: raise APIError.
        import httpx
        from anthropic import APIError

        raise APIError(
            "Overloaded",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            body=None,
        )


async def test_worker_returns_partial_summary_on_api_error_mid_loop(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    """§V41/§B6: a transient Anthropic API error from the SDK (post-retries)
    must NOT crash the drain. The worker breaks the loop, returns the
    accumulated summary w/ `partial_failure=True`, and the scheduler caller
    is free to commit what was processed.
    """
    cc, _, topic_path = await _cc_with_topic(db_session)
    # Seed 3 cards w/ DISTINCT extra tags so each yields a unique
    # filtered_tags payload (else 2nd/3rd hit cache + skip the SDK and the
    # flaky client never fires APIError).
    for i, suffix in enumerate(("alpha", "beta", "gamma"), start=101):
        card = await _seed_card_with_cc_tag(
            db_session, anki_card_id=i, cc=cc, text=f"card {suffix}"
        )
        db_session.add(
            AnkiNoteTag(
                note_id=card.note_id,
                tag_raw=f"#AK_MCAT_v2::#FirstAid::Unique::{suffix}",
                parsed_kind="unparsed",
                source="regex",
            )
        )
    await db_session.commit()

    flaky = _FlakyAnthropic(
        ok_payload={
            "decline": False,
            "topic_picks": [
                {"topic_path": topic_path, "confidence": 0.9, "rationale": "ok"},
            ],
        },
        succeed_first_n=1,  # one card succeeds, second triggers APIError
    )
    cache = _temp_cache(tmp_path)
    try:
        summary = await run(db_session, anthropic_client=flaky, cache=cache, max_cost_usd=10.0)
    finally:
        cache.close()

    assert summary.partial_failure is True
    assert summary.error is not None and "APIError" in summary.error
    # First card processed + persisted before the second call raised.
    assert summary.processed == 1
    assert summary.persisted == 1
    # The successful tag remains in the session (caller commits).
    persisted_rows = (
        (
            await db_session.execute(
                select(AnkiNoteTag).where(
                    AnkiNoteTag.source == "llm",
                    AnkiNoteTag.parsed_kind == "aamc_topic",
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(persisted_rows) == 1


def test_scheduler_constructs_anthropic_client_with_high_max_retries() -> None:
    """§V41: scheduler must override the SDK default (2) so transient 529s
    are absorbed by the client before reaching the worker.
    """
    # Use AST to inspect the construction without executing the scheduler
    # (which would need a live DB + Anthropic API key).
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "scheduler.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Find `_do_run_anki_topic_resolver` and inspect any AsyncAnthropic(...) call inside.
    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_do_run_anki_topic_resolver":
            target_fn = node
            break
    assert target_fn is not None, "scheduler missing _do_run_anki_topic_resolver"

    found_high_retries = False
    for sub in ast.walk(target_fn):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "AsyncAnthropic"
        ):
            for kw in sub.keywords:
                if kw.arg == "max_retries" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, int) and kw.value.value >= 5:
                        found_high_retries = True
    assert found_high_retries, (
        "_do_run_anki_topic_resolver must construct AsyncAnthropic(max_retries=>=5) per §V41"
    )


# --------------------------------------------------------------------------- #
# §V42 — candidate ordering groups by cc_code for prompt-cache locality
# --------------------------------------------------------------------------- #


async def test_candidate_notes_grouped_by_cc_code(db_session: AsyncSession) -> None:
    """§V42/§B7: candidate iteration must drain one CC at a time so Anthropic's
    prompt cache stays hot across same-CC calls. Without contiguous grouping
    each CC switch evicts the prior CC's cache → ~0% hit rate.
    """
    # Two CCs with notes seeded in interleaved (note_id) order to force
    # the query to perform the actual ordering rather than relying on
    # insertion order.
    cc_a = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    cc_b = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "5B"))
    ).scalar_one()

    # Seed in CC-interleaved order: A, B, A, B
    for i, cc in enumerate((cc_a, cc_b, cc_a, cc_b), start=201):
        await _seed_card_with_cc_tag(db_session, anki_card_id=i, cc=cc, text=f"content {i}" * 5)
    await db_session.commit()

    candidates = await _candidate_notes(db_session)
    cc_sequence = [cc_code for _, cc_code in candidates]

    # The candidates must be contiguous by CC — i.e., no CC reappears after
    # a different CC has been seen in between. "AAAA BBBB" is OK; "ABAB" is not.
    seen_then_left: set[str] = set()
    current: str | None = None
    for code in cc_sequence:
        if code != current:
            assert code not in seen_then_left, (
                f"candidate ordering not grouped by cc_code — saw {code!r} again "
                f"after leaving it; full sequence: {cc_sequence}"
            )
            if current is not None:
                seen_then_left.add(current)
            current = code
