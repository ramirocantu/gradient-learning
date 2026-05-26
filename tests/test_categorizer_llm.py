"""Tests for the LLM categorizer service (Ticket 3.3).

The Anthropic SDK is mocked at the boundary — `AsyncAnthropic().messages.create`
is a `MagicMock` that returns a forged Message-shaped object. No real API calls
are ever made.

Convention: any test whose assertions depend on per-token pricing (cost math,
budget caps) MUST pin `settings.CATEGORIZER_MODEL` via monkeypatch. The
production default may change (it has — 3.5 flipped Sonnet → Haiku); pinning
keeps these tests true regardless of what the runtime default is.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.services.categorizer import llm
from app.services.categorizer.outline_lookup import OutlineLookup


# --------------------------------------------------------------------------- #
# Helpers — build fake SDK responses and minimal Question stand-ins
# --------------------------------------------------------------------------- #


def _make_question(
    *,
    qid: str = "Q1",
    stem: str = "When a 5 kg box slides 2 m...",
    explanation: str = "Work equals force times distance.",
    uworld_tags=("Subject: Physics", "Chapter: 1. Motion, Force, and Energy"),
):
    # Real ORM Question has more columns; the LLM categorizer only touches qid,
    # stem_plain, explanation_plain, and uworld_aamc_tags. Use a duck-typed stub.
    return SimpleNamespace(
        qid=qid,
        stem_plain=stem,
        explanation_plain=explanation,
        uworld_aamc_tags=list(uworld_tags),
    )


def _tool_use_block(**input_data):
    """Forge an anthropic.types.ToolUseBlock-shaped object (passes isinstance)."""
    from anthropic.types import ToolUseBlock

    return ToolUseBlock(
        id="toolu_1",
        name="submit_aamc_categorization",
        input=input_data,
        type="tool_use",
    )


def _forge_message(
    *,
    tool_input: dict | None,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_create: int = 0,
):
    """Return a stand-in for anthropic.types.Message with the fields llm.py reads."""
    content = [_tool_use_block(**tool_input)] if tool_input is not None else []
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )
    return SimpleNamespace(content=content, usage=usage)


def _make_client(message):
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return client


@pytest.fixture(autouse=True)
def _reset_cache():
    llm._clear_cache_for_tests()
    yield
    llm._clear_cache_for_tests()


@pytest.fixture
def empty_lookup():
    """OutlineLookup with empty maps — categorize() doesn't touch it."""
    return OutlineLookup(sections_by_code={}, ccs_by_code={}, topics=[])


# --------------------------------------------------------------------------- #
# 1. Prompt shape
# --------------------------------------------------------------------------- #


async def test_categorize_calls_anthropic_with_expected_prompt_shape(empty_lookup):
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "skill",
                    "identifier": 2,
                    "confidence": 0.95,
                    "rationale": "calc",
                }
            ],
        }
    )
    client = _make_client(msg)

    await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    client.messages.create.assert_awaited_once()
    kwargs = client.messages.create.await_args.kwargs

    assert kwargs["model"] == llm.MODEL
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": "submit_aamc_categorization",
    }
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["name"] == "submit_aamc_categorization"

    system_blocks = kwargs["system"]
    assert any("AAMC outline for section CP" in b["text"] for b in system_blocks), (
        "system message must carry the CP outline"
    )
    assert any(b.get("cache_control") == {"type": "ephemeral"} for b in system_blocks), (
        "outline block must be marked cache_control=ephemeral"
    )
    # 3.5: system message must include the canonical identifier list with
    # at least one expected topic_path for CP ("4A >> Translational Motion").
    assert any("CANONICAL IDENTIFIERS FOR THIS SECTION" in b["text"] for b in system_blocks), (
        "system message must carry the canonical identifiers block"
    )
    assert any("4A >> Translational Motion" in b["text"] for b in system_blocks), (
        "canonical block must contain expected CP topic_path"
    )

    # T55 V44: tool schema constrains topic via integer `topic_id` enum [1..N].
    # Topic_path is no longer in the schema — server maps topic_id → canonical
    # path via the section's stable position index, surfaced as a numbered
    # list in the system block (model's reasoning surface).
    tool = kwargs["tools"][0]
    tag_props = tool["input_schema"]["properties"]["tags"]["items"]["properties"]
    assert "topic_id" in tag_props
    assert tag_props["topic_id"]["type"] == "integer"
    assert "enum" in tag_props["topic_id"]
    # Enum is positional integers — at least 1, at least covers the CP list.
    cp_n = len(tag_props["topic_id"]["enum"])
    assert cp_n > 0
    assert tag_props["topic_id"]["enum"] == list(range(1, cp_n + 1))
    # System block carries the numbered topic-path list as reasoning surface.
    system_text = "\n".join(b["text"] for b in system_blocks)
    assert "Numbered topic paths" in system_text
    assert "4A >> Translational Motion" in system_text

    user_msg = kwargs["messages"][0]
    assert user_msg["role"] == "user"
    body = user_msg["content"]
    assert "Subject: Physics" in body
    assert q.stem_plain in body
    assert q.explanation_plain in body


# --------------------------------------------------------------------------- #
# 2. Structured-output parsing
# --------------------------------------------------------------------------- #


async def test_categorize_parses_structured_output(empty_lookup):
    """3.5 canonical fields: topic_path >> content_category_code >> skill_number."""
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "topic",
                    "topic_path": "4A >> Work",
                    "confidence": 0.92,
                    "rationale": "Question computes W = F·d.",
                },
                {
                    "kind": "content_category",
                    "content_category_code": "4A",
                    "confidence": 0.85,
                    "rationale": "Falls within translational mechanics.",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "Calculation step required.",
                },
            ],
        }
    )
    client = _make_client(msg)

    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    assert result.cache_hit is False
    assert result.primary_aamc_section == "CP"
    assert len(result.suggestions) == 3
    # Topic: identifier is now the full path string.
    assert result.suggestions[0].kind == "topic"
    assert result.suggestions[0].identifier == "4A >> Work"
    assert result.suggestions[0].under_content_category == "4A"
    assert result.suggestions[1].kind == "content_category"
    assert result.suggestions[1].identifier == "4A"
    assert result.suggestions[2].kind == "skill"
    assert result.suggestions[2].identifier == 2
    assert result.extractor_version == llm.EXTRACTOR_VERSION


# --------------------------------------------------------------------------- #
# 3. Result cache
# --------------------------------------------------------------------------- #


async def test_categorize_caches_by_content_hash(empty_lookup, tmp_path):
    from app.services.categorizer.cache import CategorizerCache

    cache = CategorizerCache(tmp_path / "c.db")
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [{"kind": "skill", "identifier": 2, "confidence": 0.9, "rationale": "x"}],
        }
    )
    client = _make_client(msg)

    r1 = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup, cache=cache)
    r2 = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup, cache=cache)

    assert client.messages.create.await_count == 1
    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert r2.estimated_cost_usd == 0.0
    assert r2.cost_saved_usd > 0.0
    assert [s.identifier for s in r2.suggestions] == [s.identifier for s in r1.suggestions]
    cache.close()


# --------------------------------------------------------------------------- #
# 4. Version bump invalidates cache
# --------------------------------------------------------------------------- #


async def test_categorize_invalidates_on_version_bump(empty_lookup, tmp_path):
    from app.services.categorizer.cache import CategorizerCache

    cache = CategorizerCache(tmp_path / "c.db")
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [{"kind": "skill", "identifier": 2, "confidence": 0.9, "rationale": "x"}],
        }
    )
    client = _make_client(msg)

    r1 = await llm.categorize(
        q,
        anthropic_client=client,
        outline_lookup=empty_lookup,
        cache=cache,
        extractor_version="v1",
    )
    assert r1.cache_hit is False

    r2 = await llm.categorize(
        q,
        anthropic_client=client,
        outline_lookup=empty_lookup,
        cache=cache,
        extractor_version="v2-bumped",
    )

    assert client.messages.create.await_count == 2
    assert r2.cache_hit is False
    assert r2.extractor_version == "v2-bumped"
    cache.close()


# --------------------------------------------------------------------------- #
# 5. Section routing by Subject
# --------------------------------------------------------------------------- #


async def test_categorize_filters_outline_by_subject(empty_lookup, monkeypatch):
    """Subject: Physics → CP outline only; render is called with 'CP'."""
    seen_codes: list[str] = []
    real_render = llm.render_outline_for_section

    def spy(code: str) -> str:
        seen_codes.append(code)
        return real_render(code)

    monkeypatch.setattr(llm, "render_outline_for_section", spy)

    q = _make_question(uworld_tags=["Subject: Physics", "Chapter: X"])
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [{"kind": "skill", "identifier": 1, "confidence": 0.8, "rationale": "y"}],
        }
    )
    client = _make_client(msg)
    await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    assert seen_codes == ["CP"]


# --------------------------------------------------------------------------- #
# 6. Unrecognized subject fails fast
# --------------------------------------------------------------------------- #


async def test_categorize_fails_fast_on_unrecognized_subject(empty_lookup):
    q = _make_question(uworld_tags=["Subject: Underwater Basket Weaving"])
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "skill",
                    "identifier": 1,
                    "confidence": 0.8,
                    "rationale": "n/a",
                }
            ],
        }
    )
    client = _make_client(msg)

    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    client.messages.create.assert_not_awaited()
    assert result.suggestions == []
    assert result.primary_aamc_section is None
    assert any("Underwater Basket Weaving" in w for w in result.parse_warnings)


# --------------------------------------------------------------------------- #
# 7. Unknown topic name silently dropped (orchestrator-side concern; here we
#    assert categorize() emits the suggestion as-is — orchestrator drops it)
# --------------------------------------------------------------------------- #


async def test_categorize_emits_unknown_topic_for_orchestrator_to_drop(empty_lookup):
    """LLM may ignore the enum and emit a topic_path not in the canonical list.
    categorize() passes it through (parsed); orchestrator drops it on resolve.
    """
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "topic",
                    "topic_path": "4A >> Totally Made Up Topic",
                    "confidence": 0.7,
                    "rationale": "Hallucinated.",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "Real skill.",
                },
            ],
        }
    )
    client = _make_client(msg)
    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    assert len(result.suggestions) == 2
    assert result.suggestions[0].identifier == "4A >> Totally Made Up Topic"
    assert result.suggestions[0].under_content_category == "4A"


# --------------------------------------------------------------------------- #
# 8. Token + cost accounting
# --------------------------------------------------------------------------- #


async def test_categorize_logs_token_usage(empty_lookup, monkeypatch):
    # Pin model so the cost math below is independent of the production default.
    monkeypatch.setattr(settings, "CATEGORIZER_MODEL", "claude-sonnet-4-6")

    q = _make_question()
    # 5000 input tokens, 300 output tokens, 4000 read from cache, 0 created.
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [{"kind": "skill", "identifier": 2, "confidence": 0.9, "rationale": "x"}],
        },
        input_tokens=5000,
        output_tokens=300,
        cache_read=4000,
        cache_create=0,
    )
    client = _make_client(msg)
    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    # input_tokens (5000) at $3/M + cache_read (4000) at $0.30/M + output (300) at $15/M
    expected = (5000 / 1e6) * 3.0 + (4000 / 1e6) * 0.30 + (300 / 1e6) * 15.0
    assert result.input_tokens == 9000  # 5000 + 0 created + 4000 read
    assert result.output_tokens == 300
    assert result.estimated_cost_usd == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# Ticket 3.5: canonical identifiers
# --------------------------------------------------------------------------- #


def test_canonical_lists_rendered_per_section():
    from app.services.categorizer.outline_render import (
        canonical_identifiers_for_section,
    )

    cp = canonical_identifiers_for_section("CP")
    assert len(cp.topic_paths) > 0
    assert all(" >> " in p for p in cp.topic_paths)
    import re

    pattern = re.compile(r"^[0-9]+[A-Z] >> ")
    assert all(pattern.match(p) for p in cp.topic_paths)
    assert len(cp.content_category_codes) > 0
    assert "4A" in cp.content_category_codes
    assert cp.skill_numbers == (1, 2, 3, 4)


def test_cars_section_has_no_topic_paths():
    from app.services.categorizer.outline_render import (
        canonical_identifiers_for_section,
    )

    cars = canonical_identifiers_for_section("CARS")
    assert cars.topic_paths == ()
    assert cars.content_category_codes == ("CARS",)
    assert cars.skill_numbers == (1, 2, 3, 4)


async def test_prompt_includes_canonical_lists(empty_lookup):
    """3.5: system message carries canonical identifiers block + topic_path."""
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "x",
                }
            ],
        }
    )
    client = _make_client(msg)
    await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)
    kwargs = client.messages.create.await_args.kwargs
    system_text = "\n".join(b["text"] for b in kwargs["system"])
    assert "CANONICAL IDENTIFIERS FOR THIS SECTION" in system_text
    assert "4A >> Translational Motion" in system_text
    assert "Do not paraphrase" in system_text or "do not paraphrase" in system_text.lower()


def test_tool_schema_has_enum_constraints():
    """V44: topic enum is integer-encoded; cc + skill remain small primitive enums."""
    cp_tool = llm._tool_def_for_section("CP")
    tag_item = cp_tool["input_schema"]["properties"]["tags"]["items"]
    topic_id_prop = tag_item["properties"]["topic_id"]
    assert topic_id_prop["type"] == "integer"
    assert "enum" in topic_id_prop
    cp_n = len(topic_id_prop["enum"])
    assert cp_n > 0
    assert topic_id_prop["enum"] == list(range(1, cp_n + 1))

    cc_prop = tag_item["properties"]["content_category_code"]
    assert "enum" in cc_prop
    assert "4A" in cc_prop["enum"]
    skill_prop = tag_item["properties"]["skill_number"]
    assert skill_prop["enum"] == [1, 2, 3, 4]

    cars_tool = llm._tool_def_for_section("CARS")
    cars_topic_id = cars_tool["input_schema"]["properties"]["tags"]["items"]["properties"][
        "topic_id"
    ]
    # No enum on CARS because there are no canonical topic_paths.
    assert "enum" not in cars_topic_id


def test_parse_topic_path_well_formed():
    assert llm.parse_topic_path("4A >> Translational Motion") == (
        "4A",
        ["Translational Motion"],
    )
    assert llm.parse_topic_path("5A >> Solubility >> Ksp") == (
        "5A",
        ["Solubility", "Ksp"],
    )
    # Topic name may contain parentheses — still parsed as a single segment.
    assert llm.parse_topic_path("5D >> Acid Derivatives (Anhydrides, Amides, Esters)") == (
        "5D",
        ["Acid Derivatives (Anhydrides, Amides, Esters)"],
    )


def test_parse_topic_path_malformed_raises():
    import pytest

    for bad in ("", "   ", "no-arrow", "4A >>", ">> Topic", " >> ", None):
        with pytest.raises((ValueError, TypeError)):
            llm.parse_topic_path(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Ticket 6.8 additions
# --------------------------------------------------------------------------- #


def test_tool_schema_topic_path_enum_includes_subtopics():
    """V44 amended: subtopics are reachable via topic_id; verify by inspecting
    the section's canonical topic_paths list (server-side mapping target)."""
    from app.services.categorizer.outline_render import canonical_identifiers_for_section

    cp_paths = canonical_identifiers_for_section("CP").topic_paths
    assert "4A >> Translational Motion" in cp_paths
    sub_paths = [p for p in cp_paths if p.count(" >> ") >= 2]
    assert len(sub_paths) > 0, "Expected depth-1+ paths in section canonical list"

    cp_tool = llm._tool_def_for_section("CP")
    tag_item = cp_tool["input_schema"]["properties"]["tags"]["items"]
    assert len(tag_item["properties"]["topic_id"]["enum"]) == len(cp_paths)


def test_prompt_preamble_contains_leaf_first_rule():
    assert "evaluate the children first" in llm._SYSTEM_PROMPT_PREAMBLE


def test_parse_topic_path_multi_segment():
    assert llm.parse_topic_path("5A >> Solubility >> Ksp") == (
        "5A",
        ["Solubility", "Ksp"],
    )
    assert llm.parse_topic_path("5A >> Solubility") == ("5A", ["Solubility"])


async def test_categorize_round_trips_deep_path(empty_lookup):
    from app.services.categorizer import llm as llm_mod

    q = _make_question(uworld_tags=["Subject: General Chemistry"])
    deep_path = "5A >> Solubility >> Solubility product constant; the equilibrium expression Ksp"
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "topic",
                    "topic_path": deep_path,
                    "confidence": 0.9,
                    "rationale": "Ksp",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                },
            ],
        }
    )
    client = _make_client(msg)
    result = await llm_mod.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    topic_suggestions = [s for s in result.suggestions if s.kind == "topic"]
    assert len(topic_suggestions) == 1
    assert topic_suggestions[0].identifier == deep_path


def test_extractor_version_bumped():
    assert llm.EXTRACTOR_VERSION == "v10-strict"


def test_tool_def_strict_mode_after_int_enum():
    """V45 amended per §B10: strict mode enabled atop V44 int-encoded enum.
    additionalProperties:false on both object levels; no minimum/maximum
    on confidence (server-side clip retained)."""
    tool = llm._tool_def_for_section("CP")
    assert tool.get("strict") is True
    assert tool["input_schema"]["additionalProperties"] is False
    tag_item = tool["input_schema"]["properties"]["tags"]["items"]
    assert tag_item["additionalProperties"] is False
    conf = tag_item["properties"]["confidence"]
    assert "minimum" not in conf
    assert "maximum" not in conf
    # topic_id is integer (V44), not string (was V44 prerequisite for strict).
    assert tag_item["properties"]["topic_id"]["type"] == "integer"


async def test_categorize_resolves_topic_id_via_position_index(empty_lookup):
    """V44: tool input w/ `topic_id` resolves to the canonical path at that
    1-based position; identifier reflects the full path."""
    from app.services.categorizer.outline_render import canonical_identifiers_for_section

    cp_paths = list(canonical_identifiers_for_section("CP").topic_paths)
    pick_idx = cp_paths.index("4A >> Translational Motion") + 1

    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "topic",
                    "topic_id": pick_idx,
                    "confidence": 0.9,
                    "rationale": "translational motion",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "calc",
                },
            ],
        }
    )
    client = _make_client(msg)
    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    topic_sug = next(s for s in result.suggestions if s.kind == "topic")
    assert topic_sug.identifier == "4A >> Translational Motion"
    assert topic_sug.under_content_category == "4A"


async def test_categorize_drops_out_of_range_topic_id(empty_lookup):
    """V44: topic_id outside [1, N] is dropped with a parse warning."""
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "topic",
                    "topic_id": 999999,
                    "confidence": 0.9,
                    "rationale": "out of range",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "ok",
                },
            ],
        }
    )
    client = _make_client(msg)
    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    assert all(s.kind != "topic" for s in result.suggestions)
    assert any("topic_id out of range" in w for w in result.parse_warnings)


def test_user_message_uses_terse_delimiters():
    """V47: user-msg preambles trimmed to terse delimiters."""
    q = _make_question()
    body = llm._format_user_message(q)
    # Old verbose markdown headers gone.
    assert "## Raw UWorld taxonomy tags" not in body
    assert "## Question stem" not in body
    assert "## Explanation" not in body
    assert "Categorize this MCAT question." not in body
    # New terse markers in place.
    assert body.startswith("Tags:\n")
    assert "\nQ:\n" in body
    assert "\nExpl:\n" in body


def test_system_preamble_drops_schema_restatement():
    """V47: schema-restatement prose dropped (kind enum, tool name shoutout)."""
    assert "Submit by calling the `submit_aamc_categorization`" not in llm._SYSTEM_PROMPT_PREAMBLE
    # Empirically-validated rules retained.
    assert "evaluate the children first" in llm._SYSTEM_PROMPT_PREAMBLE
    assert "exactly one `skill`" in llm._SYSTEM_PROMPT_PREAMBLE


def test_tool_def_marks_cache_control():
    """V38: tool def carries cache_control=ephemeral so the per-section
    topic_path enum gets prompt-cached across same-section calls."""
    tool = llm._tool_def_for_section("CP")
    assert tool.get("cache_control") == {"type": "ephemeral"}


# --------------------------------------------------------------------------- #
# T55 Step 1 — server-side dedupe in parse loop (V46)
# --------------------------------------------------------------------------- #


async def test_categorize_dedupes_duplicate_tags_at_parse(empty_lookup):
    """V46: parser drops duplicate (kind, identifier) entries. First wins."""
    q = _make_question()
    msg = _forge_message(
        tool_input={
            "primary_aamc_section": "CP",
            "tags": [
                {
                    "kind": "topic",
                    "topic_path": "4A >> Work",
                    "confidence": 0.92,
                    "rationale": "first",
                },
                {
                    "kind": "topic",
                    "topic_path": "4A >> Work",  # duplicate
                    "confidence": 0.5,
                    "rationale": "second",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,
                    "confidence": 0.9,
                    "rationale": "skill",
                },
                {
                    "kind": "skill",
                    "skill_number": 2,  # duplicate
                    "confidence": 0.7,
                    "rationale": "second skill",
                },
            ],
        }
    )
    client = _make_client(msg)
    result = await llm.categorize(q, anthropic_client=client, outline_lookup=empty_lookup)

    assert len(result.suggestions) == 2
    # First occurrence retained (confidence + rationale from the first row).
    topic = next(s for s in result.suggestions if s.kind == "topic")
    skill = next(s for s in result.suggestions if s.kind == "skill")
    assert topic.confidence == pytest.approx(0.92)
    assert topic.rationale == "first"
    assert skill.confidence == pytest.approx(0.9)
    assert skill.rationale == "skill"
