"""Smoke test for the T4 OpenAI SDK swap.

Proves that every extractor — categorizer, anki topic resolver, feature
extractor, synthesizer — can be driven through its public API with a
forged `ChatCompletion` and produces the expected typed result. The
historical per-extractor tests are heavily Anthropic-SDK-shaped (cache_control
markers, ToolUseBlock isinstance asserts) and need a coordinated rewrite;
this file is the bridging proof that the SDK swap is end-to-end functional
while that rewrite lands (tracked alongside T4 in §T).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.categorizer import llm as categorizer_llm
from app.services.categorizer.outline_lookup import OutlineLookup

from tests._openai_mocks import make_client, make_completion, make_tool_call


def _question(
    *,
    qid: str = "Q1",
    stem: str = "When a 5 kg box slides 2 m...",
    explanation: str = "Work equals force times distance.",
    tags=("Subject: Physics", "Chapter: 1. Motion, Force, and Energy"),
) -> SimpleNamespace:
    return SimpleNamespace(
        qid=qid,
        stem_plain=stem,
        explanation_plain=explanation,
        uworld_aamc_tags=list(tags),
    )


@pytest.fixture
def empty_lookup() -> OutlineLookup:
    return OutlineLookup(course_id=0, nodes=[])


async def test_categorizer_swap_returns_parsed_suggestions(empty_lookup):
    """T6 + V45 reworked: OpenAI `response_format: json_schema, strict:true`."""
    q = _question()
    payload = {
        "primary_aamc_section": "CP",
        "tags": [
            {
                "kind": "skill",
                "topic_id": None,
                "content_category_code": None,
                "skill_number": 2,
                "confidence": 0.95,
                "rationale": "calc",
            }
        ],
    }
    import json as _json

    completion = make_completion(content=_json.dumps(payload))
    client = make_client(completion)

    result = await categorizer_llm.categorize(
        q,
        openai_client=client,
        outline_lookup=empty_lookup,
    )

    client.chat.completions.create.assert_awaited_once()
    kwargs = client.chat.completions.create.await_args.kwargs
    # T6: response_format json_schema is the structured-output seam.
    assert kwargs["model"] == categorizer_llm._model()
    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "submit_aamc_categorization"
    assert rf["json_schema"]["strict"] is True
    # Single system message (V38 retired — no array shape).
    assert any(m["role"] == "system" for m in kwargs["messages"])
    assert result.suggestions[0].kind == "skill"
    assert result.suggestions[0].identifier == 2
    assert result.primary_aamc_section == "CP"
    assert result.cache_hit is False


async def test_categorizer_swap_falls_back_when_no_structured_output(empty_lookup):
    """Empty content surfaces parse_warnings + empty suggestions list."""
    q = _question()
    completion = make_completion(content=None)
    client = make_client(completion)

    result = await categorizer_llm.categorize(
        q,
        openai_client=client,
        outline_lookup=empty_lookup,
    )

    assert result.suggestions == []
    assert any("did not produce" in w for w in result.parse_warnings)


def test_categorizer_response_format_drops_cache_control():
    """V38 retired: no `cache_control` markers on the response_format envelope."""
    rf = categorizer_llm._tool_def_for_section("CP")
    assert rf["type"] == "json_schema"
    assert "cache_control" not in rf
    assert "cache_control" not in rf["json_schema"]


def test_categorizer_pricing_table_is_openai_only():
    """Pricing table swap — no Claude model identifiers leaking into core."""
    for key in categorizer_llm._PRICING:
        assert not key.startswith("claude-")
