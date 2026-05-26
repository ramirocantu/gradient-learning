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
    """V45 reworked: OpenAI function tool with strict args → parsed dict."""
    q = _question()
    completion = make_completion(
        tool_calls=[
            make_tool_call(
                "submit_aamc_categorization",
                {
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
                },
            )
        ]
    )
    client = make_client(completion)

    result = await categorizer_llm.categorize(
        q,
        openai_client=client,
        outline_lookup=empty_lookup,
    )

    client.chat.completions.create.assert_awaited_once()
    kwargs = client.chat.completions.create.await_args.kwargs
    # OpenAI chat-completions shape: messages list, tools list of {type:function,...}.
    assert kwargs["model"] == categorizer_llm._model()
    assert kwargs["tool_choice"]["type"] == "function"
    assert kwargs["tool_choice"]["function"]["name"] == "submit_aamc_categorization"
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["strict"] is True
    # No Anthropic `system=` array — V38 retired, single system message instead.
    assert any(m["role"] == "system" for m in kwargs["messages"])
    # V-L1: usage cached_tokens read from prompt_tokens_details, not inferred.
    assert result.suggestions[0].kind == "skill"
    assert result.suggestions[0].identifier == 2
    assert result.primary_aamc_section == "CP"
    assert result.cache_hit is False


async def test_categorizer_swap_falls_back_when_no_tool_call(empty_lookup):
    """When the model returns prose instead of a tool call, `categorize`
    surfaces parse_warnings and an empty suggestions list (unchanged behavior
    from the Anthropic path)."""
    q = _question()
    completion = make_completion(tool_calls=[], content="(I'd refuse)")
    client = make_client(completion)

    result = await categorizer_llm.categorize(
        q,
        openai_client=client,
        outline_lookup=empty_lookup,
    )

    assert result.suggestions == []
    assert any("did not call" in w for w in result.parse_warnings)


def test_categorizer_tool_def_drops_cache_control():
    """V38 retired: no `cache_control` markers on the OpenAI function tool."""
    tool = categorizer_llm._tool_def_for_section("CP")
    assert tool["type"] == "function"
    assert "cache_control" not in tool
    assert "cache_control" not in tool["function"]


def test_categorizer_pricing_table_is_openai_only():
    """Pricing table swap — no Claude model identifiers leaking into core."""
    for key in categorizer_llm._PRICING:
        assert not key.startswith("claude-")
