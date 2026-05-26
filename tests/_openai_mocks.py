"""Shared helpers for building forged OpenAI `ChatCompletion`-shaped objects.

V16 (amended): all LLM-touching tests mock OpenAI at the SDK boundary. This
module owns the shape so individual test files can stay focused on per-extractor
assertions instead of re-forging the SDK envelope every file.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def make_tool_call(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    """Forge a single `choice.message.tool_calls[i]` entry.

    The real SDK ships `arguments` as a JSON-encoded string under strict mode;
    we honor that so the extractor's `json.loads(arguments)` path runs.
    """
    return SimpleNamespace(
        id="call_test",
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def make_completion(
    *,
    tool_calls: list[SimpleNamespace] | None = None,
    content: str | None = None,
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
    cached_tokens: int = 0,
    logprobs_tokens: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Forge a `ChatCompletion`-shaped object with the fields the services read.

    Pass `tool_calls=[make_tool_call(...)]` for the structured-output path,
    `content="..."` for the plain-completion path (synthesizer / calibrator).
    `logprobs_tokens` is a list of `{token, logprob}` dicts for T7 calibration.
    """
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
    )
    logprobs_block = None
    if logprobs_tokens is not None:
        logprobs_block = SimpleNamespace(
            content=[
                SimpleNamespace(token=tok["token"], logprob=tok["logprob"])
                for tok in logprobs_tokens
            ]
        )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason="stop",
        logprobs=logprobs_block,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(
        id="chatcmpl-test",
        choices=[choice],
        usage=usage,
        model="gpt-4.1-mini",
    )


def make_client(*completions: SimpleNamespace) -> MagicMock:
    """Build an `AsyncOpenAI`-shaped MagicMock that returns the supplied
    `ChatCompletion`s in order (or repeats the last one if more calls arrive).

    Pass a single completion for the common case; pass multiple to script
    retry/multi-call flows.
    """
    if not completions:
        completions = (make_completion(),)
    client = MagicMock()
    client.chat = MagicMock()
    if len(completions) == 1:
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=completions[0])
    else:
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=list(completions))
    return client


def client_with_error(exc: BaseException) -> MagicMock:
    """Client whose `chat.completions.create` raises `exc` on every call."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=exc)
    return client
