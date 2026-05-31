"""Tests for the OpenAI logprobs calibrator (T7, §V69 amended, §V-T3)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.llm import calibrator
from tests._openai_mocks import make_client


def _logprobs_completion(*, chosen_token: str, top: list[tuple[str, float]]) -> SimpleNamespace:
    """Forge a `ChatCompletion`-shaped object with the logprobs block the
    calibrator reads (`choice.logprobs.content[0].top_logprobs`)."""
    candidates = [SimpleNamespace(token=tok, logprob=lp, bytes=None) for tok, lp in top]
    first = SimpleNamespace(
        token=chosen_token,
        logprob=top[0][1] if top else 0.0,
        bytes=None,
        top_logprobs=candidates,
    )
    logprobs_block = SimpleNamespace(content=[first])
    choice = SimpleNamespace(
        index=0,
        message=SimpleNamespace(role="assistant", content=chosen_token, tool_calls=[]),
        finish_reason="stop",
        logprobs=logprobs_block,
    )
    usage = SimpleNamespace(
        prompt_tokens=42,
        completion_tokens=1,
        total_tokens=43,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    return SimpleNamespace(
        id="chatcmpl-calib",
        choices=[choice],
        usage=usage,
        model="gpt-4.1-mini",
    )


async def test_conf_formula_yes_dominant():
    """Conf = exp(L_yes)/(exp(L_yes)+exp(L_no)) — yes much more likely."""
    completion = _logprobs_completion(
        chosen_token="Yes",
        top=[("Yes", -0.1), ("No", -3.0), ("Maybe", -5.0)],
    )
    client = make_client(completion)

    result = await calibrator.grade_yes_no(
        prompt="Does X describe Y?",
        openai_client=client,
        model="gpt-4.1-mini",
    )

    expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-3.0))
    assert math.isclose(result.confidence, expected, rel_tol=1e-9)
    assert result.manual_review is False
    assert result.chosen_token == "Yes"


async def test_conf_below_half_triggers_manual_review():
    """V-T3: <0.5 ⇒ manual_review=true."""
    completion = _logprobs_completion(
        chosen_token="No",
        top=[("No", -0.2), ("Yes", -2.5)],
    )
    client = make_client(completion)

    result = await calibrator.grade_yes_no(
        prompt="?",
        openai_client=client,
        model="gpt-4.1-mini",
    )

    assert result.confidence < 0.5
    assert result.manual_review is True


async def test_neither_token_present_falls_back_to_zero():
    """When the calibrator's top_logprobs hold neither Yes nor No, conf=0."""
    completion = _logprobs_completion(
        chosen_token="Maybe",
        top=[("Maybe", -0.1), ("Possibly", -1.0)],
    )
    client = make_client(completion)

    result = await calibrator.grade_yes_no(
        prompt="?",
        openai_client=client,
        model="gpt-4.1-mini",
    )

    assert result.confidence == 0.0
    assert result.manual_review is True


async def test_grade_yes_no_uses_max_one_completion_token():
    """V69: discriminator runs on a plain completion with a single emitted token."""
    completion = _logprobs_completion(chosen_token="Yes", top=[("Yes", -0.05), ("No", -3.0)])
    client = make_client(completion)

    await calibrator.grade_yes_no(
        prompt="?",
        openai_client=client,
        model="gpt-4.1-mini",
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["max_completion_tokens"] == 1
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] >= 2
    # Plain completion — no response_format / tools envelope.
    assert "response_format" not in kwargs
    assert "tools" not in kwargs


async def test_grade_yes_no_passes_reasoning_effort_none_by_default():
    """V-L5: reasoning OFF is required on GPT-5.x to get logprobs back."""
    completion = _logprobs_completion(chosen_token="Yes", top=[("Yes", -0.05), ("No", -3.0)])
    client = make_client(completion)

    await calibrator.grade_yes_no(prompt="?", openai_client=client, model="gpt-5.4-nano")

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["reasoning_effort"] == "none"


async def test_grade_yes_no_omits_reasoning_effort_when_none():
    """Legacy non-reasoning models reject the flag — None omits it entirely."""
    completion = _logprobs_completion(chosen_token="Yes", top=[("Yes", -0.05), ("No", -3.0)])
    client = make_client(completion)

    await calibrator.grade_yes_no(
        prompt="?", openai_client=client, model="gpt-4o-mini", reasoning_effort=None
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in kwargs


async def test_grade_yes_no_service_tier_forwarded_and_omitted():
    """V-L5: service_tier forwarded when set, omitted (default) otherwise."""
    completion = _logprobs_completion(chosen_token="Yes", top=[("Yes", -0.05), ("No", -3.0)])

    client_flex = make_client(completion)
    await calibrator.grade_yes_no(
        prompt="?", openai_client=client_flex, model="gpt-5.4-nano", service_tier="flex"
    )
    assert client_flex.chat.completions.create.await_args.kwargs["service_tier"] == "flex"

    client_none = make_client(completion)
    await calibrator.grade_yes_no(
        prompt="?",
        openai_client=client_none,
        model="gpt-5.4-nano",  # default None
    )
    assert "service_tier" not in client_none.chat.completions.create.await_args.kwargs


async def test_case_insensitive_yes_token():
    """Tokens like 'yes' / ' yes' / 'Yes.' all count as Yes."""
    completion = _logprobs_completion(
        chosen_token="yes",
        top=[("yes", -0.1), ("No", -2.0)],
    )
    client = make_client(completion)

    result = await calibrator.grade_yes_no(
        prompt="?",
        openai_client=client,
        model="gpt-4.1-mini",
    )

    assert result.confidence > 0.5
    assert result.manual_review is False


async def test_calibrate_tag_builds_discriminator_prompt():
    """`calibrate_tag` wraps `grade_yes_no` with the standard prompt."""
    completion = _logprobs_completion(chosen_token="Yes", top=[("Yes", -0.1), ("No", -2.0)])
    client = make_client(completion)

    await calibrator.calibrate_tag(
        question_text="Stem about ATP synthesis.",
        tag_label="3A >> Bioenergetics",
        openai_client=client,
        model="gpt-4.1-mini",
    )

    kwargs = client.chat.completions.create.await_args.kwargs
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "discriminator" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "ATP synthesis" in messages[-1]["content"]
    assert "3A >> Bioenergetics" in messages[-1]["content"]
