"""LLM confidence calibrator — V69 (amended) via OpenAI logprobs.

The LLM4Tag pipeline's emitted `confidence` is the model's own self-report
and historically optimistic. V69 reads a true confidence off the
calibrator model's *next-token logprob distribution* on a binary Yes/No
discriminator question, then converts it to a [0,1] score:

    Conf = exp(L_yes) / (exp(L_yes) + exp(L_no))

`<0.5` ⇒ `manual_review=true` (V-T3). The calibrator model **must** expose
`logprobs`. On GPT-5.x that requires `reasoning_effort='none'` — reasoning
mode does not return logprobs — so the grade is run reasoning-OFF, which also
fits V69's non-reasoning constraint (V-L5: `OPENAI_CALIBRATOR_MODEL` =
`gpt-5.4-nano`, reasoning off). Legacy non-reasoning chat models (`gpt-4.1*` /
`gpt-4o*`) work too — pass `reasoning_effort=None` to omit the flag for those.
Tagging may use any model; the calibrator is a separate config knob
(`OPENAI_CALIBRATOR_MODEL`).

The grade is run on a **plain completion** (no `response_format`,
`max_completion_tokens=1`, `reasoning_effort='none'`) so the single emitted
token is readable. The SDK exposes `top_logprobs` on
`choice.logprobs.content[0]`; we look for 'Yes' / 'No' (case-insensitive) and
fall back to 0.0 when neither token makes the top-5.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


_YES_TOKEN_CANDIDATES = {"yes", "y", " yes", "yes."}
_NO_TOKEN_CANDIDATES = {"no", "n", " no", "no."}

_DEFAULT_TOP_LOGPROBS = 5
_NEG_INF = float("-inf")
# V12: GPT-5.x cannot finish a completion in max_completion_tokens=1 (even with
# reasoning_effort='none') — it 400s "max_tokens or model output limit reached".
# A small headroom (≥4 observed; 16 for safety) lets it emit; we read only the
# FIRST content token's logprob distribution, so extra budget never changes the
# Yes/No grade. gpt-4o-class tolerated 1, but 16 is harmless there too.
_CALIBRATOR_MAX_TOKENS = 16


@dataclass(frozen=True)
class CalibrationResult:
    confidence: float
    manual_review: bool
    yes_logprob: float
    no_logprob: float
    chosen_token: str | None
    raw_top_logprobs: list[dict[str, Any]]


def _extract_yes_no_logprobs(top_logprobs: list[Any]) -> tuple[float, float]:
    """Return `(L_yes, L_no)` from OpenAI's top_logprobs list.

    `top_logprobs` is `[{"token": "...", "logprob": float}, ...]`. We
    normalize the token (strip + lower) and match against the Yes/No
    candidate set. Missing tokens get -inf (i.e. exp(-inf)=0) so the
    formula stays well-defined.
    """
    l_yes = _NEG_INF
    l_no = _NEG_INF
    for entry in top_logprobs:
        token = getattr(entry, "token", None) or ""
        logprob = float(getattr(entry, "logprob", _NEG_INF) or _NEG_INF)
        norm = token.strip().lower()
        if norm in _YES_TOKEN_CANDIDATES and logprob > l_yes:
            l_yes = logprob
        elif norm in _NO_TOKEN_CANDIDATES and logprob > l_no:
            l_no = logprob
    return l_yes, l_no


def _conf_from_yes_no(l_yes: float, l_no: float) -> float:
    """`Conf = exp(L_yes) / (exp(L_yes) + exp(L_no))`.

    Numerically stable via log-sum-exp shift. When both are -inf (neither
    token surfaced) returns 0.0.
    """
    if l_yes == _NEG_INF and l_no == _NEG_INF:
        return 0.0
    m = max(l_yes, l_no)
    a = math.exp(l_yes - m) if l_yes != _NEG_INF else 0.0
    b = math.exp(l_no - m) if l_no != _NEG_INF else 0.0
    return a / (a + b)


async def grade_yes_no(
    *,
    prompt: str,
    openai_client: AsyncOpenAI,
    model: str,
    system: str | None = None,
    top_logprobs: int = _DEFAULT_TOP_LOGPROBS,
    reasoning_effort: str | None = "none",
    service_tier: str | None = None,
) -> CalibrationResult:
    """Ask the calibrator a Yes/No question, read the next-token logprob
    distribution, return a calibrated confidence score.

    `prompt` is the user-side question — should be phrased so a literal
    'Yes' or 'No' is the only sane next token. `system` is optional framing.
    Set `model` to a logprobs-capable chat model (`OPENAI_CALIBRATOR_MODEL`).

    `reasoning_effort` defaults to `'none'` — REQUIRED on GPT-5.x to get
    `logprobs` back at all (reasoning mode returns none) and to honour V69's
    non-reasoning constraint. Pass `None` to omit the flag entirely for a
    legacy non-reasoning model (`gpt-4o*` / `gpt-4.1*`) that rejects it.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": _CALIBRATOR_MAX_TOKENS,  # V12
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    if reasoning_effort is not None:
        create_kwargs["reasoning_effort"] = reasoning_effort
    if service_tier is not None:
        create_kwargs["service_tier"] = service_tier  # V-L5 Flex

    completion = await openai_client.chat.completions.create(**create_kwargs)

    choice = completion.choices[0] if completion.choices else None
    chosen_token: str | None = None
    raw: list[dict[str, Any]] = []
    l_yes = l_no = _NEG_INF
    if choice is not None and choice.logprobs and choice.logprobs.content:
        first_token = choice.logprobs.content[0]
        chosen_token = getattr(first_token, "token", None)
        candidates = getattr(first_token, "top_logprobs", []) or []
        raw = [
            {"token": getattr(c, "token", ""), "logprob": float(getattr(c, "logprob", 0.0))}
            for c in candidates
        ]
        l_yes, l_no = _extract_yes_no_logprobs(candidates)

    conf = _conf_from_yes_no(l_yes, l_no)
    return CalibrationResult(
        confidence=conf,
        manual_review=conf < 0.5,  # V-T3
        yes_logprob=l_yes,
        no_logprob=l_no,
        chosen_token=chosen_token,
        raw_top_logprobs=raw,
    )


_DISCRIMINATOR_SYSTEM = (
    "You are a strict tag-quality discriminator. Given a question and one "
    "candidate tag, answer with a single token — 'Yes' if the tag genuinely "
    "describes the question, 'No' otherwise. No prose."
)


def _build_discriminator_prompt(*, question_text: str, tag_label: str) -> str:
    return (
        f"Question:\n{question_text.strip()}\n\n"
        f"Candidate tag: {tag_label.strip()}\n\n"
        "Does this tag describe the question? Answer Yes or No."
    )


async def calibrate_tag(
    *,
    question_text: str,
    tag_label: str,
    openai_client: AsyncOpenAI,
    model: str,
    reasoning_effort: str | None = "none",
    service_tier: str | None = None,
) -> CalibrationResult:
    """Calibrate one (question, candidate_tag) pair. Wrapper around
    `grade_yes_no` that supplies the standard discriminator framing.

    `reasoning_effort` defaults to `'none'` (V-L5: gpt-5.4-nano reasoning off,
    required for logprobs); pass `None` for a legacy calibrator model.
    `service_tier` (e.g. `'flex'`, V-L5) is forwarded when set.
    """
    return await grade_yes_no(
        prompt=_build_discriminator_prompt(
            question_text=question_text,
            tag_label=tag_label,
        ),
        system=_DISCRIMINATOR_SYSTEM,
        openai_client=openai_client,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )
