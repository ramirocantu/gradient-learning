"""Grounded tagging — pick tags from retrieved candidates, calibrate each (T29).

V-L3: the model picks ONLY from the recall-layer candidate set (T28),
never free-form over the full outline. The candidate-index enum is the
grammar constraint; the numbered candidate list is the reasoning surface.

V44 (dual surface): a numbered NL candidate list (the model's reasoning
surface — bare enums tank selection quality) PLUS an int enum
``[1..N]`` over candidate positions (the grammar constraint). N =
len(candidates). The server maps ``node_index`` → ``node_id`` via the
deterministic candidate-order position index.

V45 (structured output): ``response_format: json_schema, strict:true``;
``additionalProperties:false`` on every object, every key listed in
``required``, no ``minimum``/``maximum``/``minItems``. The candidate-index
enum stays small (≤ top_k + edge fan-out) so OpenAI's enum-size limits
don't bite — no int-encoding gymnastics beyond the index itself.

V69 (calibration): the model's emitted ``confidence`` is a self-report
and historically optimistic; each picked tag is re-scored by the
calibrator (``calibrator.calibrate_tag`` — OpenAI logprobs Yes/No) into
``calibrated_confidence``; ``<0.5`` ⇒ ``manual_review`` (V-T3).

T29 produces calibrated tag *decisions*; T30 persists them to
``atomic_facts`` / ``<target>_tags``. This module does not write to the DB.

Per V16 the tagging + calibrator OpenAI clients are injected and mocked
at the SDK boundary in tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.kb.recall import RecallResult, format_candidates_for_prompt
from app.services.llm.calibrator import calibrate_tag

logger = logging.getLogger("app.services.llm.grounded")

# Bump when the prompt / schema shape changes meaningfully. Stamped onto
# every decision so T30 can persist it as ``extractor_version`` and a
# re-run under a new version is a clean cache miss.
EXTRACTOR_VERSION = "grounded-v1"
MAX_TOKENS = 2048


# --------------------------------------------------------------------------- #
# Output dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GroundedTag:
    """One calibrated tag decision grounded to a retrieved candidate.

    ``calibrated_confidence`` is the V69 logprob grade — the sole confidence
    we trust. The model's optimistic self-report is no longer collected
    (it was dead weight downstream of calibration). ``manual_review``
    follows the calibrated score (V-T3).
    """

    node_id: int
    path: str | None
    candidate_index: int        # 1..N position the model picked
    via: str                    # candidate.via ('embedding' | 'edge')
    rationale: str
    calibrated_confidence: float
    manual_review: bool


@dataclass(frozen=True)
class GroundedResult:
    tags: list[GroundedTag]
    extractor_version: str
    model: str
    calibrator_model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    parse_warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Schema / prompt construction (V44, V45)
# --------------------------------------------------------------------------- #


_SYSTEM_PREAMBLE = (
    "You tag a study item (an atomic fact or a practice question) with the "
    "outline node(s) it belongs to.\n"
    "\n"
    "Rules:\n"
    "- You MUST pick from the numbered candidate list below. Choose by "
    "`node_index`. Do NOT invent nodes or pick anything not listed.\n"
    "- Emit one tag per candidate the item genuinely belongs to; emit none if "
    "no candidate fits.\n"
    "- Prefer the most specific candidate when several overlap.\n"
)


def build_pick_schema(n_candidates: int) -> dict[str, Any]:
    """V44/V45: a strict json_schema whose ``node_index`` is an int enum
    ``[1..N]`` keyed into the numbered candidate list. Enforces V-L3 — the
    model can only return positions that exist in the retrieved set.

    Honors OpenAI strict-mode limits: ``additionalProperties:false`` on
    every object, every property in ``required``, no numeric bounds.
    """

    index_property: dict[str, Any] = {
        "type": "integer",
        "description": f"Candidate position 1..{n_candidates} from the numbered list.",
    }
    if n_candidates > 0:
        index_property["enum"] = list(range(1, n_candidates + 1))

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "submit_grounded_tags",
            "description": "Tag the item with outline nodes chosen from the candidate list.",
            "strict": True,
            "schema": {
                "type": "object",
                "required": ["tags"],
                "additionalProperties": False,
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["node_index", "rationale"],
                            "additionalProperties": False,
                            "properties": {
                                "node_index": index_property,
                                "rationale": {
                                    "type": "string",
                                    "description": "1 line.",
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def build_system_prompt(recall_result: RecallResult) -> str:
    """Preamble + the V44 numbered candidate list (reasoning surface).

    Reuses :func:`format_candidates_for_prompt` so the numbering the model
    reads is byte-for-byte the index it returns — the position index the
    server maps back to ``node_id``.
    """

    candidate_block = format_candidates_for_prompt(recall_result)
    return (
        f"{_SYSTEM_PREAMBLE}\n"
        "# Candidate outline nodes\n\n"
        f"{candidate_block}\n"
    )


def _format_user_message(entity_text: str) -> str:
    return f"Item:\n{entity_text.strip()}\n"


def _extract_structured_output(completion: Any) -> dict[str, Any] | None:
    """Parse the JSON body of a ``response_format: json_schema`` answer.

    Strict mode emits the document in ``choice.message.content``.
    """

    choices = getattr(completion, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", None) if message is not None else None
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("grounded: response content not valid JSON: %s", exc)
        return None


def _parse_picks(
    payload: dict[str, Any],
    recall_result: RecallResult,
) -> tuple[list[tuple[int, str]], list[str]]:
    """Server-side belt over the model's picks.

    Returns ``[(candidate_index, rationale), ...]`` after: range-recheck
    ``1..N`` (V-L3 — strict enum is enforced again here in case the SDK
    validation is loose) and per-candidate dedupe (first wins). Confidence
    is no longer read from the model — the calibrator (V69) is the sole
    confidence source.
    """

    n = len(recall_result.candidates)
    warnings: list[str] = []
    picks: list[tuple[int, str]] = []
    seen: set[int] = set()

    for i, raw in enumerate(payload.get("tags") or []):
        if not isinstance(raw, dict):
            warnings.append(f"pick #{i}: not an object ({type(raw).__name__})")
            continue
        idx_raw = raw.get("node_index")
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            warnings.append(f"pick #{i}: node_index not an int ({idx_raw!r})")
            continue
        if not (1 <= idx <= n):
            warnings.append(f"pick #{i}: node_index out of range ({idx}, valid 1..{n})")
            continue
        if idx in seen:
            continue
        seen.add(idx)

        rationale = raw.get("rationale") or ""
        if not isinstance(rationale, str):
            rationale = str(rationale)

        picks.append((idx, rationale.strip()))

    return picks, warnings


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


async def generate_grounded_tags(
    *,
    entity_text: str,
    recall_result: RecallResult,
    tagging_client: Any,
    calibrator_client: Any | None = None,
    tagging_model: str | None = None,
    calibrator_model: str | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
    max_tokens: int = MAX_TOKENS,
) -> GroundedResult:
    """Generate calibrated tag decisions for ``entity_text``, constrained to
    the recall candidate set (V-L3).

    Args:
        entity_text: the atomic-fact / question surface text being tagged.
        recall_result: candidates from :func:`app.services.kb.recall
            .retrieve_candidates`. Empty → no LLM call, empty result.
        tagging_client: injected ``AsyncOpenAI``-shaped client for the
            structured-output tagging call.
        calibrator_client: client for the V69 logprob grade; defaults to
            ``tagging_client`` (the calibrator model is the separate knob).
        tagging_model / calibrator_model: default to ``OPENAI_MODEL`` /
            ``OPENAI_CALIBRATOR_MODEL``.

    Returns ``GroundedResult`` — ``tags`` carry both the generation
    self-report and the V69-calibrated confidence + ``manual_review``.
    """

    resolved_tagging_model = tagging_model or settings.OPENAI_MODEL
    resolved_calibrator_model = calibrator_model or settings.OPENAI_CALIBRATOR_MODEL
    calibrator_client = calibrator_client or tagging_client
    service_tier = settings.OPENAI_SERVICE_TIER  # V-L5 Flex (None omits)

    # V-L3: nothing to pick from → don't call the model. An empty enum would
    # be an invalid strict schema anyway.
    if not recall_result.candidates:
        return GroundedResult(
            tags=[],
            extractor_version=extractor_version,
            model=resolved_tagging_model,
            calibrator_model=resolved_calibrator_model,
            parse_warnings=["no candidates retrieved — skipped LLM call"],
        )

    response_format = build_pick_schema(len(recall_result.candidates))
    system_text = build_system_prompt(recall_result)
    user_message = _format_user_message(entity_text)

    create_kwargs: dict[str, Any] = {
        "model": resolved_tagging_model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_message},
        ],
        "response_format": response_format,
    }
    if service_tier is not None:
        create_kwargs["service_tier"] = service_tier
    completion = await tagging_client.chat.completions.create(**create_kwargs)

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        # V-L1: cache-hit accounting read from cached_tokens, never inferred.
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    payload = _extract_structured_output(completion)
    if payload is None:
        return GroundedResult(
            tags=[],
            extractor_version=extractor_version,
            model=resolved_tagging_model,
            calibrator_model=resolved_calibrator_model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            parse_warnings=["LLM did not produce structured submit_grounded_tags output"],
        )

    picks, warnings = _parse_picks(payload, recall_result)

    tags: list[GroundedTag] = []
    for idx, rationale in picks:
        candidate = recall_result.candidates[idx - 1]
        tag_label = candidate.path or f"node:{candidate.node_id}"
        # V69: the logprob grade is the only confidence we keep.
        calibration = await calibrate_tag(
            question_text=entity_text,
            tag_label=tag_label,
            openai_client=calibrator_client,
            model=resolved_calibrator_model,
            service_tier=service_tier,
        )
        tags.append(
            GroundedTag(
                node_id=candidate.node_id,
                path=candidate.path,
                candidate_index=idx,
                via=candidate.via,
                rationale=rationale,
                calibrated_confidence=calibration.confidence,
                manual_review=calibration.manual_review,
            )
        )

    logger.info(
        "grounded: model=%s calibrator=%s candidates=%d picks=%d tags=%d "
        "prompt=%d cached=%d out=%d warnings=%d",
        resolved_tagging_model,
        resolved_calibrator_model,
        len(recall_result.candidates),
        len(picks),
        len(tags),
        prompt_tokens,
        cached_tokens,
        output_tokens,
        len(warnings),
    )
    return GroundedResult(
        tags=tags,
        extractor_version=extractor_version,
        model=resolved_tagging_model,
        calibrator_model=resolved_calibrator_model,
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        parse_warnings=warnings,
    )
