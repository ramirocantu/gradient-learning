"""LLM-driven AAMC categorizer.

Calls OpenAI with the relevant AAMC outline subset + the question's stem,
explanation, and raw UWorld tags. Returns 1–N tag suggestions (topic, content
category, or skill) with confidence and rationale.

Caching:
  - OpenAI prompt caching is automatic on stable prefixes — V38 retires the
    Anthropic `cache_control` markers. V42 still applies: candidate iteration
    must not switch the cached-prefix dimension between adjacent calls.
  - In-process result cache keyed on
    (stem_plain, explanation_plain, sorted UWorld tags, EXTRACTOR_VERSION).
    Bumping EXTRACTOR_VERSION invalidates the cache and re-runs the LLM.

The cache lives for the lifetime of the Python process (worker run, FastAPI
process). No Redis.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

if TYPE_CHECKING:
    from app.services.categorizer.cache import CategorizerCache

from app.config import settings
from app.models.captures import Question
from app.services.categorizer.outline_lookup import OutlineLookup
from app.services.categorizer.outline_render import (
    SUBJECT_TO_SECTION,
    canonical_identifiers_for_section,
    render_canonical_identifiers_block,
    render_outline_for_section,
)

logger = logging.getLogger(__name__)


# Bump this when the prompt changes meaningfully. The string is stamped on
# every persisted QuestionTag.extractor_version AND used to invalidate cache
# entries on lookup (without deleting them).
EXTRACTOR_VERSION = "v10-strict"
MAX_TOKENS = 4096


def _model() -> str:
    """Per-call model lookup so settings overrides propagate (esp. in tests)."""
    return settings.CATEGORIZER_MODEL


# Backwards-compat for `llm.MODEL` (tests + early scripts referenced it).
# Reflects the value at import time; runtime callers should use _model() or
# pass `model=` explicitly to `categorize()`.
MODEL = settings.CATEGORIZER_MODEL


# Pricing per million tokens. Keys are OpenAI model identifier strings.
# `cached_read` is OpenAI's automatic prompt-cache discount.
_PRICING = {
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cached_read": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_read": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_read": 0.025},
    "gpt-4o": {"input": 2.50, "output": 10.0, "cached_read": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_read": 0.075},
}


def _pricing_for(model: str) -> dict[str, float]:
    """Return pricing for a model. Falls back to gpt-4.1-mini rates with a warning."""
    if model in _PRICING:
        return _PRICING[model]
    logger.warning(
        "no pricing known for model=%r; assuming gpt-4.1-mini rates for cost estimate",
        model,
    )
    return _PRICING["gpt-4.1-mini"]


@dataclass(frozen=True)
class LlmTagSuggestion:
    kind: Literal["topic", "content_category", "skill"]
    identifier: str | int
    under_content_category: str | None
    confidence: float
    rationale: str


@dataclass(frozen=True)
class CategorizeResult:
    suggestions: list[LlmTagSuggestion]
    primary_aamc_section: str | None
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    extractor_version: str
    parse_warnings: list[str]
    # Ticket 3.4: on a cache hit, this carries the original cost stored when
    # the entry was inserted. Worker tallies these for `total_cost_saved_usd`.
    cost_saved_usd: float = 0.0
    model: str = ""


# Backwards-compat shim for tests that called `llm._clear_cache_for_tests()`
# against the old in-process dict cache. With Ticket 3.4 the real cache lives
# in `CategorizerCache` (SQLite); tests now construct/tear down their own
# temp DBs. This shim exists so removing the old import is mechanical.
def _clear_cache_for_tests() -> None:
    """Deprecated; persistent cache is per-instance now."""


def _tool_def_for_section(section_code: str) -> dict[str, Any]:
    """Build a `submit_aamc_categorization` tool definition with enum constraints
    scoped to the canonical identifiers of `section_code`.

    The enum on `topic_path` and `content_category_code` is the LLM's enforced
    closed list. The orchestrator parses the value server-side as a safety net
    in case the SDK's schema validation is loose.
    """
    ids = canonical_identifiers_for_section(section_code)
    topic_paths = list(ids.topic_paths)
    cc_codes = list(ids.content_category_codes)

    # V44: topic enum is integer IDs `[1..N]` keyed into the numbered list in
    # the system block — schema payload shrinks ~10× vs the per-section
    # string enum (CP=434, BB=619, PS=392 topic_paths, ~30-44k chars each;
    # int form is a small number array). The natural-language numbered list
    # in the system block is the model's reasoning surface (V44 finding —
    # bare enum without prose tanked anki resolver jaccard 50%); the int
    # enum is the grammar constraint. Server maps topic_id → full canonical
    # path via the deterministic `topic_paths` position index.
    n_topics = len(topic_paths)
    topic_id_property: dict[str, Any] = {
        "type": "integer",
        "description": (
            f"Required if kind='topic'. Topic ID 1..{n_topics}; see numbered list in system block."
        ),
    }
    if n_topics > 0:
        topic_id_property["enum"] = list(range(1, n_topics + 1))

    cc_property: dict[str, Any] = {
        "type": "string",
        "description": "Required if kind='content_category'. CC code (e.g. '4A').",
    }
    if cc_codes:
        cc_property["enum"] = cc_codes

    # V45 (T6): structured output via `response_format: json_schema, strict:true`.
    # JSON-schema subset honored: `additionalProperties:false` on every object,
    # no `minimum`/`maximum` on numbers, no `minItems`/`maxItems`, every key in
    # `properties` listed in `required`. V44 keeps `topic_id` an int enum so the
    # schema stays under OpenAI's enum-size + enum-string-length limits
    # (string-enum of ~500 topic_paths would exceed the per-schema enum-string
    # total length cap). V38 retired — no `cache_control`.
    schema = {
        "type": "object",
        "required": ["primary_aamc_section", "tags"],
        "additionalProperties": False,
        "properties": {
            "primary_aamc_section": {
                "type": "string",
                "enum": ["CP", "CARS", "BB", "PS"],
                "description": "Section.",
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "kind",
                        "topic_id",
                        "content_category_code",
                        "skill_number",
                        "confidence",
                        "rationale",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["topic", "content_category", "skill"],
                        },
                        "topic_id": {**topic_id_property, "type": ["integer", "null"]},
                        "content_category_code": {**cc_property, "type": ["string", "null"]},
                        "skill_number": {
                            "type": ["integer", "null"],
                            "enum": [1, 2, 3, 4, None],
                            "description": "Required if kind='skill'.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "[0,1]; clipped server-side.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "1 line.",
                        },
                    },
                },
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "submit_aamc_categorization",
            "description": "Tag question with AAMC topics, CCs, skills.",
            "strict": True,
            "schema": schema,
        },
    }


_SYSTEM_PROMPT_PREAMBLE = (
    "Tag an MCAT practice question with AAMC topics, content categories, "
    "and skills.\n"
    "\n"
    "Rules:\n"
    "- Emit the MOST SPECIFIC tag justified by the question. Prefer `topic` over "
    "`content_category` when a specific topic applies.\n"
    "- Canonical topic list mixes parents and children. ALWAYS evaluate the "
    "children first; fall back to the parent only when no child genuinely "
    "applies. Never emit both a parent and one of its own children.\n"
    "- A question may test multiple topics — even across content categories. "
    "Emit one tag per topic genuinely tested.\n"
    "- Always include exactly one `skill` tag (integer 1-4): "
    "1=Knowledge of Scientific Principles, "
    "2=Scientific Reasoning and Problem-solving, "
    "3=Reasoning about the Design and Execution of Research, "
    "4=Data-based Statistical Reasoning. "
    "Pick the skill the question primarily exercises.\n"
    "- Confidence ∈ [0.0, 1.0] per-tag.\n"
)


def parse_topic_path(path: str) -> tuple[str, list[str]]:
    """Split `'<CC_code> >> <name> [>> <child> ...]'` into `(cc_code, [name_segments])`.

    Segments are separated by ' >> ' per §V40 (the reserved delimiter avoids
    collision with ÷ notation in physics-formula topic names). Raises
    ValueError on malformed input (missing CC, empty segment, fewer than 2 parts).
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"malformed topic path: {path!r}")
    parts = [p.strip() for p in path.split(" >> ")]
    if len(parts) < 2 or not parts[0] or any(not p for p in parts[1:]):
        raise ValueError(f"malformed topic path: {path!r}")
    return parts[0], parts[1:]


def _subject_from_tags(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    for t in tags:
        if isinstance(t, str) and t.startswith("Subject: "):
            return t[len("Subject: ") :].strip()
    return None


def make_cache_key(
    stem_plain: str | None,
    explanation_plain: str | None,
    uworld_aamc_tags: list[str] | None,
    model: str,
) -> str:
    """Stable SHA-256 of (stem, explanation, sorted tags, model).

    `extractor_version` is intentionally NOT part of the hash — it's stored
    alongside in the cache and checked on lookup. Bumping the version
    invalidates existing entries without churning the keyspace.
    """
    h = hashlib.sha256()
    h.update((stem_plain or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((explanation_plain or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update("|".join(sorted(uworld_aamc_tags or [])).encode("utf-8"))
    h.update(b"\x1f")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _format_user_message(question: Question) -> str:
    """V47: terse delimiters only — user-msg preambles aren't cacheable so
    every saved byte compounds across cache hits.
    """
    raw_tags = "\n".join(f"- {t}" for t in (question.uworld_aamc_tags or [])) or "(none)"
    stem = (question.stem_plain or "").strip()
    explanation = (question.explanation_plain or "").strip()
    expl_block = f"\n\nExpl:\n{explanation}" if explanation else ""
    return f"Tags:\n{raw_tags}\n\nQ:\n{stem}{expl_block}\n"


def _extract_structured_output(completion: ChatCompletion) -> dict[str, Any] | None:
    """Return the parsed JSON body of a `response_format: json_schema` answer.

    Under strict mode the model emits a JSON document in `choice.message.content`
    that conforms to the supplied schema. We `json.loads` once and return the
    dict; downstream `_parse_tool_input` is shape-agnostic.
    """
    choice = completion.choices[0] if completion.choices else None
    if choice is None or choice.message is None or not choice.message.content:
        return None
    try:
        return json.loads(choice.message.content)
    except json.JSONDecodeError as exc:
        logger.warning("categorize: response content not valid JSON: %s", exc)
        return None


def _parse_tool_input(
    tool_input: dict[str, Any],
    *,
    topic_paths_for_section: list[str] | None = None,
) -> tuple[list[LlmTagSuggestion], str | None, list[str]]:
    """Parse the canonical-3.5 tool input shape.

    Each tag carries kind + one of {topic_id (V44), content_category_code,
    skill_number}. We unpack into LlmTagSuggestion's existing shape:
      - kind='topic'      → identifier=topic_path, under_content_category=cc_code
      - kind='content_category' → identifier=cc_code
      - kind='skill'      → identifier=int(skill_number)

    V44: `topic_id` is read first; legacy `topic_path` is accepted as a
    fallback for forge tests + old callers. `topic_paths_for_section` is the
    section's stable canonical list used to map topic_id → path; required
    for kind='topic' resolution.

    This keeps the orchestrator's resolution logic unchanged.
    """
    topic_paths_for_section = topic_paths_for_section or []
    warnings: list[str] = []
    suggestions: list[LlmTagSuggestion] = []
    # V46: dedupe by (kind, identifier) — strict mode + enum doesn't enforce
    # array uniqueness, and the model sometimes emits the same tag twice
    # (parent path duplicated, same skill emitted twice). First occurrence
    # wins. Orchestrator dedupes again at the DB-target level; this is the
    # earlier belt.
    seen_keys: set[tuple[str, str | int]] = set()
    primary = tool_input.get("primary_aamc_section")
    if primary not in {"CP", "CARS", "BB", "PS"}:
        warnings.append(f"unrecognized primary_aamc_section={primary!r}")
        primary = None

    for i, raw in enumerate(tool_input.get("tags") or []):
        if not isinstance(raw, dict):
            warnings.append(f"tag #{i}: not an object ({type(raw).__name__})")
            continue
        kind = raw.get("kind")
        if kind not in {"topic", "content_category", "skill"}:
            warnings.append(f"tag #{i}: unknown kind {kind!r}")
            continue

        identifier: str | int | None = None
        under_cc: str | None = None

        if kind == "topic":
            # V44: prefer `topic_id` (int) → map to canonical path via section
            # position index. `topic_path` (string) accepted as fallback for
            # legacy forge tests + back-compat with cached/persisted results.
            path: str | None = None
            tid = raw.get("topic_id")
            if tid is not None:
                try:
                    tid_int = int(tid)
                except (TypeError, ValueError):
                    warnings.append(f"tag #{i}: topic_id not an int ({tid!r})")
                    continue
                if not (1 <= tid_int <= len(topic_paths_for_section)):
                    warnings.append(
                        f"tag #{i}: topic_id out of range "
                        f"({tid_int}, valid 1..{len(topic_paths_for_section)})"
                    )
                    continue
                path = topic_paths_for_section[tid_int - 1]
            else:
                path = raw.get("topic_path")
                # Legacy fallback: 3.3/3.4 callers may send `identifier` + `under_content_category`.
                if not path:
                    legacy_ident = raw.get("identifier")
                    legacy_cc = raw.get("under_content_category")
                    if isinstance(legacy_ident, str) and isinstance(legacy_cc, str):
                        path = f"{legacy_cc} >> {legacy_ident}"
            if not isinstance(path, str) or not path.strip():
                warnings.append(f"tag #{i}: topic missing topic_id/topic_path")
                continue
            try:
                cc_code, _name_parts = parse_topic_path(path)
            except ValueError as exc:
                warnings.append(f"tag #{i}: {exc}")
                continue
            identifier = path  # full path; orchestrator resolves via topic_id_by_path
            under_cc = cc_code
        elif kind == "content_category":
            code = raw.get("content_category_code") or raw.get("identifier")
            if not isinstance(code, str) or not code.strip():
                warnings.append(f"tag #{i}: content_category missing content_category_code")
                continue
            identifier = code.strip()
        else:  # skill
            num = raw.get("skill_number")
            if num is None:
                num = raw.get("identifier")
            try:
                num = int(num)
            except (TypeError, ValueError):
                warnings.append(f"tag #{i}: skill_number not an int ({num!r})")
                continue
            if not 1 <= num <= 4:
                warnings.append(f"tag #{i}: skill out of range ({num})")
                continue
            identifier = num

        confidence_raw = raw.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            warnings.append(f"tag #{i}: bad confidence {confidence_raw!r}; defaulting to 0.5")
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        rationale = raw.get("rationale") or ""
        if not isinstance(rationale, str):
            rationale = str(rationale)

        dedupe_key: tuple[str, str | int] = (kind, identifier)  # type: ignore[assignment]
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        suggestions.append(
            LlmTagSuggestion(
                kind=kind,  # type: ignore[arg-type]
                identifier=identifier,  # type: ignore[arg-type]
                under_content_category=under_cc,
                confidence=confidence,
                rationale=rationale.strip(),
            )
        )

    return suggestions, primary, warnings


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    cached_input_read: int,
    *,
    model: str,
) -> float:
    """Estimate USD cost from OpenAI's `usage` breakdown.

    V-L1: OpenAI `usage.prompt_tokens` includes cached tokens; the cached
    slice is read from `prompt_tokens_details.cached_tokens`. Pass
    `input_tokens` as the uncached prompt slice (prompt_tokens − cached) and
    `cached_input_read` as `cached_tokens`; both are billed at their own rate.
    """
    p = _pricing_for(model)
    return (
        (input_tokens / 1_000_000) * p["input"]
        + (cached_input_read / 1_000_000) * p["cached_read"]
        + (output_tokens / 1_000_000) * p["output"]
    )


async def categorize(
    question: Question,
    *,
    openai_client: AsyncOpenAI,
    outline_lookup: OutlineLookup,  # noqa: ARG001 — accepted per spec, not currently used
    cache: "CategorizerCache | None" = None,
    extractor_version: str = EXTRACTOR_VERSION,
    model: str | None = None,
) -> CategorizeResult:
    """Call the LLM to produce AAMC tag suggestions for one question.

    `cache`, when provided, is consulted first; on miss the LLM is called and
    the result is persisted via `cache.put`. `extractor_version` is the
    invalidation knob — same model + same key + different version = cache miss
    + fresh call. `model` defaults to `settings.CATEGORIZER_MODEL`.
    """
    resolved_model = model or _model()
    subject = _subject_from_tags(question.uworld_aamc_tags)
    section_code = SUBJECT_TO_SECTION.get(subject) if subject else None
    if section_code is None:
        msg = f"unrecognized or missing Subject (got {subject!r})"
        logger.warning("categorize qid=%s: %s — not calling LLM", question.qid, msg)
        return CategorizeResult(
            suggestions=[],
            primary_aamc_section=None,
            cache_hit=False,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            extractor_version=extractor_version,
            parse_warnings=[msg],
            cost_saved_usd=0.0,
            model=resolved_model,
        )

    cache_key = make_cache_key(
        question.stem_plain,
        question.explanation_plain,
        question.uworld_aamc_tags,
        resolved_model,
    )
    if cache is not None:
        cached = cache.get(cache_key, extractor_version)
        if cached is not None:
            original_cost = cache.lookup_cost(cache_key)
            logger.debug(
                "categorize qid=%s: persistent cache hit (saved ~$%.4f)",
                question.qid,
                original_cost,
            )
            return CategorizeResult(
                suggestions=cached.suggestions,
                primary_aamc_section=cached.primary_aamc_section,
                cache_hit=True,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                estimated_cost_usd=0.0,
                extractor_version=extractor_version,
                parse_warnings=list(cached.parse_warnings),
                cost_saved_usd=original_cost,
                model=resolved_model,
            )

    outline_md = render_outline_for_section(section_code)
    canonical_block = render_canonical_identifiers_block(section_code)
    # V44: numbered topic_path list = model's reasoning surface for topic
    # selection. Tool schema's `topic_id` integer enum is the grammar
    # constraint; this prose is what the model deliberates over.
    section_topic_paths = list(canonical_identifiers_for_section(section_code).topic_paths)
    numbered_topic_block = (
        "# Numbered topic paths for this section\n\n"
        "When kind='topic', pick by `topic_id` from this list:\n"
        + "\n".join(f"{i}. {p}" for i, p in enumerate(section_topic_paths, start=1))
        if section_topic_paths
        else ""
    )
    system_block_2_text = (
        f"# AAMC outline for section {section_code}\n\n{outline_md}\n\n{canonical_block}"
    )
    if numbered_topic_block:
        system_block_2_text += f"\n\n{numbered_topic_block}"
    # V38 retired: OpenAI auto-caches stable prefixes — concat into one system
    # message preserving order (preamble first, then the cacheable outline).
    system_text = _SYSTEM_PROMPT_PREAMBLE + "\n\n" + system_block_2_text
    user_message = _format_user_message(question)
    response_format = _tool_def_for_section(section_code)

    response = await openai_client.chat.completions.create(
        model=resolved_model,
        max_completion_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_message},
        ],
        response_format=response_format,
    )

    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_input_read = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_input_read = int(getattr(details, "cached_tokens", 0) or 0)
    # V-L1: cache-hit accounting from `cached_tokens`, never inferred.
    uncached_input = max(prompt_tokens - cached_input_read, 0)
    cost = _compute_cost(
        uncached_input,
        output_tokens,
        cached_input_read,
        model=resolved_model,
    )

    tool_args = _extract_structured_output(response)
    if tool_args is None:
        msg = "LLM did not produce structured submit_aamc_categorization output"
        logger.warning("categorize qid=%s: %s", question.qid, msg)
        result = CategorizeResult(
            suggestions=[],
            primary_aamc_section=None,
            cache_hit=False,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            extractor_version=extractor_version,
            parse_warnings=[msg],
            cost_saved_usd=0.0,
            model=resolved_model,
        )
        if cache is not None:
            cache.put(cache_key, result, extractor_version, model=resolved_model)
        return result

    suggestions, primary, warnings = _parse_tool_input(
        tool_args,
        topic_paths_for_section=section_topic_paths,
    )

    result = CategorizeResult(
        suggestions=suggestions,
        primary_aamc_section=primary,
        cache_hit=False,
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        extractor_version=extractor_version,
        parse_warnings=warnings,
        cost_saved_usd=0.0,
        model=resolved_model,
    )
    if cache is not None:
        cache.put(cache_key, result, extractor_version, model=resolved_model)

    logger.info(
        "categorize qid=%s model=%s section=%s prompt=%d cache_read=%d "
        "out=%d cost=$%.4f suggestions=%d warnings=%d",
        question.qid,
        resolved_model,
        section_code,
        prompt_tokens,
        cached_input_read,
        output_tokens,
        cost,
        len(suggestions),
        len(warnings),
    )
    return result
