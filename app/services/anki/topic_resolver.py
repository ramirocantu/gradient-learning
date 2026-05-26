"""LLM topic-level resolver for AnKing cards parsed at CC granularity (SPEC §T32).

Mirrors `app/services/categorizer/llm.py` deliberately — same model
(`settings.ANKI_TOPIC_RESOLVER_MODEL` defaulting to Haiku), same tool-use
structured-output discipline, same SQLite cache via `LlmCacheBase`, same
extractor_version invalidation knob. Scope is narrower:

- Input: card stem/extra text + the single `aamc_cc` CC's topic-paths.
- Output: one `topic_path` from that CC's closed candidate set + confidence.
- Persistence (by the orchestrator in `worker.py`): one `anki_card_tags`
  row with `parsed_kind='aamc_topic'`, `source='llm'`, `topic_id`
  populated, `confidence`/`rationale`/`extractor_version` stamped.

Confidence threshold (`settings.ANKI_TOPIC_RESOLVER_CONFIDENCE_THRESHOLD`,
default 0.5) is checked by the orchestrator, not here.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from anthropic.types import Message, ToolUseBlock

if TYPE_CHECKING:
    from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache

from app.config import settings
from app.services.categorizer.outline_render import canonical_identifiers_for_section

logger = logging.getLogger(__name__)


# Bump when prompt/tool schema changes meaningfully. Stamped on every
# persisted anki_card_tags.extractor_version + cache key. Older-version rows
# are NOT auto-invalidated — worker's candidate query excludes any card with
# an aamc_topic LLM tag regardless of version, so prior-version rows survive
# untouched and only never-resolved cards get the new-version stamp.
#
# v6 trims (vs v5):
#   (A) enum bullet list previously duplicated inside the system block removed
#       — the tool-schema enum is in context and is the binding constraint
#       (§V25), and the duplicate cost ~3.4k tok @ CC 3B.
#   (B) tool-schema enum + `topic_picks` emit RELATIVE paths (everything after
#       `"{cc_code} >> "`); server reconstructs the full canonical path
#       before validation. Saves ~5 tok × N entries per CC.
# v7 trim:
#   (F) `_system_block_for_cc` prose compacted from ~380 → ~150 tok by dropping
#       restatement of constraints already encoded in the tool schema +
#       removing AnKing-curator backstory. Rules block retained verbatim
#       (multi-pick semantics, leaf preference, decline path, threshold).
# v8 strict:
#   `strict: true` on the tool def enables grammar-constrained sampling per
#   Anthropic structured outputs (Haiku 4.5 supported). Output is guaranteed
#   schema-conformant: enum membership on the topic field, required-field
#   presence, types — all enforced at sample time, not validated post-hoc.
#   Required schema adjustments:
#     - `additionalProperties: false` on every object (input_schema +
#       pick_item_schema).
#     - Removed `minimum`/`maximum` on confidence (numerical constraints
#       unsupported under strict). Bound advertised in description; we still
#       clip to ≥ threshold server-side.
#     - Removed `maxItems` on `topic_picks` (array constraints beyond
#       `minItems` 0/1 unsupported). MAX_TOPIC_PICKS still enforced via the
#       server-side slice on parse.
#   Server-side belt (path-set recheck, picks slice, threshold filter) kept
#   as defense-in-depth.
# v9 int-encoded enum + canonical list restored:
#   v6/v7/v8 trimmed away the natural-language canonical topic list from the
#   system block — empirical 10-card compare @ CC 3B showed mean jaccard 0.45
#   and 30% set-equality vs v5: model picked parent paths over leaves
#   (`Immune System` ×2 instead of `Macrophages` + `Innate vs adaptive`) and
#   wrong-branch leaves (`Endothelial cells` for a tricuspid valve card).
#   Root cause: tool-schema `enum` is consumed as API contract, not as
#   reasoning material — Claude needs natural-language prose to deliberate
#   over the candidate set. v9 restores the list AND keeps the trim by
#   integer-encoding the enum:
#     - System block carries a numbered list (`1. {rel_path}`...) — full
#       reasoning trellis.
#     - Tool schema's `topic_id` is an INTEGER in `[1, N]` instead of a
#       full-string enum — schema shrinks ~10× vs string-enum.
#     - Server maps `topic_id` back to the canonical full path via the
#       deterministic `_topic_paths_for_cc(cc)` index (1-based).
#   Also adds:
#     - Server-side DEDUPE of duplicate `topic_id` entries in the parse loop
#       (v8 sample showed model emitting same parent twice in `topic_picks`).
# v10 aggressive wrapper trim (S1+S2+S3+S4):
#   Strip everything the model doesn't need to produce a logic-based tagging
#   decision. Goal: only context that helps the call serve its goal.
#     - S1 system block prose: drop AnKing/AnkiHub backstory, regex pretagging
#       explainer, FirstAid/Bootcamp tag-suffix tutorial, server-side mechanics
#       explainers. Rules compressed to single-line imperatives. Numbered
#       canonical list stays intact (v9 finding: removing it tanks quality).
#     - S2 tool def descriptions: minimal. `topic_id`: "Topic ID 1..N; see
#       list in system." `confidence`: "[0,1]; <0.5 dropped." `rationale`:
#       "1 line." `tool.description`: one sentence. Schema-redundant prose
#       (NEVER emit same ID twice, must pick from enum, etc.) removed —
#       enum + server-side dedupe enforce these.
#     - S3 user message preambles: dropped tag-list backstory + card-text
#       backstory. Just `Tags:\n…` and `Card:\n…`. Affects per-call uncached
#       tokens (only path that compounds across cache hits).
#     - S4 dropped schema-redundant rules from the rules block.
EXTRACTOR_VERSION = "v10-prompt-trim"
MAX_TOKENS = 1024
MAX_TOPIC_PICKS = 5
# Stripped card text truncation cap. Empirically (§B8) the tag list alone
# under-specifies topic (parents + wrong branches); the card text disambiguates.
CARD_TEXT_MAX_LEN = 2000
# Skip threshold: a card with empty filtered tags AND a too-short card text
# carries no resolvable signal.
MIN_RESOLVABLE_TEXT_LEN = 30

# Mirror categorizer pricing table to avoid duplication; falls back to Sonnet
# rates with a warning. Keep this table in sync with categorizer.llm._PRICING.
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "cached_read": 0.10,
        "output": 5.00,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "cached_read": 0.30,
        "output": 15.00,
    },
}


def _pricing_for(model: str) -> dict[str, float]:
    if model in _PRICING:
        return _PRICING[model]
    logger.warning(
        "no pricing known for model=%r in anki topic resolver; assuming Sonnet rates",
        model,
    )
    return _PRICING["claude-sonnet-4-6"]


def _model() -> str:
    return settings.ANKI_TOPIC_RESOLVER_MODEL


@dataclass(frozen=True)
class TopicPick:
    """One topic the LLM thinks the card covers under the given CC.

    A single AnKing card may cover multiple AAMC topics within the same CC
    (e.g. a half-life card touching both `Half-life` and `Exponential decay`).
    `resolve_topic` returns a list of these; the worker persists one
    `anki_card_tags` row per pick with confidence ≥ threshold.
    """

    topic_path: str  # "<CC> >> <topic_name>" or deeper
    confidence: float
    rationale: str


# Back-compat alias — the cache module currently imports `TopicSuggestion`.
TopicSuggestion = TopicPick


@dataclass(frozen=True)
class ResolveResult:
    picks: list[TopicPick]  # empty when LLM declined or no candidates
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    extractor_version: str
    cost_saved_usd: float = 0.0
    model: str = ""


def _topic_paths_for_cc(cc_code: str) -> list[str]:
    """Return canonical topic-paths under the given CC.

    AAMC outline `canonical_identifiers_for_section` returns paths shaped
    `<CC_CODE> >> <topic_name>[>> ...]` (§V40). Filter to those starting
    with our CC. The section the CC belongs to is irrelevant for filtering —
    `cc_code` is unique across the outline.
    """
    # The outline render module caches per-section; iterate all 4 known
    # sections and filter. Cheap (cached + small set).
    paths: list[str] = []
    for section_code in ("CP", "CARS", "BB", "PS"):
        ids = canonical_identifiers_for_section(section_code)
        prefix = f"{cc_code} >> "
        paths.extend(p for p in ids.topic_paths if p.startswith(prefix))
    return paths


def _rel_paths_for_cc(cc_code: str, topic_paths: list[str]) -> list[str]:
    """Strip the `'{cc_code} >> '` prefix from each path so the tool-schema
    enum can ship rel-only entries. The prefix is constant per CC, so
    repeating it on every enum value wastes ~5 tokens × N entries.

    Server re-prepends the prefix when validating model output back into a
    canonical full path (`_to_full_path`).
    """
    prefix = f"{cc_code} >> "
    return [p[len(prefix) :] for p in topic_paths if p.startswith(prefix)]


def _to_full_path(cc_code: str, raw: str) -> str:
    """Normalize a model-emitted `topic_path` value back to canonical form.

    Tolerant of both shapes:
    - `'<rel>'` (expected — v6 schema enum is rel-only)
    - `'<cc> >> <rel>'` (model echoing full path; accepted defensively)
    """
    if not raw:
        return raw
    prefix = f"{cc_code} >> "
    if raw.startswith(prefix):
        return raw
    return prefix + raw


def _tool_def_for_cc(cc_code: str, topic_paths: list[str]) -> dict[str, Any]:
    """Tool with `topic_picks` array — each pick is one `topic_id` integer
    pointing into the numbered canonical list in the system block.

    v9: schema-side enum is integer IDs `[1..N]` instead of full path strings.
    The canonical natural-language list (with each path numbered) lives in the
    system block as the model's reasoning surface. The server maps ID → full
    canonical path via the stable position index from `_topic_paths_for_cc`.
    This recovers v5's leaf-selection quality without re-introducing the
    ~3.4k-token string-enum payload on the tool def.

    Strict mode constraints honored: `additionalProperties: false` on every
    object; no `minimum`/`maximum`/`maxItems`. Integer `enum` of primitives
    is first-class under structured outputs.
    """
    n = len(topic_paths)
    topic_id_property: dict[str, Any] = {
        "type": "integer",
        "description": f"Topic ID 1..{n}; see numbered list in system.",
    }
    if n > 0:
        topic_id_property["enum"] = list(range(1, n + 1))

    pick_item_schema: dict[str, Any] = {
        "type": "object",
        "required": ["topic_id", "confidence", "rationale"],
        "additionalProperties": False,
        "properties": {
            "topic_id": topic_id_property,
            "confidence": {
                "type": "number",
                "description": "[0,1]; <0.5 dropped.",
            },
            "rationale": {
                "type": "string",
                "description": "1 line.",
            },
        },
    }

    return {
        "name": "submit_anki_topic",
        "description": f"Tag card with AAMC topic IDs under {cc_code}.",
        # v8: grammar-constrained sampling. See module-level EXTRACTOR_VERSION
        # comment for the JSON-schema subset constraints satisfied below.
        "strict": True,
        "input_schema": {
            "type": "object",
            "required": ["decline", "topic_picks"],
            "additionalProperties": False,
            "properties": {
                "decline": {
                    "type": "boolean",
                    "description": "No topic fits.",
                },
                "topic_picks": {
                    "type": "array",
                    "items": pick_item_schema,
                    "description": "Distinct topic IDs.",
                },
            },
        },
        # Tool def carries the per-CC integer enum + descriptions. v9's int
        # enum cuts the tool-JSON payload ~10× vs v8's string enum, but the
        # cache_control still earns its keep on the descriptions + tool-use
        # system prompt overhead Anthropic injects (§V38, §B4).
        "cache_control": {"type": "ephemeral"},
    }


def make_cache_key(tag_payload: str, card_text: str, cc_code: str, model: str) -> str:
    """Cache key for one (tag_payload, card_text, CC, model) quadruple.

    Per §V25 (re-amended hybrid), the LLM sees both the filtered tag list and
    the stripped card text — so the cache key must hash both. Two cards with
    the same filtered tag set AND the same stripped text under the same CC
    hit the same cache row.
    """
    h = hashlib.sha256()
    h.update(tag_payload.encode("utf-8"))
    h.update(b"|")
    h.update(card_text.encode("utf-8"))
    h.update(b"|")
    h.update(cc_code.encode("utf-8"))
    h.update(b"|")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    cached_read: int,
    *,
    model: str,
) -> float:
    p = _pricing_for(model)
    return (
        (input_tokens / 1_000_000) * p["input"]
        + (cached_read / 1_000_000) * p["cached_read"]
        + (output_tokens / 1_000_000) * p["output"]
    )


def _extract_tool_call(message: Message) -> ToolUseBlock | None:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            return block
    return None


def _system_block_for_cc(cc_code: str, topic_paths: list[str]) -> str:
    """v10: aggressive wrapper trim. Provider-specific backstory (AnKing,
    AnkiHub, FirstAid/Bootcamp tag suffixes, regex pretagging) is removed —
    model doesn't need to know where signals come from, only how to weight
    them. Server-side mechanics ("re-attached server-side", "dropped
    server-side") removed too — not the model's concern. Rules compressed
    to single-line imperatives; schema-redundant rules dropped (the enum
    constrains `topic_id`, the parse loop dedupes).

    The numbered canonical list remains intact as the reasoning surface
    (§v9 finding: removing it caused 50% jaccard regression).
    """
    rel_paths = _rel_paths_for_cc(cc_code, topic_paths)
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(rel_paths, start=1))

    return (
        f"Pick AAMC topic(s) under content category {cc_code} that the card covers. "
        f"Two signals supplied below: filtered tag list + stripped card text. Tags "
        f"narrow the area; card text disambiguates sibling topics. On conflict, "
        f"prefer the text.\n\n"
        f"Topics under {cc_code} (pick by ID):\n"
        f"{numbered}\n\n"
        f"Rules:\n"
        f"- Up to {MAX_TOPIC_PICKS} picks, one per DISTINCT topic.\n"
        f"- Prefer LEAF over parent paths when card content supports it (deeper "
        f"  paths contain ` >> `).\n"
        f"- No topic fits → `decline=true` with empty `topic_picks`.\n"
        f"- Confidence reflects BOTH signals; <0.5 dropped. Prefer fewer high-"
        f"  confidence picks.\n"
    )


def _format_tag_payload(filtered_tags: list[str]) -> str:
    """v10: terse delimiter only — model doesn't need to know tag taxonomy
    backstory. Empty case still labeled so the user message stays parseable.
    """
    if not filtered_tags:
        return "Tags: (none)"
    return "Tags:\n" + "\n".join(f"- {t}" for t in filtered_tags)


def _format_card_text_payload(card_text: str) -> str:
    """v10: terse delimiter only."""
    if not card_text:
        return "Card: (empty)"
    return "Card:\n" + card_text


def _build_user_message(filtered_tags: list[str], card_text: str) -> str:
    return _format_tag_payload(filtered_tags) + "\n\n" + _format_card_text_payload(card_text)


async def resolve_topic(
    *,
    filtered_tags: list[str],
    card_text: str,
    cc_code: str,
    anthropic_client: AsyncAnthropic,
    cache: "AnkiTopicResolverCache | None" = None,
    extractor_version: str = EXTRACTOR_VERSION,
    model: str | None = None,
) -> ResolveResult:
    """Resolve one Anki card to one or more AAMC topics under `cc_code`.

    Input: the card's FILTERED AnKing tag list (per §V25). `cache`, when
    provided, is consulted first. On miss, the LLM is called and the result
    stored. Empty topic_paths under the CC (CARS) → returns ResolveResult
    with empty picks.
    """
    resolved_model = model or _model()
    topic_paths = _topic_paths_for_cc(cc_code)
    if not topic_paths:
        return ResolveResult(
            picks=[],
            cache_hit=False,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            extractor_version=extractor_version,
            cost_saved_usd=0.0,
            model=resolved_model,
        )

    tag_payload = _format_tag_payload(filtered_tags)
    cache_key = make_cache_key(tag_payload, card_text, cc_code, resolved_model)
    if cache is not None:
        cached = cache.get(cache_key, extractor_version)
        if cached is not None:
            original_cost = cache.lookup_cost(cache_key)
            return ResolveResult(
                picks=cached,
                cache_hit=True,
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                extractor_version=extractor_version,
                cost_saved_usd=original_cost,
                model=resolved_model,
            )

    system_block = _system_block_for_cc(cc_code, topic_paths)
    tool = _tool_def_for_cc(cc_code, topic_paths)

    msg = await anthropic_client.messages.create(
        model=resolved_model,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_block,
                # The CC's topic list is the only large reusable block across
                # calls within the same CC; mark it cacheable.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_anki_topic"},
        messages=[{"role": "user", "content": _build_user_message(filtered_tags, card_text)}],
    )

    tool_block = _extract_tool_call(msg)
    picks: list[TopicPick] = []
    if tool_block is not None:
        payload = tool_block.input or {}
        declined = bool(payload.get("decline", False))
        if not declined:
            raw_picks = payload.get("topic_picks") or []
            if isinstance(raw_picks, list):
                # v9: dedupe by topic_id across the pick array (the prompt
                # tells the model not to repeat, but observed v8 behavior
                # was duplicate parent IDs; belt-and-suspenders here).
                seen_ids: set[int] = set()
                for raw in raw_picks[:MAX_TOPIC_PICKS]:
                    if not isinstance(raw, dict):
                        continue
                    raw_id = raw.get("topic_id")
                    if raw_id is None:
                        # v6/v7/v8 path-string fallback: only used if some
                        # caller wires up an old fake. Map the path back to
                        # an id via the canonical list, then proceed.
                        raw_path = raw.get("topic_path") or ""
                        if not raw_path:
                            continue
                        full_path = _to_full_path(cc_code, raw_path)
                        try:
                            raw_id = topic_paths.index(full_path) + 1
                        except ValueError:
                            continue
                    try:
                        topic_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if not (1 <= topic_id <= len(topic_paths)):
                        continue
                    if topic_id in seen_ids:
                        continue
                    seen_ids.add(topic_id)
                    full_path = topic_paths[topic_id - 1]
                    confidence = float(raw.get("confidence", 0.0))
                    rationale = str(raw.get("rationale", ""))
                    picks.append(
                        TopicPick(
                            topic_path=full_path,
                            confidence=confidence,
                            rationale=rationale,
                        )
                    )

    usage = getattr(msg, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cost = _compute_cost(input_tokens, output_tokens, cached_read, model=resolved_model)

    result = ResolveResult(
        picks=picks,
        cache_hit=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        extractor_version=extractor_version,
        cost_saved_usd=0.0,
        model=resolved_model,
    )
    if cache is not None and picks:
        cache.put(cache_key, picks, extractor_version, model=resolved_model, cost=cost)
    return result
