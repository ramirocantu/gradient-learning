"""LLM-driven content-agnostic feature extraction for MCAT questions.

Phase 4.2. Calls Claude Sonnet with the stem, choices, explanation, passage
(if any), and the pre-computed mechanical features. Returns judgment-call
features (reasoning_type, distractor_difficulty, trap_distractor_present,
key_concept_summary, etc.) via structured tool-use output.

Caching:
  - Anthropic prompt caching on the system block (`cache_control` set to
    "ephemeral") — feature schema description amortizes across questions.
  - Persistent SQLite cache keyed on (stem, explanation, passage[:3000],
    mechanical features, model). `EXTRACTOR_VERSION` is stored alongside
    and checked on lookup — bumping it invalidates without churning the
    keyspace.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from anthropic import AsyncAnthropic
from anthropic.types import Message, ToolUseBlock

if TYPE_CHECKING:
    from app.services.analyzer.cache import FeatureExtractorCache

from app.config import settings
from app.models.captures import Passage, Question
from app.services.analyzer.mechanical_features import MechanicalFeatures

logger = logging.getLogger(__name__)


# Bump on prompt or schema change. Stamped on every QuestionFeatures.extractor_version
# AND used to invalidate cache entries on lookup (without deleting them).
EXTRACTOR_VERSION = "features-v3-clean-explanation"
MAX_TOKENS = 2048
PASSAGE_TRUNCATE_CHARS = 3000


def _model() -> str:
    return settings.FEATURE_EXTRACTOR_MODEL


MODEL = settings.FEATURE_EXTRACTOR_MODEL


# Pricing per million tokens; mirrors categorizer.llm._PRICING.
_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cached_read": 0.30},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cached_read": 0.10},
}


def _pricing_for(model: str) -> dict[str, float]:
    if model in _PRICING:
        return _PRICING[model]
    logger.warning(
        "no pricing known for model=%r; assuming Sonnet rates for cost estimate",
        model,
    )
    return _PRICING["claude-sonnet-4-6"]


ReasoningType = Literal["recall", "comprehension", "application", "analysis", "inference"]
PassageType = Literal["experimental", "descriptive", "hypothesis_driven"]
DifficultyLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class LlmJudgmentFeatures:
    reasoning_type: ReasoningType
    requires_calculation: bool
    calculation_steps: int
    passage_type: PassageType | None
    distractor_difficulty: DifficultyLevel
    trap_distractor_present: bool
    common_misconception: str | None
    jargon_density: DifficultyLevel
    key_concept_summary: str
    involves_graph_or_figure: bool
    involves_data_table: bool


@dataclass(frozen=True)
class ExtractFeaturesResult:
    features: LlmJudgmentFeatures
    cache_hit: bool
    cost_saved_usd: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    extractor_version: str
    model: str
    parse_warnings: list[str]


_TOOL = {
    "name": "submit_question_features",
    "description": "Submit content-agnostic features about an MCAT question.",
    "input_schema": {
        "type": "object",
        "required": [
            "reasoning_type",
            "requires_calculation",
            "calculation_steps",
            "distractor_difficulty",
            "trap_distractor_present",
            "common_misconception",
            "jargon_density",
            "key_concept_summary",
            "involves_graph_or_figure",
            "involves_data_table",
        ],
        "properties": {
            "reasoning_type": {
                "type": "string",
                "enum": [
                    "recall",
                    "comprehension",
                    "application",
                    "analysis",
                    "inference",
                ],
                "description": (
                    "Highest-order reasoning the question demands. "
                    "recall=fact lookup; comprehension=interpreting a given concept; "
                    "application=using a principle in a new context; "
                    "analysis=breaking a scenario into components; "
                    "inference=drawing conclusions from given evidence."
                ),
            },
            "requires_calculation": {
                "type": "boolean",
                "description": "True iff the solver must perform arithmetic or algebra.",
            },
            "calculation_steps": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Approximate count of distinct calculation steps. "
                    "Set to 0 when requires_calculation=false."
                ),
            },
            "passage_type": {
                "type": "string",
                "enum": ["experimental", "descriptive", "hypothesis_driven"],
                "description": (
                    "Set only when the question is passage-based. "
                    "experimental=passage describes an experiment with methods/results; "
                    "descriptive=expository content; "
                    "hypothesis_driven=passage frames competing hypotheses."
                ),
            },
            "distractor_difficulty": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": (
                    "How challenging the wrong answers are to rule out. "
                    "low=obviously wrong; medium=plausible on a quick read; "
                    "high=closely matched, requires careful reasoning."
                ),
            },
            "trap_distractor_present": {
                "type": "boolean",
                "description": (
                    "True if one wrong answer specifically targets a known misstep "
                    "(wrong-direction inference, common formula confusion, etc.)."
                ),
            },
            "common_misconception": {
                "type": "string",
                "description": (
                    "One-sentence description of the specific misconception the wrong "
                    "answers exploit. Empty string when no specific misconception is "
                    "involved — do NOT confabulate one for every question."
                ),
            },
            "jargon_density": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": (
                    "Density of technical vocabulary in stem + choices "
                    "(not the passage). low=general terms; medium=several discipline "
                    "terms; high=dense field-specific terminology."
                ),
            },
            "key_concept_summary": {
                "type": "string",
                "description": (
                    "One sentence, <= 25 words, of the form: "
                    "'Tests [skill/knowledge] of [topic] in context of [scenario].' "
                    "Used by downstream synthesis to describe what the student is "
                    "getting tripped up on."
                ),
            },
            "involves_graph_or_figure": {
                "type": "boolean",
                "description": (
                    "`True` only if interpreting a substantive visual element "
                    "(graph, diagram, chemical structure, anatomical figure, "
                    "experimental apparatus, etc.) is necessary to answer the question. "
                    "Use the explanation as your primary evidence — if it refers to "
                    "'the figure,' 'the graph,' 'the diagram,' or otherwise treats a visual "
                    "as load-bearing, set `True`. If the only images are icons, formulas "
                    "rendered as images, decorative MathML, or layout elements, set `False`. "
                    "UWorld embeds inline images on every page; do not flag those as figures."
                ),
            },
            "involves_data_table": {
                "type": "boolean",
                "description": (
                    "`True` only if interpreting tabular data is required to answer the "
                    "question (e.g., a results table from an experiment, a periodic-table "
                    "excerpt, a thermodynamic data table). `False` if any table in the HTML "
                    "is decorative, used for layout, or just renders math/chemistry symbols."
                ),
            },
        },
    },
}


_SYSTEM_PROMPT = (
    "You are an MCAT question analyst. Given an MCAT practice question, "
    "submit content-agnostic features that describe HOW the question is structured "
    "and what kind of reasoning it demands — independent of which AAMC topic it covers.\n"
    "\n"
    "You will receive a preamble of MECHANICAL FACTS already determined by code "
    "(question_format, has_negative_phrasing, passage_length_bucket). Treat those "
    "as ground truth and DO NOT re-derive or contradict them. Use them to inform "
    "your judgment-call features below.\n"
    "\n"
    "Guidelines:\n"
    "- reasoning_type: pick the HIGHEST-order reasoning the question actually "
    "demands. Most MCAT items land at application or analysis; reserve recall "
    "for pure fact lookup and inference for items that require concluding beyond "
    "what is stated.\n"
    "- requires_calculation: True only if numeric or symbolic arithmetic is needed "
    "to choose the correct answer. Reading a number off a graph is NOT calculation.\n"
    "- calculation_steps: integer count of distinct ops (lookups, conversions, "
    "arithmetic ops). Set to 0 when requires_calculation=false.\n"
    "- passage_type: ONLY set when question_format='passage_based'. Leave empty "
    "string '' for discrete items — the post-processor will normalize.\n"
    "- distractor_difficulty: think about whether a well-prepared student could "
    "rule out wrong answers by elimination (low) vs. would need the precise "
    "concept (high).\n"
    "- trap_distractor_present: true when one wrong answer is specifically the "
    "result of a typical student error (wrong sign, off-by-one, formula swap).\n"
    "- common_misconception: empty string when no specific misconception is "
    "exploited. Do NOT invent one. Most questions do not have one.\n"
    "- jargon_density: judged on stem + choices, NOT the passage.\n"
    "- key_concept_summary: one sentence, <= 25 words, in the form "
    "'Tests [skill/knowledge] of [topic] in context of [scenario].'\n"
    "- involves_graph_or_figure: True ONLY if interpreting a substantive visual "
    "(graph, diagram, chemical structure, anatomical figure, experimental apparatus) "
    "is necessary to answer. Use the explanation as your primary evidence — if it "
    "explicitly refers to 'the figure,' 'the graph,' or 'the diagram,' set True. "
    "UWorld embeds inline images on every page for UI purposes; those are NOT figures.\n"
    "- involves_data_table: True ONLY if interpreting tabular data (results table, "
    "periodic-table excerpt, thermodynamic data table) is required. False if any "
    "table is decorative, layout, or renders math/chemistry symbols.\n"
    "\n"
    "Submit by calling the submit_question_features tool. Do not reply with prose."
)


def _format_mechanical_block(m: MechanicalFeatures) -> str:
    pl = m.passage_length_bucket or "(not applicable — discrete)"
    return (
        "## Mechanical facts (ground truth — do not contradict)\n"
        f"- question_format: {m.question_format}\n"
        f"- has_negative_phrasing: {str(m.has_negative_phrasing).lower()}\n"
        f"- passage_length_bucket: {pl}\n"
    )


def _format_choices(question: Question) -> str:
    lines: list[str] = []
    for c in question.choices or []:
        if not isinstance(c, dict):
            continue
        key = str(c.get("key", "?"))
        plain = str(c.get("plain", "")).strip()
        marker = " (correct)" if key == question.correct_choice else ""
        lines.append(f"- ({key}){marker} {plain}")
    return "\n".join(lines) if lines else "(no choices)"


def _truncate_passage(plain: str | None) -> str:
    if not plain:
        return ""
    if len(plain) <= PASSAGE_TRUNCATE_CHARS:
        return plain
    return plain[:PASSAGE_TRUNCATE_CHARS] + "\n... [truncated]"


def _format_user_message(
    question: Question,
    passage: Passage | None,
    mechanical: MechanicalFeatures,
) -> str:
    stem = (question.stem_plain or "").strip()
    explanation = (question.explanation_plain or "").strip()
    parts: list[str] = [
        _format_mechanical_block(mechanical),
        "",
        "## Question stem",
        stem or "(empty)",
        "",
        "## Choices",
        _format_choices(question),
    ]
    if explanation:
        parts.extend(["", "## Explanation", explanation])
    if passage is not None:
        truncated = _truncate_passage(passage.plain_text)
        if truncated:
            parts.extend(["", "## Passage", truncated])
    return "\n".join(parts)


def make_features_cache_key(
    stem_plain: str | None,
    explanation_plain: str | None,
    passage_plain: str | None,
    mechanical: MechanicalFeatures,
    model: str,
) -> str:
    """Stable SHA-256 over (stem, explanation, passage[:3000], mechanical, model).

    Including mechanical features in the key means a bug fix in the
    mechanical-features regex naturally invalidates affected entries.
    `extractor_version` is checked at lookup time, not folded into the hash.
    """
    h = hashlib.sha256()
    h.update((stem_plain or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((explanation_plain or "").encode("utf-8"))
    h.update(b"\x1f")
    passage_part = _truncate_passage(passage_plain)
    h.update(passage_part.encode("utf-8"))
    h.update(b"\x1f")
    h.update(json.dumps(asdict(mechanical), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(b"\x1f")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _extract_tool_call(message: Message) -> ToolUseBlock | None:
    for block in message.content or []:
        if isinstance(block, ToolUseBlock) and block.name == "submit_question_features":
            return block
    return None


def _parse_tool_input(
    tool_input: dict[str, Any],
    mechanical: MechanicalFeatures,
) -> tuple[LlmJudgmentFeatures, list[str]]:
    """Parse the tool-use payload into LlmJudgmentFeatures + apply post-processing.

    Post-processing rules per the kickoff:
      - passage_type forced to None when question_format='discrete'.
      - passage_type defaults to 'descriptive' (with warning) when passage_based
        but LLM emitted nothing usable.
      - common_misconception '' or whitespace-only → None.
      - calculation_steps clamped to 0 when requires_calculation=False.
    """
    warnings: list[str] = []

    reasoning_type_raw = tool_input.get("reasoning_type")
    if reasoning_type_raw not in {
        "recall",
        "comprehension",
        "application",
        "analysis",
        "inference",
    }:
        warnings.append(f"unknown reasoning_type={reasoning_type_raw!r}; defaulting to application")
        reasoning_type: ReasoningType = "application"
    else:
        reasoning_type = reasoning_type_raw  # type: ignore[assignment]

    requires_calc = bool(tool_input.get("requires_calculation", False))

    try:
        calc_steps = int(tool_input.get("calculation_steps", 0))
    except (TypeError, ValueError):
        warnings.append(
            f"calculation_steps not an int ({tool_input.get('calculation_steps')!r}); defaulting to 0"
        )
        calc_steps = 0
    if calc_steps < 0:
        warnings.append(f"calculation_steps negative ({calc_steps}); clamped to 0")
        calc_steps = 0
    if not requires_calc and calc_steps != 0:
        warnings.append(
            f"requires_calculation=false but calculation_steps={calc_steps}; clamped to 0"
        )
        calc_steps = 0

    pt_raw = tool_input.get("passage_type")
    pt_valid = pt_raw in {"experimental", "descriptive", "hypothesis_driven"}
    passage_type: PassageType | None
    if mechanical.question_format == "discrete":
        if pt_valid:
            warnings.append(
                f"discrete question but LLM emitted passage_type={pt_raw!r}; forced to None"
            )
        passage_type = None
    else:
        if pt_valid:
            passage_type = pt_raw  # type: ignore[assignment]
        else:
            warnings.append(
                f"passage_based question but passage_type={pt_raw!r}; defaulting to 'descriptive'"
            )
            passage_type = "descriptive"

    dd_raw = tool_input.get("distractor_difficulty")
    if dd_raw not in {"low", "medium", "high"}:
        warnings.append(f"unknown distractor_difficulty={dd_raw!r}; defaulting to 'medium'")
        distractor_difficulty: DifficultyLevel = "medium"
    else:
        distractor_difficulty = dd_raw  # type: ignore[assignment]

    trap = bool(tool_input.get("trap_distractor_present", False))

    misc_raw = tool_input.get("common_misconception")
    if not isinstance(misc_raw, str):
        common_misconception: str | None = None
    else:
        stripped = misc_raw.strip()
        common_misconception = stripped or None

    jd_raw = tool_input.get("jargon_density")
    if jd_raw not in {"low", "medium", "high"}:
        warnings.append(f"unknown jargon_density={jd_raw!r}; defaulting to 'medium'")
        jargon_density: DifficultyLevel = "medium"
    else:
        jargon_density = jd_raw  # type: ignore[assignment]

    summary_raw = tool_input.get("key_concept_summary", "")
    if not isinstance(summary_raw, str):
        summary_raw = str(summary_raw)
    summary = summary_raw.strip()
    if not summary:
        warnings.append("key_concept_summary empty; defaulting to placeholder")
        summary = "Tests an MCAT concept in context of the question scenario."

    involves_graph = bool(tool_input.get("involves_graph_or_figure", False))
    involves_table = bool(tool_input.get("involves_data_table", False))

    return (
        LlmJudgmentFeatures(
            reasoning_type=reasoning_type,
            requires_calculation=requires_calc,
            calculation_steps=calc_steps,
            passage_type=passage_type,
            distractor_difficulty=distractor_difficulty,
            trap_distractor_present=trap,
            common_misconception=common_misconception,
            jargon_density=jargon_density,
            key_concept_summary=summary,
            involves_graph_or_figure=involves_graph,
            involves_data_table=involves_table,
        ),
        warnings,
    )


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    cached_input_read: int,
    *,
    model: str,
) -> float:
    p = _pricing_for(model)
    return (
        (input_tokens / 1_000_000) * p["input"]
        + (cached_input_read / 1_000_000) * p["cached_read"]
        + (output_tokens / 1_000_000) * p["output"]
    )


async def extract_judgment_features(
    question: Question,
    passage: Passage | None,
    mechanical: MechanicalFeatures,
    *,
    anthropic_client: AsyncAnthropic,
    cache: "FeatureExtractorCache | None" = None,
    extractor_version: str = EXTRACTOR_VERSION,
    model: str | None = None,
) -> ExtractFeaturesResult:
    """Call the LLM (or cache) to produce LlmJudgmentFeatures for one question.

    `cache`, when provided, is consulted first; on miss the LLM is called and
    the result persisted. `extractor_version` is the invalidation knob — same
    key + different version = cache miss + fresh call.
    """
    resolved_model = model or _model()

    cache_key = make_features_cache_key(
        question.stem_plain,
        question.explanation_plain,
        passage.plain_text if passage is not None else None,
        mechanical,
        resolved_model,
    )
    if cache is not None:
        cached = cache.get(cache_key, extractor_version)
        if cached is not None:
            original_cost = cache.lookup_cost(cache_key)
            logger.debug(
                "extract_features qid=%s: persistent cache hit (saved ~$%.4f)",
                question.qid,
                original_cost,
            )
            return ExtractFeaturesResult(
                features=cached.features,
                cache_hit=True,
                cost_saved_usd=original_cost,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                estimated_cost_usd=0.0,
                extractor_version=extractor_version,
                model=resolved_model,
                parse_warnings=list(cached.parse_warnings),
            )

    system_blocks = [
        {
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
    ]
    user_message = _format_user_message(question, passage, mechanical)

    response = await anthropic_client.messages.create(
        model=resolved_model,
        max_tokens=MAX_TOKENS,
        system=system_blocks,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "submit_question_features"},
        messages=[{"role": "user", "content": user_message}],
    )

    input_tokens = response.usage.input_tokens or 0
    output_tokens = response.usage.output_tokens or 0
    cached_input_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cost = _compute_cost(
        input_tokens + cache_creation,
        output_tokens,
        cached_input_read,
        model=resolved_model,
    )

    tool_call = _extract_tool_call(response)
    if tool_call is None:
        msg = "LLM did not call submit_question_features"
        logger.warning("extract_features qid=%s: %s", question.qid, msg)
        # Fall back to a safe default block so downstream UPSERT can still proceed.
        fallback = LlmJudgmentFeatures(
            reasoning_type="application",
            requires_calculation=False,
            calculation_steps=0,
            passage_type=("descriptive" if mechanical.question_format == "passage_based" else None),
            distractor_difficulty="medium",
            trap_distractor_present=False,
            common_misconception=None,
            jargon_density="medium",
            key_concept_summary="Tests an MCAT concept in context of the question scenario.",
            involves_graph_or_figure=False,
            involves_data_table=False,
        )
        result = ExtractFeaturesResult(
            features=fallback,
            cache_hit=False,
            cost_saved_usd=0.0,
            input_tokens=input_tokens + cache_creation + cached_input_read,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            extractor_version=extractor_version,
            model=resolved_model,
            parse_warnings=[msg],
        )
        if cache is not None:
            cache.put(cache_key, result, extractor_version, model=resolved_model)
        return result

    features, warnings = _parse_tool_input(tool_call.input, mechanical)  # type: ignore[arg-type]

    result = ExtractFeaturesResult(
        features=features,
        cache_hit=False,
        cost_saved_usd=0.0,
        input_tokens=input_tokens + cache_creation + cached_input_read,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        extractor_version=extractor_version,
        model=resolved_model,
        parse_warnings=warnings,
    )
    if cache is not None:
        cache.put(cache_key, result, extractor_version, model=resolved_model)

    logger.info(
        "extract_features qid=%s model=%s in=%d cache_create=%d cache_read=%d "
        "out=%d cost=$%.4f warnings=%d",
        question.qid,
        resolved_model,
        input_tokens,
        cache_creation,
        cached_input_read,
        output_tokens,
        cost,
        len(warnings),
    )
    return result
