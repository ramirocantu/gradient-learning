"""LLM insight synthesizer — Phase 4.5.

Takes an InsightReport from analyze() and produces readable markdown prose
that a tutor would write. Uses Claude Sonnet 4.6 with ephemeral system-prompt
caching. Cache keyed on (report content hash, model). EXTRACTOR_VERSION is
stored separately and checked at lookup — bumping it invalidates cached entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass

from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analyzer.patterns import AnalysisFilter, InsightReport, analyze
from app.services.analyzer.synthesizer_cache import SynthesizerCache

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "synthesizer-v1-claude-sonnet-4-6"
MODEL = "claude-sonnet-4-6"
TARGET_OUTPUT_TOKENS = 800

_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cached_read": 0.30},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cached_read": 0.10},
}

_SYSTEM_PROMPT = """\
You are an experienced MCAT tutor analyzing a student's practice performance.
You're given a structured pattern-analysis report. Your job is to turn it into
readable, actionable prose the student will actually use.

Voice and tone:
- Direct and specific. Cite actual feature names, accuracy percentages, and qids.
- Honest about confidence. When sample sizes are small (n < 8 in either
  with-group or without-group), explicitly flag the finding as preliminary
  using a phrase like "based on a small sample" or "this needs more data
  to confirm."
- Tutor voice, not academic. "You're getting tripped up by X" — not
  "The student exhibits decreased accuracy on questions involving X."
- Encouraging, not punishing. If positive findings exist (features where the
  student does better than baseline), surface them — "what's working" matters
  as much as "what's wrong."

Format: Markdown. Structure:

## Quick read
[1 short paragraph: overall accuracy, biggest pattern, total sample size.]

## What's hurting you
[For each of the top 3 most-negative confident_delta findings:]
### [Feature name in plain English]: [accuracy_with]% vs [accuracy_without]% without
[2-3 sentences: what the feature means, why this pattern might exist,
  concrete study recommendation. Cite up to 2 representative qids.]

## What's working
[Mention any findings with positive confident_delta (features where accuracy
 is meaningfully ABOVE baseline). 1-2 sentences each. Maximum 2 entries.]

## Caveats
[List any small-sample warnings, uncategorized-question count if non-zero,
 and the coverage gap if questions_without_features > 0. 1-3 bullet items.]

Target length: 300-600 words total. Don't pad. If the report has fewer than
3 negative findings, write fewer subsections — never invent material.

Do not include conclusions or "good luck" closers. Stop at the last caveat.\
"""

_FALLBACK_MARKDOWN = """\
## Quick read

The analyzer ran but the LLM didn't produce a parseable response. \
Try re-running with `?bust_cache=1` or check the underlying report at \
`/api/v1/analyzer/patterns`.\
"""

_NO_DATA_MARKDOWN = """\
## Quick read

Not enough data yet. No findings met the minimum sample size threshold for \
this filter. Attempt more questions in this category and re-run.\
"""


def _pricing_for(model: str) -> dict[str, float]:
    if model in _PRICING:
        return _PRICING[model]
    logger.warning("no pricing known for model=%r; using Sonnet rates", model)
    return _PRICING["claude-sonnet-4-6"]


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


@dataclass(frozen=True)
class InsightSynthesis:
    markdown: str
    report: InsightReport
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    cost_saved_usd: float
    extractor_version: str
    model: str


def make_synthesis_cache_key(report: InsightReport, model: str) -> str:
    """SHA-256 of canonical JSON of InsightReport content + model."""
    payload = json.dumps(asdict(report), sort_keys=True, default=str)
    h = hashlib.sha256()
    h.update(payload.encode("utf-8"))
    h.update(b"\x1f")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _format_filter_section(af: AnalysisFilter) -> str:
    return (
        f"section: {af.section_code or '(all)'}\n"
        f"content_category: {af.content_category_code or '(all)'}\n"
        f"topic_id: {af.topic_id or '(all)'}\n"
        f"skill: {af.skill or '(all)'}\n"
        f"since: {af.since or '(all time)'}\n"
        f"until: {af.until or '(all time)'}\n"
        f"min_sample_size: {af.min_sample_size}"
    )


def _format_user_message(report: InsightReport) -> str:
    baseline_pct = f"{report.baseline_accuracy * 100:.1f}%"
    lines: list[str] = [
        "## Analysis Scope",
        _format_filter_section(report.filter_applied),
        "",
        "## Baseline",
        f"accuracy: {baseline_pct} ({report.total_attempts_in_scope} attempts, "
        f"{report.total_questions_in_scope} questions)",
        f"wilson_lower_bound: {report.baseline_wilson_lower:.3f}",
        "",
    ]

    top_findings = report.findings[:8]

    if top_findings:
        lines.append("## Top Findings (sorted by impact, most negative confident_delta first)")
        lines.append("")
        for i, ff in enumerate(top_findings, 1):
            acc_with_pct = f"{ff.accuracy_with * 100:.1f}%"
            acc_without_pct = f"{ff.accuracy_without * 100:.1f}%"
            qids_str = ", ".join(ff.representative_missed_qids) or "(none missed)"
            small_n = min(ff.attempts_with, ff.attempts_without) < 8
            lines.extend(
                [
                    f"### Finding {i}: {ff.feature_name} = {ff.feature_value}",
                    f"accuracy WITH: {acc_with_pct} ({ff.correct_with}/{ff.attempts_with} correct)",
                    f"accuracy WITHOUT: {acc_without_pct} "
                    f"({ff.correct_without}/{ff.attempts_without} correct)",
                    f"accuracy_delta: {ff.accuracy_delta * 100:.1f}%",
                    f"confident_delta: {ff.confident_delta:.3f}",
                    f"representative missed qids: {qids_str}",
                    f"small_sample_flag: "
                    f"{'YES — n < 8 in at least one group' if small_n else 'no'}",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Findings",
                "No findings met the minimum sample size threshold.",
                "",
            ]
        )

    lines.extend(
        [
            "## Coverage",
            f"questions_with_features: {report.coverage.questions_with_features}",
            f"questions_without_features: {report.coverage.questions_without_features}",
            f"feature_extractor_version: {report.coverage.feature_extractor_version}",
        ]
    )

    return "\n".join(lines)


def _is_valid_markdown(text: str) -> bool:
    return "##" in text and len(text.strip()) >= 100


async def synthesize(
    report: InsightReport,
    *,
    anthropic_client: AsyncAnthropic,
    cache: SynthesizerCache,
    bust_cache: bool = False,
    run_llm: bool = True,
    extractor_version: str = EXTRACTOR_VERSION,
    model: str = MODEL,
) -> InsightSynthesis | None:
    """Synthesize an InsightReport into readable markdown prose.

    When run_llm=False and no cached result exists, returns None instead of
    calling the LLM. Use this for page loads where LLM should only fire on
    explicit user action.
    """
    if not report.findings:
        logger.info("synthesize: no findings in report, returning no-data fallback")
        return InsightSynthesis(
            markdown=_NO_DATA_MARKDOWN,
            report=report,
            cache_hit=False,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            cost_saved_usd=0.0,
            extractor_version=extractor_version,
            model=model,
        )

    cache_key = make_synthesis_cache_key(report, model)

    if not bust_cache:
        cached = cache.get(cache_key, extractor_version)
        if cached is not None:
            original_cost = cache.lookup_cost(cache_key)
            logger.debug("synthesize: cache hit (saved ~$%.4f)", original_cost)
            return InsightSynthesis(
                markdown=cached.markdown,
                report=report,
                cache_hit=True,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                estimated_cost_usd=0.0,
                cost_saved_usd=original_cost,
                extractor_version=extractor_version,
                model=model,
            )

    if not run_llm:
        logger.debug("synthesize: cache miss and run_llm=False, returning None")
        return None

    system_blocks = [
        {
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_message = _format_user_message(report)

    response = await anthropic_client.messages.create(
        model=model,
        max_tokens=TARGET_OUTPUT_TOKENS,
        system=system_blocks,
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
        model=model,
    )

    raw_text = ""
    if response.content:
        first = response.content[0]
        if hasattr(first, "text"):
            raw_text = first.text

    markdown = raw_text.strip()

    if not _is_valid_markdown(markdown):
        logger.warning(
            "synthesize: LLM response malformed or empty (len=%d); using fallback",
            len(markdown),
        )
        markdown = _FALLBACK_MARKDOWN

    logger.info(
        "synthesize model=%s in=%d cache_create=%d cache_read=%d out=%d cost=$%.4f",
        model,
        input_tokens,
        cache_creation,
        cached_input_read,
        output_tokens,
        cost,
    )

    total_input = input_tokens + cache_creation + cached_input_read
    cache.put(
        cache_key,
        markdown,
        extractor_version,
        model=model,
        input_tokens=total_input,
        output_tokens=output_tokens,
        cost_estimate_usd=cost,
    )

    return InsightSynthesis(
        markdown=markdown,
        report=report,
        cache_hit=False,
        input_tokens=total_input,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        cost_saved_usd=0.0,
        extractor_version=extractor_version,
        model=model,
    )


async def insights_for_filter(
    filter: AnalysisFilter,
    session: AsyncSession,
    *,
    anthropic_client: AsyncAnthropic,
    cache: SynthesizerCache,
    bust_cache: bool = False,
    run_llm: bool = True,
    extractor_version: str = EXTRACTOR_VERSION,
    model: str = MODEL,
) -> InsightSynthesis | None:
    """Convenience wrapper: analyze(), then synthesize()."""
    report = await analyze(filter, session)
    return await synthesize(
        report,
        anthropic_client=anthropic_client,
        cache=cache,
        bust_cache=bust_cache,
        run_llm=run_llm,
        extractor_version=extractor_version,
        model=model,
    )
