"""Side-by-side comparison of Sonnet 4.6 vs Haiku 4.5 on the categorizer task.

Read-only against `question_tags` — never writes production state. Each
question is categorized twice (once per model) and the per-question tag
sets are compared via Jaccard overlap (skill tags excluded; they're
mechanical and would always match).

Writes a Markdown report to stdout (or `--output PATH`). The "Verdict"
section is left as a template for the human to fill in.

CLI:
    python -m scripts.eval_categorizer_models --sample 20
    python -m scripts.eval_categorizer_models --sample 20 --output eval-results.md
    python -m scripts.eval_categorizer_models --sample 20 --stratify
    python -m scripts.eval_categorizer_models --sample 5 --yes  # skip confirm
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.captures import Question
from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.llm import LlmTagSuggestion, categorize
from app.services.categorizer.outline_lookup import OutlineLookup

logger = logging.getLogger(__name__)


SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
COST_ESTIMATE_PER_QUESTION = 0.015


@dataclass
class PerModelOutcome:
    suggestions: list[LlmTagSuggestion]
    cost_usd: float
    runtime_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass
class QuestionEval:
    qid: str
    subject: str
    chapter: str | None
    sonnet: PerModelOutcome
    haiku: PerModelOutcome
    jaccard_excl_skill: float


@dataclass
class AggregateMetrics:
    sonnet_total_cost: float = 0.0
    haiku_total_cost: float = 0.0
    sonnet_total_runtime: float = 0.0
    haiku_total_runtime: float = 0.0
    sonnet_tag_count_excl_skill: int = 0
    haiku_tag_count_excl_skill: int = 0
    sonnet_confidence_sum: float = 0.0
    haiku_confidence_sum: float = 0.0
    sonnet_confidence_n: int = 0
    haiku_confidence_n: int = 0
    jaccard_values: list[float] = field(default_factory=list)


def _subject_from_tags(tags: list[str] | None) -> str:
    if not tags:
        return "(unknown)"
    for t in tags:
        if isinstance(t, str) and t.startswith("Subject: "):
            return t[len("Subject: ") :].strip()
    return "(unknown)"


def _chapter_from_tags(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    for t in tags:
        if isinstance(t, str) and t.startswith("Chapter: "):
            return t[len("Chapter: ") :].strip()
    return None


def _suggestion_key(s: LlmTagSuggestion) -> tuple[str, str]:
    """Key used for Jaccard set construction. Skill kind is excluded by caller."""
    ident = str(s.identifier).strip().lower()
    if s.kind == "topic" and s.under_content_category:
        ident = f"{s.under_content_category.upper()}/{ident}"
    return (s.kind, ident)


def _jaccard_excl_skill(a: list[LlmTagSuggestion], b: list[LlmTagSuggestion]) -> float:
    set_a = {_suggestion_key(s) for s in a if s.kind != "skill"}
    set_b = {_suggestion_key(s) for s in b if s.kind != "skill"}
    if not set_a and not set_b:
        return 1.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)


async def _sample_questions(
    session: AsyncSession, *, sample: int, stratify: bool
) -> list[Question]:
    if not stratify:
        stmt = (
            select(Question)
            .where(Question.uworld_aamc_tags.is_not(None))
            .order_by(func.random())
            .limit(sample)
        )
        return list((await session.execute(stmt)).scalars().all())

    # Stratify: bucket by Subject and round-robin until we have `sample`.
    all_q = list(
        (await session.execute(select(Question).where(Question.uworld_aamc_tags.is_not(None))))
        .scalars()
        .all()
    )
    by_subject: dict[str, list[Question]] = {}
    for q in all_q:
        by_subject.setdefault(_subject_from_tags(q.uworld_aamc_tags), []).append(q)

    import random

    rng = random.Random(0xDEAD_BEEF)
    for v in by_subject.values():
        rng.shuffle(v)

    picked: list[Question] = []
    while len(picked) < sample and by_subject:
        empty: list[str] = []
        for subj, bucket in by_subject.items():
            if not bucket:
                empty.append(subj)
                continue
            picked.append(bucket.pop())
            if len(picked) >= sample:
                break
        for e in empty:
            del by_subject[e]
    return picked


async def _run_once(
    question: Question,
    *,
    model: str,
    client: AsyncAnthropic,
    lookup: OutlineLookup,
    cache: CategorizerCache,
    extractor_version: str,
) -> PerModelOutcome:
    started = time.perf_counter()
    cat = await categorize(
        question,
        anthropic_client=client,
        outline_lookup=lookup,
        cache=cache,
        extractor_version=extractor_version,
        model=model,
    )
    elapsed = time.perf_counter() - started
    return PerModelOutcome(
        suggestions=list(cat.suggestions),
        cost_usd=cat.estimated_cost_usd,
        runtime_seconds=elapsed,
        input_tokens=cat.input_tokens,
        output_tokens=cat.output_tokens,
    )


def _fmt_sugg(s: LlmTagSuggestion) -> str:
    if s.kind == "topic":
        prefix = f"{s.kind} {s.confidence:.2f}"
        cc = s.under_content_category or ""
        body = f"{cc + ' / ' if cc else ''}{s.identifier}"
    elif s.kind == "content_category":
        prefix = f"{s.kind} {s.confidence:.2f}"
        body = str(s.identifier)
    else:
        prefix = f"{s.kind} {s.confidence:.2f}"
        body = str(s.identifier)
    rat = (s.rationale or "").strip().replace("\n", " ")
    if len(rat) > 160:
        rat = rat[:157] + "…"
    return f"[{prefix}] {body} — {rat}"


def _avg(num: float, den: float) -> float:
    return num / den if den else 0.0


def _build_markdown(
    *,
    evaluations: list[QuestionEval],
    aggregate: AggregateMetrics,
    sample: int,
    stratified: bool,
    started_iso: str,
    sonnet_model: str,
    haiku_model: str,
) -> str:
    subjects = sorted({e.subject for e in evaluations})
    n = len(evaluations)

    def _bucket(j: float) -> str:
        if j >= 1.0 - 1e-9:
            return "perfect"
        if j >= 0.75:
            return "strong"
        if j >= 0.5:
            return "partial"
        return "weak"

    buckets = {"perfect": 0, "strong": 0, "partial": 0, "weak": 0}
    for j in aggregate.jaccard_values:
        buckets[_bucket(j)] += 1

    def _pct(k: int) -> str:
        return f"{(100 * k / n):.0f}%" if n else "—"

    mean_j = sum(aggregate.jaccard_values) / n if n else 0.0
    ratio = (
        aggregate.sonnet_total_cost / aggregate.haiku_total_cost
        if aggregate.haiku_total_cost
        else 0.0
    )

    lines = [
        f"# Categorizer Model Comparison: {sonnet_model} vs {haiku_model}",
        "",
        f"Generated: {started_iso}",
        f"Sample size: {n} questions",
        f"Stratified: {'yes' if stratified else 'no'}",
        f"Subjects represented: {', '.join(subjects) if subjects else '(none)'}",
        "",
        "## Aggregate",
        "",
        "| Metric | Sonnet 4.6 | Haiku 4.5 | Ratio |",
        "|---|---|---|---|",
        f"| Total cost | ${aggregate.sonnet_total_cost:.4f} | ${aggregate.haiku_total_cost:.4f} | {ratio:.2f}× |",
        f"| Avg cost per question | ${_avg(aggregate.sonnet_total_cost, n):.4f} | ${_avg(aggregate.haiku_total_cost, n):.4f} | |",
        f"| Avg tags per question (excl. skill) | {_avg(aggregate.sonnet_tag_count_excl_skill, n):.2f} | {_avg(aggregate.haiku_tag_count_excl_skill, n):.2f} | |",
        f"| Avg confidence | {_avg(aggregate.sonnet_confidence_sum, aggregate.sonnet_confidence_n):.3f} | {_avg(aggregate.haiku_confidence_sum, aggregate.haiku_confidence_n):.3f} | |",
        f"| Avg runtime (s) | {_avg(aggregate.sonnet_total_runtime, n):.2f} | {_avg(aggregate.haiku_total_runtime, n):.2f} | |",
        "",
        "## Overlap analysis (excluding auto-parsed skill tags)",
        "",
        "| Match level | Count | % |",
        "|---|---|---|",
        f"| Perfect (Jaccard = 1.0) | {buckets['perfect']} | {_pct(buckets['perfect'])} |",
        f"| Strong (≥ 0.75) | {buckets['strong']} | {_pct(buckets['strong'])} |",
        f"| Partial (≥ 0.5) | {buckets['partial']} | {_pct(buckets['partial'])} |",
        f"| Weak (< 0.5) | {buckets['weak']} | {_pct(buckets['weak'])} |",
        "",
        f"Mean Jaccard: {mean_j:.3f}",
        "",
        "## Per-question detail",
        "",
    ]

    for e in evaluations:
        chap = f", Chapter: {e.chapter}" if e.chapter else ""
        lines.append(f"### qid={e.qid} — Subject: {e.subject}{chap}")
        lines.append("")
        lines.append(
            f"**Sonnet (cost ${e.sonnet.cost_usd:.4f}, runtime {e.sonnet.runtime_seconds:.1f}s):**"
        )
        if not e.sonnet.suggestions:
            lines.append("- (no suggestions)")
        for s in e.sonnet.suggestions:
            lines.append(f"- {_fmt_sugg(s)}")
        lines.append("")
        lines.append(
            f"**Haiku (cost ${e.haiku.cost_usd:.4f}, runtime {e.haiku.runtime_seconds:.1f}s):**"
        )
        if not e.haiku.suggestions:
            lines.append("- (no suggestions)")
        for s in e.haiku.suggestions:
            lines.append(f"- {_fmt_sugg(s)}")
        sonnet_non_skill = [s for s in e.sonnet.suggestions if s.kind != "skill"]
        haiku_non_skill = [s for s in e.haiku.suggestions if s.kind != "skill"]
        sonnet_set = {_suggestion_key(s) for s in sonnet_non_skill}
        matched = sum(1 for s in haiku_non_skill if _suggestion_key(s) in sonnet_set)
        lines.append("")
        lines.append(
            f"Overlap: {e.jaccard_excl_skill:.2f} "
            f"({matched} of {len(sonnet_non_skill)} Sonnet tags matched by Haiku)"
        )
        lines.append("")

    lines.extend(
        [
            "## Verdict (manual)",
            "",
            "> Fill in after eyeballing. Look for: cases where Haiku missed multi-topic structure",
            "> Sonnet caught; cases where Haiku picked clearly wrong tags; whether the cost",
            "> reduction justifies any quality loss.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_eval(
    session: AsyncSession,
    *,
    sample: int,
    stratify: bool,
    client: AsyncAnthropic,
    cache: CategorizerCache,
    eval_version: str,
) -> tuple[list[QuestionEval], AggregateMetrics, str]:
    """Core evaluation loop. Accepts an already-open session so tests can inject a test DB."""
    started_iso = datetime.now(tz=timezone.utc).isoformat()
    aggregate = AggregateMetrics()
    evaluations: list[QuestionEval] = []

    lookup = await OutlineLookup.load(session)
    questions = await _sample_questions(session, sample=sample, stratify=stratify)
    if not questions:
        return evaluations, aggregate, started_iso

    for i, q in enumerate(questions, start=1):
        logger.info("evaluating %d/%d qid=%s", i, len(questions), q.qid)
        sonnet = await _run_once(
            q,
            model=SONNET_MODEL,
            client=client,
            lookup=lookup,
            cache=cache,
            extractor_version=eval_version,
        )
        haiku = await _run_once(
            q,
            model=HAIKU_MODEL,
            client=client,
            lookup=lookup,
            cache=cache,
            extractor_version=eval_version,
        )
        j = _jaccard_excl_skill(sonnet.suggestions, haiku.suggestions)
        evaluations.append(
            QuestionEval(
                qid=q.qid,
                subject=_subject_from_tags(q.uworld_aamc_tags),
                chapter=_chapter_from_tags(q.uworld_aamc_tags),
                sonnet=sonnet,
                haiku=haiku,
                jaccard_excl_skill=j,
            )
        )
        aggregate.sonnet_total_cost += sonnet.cost_usd
        aggregate.haiku_total_cost += haiku.cost_usd
        aggregate.sonnet_total_runtime += sonnet.runtime_seconds
        aggregate.haiku_total_runtime += haiku.runtime_seconds
        for s in sonnet.suggestions:
            if s.kind != "skill":
                aggregate.sonnet_tag_count_excl_skill += 1
            aggregate.sonnet_confidence_sum += s.confidence
            aggregate.sonnet_confidence_n += 1
        for s in haiku.suggestions:
            if s.kind != "skill":
                aggregate.haiku_tag_count_excl_skill += 1
            aggregate.haiku_confidence_sum += s.confidence
            aggregate.haiku_confidence_n += 1
        aggregate.jaccard_values.append(j)

    return evaluations, aggregate, started_iso


async def main_async(args: argparse.Namespace) -> int:
    estimate = args.sample * COST_ESTIMATE_PER_QUESTION * 2  # two models
    print(
        f"About to run {args.sample} questions × 2 models (~${estimate:.2f} estimated).",
        file=sys.stderr,
    )
    if not args.yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.", file=sys.stderr)
            return 1

    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    # Eval-only extractor_version keyed on timestamp so eval doesn't pollute
    # the production cache AND re-running the script always calls the LLM.
    eval_version = f"eval-{int(time.time())}"
    cache = CategorizerCache(settings.CATEGORIZER_CACHE_PATH)

    try:
        async with AsyncSessionLocal() as session:
            evaluations, aggregate, started_iso = await run_eval(
                session,
                sample=args.sample,
                stratify=args.stratify,
                client=client,
                cache=cache,
                eval_version=eval_version,
            )

        if not evaluations:
            print("No questions matched the sample criteria.", file=sys.stderr)
            return 1

        markdown = _build_markdown(
            evaluations=evaluations,
            aggregate=aggregate,
            sample=args.sample,
            stratified=args.stratify,
            started_iso=started_iso,
            sonnet_model=SONNET_MODEL,
            haiku_model=HAIKU_MODEL,
        )

        sys.stdout.write(markdown + "\n")
        if args.output:
            Path(args.output).write_text(markdown + "\n", encoding="utf-8")
            print(f"\nReport also written to {args.output}", file=sys.stderr)
    finally:
        cache.clear(extractor_version=eval_version)
        cache.close()
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--yes", action="store_true", help="Skip the cost-estimate confirmation.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(_cli())
