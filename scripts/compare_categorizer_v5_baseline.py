"""Compare the persisted production categorizer baseline against the latest
extractor's live picks on the same questions.

Baseline = whatever is already in `question_tags` w/ `source='llm'` and
`extractor_version` matching `--baseline-version` (default
`v4-clean-explanation`, the most recent corpus drain at the time T55 was
opened). No re-call needed; the rows are the historical record.

Latest = the current `app.services.categorizer.llm.categorize` (whatever
`EXTRACTOR_VERSION` is in the source tree right now — v6 → v7 → ...).
Called live against Anthropic w/ `cache=None` so we always hit the API
and get true per-call token + cost numbers. The current SQLite cache is
not touched.

Outputs per question:
  - baseline identifiers ({(kind, target_id)} set) sourced from DB
  - latest identifiers (live API call) + token + cost report
  - set diff

Aggregate:
  - exact set-equality rate
  - mean jaccard
  - per-section breakdown
  - per-question cost

Use to validate that prompt/schema edits don't degrade pick quality
BEFORE bumping `EXTRACTOR_VERSION` and re-extracting the deck.

Usage:
    cd backend
    uv run python -m scripts.compare_categorizer_v5_baseline \
        [--n=12] [--baseline-version=v4-clean-explanation] [--section=CP]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from random import sample

from anthropic import AsyncAnthropic
from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.captures import Question, QuestionTag
from app.models.outline import ContentCategory, Topic
from app.services.categorizer.llm import EXTRACTOR_VERSION, categorize
from app.services.categorizer.outline_lookup import OutlineLookup
from app.services.categorizer.outline_render import SUBJECT_TO_SECTION

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_VERSION = "v4-clean-explanation"

# Sonnet 4.6 pricing for one-shot cost display (matches llm._PRICING).
_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "cached_read": 0.30, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "cached_read": 0.10, "output": 5.0},
}


def _identifier_label(
    kind: str,
    target_id: int,
    topic_path_by_id: dict[int, str],
    cc_code_by_id: dict[int, str],
) -> str:
    if kind == "topic":
        return f"topic:{topic_path_by_id.get(target_id, f'<id={target_id}>')}"
    if kind == "content_category":
        return f"cc:{cc_code_by_id.get(target_id, f'<id={target_id}>')}"
    return f"skill:{target_id}"


@dataclass
class QuestionComparison:
    question_id: int
    qid: str
    section: str | None
    baseline: set[tuple[str, int]] = field(default_factory=set)
    latest: set[tuple[str, int]] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    cost: float = 0.0
    error: str | None = None

    @property
    def jaccard(self) -> float:
        a, b = self.baseline, self.latest
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 1.0
        return len(a & b) / len(union)


async def _gather_baseline(
    session,
    *,
    baseline_version: str,
    section_filter: str | None,
) -> dict[int, tuple[Question, set[tuple[str, int]]]]:
    """Map question_id → (Question, {(kind, target_id)}) for all rows at the
    given baseline version. `section_filter` (CP/CARS/BB/PS) trims via the
    question's Subject tag.
    """
    stmt = (
        select(Question, QuestionTag)
        .join(QuestionTag, QuestionTag.question_id == Question.id)
        .where(QuestionTag.source == "llm")
        .where(QuestionTag.extractor_version == baseline_version)
    )
    rows = (await session.execute(stmt)).all()

    grouped: dict[int, tuple[Question, set[tuple[str, int]]]] = {}
    for q, t in rows:
        if section_filter is not None:
            subject = next(
                (
                    s[len("Subject: ") :].strip()
                    for s in (q.uworld_aamc_tags or [])
                    if isinstance(s, str) and s.startswith("Subject: ")
                ),
                None,
            )
            sec = SUBJECT_TO_SECTION.get(subject) if subject else None
            if sec != section_filter:
                continue

        if t.topic_id is not None:
            ident = ("topic", t.topic_id)
        elif t.content_category_id is not None:
            ident = ("content_category", t.content_category_id)
        elif t.skill is not None:
            ident = ("skill", int(t.skill))
        else:
            continue

        if q.id not in grouped:
            grouped[q.id] = (q, set())
        grouped[q.id][1].add(ident)

    return grouped


async def _run_latest(
    question: Question,
    *,
    client: AsyncAnthropic,
    lookup: OutlineLookup,
) -> tuple[set[tuple[str, int]], int, int, int, float, str | None]:
    try:
        result = await categorize(
            question,
            anthropic_client=client,
            outline_lookup=lookup,
            cache=None,
        )
    except Exception as exc:  # noqa: BLE001
        return set(), 0, 0, 0, 0.0, f"{type(exc).__name__}: {exc}"

    latest: set[tuple[str, int]] = set()
    for s in result.suggestions:
        if s.kind == "topic":
            tid = lookup.topic_id_by_path(str(s.identifier))
            if tid is not None:
                latest.add(("topic", tid))
        elif s.kind == "content_category":
            cc_id = lookup.content_category_id(str(s.identifier))
            if cc_id is not None:
                latest.add(("content_category", cc_id))
        else:
            try:
                latest.add(("skill", int(s.identifier)))
            except (TypeError, ValueError):
                continue

    return (
        latest,
        result.input_tokens,
        result.output_tokens,
        0,  # categorize folds cache_read into input_tokens already
        result.estimated_cost_usd,
        None,
    )


def _format_set(
    s: set[tuple[str, int]],
    topic_path_by_id: dict[int, str],
    cc_code_by_id: dict[int, str],
) -> str:
    if not s:
        return "(empty)"
    return ", ".join(
        sorted(_identifier_label(kind, tid, topic_path_by_id, cc_code_by_id) for kind, tid in s)
    )


def _print_comparison(
    cmp: QuestionComparison,
    topic_path_by_id: dict[int, str],
    cc_code_by_id: dict[int, str],
) -> None:
    print("─" * 88)
    print(f"qid={cmp.qid} section={cmp.section}  (Q.id={cmp.question_id})")
    print(
        f"  baseline ({len(cmp.baseline)} tags): {_format_set(cmp.baseline, topic_path_by_id, cc_code_by_id)}"
    )
    if cmp.error:
        print(f"  latest: ERROR {cmp.error}")
        return
    print(
        f"  latest   ({len(cmp.latest)} tags): {_format_set(cmp.latest, topic_path_by_id, cc_code_by_id)}"
    )
    common = cmp.baseline & cmp.latest
    only_base = cmp.baseline - cmp.latest
    only_late = cmp.latest - cmp.baseline
    if cmp.baseline == cmp.latest and cmp.baseline:
        print("  → IDENTICAL set")
    elif not cmp.baseline and not cmp.latest:
        print("  → both empty")
    else:
        if common:
            print(f"  common: {_format_set(common, topic_path_by_id, cc_code_by_id)}")
        if only_base:
            print(f"  only baseline: {_format_set(only_base, topic_path_by_id, cc_code_by_id)}")
        if only_late:
            print(f"  only latest:   {_format_set(only_late, topic_path_by_id, cc_code_by_id)}")
    print(
        f"  tokens: in={cmp.input_tokens} out={cmp.output_tokens}  cost=${cmp.cost:.5f}  jaccard={cmp.jaccard:.2f}"
    )


def _print_aggregate(
    baseline_version: str,
    latest_version: str,
    comparisons: list[QuestionComparison],
    topic_path_by_id: dict[int, str],
    cc_code_by_id: dict[int, str],
) -> None:
    print()
    print("=" * 88)
    print(f"Aggregate over {len(comparisons)} questions")
    print(f"  baseline version = {baseline_version!r}")
    print(f"  latest version   = {latest_version!r}")

    errs = [c for c in comparisons if c.error]
    ok = [c for c in comparisons if not c.error]
    if errs:
        print(f"  errors: {len(errs)} question(s)")
        for c in errs:
            print(f"    qid={c.qid} err={c.error!r}")
    if not ok:
        print("  no successful latest-arm calls; nothing to aggregate.")
        return

    exact_eq = sum(1 for c in ok if c.baseline == c.latest)
    mean_jac = sum(c.jaccard for c in ok) / len(ok)
    total_base = sum(len(c.baseline) for c in ok)
    total_late = sum(len(c.latest) for c in ok)
    added = sum(len(c.latest - c.baseline) for c in ok)
    dropped = sum(len(c.baseline - c.latest) for c in ok)
    cost = sum(c.cost for c in ok)
    tin = sum(c.input_tokens for c in ok)
    tout = sum(c.output_tokens for c in ok)

    by_sec: dict[str | None, list[float]] = {}
    for c in ok:
        by_sec.setdefault(c.section, []).append(c.jaccard)

    print(f"  exact set-equality: {exact_eq}/{len(ok)} ({100 * exact_eq / len(ok):.0f}%)")
    print(f"  mean jaccard:       {mean_jac:.3f}")
    print(f"  tags: baseline={total_base} latest={total_late} (added={added}, dropped={dropped})")
    print(
        f"  latest tokens: in={tin} out={tout}  total cost ≈ ${cost:.4f} "
        f"({cost / len(ok):.5f}/question)"
    )

    if by_sec:
        print("  per-section mean jaccard:")
        for sec, vals in sorted(by_sec.items(), key=lambda kv: kv[0] or "ZZ"):
            print(f"    {sec or '<unknown>'}: {sum(vals) / len(vals):.3f}  (n={len(vals)})")

    added_lbl: Counter = Counter()
    dropped_lbl: Counter = Counter()
    for c in ok:
        for ident in c.latest - c.baseline:
            added_lbl[_identifier_label(ident[0], ident[1], topic_path_by_id, cc_code_by_id)] += 1
        for ident in c.baseline - c.latest:
            dropped_lbl[_identifier_label(ident[0], ident[1], topic_path_by_id, cc_code_by_id)] += 1
    if added_lbl:
        print("  top latest-only tags:")
        for label, n in added_lbl.most_common(5):
            print(f"    +{n}  {label}")
    if dropped_lbl:
        print("  top baseline-only tags (latest dropped):")
        for label, n in dropped_lbl.most_common(5):
            print(f"    -{n}  {label}")


async def main(
    n_questions: int,
    baseline_version: str,
    section_filter: str | None,
    concurrency: int,
) -> None:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY, max_retries=5)
    async with AsyncSessionLocal() as session:
        baseline = await _gather_baseline(
            session,
            baseline_version=baseline_version,
            section_filter=section_filter,
        )
        lookup = await OutlineLookup.load(session)
        # Build label maps for output rendering.
        topic_rows = (await session.execute(select(Topic.id, Topic.name))).all()
        cc_rows = (await session.execute(select(ContentCategory.id, ContentCategory.code))).all()
        # Reconstruct canonical paths via the lookup's reverse map if possible;
        # else fall back to bare topic name (sufficient for diff rendering).
        topic_path_by_id = {tid: name for tid, name in topic_rows}
        cc_code_by_id = {cid: code for cid, code in cc_rows}

    if not baseline:
        print(
            f"No baseline rows at extractor_version={baseline_version!r}"
            + (f" / section={section_filter!r}" if section_filter else "")
        )
        return

    items = list(baseline.values())
    chosen = sample(items, min(n_questions, len(items)))
    print(
        f"\nComparing baseline {baseline_version!r} (persisted) vs latest "
        f"{EXTRACTOR_VERSION!r} (live, no cache) on {len(chosen)} of "
        f"{len(items)} eligible questions"
        + (f" (section={section_filter})" if section_filter else "")
        + "\n"
    )

    # Sort by section so each section drains contiguously — Anthropic
    # prompt-cache prefix hot inside the latest arm (§V42).
    def _section_of(q: Question) -> str | None:
        subject = next(
            (
                s[len("Subject: ") :].strip()
                for s in (q.uworld_aamc_tags or [])
                if isinstance(s, str) and s.startswith("Subject: ")
            ),
            None,
        )
        return SUBJECT_TO_SECTION.get(subject) if subject else None

    chosen.sort(key=lambda pair: (_section_of(pair[0]) or "ZZ", pair[0].id))

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(q: Question, base_set: set[tuple[str, int]]) -> QuestionComparison:
        async with sem:
            (latest, tin, tout, cread, cost, err) = await _run_latest(
                q, client=client, lookup=lookup
            )
            return QuestionComparison(
                question_id=q.id,
                qid=q.qid,
                section=_section_of(q),
                baseline=base_set,
                latest=latest,
                input_tokens=tin,
                output_tokens=tout,
                cached_read_tokens=cread,
                cost=cost,
                error=err,
            )

    comparisons = await asyncio.gather(*[_bounded(q, base_set) for (q, base_set) in chosen])

    for cmp in sorted(comparisons, key=lambda c: (c.section or "ZZ", c.question_id)):
        _print_comparison(cmp, topic_path_by_id, cc_code_by_id)

    _print_aggregate(
        baseline_version, EXTRACTOR_VERSION, comparisons, topic_path_by_id, cc_code_by_id
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12, help="questions to sample (default 12)")
    parser.add_argument(
        "--baseline-version",
        default=DEFAULT_BASELINE_VERSION,
        help=(
            "extractor_version of the persisted baseline tags to diff against "
            f"(default {DEFAULT_BASELINE_VERSION!r})"
        ),
    )
    parser.add_argument(
        "--section",
        default=None,
        help="restrict comparison to one section (CP/CARS/BB/PS). Default: any.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="max in-flight Anthropic calls (default 3 — modest to share cache prefix)",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            n_questions=args.n,
            baseline_version=args.baseline_version,
            section_filter=args.section,
            concurrency=args.concurrency,
        )
    )
