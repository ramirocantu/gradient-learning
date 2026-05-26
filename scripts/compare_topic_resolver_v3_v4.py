"""Compare the persisted production resolver baseline against the latest
extractor's live picks on the same (card, CC) pairs.

Baseline = whatever is already in `anki_card_tags` w/ `source='llm'`,
`parsed_kind='aamc_topic'`, and `extractor_version` matching `--baseline-version`
(default `v5-tags-plus-text-multi-topic` — the pre-token-optimization production
state). No re-call needed; the rows are the historical record of what
production picked, paid for already.

Latest = the current `resolve_topic` (whatever `EXTRACTOR_VERSION` is in the
source tree right now — v6 → v7 → v8 → ...). Called live against Anthropic
w/ `cache=None` so we always hit the API and get true per-call token + cost
numbers. The current cache is not touched — keys are namespaced by
`extractor_version` so a baseline-version test run can't contaminate
production cache rows for any other version.

Outputs per card:
  - baseline picks (topic_path, confidence, rationale) sourced from DB
  - latest picks (live API call) + token + cost report
  - set diff (identical / common / only-baseline / only-latest)

Aggregate at the end:
  - exact set-equality rate
  - mean jaccard
  - net new picks vs net dropped picks
  - per-card $ for the latest arm (baseline is sunk cost)

Use to validate that prompt/schema edits don't degrade pick quality BEFORE
bumping `EXTRACTOR_VERSION` and re-extracting the deck.

Usage:
    cd backend
    uv run python -m scripts.compare_topic_resolver_v3_v4 \
        [--n=10] [--baseline-version=v5-tags-plus-text-multi-topic] [--cc=3B]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from random import sample

from anthropic import AsyncAnthropic
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.anki import AnkiCard, AnkiCardTag
from app.models.outline import ContentCategory
from app.services.anki.topic_resolver import (
    EXTRACTOR_VERSION,
    TopicPick,
    resolve_topic,
)
from app.services.anki.topic_resolver_worker import _card_text, _filter_anking_tags

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_VERSION = "v5-tags-plus-text-multi-topic"

# Haiku 4.5 pricing for one-shot cost display (matches topic_resolver._PRICING).
_PRICE_IN_PER_M = 1.00
_PRICE_OUT_PER_M = 5.00


@dataclass
class BaselineRow:
    topic_path: str
    confidence: float
    rationale: str


def _parse_topic_path_from_synthetic_raw(tag_raw: str) -> str | None:
    """Worker stores `tag_raw = f"__llm_topic__::{extractor_version}::{topic_path}"`.

    Recover the topic_path from that synthetic. Returns None on malformed input
    (defensive — shouldn't happen on rows written by the worker).
    """
    prefix = "__llm_topic__::"
    if not tag_raw.startswith(prefix):
        return None
    rest = tag_raw[len(prefix) :]
    sep = rest.find("::")
    if sep < 0:
        return None
    return rest[sep + 2 :]


@dataclass
class CardComparison:
    anki_card_id: int
    cc_code: str
    baseline: list[BaselineRow]
    latest: list[TopicPick]
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int
    error: str | None = None

    @property
    def latest_cost_usd(self) -> float:
        return (
            self.input_tokens * _PRICE_IN_PER_M + self.output_tokens * _PRICE_OUT_PER_M
        ) / 1_000_000

    @property
    def baseline_paths(self) -> set[str]:
        return {p.topic_path for p in self.baseline}

    @property
    def latest_paths(self) -> set[str]:
        return {p.topic_path for p in self.latest}

    @property
    def jaccard(self) -> float:
        a, b = self.baseline_paths, self.latest_paths
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 1.0
        return len(a & b) / len(union)


async def _gather_candidates(
    session,
    *,
    baseline_version: str,
    cc_filter: str | None,
) -> list[tuple[AnkiCard, str, list[BaselineRow]]]:
    """Pick cards that have at least one persisted aamc_topic row at the
    baseline version. Returns (card, cc_code, baseline_rows[]) tuples.
    """
    cc_tag = AnkiCardTag.__table__.alias("cc_tag")
    llm_tag = AnkiCardTag.__table__.alias("llm_tag")
    cc = ContentCategory.__table__.alias("cc")

    stmt = (
        select(AnkiCard, cc.c.code, llm_tag.c.tag_raw, llm_tag.c.confidence, llm_tag.c.rationale)
        .join(cc_tag, cc_tag.c.anki_card_id == AnkiCard.id)
        .join(cc, cc.c.id == cc_tag.c.content_category_id)
        .join(
            llm_tag,
            (llm_tag.c.anki_card_id == AnkiCard.id)
            & (llm_tag.c.content_category_id == cc_tag.c.content_category_id)
            & (llm_tag.c.source == "llm")
            & (llm_tag.c.parsed_kind == "aamc_topic")
            & (llm_tag.c.extractor_version == baseline_version),
        )
        .where(cc_tag.c.parsed_kind == "aamc_cc")
        .options(selectinload(AnkiCard.tags))
    )
    if cc_filter is not None:
        stmt = stmt.where(cc.c.code == cc_filter)

    rows = (await session.execute(stmt)).all()

    # Group rows by (card, cc_code) so we collect all baseline picks per pair.
    grouped: dict[tuple[int, str], tuple[AnkiCard, list[BaselineRow]]] = {}
    for card, cc_code, tag_raw, confidence, rationale in rows:
        topic_path = _parse_topic_path_from_synthetic_raw(tag_raw or "")
        if topic_path is None:
            continue
        key = (card.id, cc_code)
        if key not in grouped:
            grouped[key] = (card, [])
        grouped[key][1].append(
            BaselineRow(
                topic_path=topic_path,
                confidence=float(confidence or 0.0),
                rationale=str(rationale or ""),
            )
        )

    return [(card, cc_code, picks) for (_, cc_code), (card, picks) in grouped.items()]


async def _run_latest(card: AnkiCard, cc_code: str, client: AsyncAnthropic) -> CardComparison:
    filtered_tags = _filter_anking_tags(card)
    card_text = _card_text(card)
    try:
        result = await resolve_topic(
            filtered_tags=filtered_tags,
            card_text=card_text,
            cc_code=cc_code,
            anthropic_client=client,
            cache=None,  # always hit the API for a clean live measurement
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic script, want full error
        return CardComparison(
            anki_card_id=card.anki_card_id,
            cc_code=cc_code,
            baseline=[],
            latest=[],
            input_tokens=0,
            output_tokens=0,
            cached_read_tokens=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    return CardComparison(
        anki_card_id=card.anki_card_id,
        cc_code=cc_code,
        baseline=[],
        latest=list(result.picks),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_read_tokens=0,
    )


def _format_baseline(picks: list[BaselineRow]) -> str:
    if not picks:
        return "    (no baseline picks — declined or below threshold)"
    return "\n".join(
        f"    {p.topic_path!r}  (conf={p.confidence:.2f})\n      {p.rationale!r}" for p in picks
    )


def _format_latest(picks: list[TopicPick]) -> str:
    if not picks:
        return "    (declined or no picks)"
    return "\n".join(
        f"    {p.topic_path!r}  (conf={p.confidence:.2f})\n      {p.rationale!r}" for p in picks
    )


def _print_comparison(cmp: CardComparison) -> None:
    print("─" * 88)
    print(f"anki_card_id={cmp.anki_card_id}  CC={cmp.cc_code}")
    print(f"  baseline ({len(cmp.baseline)} picks):")
    print(_format_baseline(cmp.baseline))
    print(f"  latest ({len(cmp.latest)} picks):")
    if cmp.error:
        print(f"    ERROR: {cmp.error}")
    print(_format_latest(cmp.latest))
    if not cmp.error:
        print(
            f"    tokens: in={cmp.input_tokens} out={cmp.output_tokens}  "
            f"cost=${cmp.latest_cost_usd:.5f}"
        )

    bp, lp = cmp.baseline_paths, cmp.latest_paths
    common = bp & lp
    only_baseline = bp - lp
    only_latest = lp - bp
    if bp == lp and bp:
        print(f"  → IDENTICAL set: {sorted(bp)}")
    elif not bp and not lp:
        print("  → both empty")
    else:
        if common:
            print(f"  → COMMON: {sorted(common)}")
        if only_baseline:
            print(f"    only baseline: {sorted(only_baseline)}")
        if only_latest:
            print(f"    only latest:   {sorted(only_latest)}")
    print(f"  jaccard: {cmp.jaccard:.2f}")


def _print_aggregate(
    baseline_version: str,
    latest_version: str,
    comparisons: list[CardComparison],
) -> None:
    print()
    print("=" * 88)
    print(f"Aggregate over {len(comparisons)} cards")
    print(f"  baseline version = {baseline_version!r}")
    print(f"  latest version   = {latest_version!r}")

    errs = [c for c in comparisons if c.error]
    ok = [c for c in comparisons if not c.error]
    if errs:
        print(f"  errors: {len(errs)} card(s)")
        for c in errs:
            print(f"    anki_card_id={c.anki_card_id} cc={c.cc_code} err={c.error!r}")
    if not ok:
        print("  no successful latest-arm calls; nothing to aggregate.")
        return

    exact_eq = sum(1 for c in ok if c.baseline_paths == c.latest_paths)
    mean_jaccard = sum(c.jaccard for c in ok) / len(ok)
    total_baseline_picks = sum(len(c.baseline) for c in ok)
    total_latest_picks = sum(len(c.latest) for c in ok)
    added = sum(len(c.latest_paths - c.baseline_paths) for c in ok)
    dropped = sum(len(c.baseline_paths - c.latest_paths) for c in ok)
    total_cost = sum(c.latest_cost_usd for c in ok)
    total_in = sum(c.input_tokens for c in ok)
    total_out = sum(c.output_tokens for c in ok)

    cc_jaccard: dict[str, list[float]] = {}
    for c in ok:
        cc_jaccard.setdefault(c.cc_code, []).append(c.jaccard)

    print(f"  exact set-equality: {exact_eq}/{len(ok)} ({100 * exact_eq / len(ok):.0f}%)")
    print(f"  mean jaccard:       {mean_jaccard:.3f}")
    print(
        f"  picks: baseline={total_baseline_picks} latest={total_latest_picks} "
        f"(added={added}, dropped={dropped})"
    )
    print(
        f"  latest tokens: in={total_in} out={total_out}  total cost ≈ ${total_cost:.4f} "
        f"({total_cost / len(ok):.5f}/card)"
    )

    # CC-level breakdown for jaccard (only CCs w/ ≥3 cards in the sample).
    eligible = {k: v for k, v in cc_jaccard.items() if len(v) >= 3}
    if eligible:
        print("  per-CC mean jaccard (CCs w/ ≥3 sample cards):")
        for cc_code, vals in sorted(eligible.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"    {cc_code}: {sum(vals) / len(vals):.3f}  (n={len(vals)})")

    # Decline-flip stats — useful for catching regressions where a card had
    # baseline picks but latest declines (or vice versa).
    flipped_to_decline = sum(1 for c in ok if c.baseline and not c.latest)
    flipped_from_decline = sum(1 for c in ok if not c.baseline and c.latest)
    if flipped_to_decline or flipped_from_decline:
        print(
            f"  decline flips: "
            f"baseline→latest_declined={flipped_to_decline}, "
            f"latest_picked_up={flipped_from_decline}"
        )

    # Top topic_path delta per direction.
    added_paths = Counter()
    dropped_paths = Counter()
    for c in ok:
        added_paths.update(c.latest_paths - c.baseline_paths)
        dropped_paths.update(c.baseline_paths - c.latest_paths)
    if added_paths:
        print("  top latest-only picks:")
        for path, n in added_paths.most_common(5):
            print(f"    +{n}  {path}")
    if dropped_paths:
        print("  top baseline-only picks (latest dropped):")
        for path, n in dropped_paths.most_common(5):
            print(f"    -{n}  {path}")


async def main(
    n_cards: int, baseline_version: str, cc_filter: str | None, concurrency: int
) -> None:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY, max_retries=5)
    async with AsyncSessionLocal() as session:
        candidates = await _gather_candidates(
            session, baseline_version=baseline_version, cc_filter=cc_filter
        )

    if not candidates:
        print(
            f"No candidate (card, cc) pairs found w/ persisted baseline "
            f"extractor_version={baseline_version!r}"
            + (f" and cc={cc_filter!r}" if cc_filter else "")
            + ". Has any production drain run for that version?"
        )
        return

    chosen = sample(candidates, min(n_cards, len(candidates)))
    print(
        f"\nComparing baseline {baseline_version!r} (persisted) vs latest "
        f"{EXTRACTOR_VERSION!r} (live, no cache) on {len(chosen)} of "
        f"{len(candidates)} eligible cards" + (f" (cc={cc_filter})" if cc_filter else "") + "\n"
    )

    # CC-sorted dispatch keeps Anthropic's prompt-cache prefix hot inside the
    # latest arm too — mirrors the worker's §V42 ordering.
    chosen.sort(key=lambda r: r[1])

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(card: AnkiCard, cc_code: str, baseline: list[BaselineRow]):
        async with sem:
            cmp = await _run_latest(card, cc_code, client)
            cmp.baseline = baseline
            return cmp

    comparisons = await asyncio.gather(
        *[_bounded(card, cc_code, baseline) for card, cc_code, baseline in chosen]
    )

    # Print in the same CC-sorted order we dispatched.
    for cmp in sorted(comparisons, key=lambda c: (c.cc_code, c.anki_card_id)):
        _print_comparison(cmp)

    _print_aggregate(baseline_version, EXTRACTOR_VERSION, comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="number of cards to sample (default 10)")
    parser.add_argument(
        "--baseline-version",
        default=DEFAULT_BASELINE_VERSION,
        help=(
            "extractor_version of the persisted baseline picks to diff against "
            f"(default {DEFAULT_BASELINE_VERSION!r})"
        ),
    )
    parser.add_argument(
        "--cc",
        default=None,
        help="restrict comparison to one CC code (e.g. 3B). Default: any CC.",
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
            n_cards=args.n,
            baseline_version=args.baseline_version,
            cc_filter=args.cc,
            concurrency=args.concurrency,
        )
    )
