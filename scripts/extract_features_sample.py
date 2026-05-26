"""Ticket 4.2 smoke run: extract features for a small sample of live questions.

NOT the batch worker (that's Ticket 4.3). Picks a varied sample (discrete +
passage-based + at least one with images), runs the extractor, persists rows
to question_features, prints a cost summary and per-question feature dump.

Usage:
    python -m scripts.extract_features_sample
    python -m scripts.extract_features_sample --limit 10
    python -m scripts.extract_features_sample --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import asdict
from typing import Any

from anthropic import AsyncAnthropic
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.captures import Question
from app.services.analyzer import extract_features_for_question
from app.services.analyzer.cache import FeatureExtractorCache

logger = logging.getLogger("extract_features_sample")


async def _pick_sample(session, limit: int) -> list[int]:
    """Pick up to `limit` question_ids, prioritizing variety.

    Strategy: take a handful of discrete + a handful of passage-based, plus
    anything with an <img tag. Order doesn't matter — the extractor runs
    serially.
    """
    by_passage = (
        (
            await session.execute(
                select(Question.id)
                .where(Question.passage_id.is_not(None))
                .order_by(Question.id)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    by_discrete = (
        (
            await session.execute(
                select(Question.id)
                .where(Question.passage_id.is_(None))
                .order_by(Question.id)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    ids: list[int] = []
    # Interleave so we hit both kinds early.
    for a, b in zip(by_discrete, by_passage):
        ids.append(a)
        ids.append(b)
        if len(ids) >= limit:
            break
    for extra in by_discrete[len(ids) // 2 :] + by_passage[len(ids) // 2 :]:
        if len(ids) >= limit:
            break
        if extra not in ids:
            ids.append(extra)
    return ids[:limit]


def _dump_features(qid: str, result) -> dict[str, Any]:
    if not result.persisted:
        return {
            "qid": qid,
            "persisted": False,
            "skipped_reason": result.skipped_reason,
        }
    mech = asdict(result.mechanical) if result.mechanical else None
    feats = asdict(result.features) if result.features else None
    return {
        "qid": qid,
        "persisted": True,
        "mechanical": mech,
        "judgment": feats,
        "cache_hit": result.cache_hit,
        "cost_estimate_usd": round(result.cost_estimate_usd, 6),
        "cost_saved_usd": round(result.cost_saved_usd, 6),
    }


async def run(*, limit: int, dry_run: bool) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    cache = FeatureExtractorCache(settings.FEATURE_EXTRACTOR_CACHE_PATH)

    total_cost = 0.0
    total_saved = 0.0
    persisted = 0
    skipped = 0
    dumps: list[dict[str, Any]] = []

    try:
        async with AsyncSessionLocal() as session:
            ids = await _pick_sample(session, limit)
            logger.info("sample question_ids: %s", ids)

            for qid in ids:
                # Resolve qid (text) via a small lookup for human-readable logging.
                q = (
                    await session.execute(
                        select(Question)
                        .options(selectinload(Question.passage))
                        .where(Question.id == qid)
                    )
                ).scalar_one()

                result = await extract_features_for_question(
                    qid,
                    session,
                    anthropic_client=anthropic_client,
                    cache=cache,
                )

                if result.persisted:
                    persisted += 1
                    total_cost += result.cost_estimate_usd
                    total_saved += result.cost_saved_usd
                else:
                    skipped += 1
                dumps.append(_dump_features(q.qid, result))

            if dry_run:
                logger.info("dry-run: rolling back")
                await session.rollback()
            else:
                await session.commit()

    finally:
        cache.close()

    print("\n===== SAMPLE FEATURE DUMPS =====")
    for d in dumps:
        print()
        for k, v in d.items():
            print(f"  {k}: {v}")

    print("\n===== COST SUMMARY =====")
    print(f"  questions_persisted:   {persisted}")
    print(f"  questions_skipped:     {skipped}")
    print(f"  total_cost_usd:        ${total_cost:.4f}")
    print(f"  total_saved_usd:       ${total_saved:.4f}")
    print(f"  model:                 {settings.FEATURE_EXTRACTOR_MODEL}")
    print(f"  cache_path:            {settings.FEATURE_EXTRACTOR_CACHE_PATH}")
    print(f"  dry_run:               {dry_run}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Number of questions to extract features for (default 6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction and print results but roll back the DB transaction.",
    )
    args = parser.parse_args()
    return asyncio.run(run(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
