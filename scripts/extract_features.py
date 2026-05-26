"""Drain unextracted / stale-version questions through the feature extractor.

CLI:
    uv run python -m backend.scripts.extract_features
    uv run python -m backend.scripts.extract_features --missed-only
    uv run python -m backend.scripts.extract_features --since 2026-05-01
    uv run python -m backend.scripts.extract_features --limit 20
    uv run python -m backend.scripts.extract_features --max-cost-usd 0.50
    uv run python -m backend.scripts.extract_features --concurrency 5
    uv run python -m backend.scripts.extract_features --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from anthropic import AsyncAnthropic

from app.config import settings
from app.database import AsyncSessionLocal
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.worker import ExtractionSummary, run_extraction


async def _main(
    *,
    missed_only: bool,
    since: date | None,
    limit: int | None,
    max_cost_usd: float | None,
    concurrency: int,
    dry_run: bool,
) -> ExtractionSummary:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    cache = FeatureExtractorCache(settings.FEATURE_EXTRACTOR_CACHE_PATH)
    try:
        summary = await run_extraction(
            AsyncSessionLocal,
            anthropic_client=client,
            cache=cache,
            missed_only=missed_only,
            since=since,
            limit=limit,
            max_cost_usd=max_cost_usd,
            concurrency=concurrency,
            dry_run=dry_run,
        )
    finally:
        cache.close()
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--missed-only",
        action="store_true",
        help="Only process questions with at least one incorrect attempt.",
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only process questions first seen on or after this date.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Hard cap on number of questions to process.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Stop once accumulated LLM cost reaches this cap (USD).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent LLM calls (default 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract features and log results but do not commit to DB.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = asyncio.run(
        _main(
            missed_only=args.missed_only,
            since=args.since,
            limit=args.limit,
            max_cost_usd=args.max_cost_usd,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
    )
    print(summary.as_text())
    if summary.failed > 0:
        print(f"WARNING: {summary.failed} question(s) failed both extraction attempts.")


if __name__ == "__main__":
    _cli()
