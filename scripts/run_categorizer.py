"""Drain the needs_categorization=true queue.

Runs the LLM categorizer over every pending question and persists
QuestionTag rows via app.services.categorizer.tag_question. The
persistent SQLite-backed CategorizerCache makes re-runs of the same
content effectively free across process restarts.

CLI:
    python -m scripts.run_categorizer
    python -m scripts.run_categorizer --batch-size 50
    python -m scripts.run_categorizer --max-cost-usd 5.00
    python -m scripts.run_categorizer --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from anthropic import AsyncAnthropic

from app.config import settings
from app.database import AsyncSessionLocal
from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.worker import WorkerSummary, run
from app.startup import ensure_outline_seeded

logger = logging.getLogger(__name__)


async def _main(
    *,
    batch_size: int,
    dry_run: bool,
    max_cost_usd: float | None,
) -> WorkerSummary:
    await ensure_outline_seeded()
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    cache = CategorizerCache(settings.CATEGORIZER_CACHE_PATH)
    try:
        async with AsyncSessionLocal() as session:
            try:
                summary = await run(
                    session,
                    anthropic_client=client,
                    batch_size=batch_size,
                    dry_run=dry_run,
                    cache=cache,
                    max_cost_usd=max_cost_usd,
                )
                if not dry_run:
                    await session.commit()
                return summary
            except Exception:
                await session.rollback()
                raise
    finally:
        cache.close()


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="Stop draining once accumulated LLM cost reaches this cap (USD).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = asyncio.run(
        _main(
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            max_cost_usd=args.max_cost_usd,
        )
    )
    print(f"SUMMARY: {summary.as_text()}")
    if summary.failure_qids:
        print(f"FAILURES: {summary.failure_qids}")


if __name__ == "__main__":
    _cli()
