"""CLI wrapper for the topic resolver Batches API path (SPEC §T51).

Three subcommands:

    submit          Build pending requests + submit batch + record row.
    status <id>     Poll Anthropic by llm_batch_runs.id, print + update.
    finalize <id>   Poll → stream results → write tags → update row.

Usage:
    cd backend
    uv run python -m scripts.run_anki_topic_resolver_batch submit
    uv run python -m scripts.run_anki_topic_resolver_batch status 1
    uv run python -m scripts.run_anki_topic_resolver_batch finalize 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from anthropic import AsyncAnthropic
from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.llm_batch import LlmBatchRun
from app.services.anki.topic_resolver_batch import (
    finalize_topic_resolver_batch,
    submit_topic_resolver_batch,
)
from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache
from app.services.llm.batch import get_batch_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


async def cmd_submit() -> None:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY, max_retries=5)
    cache = AnkiTopicResolverCache(settings.ANKI_TOPIC_RESOLVER_CACHE_PATH)
    try:
        async with AsyncSessionLocal() as session:
            row, build = await submit_topic_resolver_batch(session, client=client, cache=cache)
            await session.commit()
            print(
                f"submitted batch id={row.anthropic_batch_id} "
                f"run_id={row.id} items={row.total_requests} "
                f"skipped(cache_hit={build.skipped_cache_hits}, "
                f"empty_signal={build.skipped_empty_signal}, "
                f"no_candidates={build.skipped_no_candidates}, "
                f"duplicate={build.skipped_duplicate})"
            )
            print(
                f"poll with: uv run python -m scripts.run_anki_topic_resolver_batch status {row.id}"
            )
    finally:
        cache.close()


async def cmd_status(run_id: int) -> None:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    async with AsyncSessionLocal() as session:
        run = (
            await session.execute(select(LlmBatchRun).where(LlmBatchRun.id == run_id))
        ).scalar_one_or_none()
        if run is None:
            print(f"no llm_batch_runs row with id={run_id}")
            return
        batch = await get_batch_status(client, run.anthropic_batch_id)
        counts = batch.request_counts
        # Refresh DB row in-place.
        run.processing_status = batch.processing_status
        run.succeeded_count = int(getattr(counts, "succeeded", 0) or 0)
        run.errored_count = int(getattr(counts, "errored", 0) or 0)
        run.canceled_count = int(getattr(counts, "canceled", 0) or 0)
        run.expired_count = int(getattr(counts, "expired", 0) or 0)
        run.processing_count = int(getattr(counts, "processing", 0) or 0)
        if batch.ended_at is not None:
            run.ended_at = batch.ended_at
        await session.commit()
        print(
            f"batch={run.anthropic_batch_id} status={batch.processing_status} "
            f"counts: processing={run.processing_count} succeeded={run.succeeded_count} "
            f"errored={run.errored_count} canceled={run.canceled_count} "
            f"expired={run.expired_count}"
        )
        if batch.processing_status in {"ended", "canceled", "expired"}:
            print(
                f"batch terminal — finalize with: "
                f"uv run python -m scripts.run_anki_topic_resolver_batch finalize {run.id}"
            )


async def cmd_finalize(run_id: int) -> None:
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY, max_retries=5)
    cache = AnkiTopicResolverCache(settings.ANKI_TOPIC_RESOLVER_CACHE_PATH)
    try:
        async with AsyncSessionLocal() as session:
            persist = await finalize_topic_resolver_batch(
                session, run_id=run_id, client=client, cache=cache
            )
            await session.commit()
            print(
                f"finalized run_id={run_id}: "
                f"succeeded={persist.succeeded} errored={persist.errored} "
                f"canceled={persist.canceled} expired={persist.expired} "
                f"persisted_rows={persist.persisted_rows} "
                f"low_conf={persist.skipped_low_confidence} "
                f"declined={persist.declined} "
                f"unresolved={persist.unresolved_paths} "
                f"cost=${persist.total_cost_usd:.4f}"
            )
    finally:
        cache.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("submit")
    s_status = sub.add_parser("status")
    s_status.add_argument("run_id", type=int)
    s_final = sub.add_parser("finalize")
    s_final.add_argument("run_id", type=int)
    args = parser.parse_args()

    if args.cmd == "submit":
        asyncio.run(cmd_submit())
    elif args.cmd == "status":
        asyncio.run(cmd_status(args.run_id))
    elif args.cmd == "finalize":
        asyncio.run(cmd_finalize(args.run_id))


if __name__ == "__main__":
    main()
