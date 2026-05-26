"""APScheduler background jobs for categorizer and feature extraction.

Jobs write TaskRun rows to Postgres for history/monitoring. An in-flight
guard prevents concurrent runs of the same job — APScheduler's interval
trigger can fire a second instance if the first run exceeds the interval.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings
from app.services.llm.client import build_openai_client
from app.database import AsyncSessionLocal
from app.models.task_run import TaskRun, TaskRunStatus
from app.services.anki.assignment import run_complete_unlocked, run_unlock_due
from app.services.anki.review import run_review_due
from app.services.anki.client import AnkiConnectClient
from app.services.anki.sync import sync_deck
from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache
from app.services.anki.topic_resolver_worker import (
    make_summary_text as make_anki_topic_summary_text,
    run as run_anki_topic_resolver,
)
from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.outline_lookup import OutlineLookup
from app.services.categorizer.worker import run as run_categorizer
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.worker import run_extraction

logger = logging.getLogger(__name__)

_inflight: set[str] = set()
_lock = asyncio.Lock()

scheduler = AsyncIOScheduler()


# --------------------------------------------------------------------------- #
# Categorizer job
# --------------------------------------------------------------------------- #


async def _do_run_categorizer() -> None:
    # V41: max_retries=5 absorbs transient OpenAI errors at the SDK layer.
    client = build_openai_client(max_retries=5)
    cache = CategorizerCache(settings.CATEGORIZER_CACHE_PATH)
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_categorizer",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        async with AsyncSessionLocal() as session:
            lookup = await OutlineLookup.load(session)
            summary = await run_categorizer(
                session,
                openai_client=client,
                cache=cache,
                max_cost_usd=settings.CATEGORIZER_PER_RUN_BUDGET_USD,
                lookup=lookup,
            )
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.processed
                run_row.cost_usd = summary.total_cost_usd
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info("run_categorizer finished: %s", summary.as_text())
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_categorizer failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record categorizer failure in task_runs")
    finally:
        cache.close()


async def run_categorizer_job() -> None:
    async with _lock:
        if "run_categorizer" in _inflight:
            logger.warning("run_categorizer already in-flight, skipping")
            return
        _inflight.add("run_categorizer")
    try:
        await _do_run_categorizer()
    finally:
        async with _lock:
            _inflight.discard("run_categorizer")


# --------------------------------------------------------------------------- #
# Feature extraction job
# --------------------------------------------------------------------------- #


async def _do_run_feature_extraction() -> None:
    client = build_openai_client(max_retries=5)
    cache = FeatureExtractorCache(settings.FEATURE_EXTRACTOR_CACHE_PATH)
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_feature_extraction",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        summary = await run_extraction(
            AsyncSessionLocal,
            openai_client=client,
            cache=cache,
            missed_only=False,
        )

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.processed
                run_row.cost_usd = summary.total_cost_usd
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info("run_feature_extraction finished: processed=%d", summary.processed)
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_feature_extraction failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record feature_extraction failure in task_runs")
    finally:
        cache.close()


async def run_feature_extraction_job() -> None:
    async with _lock:
        if "run_feature_extraction" in _inflight:
            logger.warning("run_feature_extraction already in-flight, skipping")
            return
        _inflight.add("run_feature_extraction")
    try:
        await _do_run_feature_extraction()
    finally:
        async with _lock:
            _inflight.discard("run_feature_extraction")


# --------------------------------------------------------------------------- #
# Anki sync job (SPEC §T4)
# --------------------------------------------------------------------------- #


async def _do_run_anki_sync() -> None:
    client = AnkiConnectClient(settings.ANKICONNECT_URL)
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_anki_sync",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        async with AsyncSessionLocal() as session:
            summary = await sync_deck(session, client, deck_name=settings.ANKI_DECK_NAME)
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                # AnkiConnect unreachable is not a job failure — V4 says the
                # sync just no-ops and reports "anki_not_running". Distinguish
                # via error_text so /admin can show a hint without alerting.
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.synced_cards
                run_row.error_text = summary.error
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "run_anki_sync finished: synced=%d linked_qids=%d reviews=%d error=%s",
            summary.synced_cards,
            summary.linked_qids,
            summary.reviews_synced,
            summary.error,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_anki_sync failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record anki_sync failure in task_runs")
    finally:
        await client.aclose()


async def run_anki_sync_job() -> None:
    async with _lock:
        if "run_anki_sync" in _inflight:
            logger.warning("run_anki_sync already in-flight, skipping")
            return
        _inflight.add("run_anki_sync")
    try:
        await _do_run_anki_sync()
    finally:
        async with _lock:
            _inflight.discard("run_anki_sync")


# --------------------------------------------------------------------------- #
# Anki topic resolver job (SPEC §T32)
# --------------------------------------------------------------------------- #


async def _do_run_anki_topic_resolver() -> None:
    # V41 (amended): max_retries≥5 absorbs transient OpenAI errors at the SDK
    # boundary (429 rate-limit, 5xx, transport blips) before reaching the
    # worker's per-card try/except.
    client = build_openai_client(max_retries=5)
    cache = AnkiTopicResolverCache(settings.ANKI_TOPIC_RESOLVER_CACHE_PATH)
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_anki_topic_resolver",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        async with AsyncSessionLocal() as session:
            summary = await run_anki_topic_resolver(
                session,
                openai_client=client,
                cache=cache,
                max_cost_usd=settings.ANKI_TOPIC_RESOLVER_PER_RUN_BUDGET_USD,
            )
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.persisted
                run_row.cost_usd = summary.total_cost_usd
                run_row.finished_at = datetime.now(timezone.utc)
                # §V41: surface partial-failure so /admin run history shows it.
                if summary.partial_failure and summary.error:
                    run_row.error_text = (
                        f"partial: broke on transient API error after "
                        f"{summary.processed} cards — {summary.error}"
                    )[:2000]
            await session.commit()

        logger.info("run_anki_topic_resolver finished: %s", make_anki_topic_summary_text(summary))
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_anki_topic_resolver failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record anki_topic_resolver failure in task_runs")
    finally:
        cache.close()


async def run_anki_topic_resolver_job() -> None:
    async with _lock:
        if "run_anki_topic_resolver" in _inflight:
            logger.warning("run_anki_topic_resolver already in-flight, skipping")
            return
        _inflight.add("run_anki_topic_resolver")
    try:
        await _do_run_anki_topic_resolver()
    finally:
        async with _lock:
            _inflight.discard("run_anki_topic_resolver")


# --------------------------------------------------------------------------- #
# Anki assignment unlock job (SPEC §T63, V51 + V55)
# --------------------------------------------------------------------------- #


async def _do_run_anki_assignment_unlock() -> None:
    """Process every pending assignment whose `scheduled_unlock_at ≤ now`.

    Per V55 the AnkiConnect write is retried on the next tick on failure
    (unsuspend is idempotent); we record each attempt in `anki_writes` so
    /admin can show the retry trail. The TaskRun status is always
    `succeeded` once the loop reaches `commit()`, even if individual
    assignments failed — the per-assignment failure counter on
    `anki_assignments` is the real durability signal, not the TaskRun.
    """
    client = AnkiConnectClient(settings.ANKICONNECT_URL)
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_anki_assignment_unlock",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        async with AsyncSessionLocal() as session:
            summary = await run_unlock_due(session, client)

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.succeeded
                run_row.finished_at = datetime.now(timezone.utc)
                if summary.failed:
                    run_row.error_text = (
                        f"{summary.succeeded}/{summary.processed} unlocked; "
                        f"{summary.failed} failed ({summary.terminal_failed} terminal)"
                    )[:2000]
            await session.commit()

        logger.info(
            "run_anki_assignment_unlock processed=%d succeeded=%d failed=%d terminal=%d",
            summary.processed,
            summary.succeeded,
            summary.failed,
            summary.terminal_failed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_anki_assignment_unlock failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record anki_assignment_unlock failure in task_runs")
    finally:
        await client.aclose()


async def run_anki_assignment_unlock_job() -> None:
    async with _lock:
        if "run_anki_assignment_unlock" in _inflight:
            logger.warning("run_anki_assignment_unlock already in-flight, skipping")
            return
        _inflight.add("run_anki_assignment_unlock")
    try:
        await _do_run_anki_assignment_unlock()
    finally:
        async with _lock:
            _inflight.discard("run_anki_assignment_unlock")


# --------------------------------------------------------------------------- #
# Anki assignment auto-complete job (SPEC §T64, V51)
# --------------------------------------------------------------------------- #


async def _do_run_anki_assignment_complete() -> None:
    """Flip each `unlocked` assignment to `completed` once every card in
    its snapshot has at least one review after `actual_unlock_at` (V51).
    No AnkiConnect call — purely DB-driven from `anki_card_reviews`
    populated by the existing sync job (T36)."""
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_anki_assignment_complete",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        async with AsyncSessionLocal() as session:
            summary = await run_complete_unlocked(session)
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.completed
                run_row.finished_at = datetime.now(timezone.utc)
                if summary.still_unlocked:
                    run_row.error_text = (
                        f"{summary.completed}/{summary.processed} completed; "
                        f"{summary.still_unlocked} still unlocked"
                    )[:2000]
            await session.commit()

        logger.info(
            "run_anki_assignment_complete processed=%d completed=%d still_unlocked=%d",
            summary.processed,
            summary.completed,
            summary.still_unlocked,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_anki_assignment_complete failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record anki_assignment_complete failure in task_runs")


async def run_anki_assignment_complete_job() -> None:
    async with _lock:
        if "run_anki_assignment_complete" in _inflight:
            logger.warning("run_anki_assignment_complete already in-flight, skipping")
            return
        _inflight.add("run_anki_assignment_complete")
    try:
        await _do_run_anki_assignment_complete()
    finally:
        async with _lock:
            _inflight.discard("run_anki_assignment_complete")


# --------------------------------------------------------------------------- #
# Anki review job (SPEC §T76, V53 amended + V55)
# --------------------------------------------------------------------------- #


async def _do_run_anki_review() -> None:
    """Fire `createFilteredDeck` + chain `addTags` for every pending
    review whose review_date is on or before today. V55 retry-with-cap
    on createFilteredDeck failure; addTags failure ⊥ load-bearing
    (audit-only per V50). Per-row commit so transient mid-batch
    failures don't poison the rest of the batch."""
    client = AnkiConnectClient(settings.ANKICONNECT_URL)
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_anki_review",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        async with AsyncSessionLocal() as session:
            summary = await run_review_due(session, client)

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = summary.pushed
                run_row.finished_at = datetime.now(timezone.utc)
                if summary.failed:
                    run_row.error_text = (
                        f"{summary.pushed}/{summary.processed} pushed; "
                        f"{summary.failed} failed ({summary.terminal_failed} terminal)"
                    )[:2000]
            await session.commit()

        logger.info(
            "run_anki_review processed=%d pushed=%d failed=%d terminal=%d",
            summary.processed,
            summary.pushed,
            summary.failed,
            summary.terminal_failed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_anki_review failed: %s", exc)
        if run_id is not None:
            try:
                async with AsyncSessionLocal() as session:
                    run_row = await session.get(TaskRun, run_id)
                    if run_row is not None:
                        run_row.status = TaskRunStatus.failed
                        run_row.error_text = str(exc)[:2000]
                        run_row.finished_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception("failed to record anki_review failure in task_runs")
    finally:
        await client.aclose()


async def run_anki_review_job() -> None:
    async with _lock:
        if "run_anki_review" in _inflight:
            logger.warning("run_anki_review already in-flight, skipping")
            return
        _inflight.add("run_anki_review")
    try:
        await _do_run_anki_review()
    finally:
        async with _lock:
            _inflight.discard("run_anki_review")


# --------------------------------------------------------------------------- #
# Scheduler lifecycle
# --------------------------------------------------------------------------- #


def start_scheduler() -> None:
    if not settings.SCHEDULER_ENABLED:
        return
    scheduler.add_job(
        run_categorizer_job,
        "interval",
        minutes=settings.CATEGORIZER_INTERVAL_MINUTES,
        id="run_categorizer",
        replace_existing=True,
    )
    scheduler.add_job(
        run_feature_extraction_job,
        "interval",
        minutes=settings.FEATURE_EXTRACTION_INTERVAL_MINUTES,
        id="run_feature_extraction",
        replace_existing=True,
    )
    scheduler.add_job(
        run_anki_sync_job,
        "interval",
        minutes=settings.ANKI_SYNC_INTERVAL_MINUTES,
        id="run_anki_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_anki_topic_resolver_job,
        "interval",
        minutes=settings.ANKI_TOPIC_RESOLVER_INTERVAL_MINUTES,
        id="run_anki_topic_resolver",
        replace_existing=True,
    )
    scheduler.add_job(
        run_anki_assignment_unlock_job,
        "interval",
        minutes=settings.ANKI_ASSIGNMENT_UNLOCK_INTERVAL_MINUTES,
        id="run_anki_assignment_unlock",
        replace_existing=True,
    )
    scheduler.add_job(
        run_anki_assignment_complete_job,
        "cron",
        hour=settings.ANKI_ASSIGNMENT_COMPLETE_CRON_HOUR,
        minute=settings.ANKI_ASSIGNMENT_COMPLETE_CRON_MINUTE,
        id="run_anki_assignment_complete",
        replace_existing=True,
    )
    scheduler.add_job(
        run_anki_review_job,
        "interval",
        minutes=settings.ANKI_REVIEW_PUSH_INTERVAL_MINUTES,
        id="run_anki_review",
        replace_existing=True,
    )
    scheduler.start()


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
