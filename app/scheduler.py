"""APScheduler background jobs for Anki sync, assignment, and review.

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
from app.database import AsyncSessionLocal
from app.models.task_run import TaskRun, TaskRunStatus
from app.services.anki.assignment import run_complete_unlocked, run_unlock_due
from app.services.anki.review import run_review_due
from app.services.anki.client import AnkiConnectClient
from app.services.anki.sync import sync_deck
from app.services.kb.inbox import poll_inbox
from app.services.kb.jobs import embed_pending, tag_pending
from app.services.kb.notion import sync_pending_nodes
from app.services.llm.client import build_openai_client

logger = logging.getLogger(__name__)

_inflight: set[str] = set()
_lock = asyncio.Lock()

scheduler = AsyncIOScheduler()


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
# PDF inbox ingest job (T51, V-KB1, V41)
# --------------------------------------------------------------------------- #


async def _do_run_pdf_ingest() -> None:
    """Poll ``PDF_INBOX_DIR/<slug>/*.pdf`` → vision-ingest each (V-KB1).

    Skips (no-op, ``succeeded``) when no OpenAI key is configured — ingest
    needs the vision + extraction calls. V41: ``poll_inbox`` isolates per-file
    failures into ``report.failures`` so the run still reaches ``commit()`` and
    is marked ``succeeded`` (partial), resuming next tick via the SHA filter.
    """
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_pdf_ingest",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        if not settings.OPENAI_API_KEY:
            async with AsyncSessionLocal() as session:
                run_row = await session.get(TaskRun, run_id)
                if run_row is not None:
                    run_row.status = TaskRunStatus.succeeded
                    run_row.error_text = "openai unconfigured — pdf ingest skipped"
                    run_row.finished_at = datetime.now(timezone.utc)
                await session.commit()
            logger.info("run_pdf_ingest: OPENAI_API_KEY unset; skipping")
            return

        client = build_openai_client()
        async with AsyncSessionLocal() as session:
            report = await poll_inbox(
                session,
                vision_client=client,
                extract_client=client,
                inbox_dir=settings.PDF_INBOX_DIR,
            )
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = report.new_facts
                if report.failures or report.files_skipped:
                    run_row.error_text = (
                        f"ingested={report.files_ingested} reused={report.files_reused} "
                        f"skipped={report.files_skipped} failures={len(report.failures)}"
                    )[:2000]
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "run_pdf_ingest finished: seen=%d ingested=%d reused=%d skipped=%d "
            "new_facts=%d failures=%d",
            report.files_seen,
            report.files_ingested,
            report.files_reused,
            report.files_skipped,
            report.new_facts,
            len(report.failures),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pdf_ingest failed: %s", exc)
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
                logger.exception("failed to record pdf_ingest failure in task_runs")


async def run_pdf_ingest_job() -> None:
    async with _lock:
        if "run_pdf_ingest" in _inflight:
            logger.warning("run_pdf_ingest already in-flight, skipping")
            return
        _inflight.add("run_pdf_ingest")
    try:
        await _do_run_pdf_ingest()
    finally:
        async with _lock:
            _inflight.discard("run_pdf_ingest")


# --------------------------------------------------------------------------- #
# Notion write-out job (T51, V-N1, V-N2, V41)
# --------------------------------------------------------------------------- #


async def _do_run_notion_sync() -> None:
    """One-way mirror tagged atomic facts → Notion (V-N1, V-N2).

    Skips (no-op, ``succeeded``) when Notion is unconfigured. Empty until the
    categorizer (T50) sets ``atomic_facts.node_id`` — only tagged facts have a
    node page to live on. V41: per-node failures isolated; run still
    ``succeeded`` (partial).
    """
    run_id: int | None = None
    notion_client = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_notion_sync",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        if not settings.NOTION_API_TOKEN or not settings.NOTION_WIKI_DB_ID:
            async with AsyncSessionLocal() as session:
                run_row = await session.get(TaskRun, run_id)
                if run_row is not None:
                    run_row.status = TaskRunStatus.succeeded
                    run_row.error_text = "notion unconfigured — sync skipped"
                    run_row.finished_at = datetime.now(timezone.utc)
                await session.commit()
            logger.info("run_notion_sync: NOTION_API_TOKEN/WIKI_DB_ID unset; skipping")
            return

        from notion_client import AsyncClient

        notion_client = AsyncClient(auth=settings.NOTION_API_TOKEN)
        async with AsyncSessionLocal() as session:
            report = await sync_pending_nodes(
                session,
                notion_client=notion_client,
                notion_wiki_db_id=settings.NOTION_WIKI_DB_ID,
            )
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = report.nodes_synced
                if report.failures:
                    run_row.error_text = (
                        f"nodes={report.nodes_synced} pages_created={report.pages_created} "
                        f"blocks={report.blocks_appended} failures={len(report.failures)}"
                    )[:2000]
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "run_notion_sync finished: nodes=%d pages_created=%d blocks=%d failures=%d",
            report.nodes_synced,
            report.pages_created,
            report.blocks_appended,
            len(report.failures),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_notion_sync failed: %s", exc)
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
                logger.exception("failed to record notion_sync failure in task_runs")
    finally:
        if notion_client is not None:
            try:
                await notion_client.aclose()
            except Exception:  # noqa: BLE001
                pass


async def run_notion_sync_job() -> None:
    async with _lock:
        if "run_notion_sync" in _inflight:
            logger.warning("run_notion_sync already in-flight, skipping")
            return
        _inflight.add("run_notion_sync")
    try:
        await _do_run_notion_sync()
    finally:
        async with _lock:
            _inflight.discard("run_notion_sync")


# --------------------------------------------------------------------------- #
# Embedding job (T50, V-E1, V41)
# --------------------------------------------------------------------------- #


async def _do_run_embed() -> None:
    """Embed un-embedded outline_nodes / atomic_facts / questions (V-E1).

    No-op (``succeeded``) without an OpenAI key. V41: ``embed_pending``
    isolates per-item failures so the run still commits + succeeds (partial).
    Outline-node vectors are the recall candidate index, so this gates
    tagging — it runs on a slightly tighter interval than the tag job.
    """
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_embed",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        if not settings.OPENAI_API_KEY:
            async with AsyncSessionLocal() as session:
                run_row = await session.get(TaskRun, run_id)
                if run_row is not None:
                    run_row.status = TaskRunStatus.succeeded
                    run_row.error_text = "openai unconfigured — embed skipped"
                    run_row.finished_at = datetime.now(timezone.utc)
                await session.commit()
            logger.info("run_embed: OPENAI_API_KEY unset; skipping")
            return

        client = build_openai_client()
        async with AsyncSessionLocal() as session:
            report = await embed_pending(session, openai_client=client)
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = report.embedded
                if report.failures:
                    run_row.error_text = (
                        f"embedded={report.embedded} reused={report.reused} "
                        f"failures={len(report.failures)}"
                    )[:2000]
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "run_embed finished: embedded=%d reused=%d failures=%d",
            report.embedded,
            report.reused,
            len(report.failures),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_embed failed: %s", exc)
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
                logger.exception("failed to record embed failure in task_runs")


async def run_embed_job() -> None:
    async with _lock:
        if "run_embed" in _inflight:
            logger.warning("run_embed already in-flight, skipping")
            return
        _inflight.add("run_embed")
    try:
        await _do_run_embed()
    finally:
        async with _lock:
            _inflight.discard("run_embed")


# --------------------------------------------------------------------------- #
# Grounded-tag job — the categorizer (T50, V-L3, V69, V-T2/V-T3, V41)
# --------------------------------------------------------------------------- #


async def _do_run_grounded_tag() -> None:
    """Categorize untagged atomic_facts + single-course questions (V-L3).

    Recall → grounded pick → inline V69 calibration → persist
    (``atomic_facts.node_id`` denormalized for the notion + read paths). No-op
    (``succeeded``) without an OpenAI key; empty until embeddings exist
    (recall returns no candidates → grounded no-ops). V41: per-entity failures
    isolated; run still succeeds (partial).
    """
    run_id: int | None = None
    try:
        async with AsyncSessionLocal() as session:
            row = TaskRun(
                job_name="run_grounded_tag",
                started_at=datetime.now(timezone.utc),
                status=TaskRunStatus.running,
                items_processed=0,
            )
            session.add(row)
            await session.flush()
            run_id = row.id
            await session.commit()

        if not settings.OPENAI_API_KEY:
            async with AsyncSessionLocal() as session:
                run_row = await session.get(TaskRun, run_id)
                if run_row is not None:
                    run_row.status = TaskRunStatus.succeeded
                    run_row.error_text = "openai unconfigured — grounded tag skipped"
                    run_row.finished_at = datetime.now(timezone.utc)
                await session.commit()
            logger.info("run_grounded_tag: OPENAI_API_KEY unset; skipping")
            return

        client = build_openai_client()
        async with AsyncSessionLocal() as session:
            report = await tag_pending(
                session, tagging_client=client, calibrator_client=client
            )
            await session.commit()

        async with AsyncSessionLocal() as session:
            run_row = await session.get(TaskRun, run_id)
            if run_row is not None:
                run_row.status = TaskRunStatus.succeeded
                run_row.items_processed = report.facts_tagged + report.questions_tagged
                notes = (
                    f"facts={report.facts_tagged} q={report.questions_tagged} "
                    f"tags={report.tags_persisted} flagged={report.manual_review_flagged} "
                    f"no_emb={report.facts_skipped_no_embedding} "
                    f"q_skipped={report.questions_skipped} failures={len(report.failures)}"
                )
                if report.failures or report.facts_skipped_no_embedding or report.questions_skipped:
                    run_row.error_text = notes[:2000]
                run_row.finished_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "run_grounded_tag finished: facts=%d q=%d tags=%d flagged=%d failures=%d",
            report.facts_tagged,
            report.questions_tagged,
            report.tags_persisted,
            report.manual_review_flagged,
            len(report.failures),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_grounded_tag failed: %s", exc)
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
                logger.exception("failed to record grounded_tag failure in task_runs")


async def run_grounded_tag_job() -> None:
    async with _lock:
        if "run_grounded_tag" in _inflight:
            logger.warning("run_grounded_tag already in-flight, skipping")
            return
        _inflight.add("run_grounded_tag")
    try:
        await _do_run_grounded_tag()
    finally:
        async with _lock:
            _inflight.discard("run_grounded_tag")


# --------------------------------------------------------------------------- #
# Scheduler lifecycle
# --------------------------------------------------------------------------- #


def start_scheduler() -> None:
    if not settings.SCHEDULER_ENABLED:
        return
    scheduler.add_job(
        run_anki_sync_job,
        "interval",
        minutes=settings.ANKI_SYNC_INTERVAL_MINUTES,
        id="run_anki_sync",
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
    scheduler.add_job(
        run_pdf_ingest_job,
        "interval",
        minutes=settings.PDF_INGEST_INTERVAL_MINUTES,
        id="run_pdf_ingest",
        replace_existing=True,
    )
    scheduler.add_job(
        run_notion_sync_job,
        "interval",
        minutes=settings.NOTION_SYNC_INTERVAL_MINUTES,
        id="run_notion_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        run_embed_job,
        "interval",
        minutes=settings.EMBED_INTERVAL_MINUTES,
        id="run_embed",
        replace_existing=True,
    )
    scheduler.add_job(
        run_grounded_tag_job,
        "interval",
        minutes=settings.GROUNDED_TAG_INTERVAL_MINUTES,
        id="run_grounded_tag",
        replace_existing=True,
    )
    scheduler.start()


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
