"""PDF inbox poller (T51, V-KB1, V41).

Walks ``PDF_INBOX_DIR/<course_slug>/*.pdf`` and runs each file through the
vision-ingest pipeline (``kb/pdf_ingest.ingest_pdf``). The per-course slug
subdirectory routes a file to its course — files dropped in the inbox root
(no slug subdir) or under an unknown slug are skipped with a WARN, never an
error (V41: one bad file ⊥ aborts the batch).

Idempotent: ``ingest_pdf`` short-circuits on a known SHA-256, so re-polling an
already-ingested file is a cheap no-op (no render, no API call).

Session-accepting core (mirrors the anki ``sync_deck`` shape) so the scheduler
wrapper owns ``TaskRun`` + ``AsyncSessionLocal`` and tests can drive this
against the savepoint-rolled-back fixture session with mocked OpenAI clients
(V16).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import Course
from app.services.kb.pdf_ingest import Renderer, ingest_pdf, render_pages

_logger = logging.getLogger("app.services.kb.inbox")


@dataclass
class InboxReport:
    files_seen: int = 0
    files_ingested: int = 0      # parsed this run (not SHA-reused)
    files_reused: int = 0        # already ingested (SHA short-circuit)
    files_skipped: int = 0       # no resolvable course slug
    new_facts: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def partial_failure(self) -> bool:
        return bool(self.failures)


async def _resolve_course(session: AsyncSession, slug: str) -> Course | None:
    return (
        await session.execute(select(Course).where(Course.slug == slug))
    ).scalar_one_or_none()


async def poll_inbox(
    session: AsyncSession,
    *,
    vision_client: Any,
    extract_client: Any | None = None,
    inbox_dir: Path,
    renderer: Renderer = render_pages,
) -> InboxReport:
    """Ingest every ``<slug>/*.pdf`` under ``inbox_dir`` (V-KB1, V41).

    Returns an :class:`InboxReport`; per-file exceptions are caught, logged
    WARN, and recorded in ``failures`` so the scheduler still reaches
    ``commit()`` and marks the run ``succeeded`` (partial). Runs inside the
    caller's transaction — the caller owns commit/rollback.
    """

    report = InboxReport()
    if not inbox_dir.exists():
        _logger.info("poll_inbox: inbox dir %s does not exist; nothing to do", inbox_dir)
        return report

    # Cache slug → Course over the run so a many-file course resolves once.
    course_cache: dict[str, Course | None] = {}

    for slug_dir in sorted(p for p in inbox_dir.iterdir() if p.is_dir()):
        slug = slug_dir.name
        for pdf_path in sorted(slug_dir.glob("*.pdf")):
            report.files_seen += 1
            if slug not in course_cache:
                course_cache[slug] = await _resolve_course(session, slug)
            course = course_cache[slug]
            if course is None:
                report.files_skipped += 1
                _logger.warning(
                    "poll_inbox: no course for slug %r; skipping %s", slug, pdf_path.name
                )
                continue
            try:
                result = await ingest_pdf(
                    session,
                    course_id=course.id,
                    path=pdf_path,
                    vision_client=vision_client,
                    extract_client=extract_client,
                    renderer=renderer,
                )
            except Exception as exc:  # noqa: BLE001 — V41 per-file isolation
                report.failures.append(f"{slug}/{pdf_path.name}: {exc}")
                _logger.warning("poll_inbox: failed on %s: %s", pdf_path, exc)
                continue
            if result.reused_pdf:
                report.files_reused += 1
            else:
                report.files_ingested += 1
                report.new_facts += result.new_facts

    _logger.info(
        "poll_inbox: seen=%d ingested=%d reused=%d skipped=%d new_facts=%d failures=%d",
        report.files_seen,
        report.files_ingested,
        report.files_reused,
        report.files_skipped,
        report.new_facts,
        len(report.failures),
    )
    return report
