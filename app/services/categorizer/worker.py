"""Categorizer drain worker — importable from both CLI scripts and the scheduler.

Extracted from scripts/run_categorizer.py (Ticket 6.9b) so the scheduler can
import it without depending on the scripts/ directory, which is not an
installable package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from openai import (
    APIError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question
from app.services.categorizer import (
    QuestionNotFoundError,
    TagQuestionResult,
    tag_question,
)
from app.services.categorizer.cache import CategorizerCache
from app.services.categorizer.outline_lookup import OutlineLookup
from app.services.categorizer.outline_render import SUBJECT_TO_SECTION


def _section_sort_key(uworld_aamc_tags: list[str] | None) -> tuple[str, int]:
    """Derive section code from `Subject:` tag for cache-prefix-stable ordering.

    Returns `(section_code, 0)` for known sections; `("ZZ", 0)` for unknown so
    those drain last. Tuple shape lets callers append a secondary key
    (first_seen_at, id) for stable tiebreak.
    """
    if not uworld_aamc_tags:
        return ("ZZ", 0)
    for t in uworld_aamc_tags:
        if isinstance(t, str) and t.startswith("Subject: "):
            subject = t[len("Subject: ") :].strip()
            sec = SUBJECT_TO_SECTION.get(subject)
            if sec is not None:
                return (sec, 0)
    return ("ZZ", 0)


logger = logging.getLogger(__name__)

TagFn = Callable[..., Awaitable[TagQuestionResult]]


@dataclass
class WorkerSummary:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    zero_result_questions: int = 0
    total_targets_persisted: int = 0
    total_targets_replaced: int = 0
    total_suggestions_unresolved: int = 0
    total_cost_usd: float = 0.0
    total_cost_saved_usd: float = 0.0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    failure_qids: list[str] = field(default_factory=list)
    dry_run: bool = False
    cost_limit_hit: bool = False
    max_cost_usd: float | None = None
    model: str = ""
    # V41 (amended): set when a transient OpenAI error broke the loop early.
    # Scheduler sees this and tags the task_run with a non-fatal note while
    # still committing succeeded work + setting status='succeeded'. Candidate
    # filter resumes the unprocessed remainder on the next run.
    partial_failure: bool = False
    error: str | None = None

    def as_text(self) -> str:
        suffix = " (DRY RUN — nothing committed)" if self.dry_run else ""
        limit_note = ""
        if self.max_cost_usd is not None:
            limit_note = (
                f" cost_limit_hit={self.cost_limit_hit} max_cost_usd=${self.max_cost_usd:.4f}"
            )
        model_note = f"model={self.model} " if self.model else ""
        return (
            f"{model_note}"
            f"processed={self.processed} succeeded={self.succeeded} "
            f"failed={self.failed} zero_result={self.zero_result_questions} "
            f"targets_persisted={self.total_targets_persisted} "
            f"targets_replaced={self.total_targets_replaced} "
            f"suggestions_unresolved={self.total_suggestions_unresolved} "
            f"cache_hits={self.cache_hit_count} cache_misses={self.cache_miss_count} "
            f"total_cost_usd=${self.total_cost_usd:.4f} "
            f"total_cost_saved_usd=${self.total_cost_saved_usd:.4f}"
            f"{limit_note}{suffix}"
        )


async def run(
    session: AsyncSession,
    *,
    openai_client: AsyncOpenAI,
    batch_size: int = 100,
    dry_run: bool = False,
    lookup: OutlineLookup | None = None,
    tag_fn: TagFn = tag_question,
    cache: CategorizerCache | None = None,
    max_cost_usd: float | None = None,
) -> WorkerSummary:
    """Drain pending questions; per-question failures are isolated via savepoints.

    If `max_cost_usd` is set, the loop stops as soon as accumulated cost
    (cache misses only — hits cost nothing) reaches the cap. Unprocessed
    questions retain `needs_categorization=true` so a follow-up run picks
    them up.
    """
    from app.config import settings

    if lookup is None:
        lookup = await OutlineLookup.load(session)
    summary = WorkerSummary(
        dry_run=dry_run,
        max_cost_usd=max_cost_usd,
        model=settings.CATEGORIZER_MODEL,
    )
    attempted: set[int] = set()

    while True:
        stmt = (
            select(Question.id, Question.qid, Question.uworld_aamc_tags)
            .where(Question.needs_categorization.is_(True))
            .order_by(Question.first_seen_at)
            .limit(batch_size)
        )
        if attempted:
            stmt = stmt.where(Question.id.notin_(attempted))
        rows = (await session.execute(stmt)).all()
        if not rows:
            break

        # V42: order by derived section so the Anthropic prompt-cache prefix
        # (per-section outline + canonical block + tool def) stays hot across
        # consecutive calls within the same section. `first_seen_at` ordering
        # is preserved as the tiebreak within each section to keep behavior
        # near the pre-T55 batch shape.
        sorted_rows = sorted(
            enumerate(rows),
            key=lambda pair: (
                _section_sort_key(pair[1][2])[0],
                pair[0],
            ),
        )

        for _orig_pos, (q_id, qid, _tags) in sorted_rows:
            attempted.add(q_id)
            summary.processed += 1
            try:
                async with session.begin_nested():
                    result = await tag_fn(
                        q_id,
                        session,
                        lookup=lookup,
                        openai_client=openai_client,
                        cache=cache,
                    )
                summary.succeeded += 1
                summary.total_targets_persisted += result.targets_persisted
                summary.total_targets_replaced += result.targets_replaced
                summary.total_suggestions_unresolved += result.suggestions_unresolved
                summary.total_cost_usd += result.cost_estimate_usd
                summary.total_cost_saved_usd += result.cost_saved_usd
                if result.cache_hit:
                    summary.cache_hit_count += 1
                else:
                    summary.cache_miss_count += 1
                if result.targets_persisted == 0:
                    summary.zero_result_questions += 1
                logger.info(
                    "tagged qid=%s persisted=%d replaced=%d unresolved=%d "
                    "cache_hit=%s cost=$%.4f saved=$%.4f",
                    qid,
                    result.targets_persisted,
                    result.targets_replaced,
                    result.suggestions_unresolved,
                    result.cache_hit,
                    result.cost_estimate_usd,
                    result.cost_saved_usd,
                )
                for w in result.categorize_result.parse_warnings:
                    logger.info("  warning qid=%s: %s", qid, w)

                if max_cost_usd is not None and summary.total_cost_usd >= max_cost_usd:
                    pending_remaining = (
                        await session.execute(
                            select(Question.id).where(
                                Question.needs_categorization.is_(True),
                                Question.id.notin_(attempted),
                            )
                        )
                    ).all()
                    summary.cost_limit_hit = True
                    logger.warning(
                        "cost limit hit: spent=$%.4f cap=$%.4f processed=%d pending=%d",
                        summary.total_cost_usd,
                        max_cost_usd,
                        summary.processed,
                        len(pending_remaining),
                    )
                    if dry_run:
                        await session.rollback()
                    return summary
            except QuestionNotFoundError:
                logger.warning("qid=%s disappeared mid-run; skipping", qid)
                summary.failed += 1
                summary.failure_qids.append(qid)
            except (APIError, RateLimitError, InternalServerError) as exc:
                # V41 amended: transient OpenAI errors break the loop early.
                # The SDK already retried max_retries times; further per-item
                # retries here would burn budget. Mark partial + return so the
                # scheduler commits the already-succeeded work and sets
                # status='succeeded'. Next scheduler tick picks up the rest
                # via the `needs_categorization=true` candidate filter.
                logger.warning(
                    "transient OpenAI error on qid=%s after SDK retries: %s; "
                    "breaking early with partial_failure",
                    qid,
                    exc,
                )
                summary.partial_failure = True
                summary.error = f"{type(exc).__name__}: {exc}"[:500]
                if dry_run:
                    await session.rollback()
                return summary
            except Exception as exc:  # noqa: BLE001
                logger.exception("failed to tag qid=%s: %s", qid, exc)
                summary.failed += 1
                summary.failure_qids.append(qid)

    if dry_run:
        await session.rollback()

    return summary
