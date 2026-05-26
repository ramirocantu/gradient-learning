"""Batch feature-extraction worker (Ticket 4.3).

Drains questions that have no QuestionFeatures row or whose
extractor_version is stale, processing them concurrently with
bounded parallelism and a cost guard.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

from anthropic import AsyncAnthropic
from sqlalchemy import exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question
from app.models.features import QuestionFeatures
from app.services.analyzer import (
    CARS_SKIPPED_REASON,
    FeatureExtractionResult,
    extract_features_for_question,
)
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION

logger = logging.getLogger(__name__)

# Fields included in the end-of-run distribution table.
_DIST_FIELDS = [
    "question_format",
    "reasoning_type",
    "requires_calculation",
    "involves_graph_or_figure",
    "involves_data_table",
    "has_negative_phrasing",
    "distractor_difficulty",
    "trap_distractor_present",
    "jargon_density",
    "passage_length_bucket",
    "passage_type",
]

SessionFactory = Callable[[], Any]  # async context manager returning AsyncSession


@dataclass
class ExtractionSummary:
    model: str
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    retried: int = 0
    skipped_cars: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost_usd: float = 0.0
    total_cost_saved_usd: float = 0.0
    cost_limit_hit: bool = False
    dry_run: bool = False
    max_cost_usd: float | None = None
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_text(self) -> str:
        n = self.succeeded
        dry_note = " (DRY RUN — nothing committed)" if self.dry_run else ""
        lines = [
            f"SUMMARY: model={self.model} processed={self.processed} "
            f"succeeded={self.succeeded} failed={self.failed} retried={self.retried} "
            f"skipped_cars={self.skipped_cars}{dry_note}",
            f"         cache_hits={self.cache_hits} cache_misses={self.cache_misses}",
            f"         total_cost_usd=${self.total_cost_usd:.2f} "
            f"total_cost_saved_usd=${self.total_cost_saved_usd:.2f} "
            f"cost_limit_hit={self.cost_limit_hit}",
        ]
        if not self.distributions or n == 0:
            return "\n".join(lines)

        lines.append("")
        lines.append(f"DISTRIBUTIONS (over {n} successfully-extracted questions):")
        for field_name in _DIST_FIELDS:
            counts = self.distributions.get(field_name, {})
            if not counts:
                continue
            parts = []
            for val, cnt in sorted(counts.items()):
                pct = cnt / n * 100
                parts.append(f"{val}={cnt} ({pct:.0f}%)")
            lines.append(f"  {field_name + ':':<32} {', '.join(parts)}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "retried": self.retried,
            "skipped_cars": self.skipped_cars,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
            "cost_limit_hit": self.cost_limit_hit,
            "dry_run": self.dry_run,
            "distributions": self.distributions,
        }


async def _query_pending_ids(
    session: AsyncSession,
    *,
    extractor_version: str,
    missed_only: bool = False,
    since: date | None = None,
    limit: int | None = None,
) -> list[tuple[int, str]]:
    """Return (question_id, qid) pairs that need (re-)extraction."""
    stmt = (
        select(Question.id, Question.qid)
        .outerjoin(QuestionFeatures, QuestionFeatures.question_id == Question.id)
        .where(
            or_(
                QuestionFeatures.id.is_(None),
                QuestionFeatures.extractor_version != extractor_version,
            )
        )
        .order_by(Question.first_seen_at)
    )
    if missed_only:
        stmt = stmt.where(
            exists().where(
                Attempt.question_id == Question.id,
                Attempt.is_correct.is_(False),
            )
        )
    if since is not None:
        since_dt = datetime(since.year, since.month, since.day, tzinfo=timezone.utc)
        stmt = stmt.where(Question.first_seen_at >= since_dt)
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = (await session.execute(stmt)).all()
    return [(int(row[0]), str(row[1])) for row in rows]


def _compute_distributions(
    results: list[FeatureExtractionResult],
) -> dict[str, dict[str, int]]:
    dists: dict[str, Counter[str]] = {}

    def _tally(key: str, val: str) -> None:
        if key not in dists:
            dists[key] = Counter()
        dists[key][val] += 1

    for r in results:
        if r.mechanical is None or r.features is None:
            continue
        _tally("question_format", r.mechanical.question_format)
        _tally("has_negative_phrasing", str(r.mechanical.has_negative_phrasing))
        _tally("passage_length_bucket", r.mechanical.passage_length_bucket or "n/a")
        _tally("reasoning_type", r.features.reasoning_type)
        _tally("requires_calculation", str(r.features.requires_calculation))
        _tally("involves_graph_or_figure", str(r.features.involves_graph_or_figure))
        _tally("involves_data_table", str(r.features.involves_data_table))
        _tally("distractor_difficulty", r.features.distractor_difficulty)
        _tally("trap_distractor_present", str(r.features.trap_distractor_present))
        _tally("jargon_density", r.features.jargon_density)
        _tally("passage_type", r.features.passage_type or "n/a")

    return {k: dict(v) for k, v in dists.items()}


async def run_extraction(
    session_factory: SessionFactory,
    *,
    anthropic_client: AsyncAnthropic,
    cache: FeatureExtractorCache | None = None,
    missed_only: bool = False,
    since: date | None = None,
    limit: int | None = None,
    max_cost_usd: float | None = None,
    concurrency: int = 5,
    dry_run: bool = False,
    model: str | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
) -> ExtractionSummary:
    """Drain questions needing extraction. Returns summary with distributions."""
    from app.config import settings

    resolved_model = model or settings.FEATURE_EXTRACTOR_MODEL
    summary = ExtractionSummary(
        model=resolved_model,
        dry_run=dry_run,
        max_cost_usd=max_cost_usd,
    )

    async with session_factory() as session:
        pending = await _query_pending_ids(
            session,
            extractor_version=extractor_version,
            missed_only=missed_only,
            since=since,
            limit=limit,
        )

    if not pending:
        return summary

    sem = asyncio.Semaphore(concurrency)
    stop_event = asyncio.Event()
    successful_results: list[FeatureExtractionResult] = []

    async def _extract_one(q_id: int, qid: str) -> None:
        async with sem:
            if stop_event.is_set():
                return
            summary.processed += 1
            last_exc: BaseException | None = None
            for attempt_num in range(2):
                try:
                    async with session_factory() as session:
                        result = await extract_features_for_question(
                            q_id,
                            session,
                            anthropic_client=anthropic_client,
                            cache=cache,
                        )
                        if not dry_run:
                            await session.commit()

                    if result.skipped_reason == CARS_SKIPPED_REASON:
                        summary.skipped_cars += 1
                        return

                    summary.succeeded += 1
                    if result.cache_hit:
                        summary.cache_hits += 1
                    else:
                        summary.cache_misses += 1
                    summary.total_cost_usd += result.cost_estimate_usd
                    summary.total_cost_saved_usd += result.cost_saved_usd

                    logger.info(
                        "qid=%s cache_hit=%s cost=$%.4f involves_graph=%s involves_table=%s",
                        qid,
                        result.cache_hit,
                        result.cost_estimate_usd,
                        result.features.involves_graph_or_figure if result.features else "N/A",
                        result.features.involves_data_table if result.features else "N/A",
                    )

                    successful_results.append(result)

                    if max_cost_usd is not None and summary.total_cost_usd >= max_cost_usd:
                        stop_event.set()
                        summary.cost_limit_hit = True
                        logger.warning(
                            "cost limit hit: spent=$%.4f cap=$%.4f",
                            summary.total_cost_usd,
                            max_cost_usd,
                        )
                    return

                except Exception as exc:
                    last_exc = exc
                    if attempt_num == 0:
                        summary.retried += 1
                        logger.warning("qid=%s attempt 1 failed: %s; retrying", qid, exc)

            summary.failed += 1
            logger.error("qid=%s failed both attempts: %s", qid, last_exc)

    tasks = [_extract_one(q_id, qid) for q_id, qid in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    summary.distributions = _compute_distributions(successful_results)
    return summary
