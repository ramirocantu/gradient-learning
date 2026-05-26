"""Batch-API adapter for the anki topic resolver (SPEC §T51).

Mirrors `topic_resolver_worker.run`'s logic but produces a single Anthropic
Batches API submission instead of N synchronous calls. 50% off input + output
tokens per Anthropic's pricing, same cache_control semantics, same multi-pick
output shape per §V25.

Three entry points:

- `build_topic_resolver_batch_requests` — iterate pending (card, cc) pairs,
  build one `BatchRequestItem` per pair. Skip cache hits + skip empty-signal
  cards (matches the sync worker's skip rule).
- `persist_topic_resolver_batch_results` — stream results from a finished
  batch, parse each `MessageBatchIndividualResponse`, idempotently write
  `anki_card_tags` rows mirroring the sync worker's persistence.
- `submit_topic_resolver_batch` — convenience: build + submit + record a
  row in `llm_batch_runs`.
- `finalize_topic_resolver_batch` — convenience: poll a batch by id, stream
  results, persist tags, update the `llm_batch_runs` row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message, ToolUseBlock
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.anki import AnkiNote, AnkiNoteTag
from app.models.llm_batch import LlmBatchRun
from app.models.outline import ContentCategory
from app.services.anki.topic_resolver import (
    EXTRACTOR_VERSION,
    MAX_TOKENS,
    MAX_TOPIC_PICKS,
    TopicPick,
    _build_user_message,
    _compute_cost,
    _format_tag_payload,
    _system_block_for_cc,
    _to_full_path,
    _tool_def_for_cc,
    _topic_paths_for_cc,
    make_cache_key,
)
from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache
from app.services.anki.topic_resolver_worker import (
    _candidate_notes,
    _filter_anking_tags,
    _note_text,
)
from app.services.categorizer.outline_lookup import OutlineLookup
from app.services.llm.batch import (
    BatchRequestItem,
    iter_batch_results,
    submit_batch,
)

logger = logging.getLogger(__name__)


# Batch processing-status values returned by the Anthropic API. We mirror
# them on `llm_batch_runs.processing_status` verbatim.
_TERMINAL_STATUSES = frozenset({"ended", "canceled", "expired"})


# `custom_id` ≤ 64 chars per Anthropic. Schema: `topic-<note_id>-<cc_code>`
# (§V75: keyed on note_id — 13-digit AnKing native id fits comfortably).
def _build_custom_id(note_id: int, cc_code: str) -> str:
    cid = f"topic-{note_id}-{cc_code}"
    if len(cid) > 64:  # extremely defensive — cc codes are ≤4 chars, ids fit
        raise ValueError(f"custom_id too long: {cid!r}")
    return cid


def _parse_custom_id(custom_id: str) -> tuple[int, str] | None:
    """Inverse of `_build_custom_id` — returns (note_id, cc_code) or None
    on malformed input (defensive — should never happen on results we submitted)."""
    if not custom_id.startswith("topic-"):
        return None
    rest = custom_id[len("topic-") :]
    last_dash = rest.rfind("-")
    if last_dash <= 0:
        return None
    try:
        return int(rest[:last_dash]), rest[last_dash + 1 :]
    except ValueError:
        return None


@dataclass
class BatchBuildSummary:
    items: list[BatchRequestItem]
    skipped_cache_hits: int
    skipped_empty_signal: int
    skipped_no_candidates: int  # CARS, etc.
    skipped_duplicate: int = 0


async def build_topic_resolver_batch_requests(
    session: AsyncSession,
    *,
    cache: AnkiTopicResolverCache | None = None,
    model: str | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
) -> BatchBuildSummary:
    """Walk pending (card, cc) pairs and build batch items for those that
    actually need an LLM call. Skips:
      - cache hits (same logic as sync worker — cache key on tag_payload +
        card_text + cc + model)
      - empty-signal cards (no filtered tags AND card text < MIN length)
      - CARS / no-candidate CCs

    The returned items embed the same `params` shape used by the sync
    resolver — system + tools w/ cache_control, tool_choice, single user
    message rendering filtered tags + card text.
    """
    resolved_model = model or settings.ANKI_TOPIC_RESOLVER_MODEL

    items: list[BatchRequestItem] = []
    skipped_cache = 0
    skipped_empty = 0
    skipped_no_candidates = 0
    skipped_duplicate = 0
    seen_pairs: set[tuple[int, str]] = set()

    candidates = await _candidate_notes(session)
    for note, cc_code in candidates:
        pair = (note.note_id, cc_code)
        if pair in seen_pairs:
            # Multiple `aamc_cc` tag_raw rows on the same note can resolve to
            # the same content_category_id (uniqueness is on tag_raw, not cc_id);
            # dedupe so we don't emit a duplicate custom_id, which the Anthropic
            # batch API rejects.
            skipped_duplicate += 1
            continue
        seen_pairs.add(pair)

        topic_paths = _topic_paths_for_cc(cc_code)
        if not topic_paths:
            skipped_no_candidates += 1
            continue

        filtered_tags = _filter_anking_tags(note)
        note_text = _note_text(note)
        from app.services.anki.topic_resolver import MIN_RESOLVABLE_TEXT_LEN

        if not filtered_tags and len(note_text) < MIN_RESOLVABLE_TEXT_LEN:
            skipped_empty += 1
            continue

        if cache is not None:
            tag_payload = _format_tag_payload(filtered_tags)
            cache_key = make_cache_key(tag_payload, note_text, cc_code, resolved_model)
            if cache.get(cache_key, extractor_version) is not None:
                skipped_cache += 1
                continue

        params: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": MAX_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": _system_block_for_cc(cc_code, topic_paths),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [_tool_def_for_cc(cc_code, topic_paths)],
            "tool_choice": {"type": "tool", "name": "submit_anki_topic"},
            "messages": [
                {
                    "role": "user",
                    "content": _build_user_message(filtered_tags, note_text),
                }
            ],
        }
        items.append(
            BatchRequestItem(
                custom_id=_build_custom_id(note.note_id, cc_code),
                params=params,
            )
        )

    return BatchBuildSummary(
        items=items,
        skipped_cache_hits=skipped_cache,
        skipped_empty_signal=skipped_empty,
        skipped_no_candidates=skipped_no_candidates,
        skipped_duplicate=skipped_duplicate,
    )


@dataclass
class BatchPersistSummary:
    succeeded: int
    errored: int
    canceled: int
    expired: int
    persisted_rows: int
    skipped_low_confidence: int
    declined: int
    total_cost_usd: float
    unresolved_paths: int  # picks whose path didn't resolve via lookup


def _extract_tool_input(message: Message) -> dict[str, Any] | None:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            return block.input or {}
    return None


async def persist_topic_resolver_batch_results(
    session: AsyncSession,
    *,
    anthropic_batch_id: str,
    client: AsyncAnthropic,
    cache: AnkiTopicResolverCache | None = None,
    lookup: OutlineLookup | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
    model: str | None = None,
) -> BatchPersistSummary:
    """Stream a finished batch's results, parse each, and write
    `anki_card_tags` rows. Mirrors `topic_resolver_worker.run`'s persistence
    logic so the sync + batch code paths agree on schema + idempotency.

    Caller is responsible for committing the session.
    """
    resolved_model = model or settings.ANKI_TOPIC_RESOLVER_MODEL
    threshold = settings.ANKI_TOPIC_RESOLVER_CONFIDENCE_THRESHOLD
    if lookup is None:
        lookup = await OutlineLookup.load(session)

    # Resolve CC code → id once (small set; one row per batch result then
    # cheap dict lookup keeps the loop tight).
    cc_codes_to_id: dict[str, int] = {
        row.code: row.id for row in (await session.execute(select(ContentCategory))).scalars()
    }

    # Memoized per-note lookup. Avoid preloading the entire AnkiNote table —
    # batches touch a small subset, so fetch on first miss and cache (§V75:
    # resolution + content are note-level).
    notes_by_id: dict[int, AnkiNote] = {}

    async def _get_note(note_id: int) -> AnkiNote | None:
        cached = notes_by_id.get(note_id)
        if cached is not None:
            return cached
        note = (
            await session.execute(
                select(AnkiNote)
                .where(AnkiNote.note_id == note_id)
                .options(selectinload(AnkiNote.tags))
            )
        ).scalar_one_or_none()
        if note is not None:
            notes_by_id[note_id] = note
        return note

    succeeded = errored = canceled = expired = 0
    persisted = low_conf = declined_by_llm = unresolved = 0
    total_cost = 0.0

    async for item in iter_batch_results(client, anthropic_batch_id):
        result = item.result
        result_type = getattr(result, "type", None)
        if result_type == "errored":
            errored += 1
            err = getattr(result, "error", None)
            logger.warning(
                "batch result errored custom_id=%s err=%r",
                item.custom_id,
                err,
            )
            continue
        if result_type == "canceled":
            canceled += 1
            continue
        if result_type == "expired":
            expired += 1
            continue
        if result_type != "succeeded":
            logger.warning("batch result unknown type %r", result_type)
            continue

        succeeded += 1
        parsed = _parse_custom_id(item.custom_id)
        if parsed is None:
            logger.warning(
                "batch result custom_id %r failed to parse; skipping",
                item.custom_id,
            )
            continue
        note_id, cc_code = parsed

        note = await _get_note(note_id)
        cc_id = cc_codes_to_id.get(cc_code)
        if note is None or cc_id is None:
            logger.warning(
                "batch result custom_id=%s could not resolve note/cc (note=%s cc_id=%s)",
                item.custom_id,
                note,
                cc_id,
            )
            continue

        message = getattr(result, "message", None)
        if message is None:
            continue

        # Cost accounting per result. Batches charge 50% of normal, but the
        # raw usage stays the same — sync caller can apply the discount
        # in the summary if desired.
        usage = getattr(message, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cost = (
            _compute_cost(input_tokens, output_tokens, cached_read, model=resolved_model) * 0.5
        )  # Batches API discount
        total_cost += cost

        payload = _extract_tool_input(message) or {}
        if payload.get("decline"):
            declined_by_llm += 1
            continue

        raw_picks = payload.get("topic_picks") or []
        if not isinstance(raw_picks, list):
            declined_by_llm += 1
            continue

        topic_paths = _topic_paths_for_cc(cc_code)

        # Parse + threshold-filter picks. v9: schema field is `topic_id`
        # (integer in [1, N]); server maps id → full canonical path via the
        # deterministic position index. Dedupe by topic_id across picks (the
        # prompt asks for distinct topics; defensive). Old (v6-v8)
        # `topic_path` field is still accepted defensively in case a cached
        # message replay sneaks through.
        picks: list[TopicPick] = []
        seen_ids: set[int] = set()
        for raw in raw_picks[:MAX_TOPIC_PICKS]:
            if not isinstance(raw, dict):
                continue
            raw_id = raw.get("topic_id")
            if raw_id is None:
                raw_path = raw.get("topic_path") or ""
                if not raw_path:
                    continue
                full_path = _to_full_path(cc_code, raw_path)
                try:
                    raw_id = topic_paths.index(full_path) + 1
                except ValueError:
                    continue
            try:
                topic_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if not (1 <= topic_id <= len(topic_paths)):
                continue
            if topic_id in seen_ids:
                continue
            seen_ids.add(topic_id)
            full_path = topic_paths[topic_id - 1]
            picks.append(
                TopicPick(
                    topic_path=full_path,
                    confidence=float(raw.get("confidence", 0.0)),
                    rationale=str(raw.get("rationale", "")),
                )
            )

        if not picks:
            declined_by_llm += 1
            continue

        accepted_picks = [p for p in picks if p.confidence >= threshold]
        if not accepted_picks:
            low_conf += 1
            continue

        # Write to cache so future sync runs see the same answer without
        # paying again.
        if cache is not None:
            filtered_tags = _filter_anking_tags(note)
            tag_payload = _format_tag_payload(filtered_tags)
            cache_key = make_cache_key(tag_payload, _note_text(note), cc_code, resolved_model)
            cache.put(cache_key, picks, extractor_version, model=resolved_model, cost=cost)

        # Idempotent re-write per (note, cc) — same shape as sync worker.
        await session.execute(
            delete(AnkiNoteTag).where(
                AnkiNoteTag.note_id == note.note_id,
                AnkiNoteTag.source == "llm",
                AnkiNoteTag.parsed_kind == "aamc_topic",
                AnkiNoteTag.content_category_id == cc_id,
            )
        )
        for pick in accepted_picks:
            topic_id = lookup.topic_id_by_path(pick.topic_path)
            if topic_id is None:
                unresolved += 1
                logger.warning(
                    "batch pick path %r did not resolve (custom_id=%s)",
                    pick.topic_path,
                    item.custom_id,
                )
                continue
            synthetic_raw = f"__llm_topic__::{extractor_version}::{pick.topic_path}"
            session.add(
                AnkiNoteTag(
                    note_id=note.note_id,
                    tag_raw=synthetic_raw,
                    topic_id=topic_id,
                    content_category_id=cc_id,
                    question_qid=None,
                    skill_number=None,
                    parsed_kind="aamc_topic",
                    source="llm",
                    confidence=pick.confidence,
                    rationale=pick.rationale,
                    extractor_version=extractor_version,
                )
            )
            persisted += 1

    await session.flush()
    return BatchPersistSummary(
        succeeded=succeeded,
        errored=errored,
        canceled=canceled,
        expired=expired,
        persisted_rows=persisted,
        skipped_low_confidence=low_conf,
        declined=declined_by_llm,
        total_cost_usd=total_cost,
        unresolved_paths=unresolved,
    )


async def submit_topic_resolver_batch(
    session: AsyncSession,
    *,
    client: AsyncAnthropic,
    cache: AnkiTopicResolverCache | None = None,
    model: str | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
) -> tuple[LlmBatchRun, BatchBuildSummary]:
    """Build pending requests + submit + persist a row in `llm_batch_runs`.

    Returns the new `LlmBatchRun` (uncommitted — caller commits) and the
    build summary so the caller can report skip counts.
    """
    resolved_model = model or settings.ANKI_TOPIC_RESOLVER_MODEL
    build = await build_topic_resolver_batch_requests(
        session,
        cache=cache,
        model=resolved_model,
        extractor_version=extractor_version,
    )
    if not build.items:
        raise ValueError(
            f"submit_topic_resolver_batch: no items to submit "
            f"(cache_hits={build.skipped_cache_hits}, "
            f"empty_signal={build.skipped_empty_signal}, "
            f"no_candidates={build.skipped_no_candidates}, "
            f"duplicate={build.skipped_duplicate})"
        )
    batch = await submit_batch(client, build.items)
    counts = batch.request_counts
    row = LlmBatchRun(
        anthropic_batch_id=batch.id,
        extractor="anki_topic_resolver",
        extractor_version=extractor_version,
        model=resolved_model,
        submitted_at=datetime.now(timezone.utc),
        processing_status=batch.processing_status,
        total_requests=len(build.items),
        succeeded_count=int(getattr(counts, "succeeded", 0) or 0),
        errored_count=int(getattr(counts, "errored", 0) or 0),
        canceled_count=int(getattr(counts, "canceled", 0) or 0),
        expired_count=int(getattr(counts, "expired", 0) or 0),
        processing_count=int(getattr(counts, "processing", 0) or 0),
    )
    session.add(row)
    await session.flush()
    return row, build


async def finalize_topic_resolver_batch(
    session: AsyncSession,
    *,
    run_id: int,
    client: AsyncAnthropic,
    cache: AnkiTopicResolverCache | None = None,
    lookup: OutlineLookup | None = None,
) -> BatchPersistSummary:
    """Poll the batch by id, stream results, write `anki_card_tags`, update
    `llm_batch_runs` row. Caller commits.

    Raises if the batch hasn't reached a terminal status — caller decides
    whether to re-poll later.
    """
    run = (await session.execute(select(LlmBatchRun).where(LlmBatchRun.id == run_id))).scalar_one()

    from app.services.llm.batch import get_batch_status

    batch = await get_batch_status(client, run.anthropic_batch_id)
    run.processing_status = batch.processing_status
    counts = batch.request_counts
    run.succeeded_count = int(getattr(counts, "succeeded", 0) or 0)
    run.errored_count = int(getattr(counts, "errored", 0) or 0)
    run.canceled_count = int(getattr(counts, "canceled", 0) or 0)
    run.expired_count = int(getattr(counts, "expired", 0) or 0)
    run.processing_count = int(getattr(counts, "processing", 0) or 0)
    if batch.ended_at is not None:
        run.ended_at = batch.ended_at

    if batch.processing_status not in _TERMINAL_STATUSES:
        raise RuntimeError(
            f"batch {run.anthropic_batch_id} not in a terminal status "
            f"(processing_status={batch.processing_status!r}); re-poll later"
        )

    persist = await persist_topic_resolver_batch_results(
        session,
        anthropic_batch_id=run.anthropic_batch_id,
        client=client,
        cache=cache,
        lookup=lookup,
        extractor_version=run.extractor_version,
        model=run.model,
    )
    run.total_cost_usd = Decimal(f"{persist.total_cost_usd:.4f}")
    run.result_persisted_at = datetime.now(timezone.utc)
    return persist
