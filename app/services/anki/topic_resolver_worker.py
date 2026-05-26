"""Anki topic-resolver orchestrator (SPEC §T32).

Pulls cards that have an `aamc_cc` tag but no `aamc_topic` LLM tag, calls
the resolver, and persists `anki_card_tags` rows with `source='llm'`,
`parsed_kind='aamc_topic'`, `topic_id` populated. Idempotent re-run: deletes
existing `source='llm'` rows for each card before re-inserting, mirroring the
QuestionTag/`tag_question` pattern.

Cost-capped per run (settings.ANKI_TOPIC_RESOLVER_PER_RUN_BUDGET_USD).
Confidence threshold (settings.ANKI_TOPIC_RESOLVER_CONFIDENCE_THRESHOLD) is
checked here, not inside `resolve_topic`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from anthropic import APIError, AsyncAnthropic
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.anki import AnkiNote, AnkiNoteTag
from app.models.outline import ContentCategory
from app.services.anki.topic_resolver import (
    CARD_TEXT_MAX_LEN,
    EXTRACTOR_VERSION,
    MIN_RESOLVABLE_TEXT_LEN,
    resolve_topic,
)
from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache
from app.services.categorizer.outline_lookup import OutlineLookup

logger = logging.getLogger(__name__)


@dataclass
class ResolverSummary:
    processed: int
    persisted: int
    skipped_low_confidence: int
    declined_by_llm: int
    cache_hits: int
    total_cost_usd: float
    total_cost_saved_usd: float
    error: str | None = None
    # §V41: True when the loop broke early on a transient Anthropic API error
    # (529/429/5xx) after the SDK's own retries had been exhausted. Distinct
    # from `error` (which is a hard failure of the whole run); a partial run
    # still commits accumulated progress and is recorded as `succeeded`.
    partial_failure: bool = False


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_UWORLD_QID_TAG_RE = re.compile(r"^#AK_MCAT_v2::#UWorld::\d+$")
# Internal Anki state tags that carry no semantic content. Match exact strings
# (with or without optional prefixes) — the user is free to add custom tags
# but these three are anki-native.
_ANKI_INTERNAL_TAGS = frozenset({"marked", "leech", "duplicate"})


def _filter_anking_tags(note: AnkiNote) -> list[str]:
    """Filter the note's raw AnKing tag strings per §V25 (§V75: note-level).

    KEEP: all taxonomy-bearing tags (`#AK_MCAT_v2::#AAMC::*`, `#FirstAid::*`,
    `#Bootcamp::*`, `#Pixorize::*`, `#Sketchy::*`, etc.).
    DROP: UWorld qid-only tag (carries no topic signal), anki-internal tags
    (`marked`, `leech`, `duplicate`), any tag that is just the deck name, and
    LLM-derived aamc_topic rows persisted by this worker (source='llm',
    parsed_kind='aamc_topic') — feeding those back into the prompt on a
    later-CC re-run would bias the resolver toward its own prior picks.

    Tags emitted in case-insensitive lexical order of `tag_raw` for cache key
    stability — the ORM relationship has no `order_by`, so DB-row order is not
    guaranteed across queries; without a deterministic sort here, two equivalent
    tag sets would hash to different cache rows on different reads. The
    (note_id, tag_raw) unique constraint guarantees ties are impossible.
    """
    deck_name = (note.deck_name or "").strip()
    sorted_tags = sorted(note.tags or [], key=lambda t: (t.tag_raw or "").strip().lower())
    filtered: list[str] = []
    for tag in sorted_tags:
        if tag.source == "llm" and tag.parsed_kind == "aamc_topic":
            continue
        raw = (tag.tag_raw or "").strip()
        if not raw:
            continue
        if raw in _ANKI_INTERNAL_TAGS:
            continue
        if _UWORLD_QID_TAG_RE.match(raw):
            continue
        if deck_name and raw == deck_name:
            continue
        filtered.append(raw)
    return filtered


def _card_tag_payload(note: AnkiNote) -> list[str]:
    """Returns the filtered tag list shipped to the LLM as part of the user
    message body (§V25). Note-scoped per §V75."""
    return _filter_anking_tags(note)


def _strip_html_to_plain(html: str) -> str:
    """Crude HTML → plain conversion. AnKing fields are MathJax/markdown-heavy
    so we don't bother with typographic-punctuation normalization."""
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    )
    return _WS_RE.sub(" ", text).strip()


def _note_text(note: AnkiNote) -> str:
    """Pull AnKing's primary content fields (Text + Extra + Lecture Notes)
    into a single stripped + truncated plain-text blob (§V25 hybrid input).
    §V75: content lives on the note.
    """
    fields = note.fields_json or {}
    parts: list[str] = []
    for key in ("Text", "Extra", "Lecture Notes"):
        field = fields.get(key)
        if not field:
            continue
        value = field.get("value") if isinstance(field, dict) else None
        if not value:
            continue
        plain = _strip_html_to_plain(str(value))
        if plain:
            parts.append(f"[{key}] {plain}")
    text = "\n\n".join(parts)
    if len(text) > CARD_TEXT_MAX_LEN:
        text = text[:CARD_TEXT_MAX_LEN] + "…[truncated]"
    return text


async def _candidate_notes(session: AsyncSession) -> list[tuple[AnkiNote, str]]:
    """Notes w/ an aamc_cc tag NOT yet LLM-tagged at topic level (§V75).

    Returns list of (AnkiNote, cc_code) — one entry per (note, cc) pair so a
    note with multiple aamc_cc tags gets one resolver call per CC. Note-scoped
    per §V75: one LLM resolution per note, ⊥ once per card (the pre-T93
    per-card fan-out re-resolved the same note text N times).

    §V42: Sort by cc_code first so each CC drains contiguously and Anthropic
    prompt-cache prefix matches stay hot across adjacent calls. Cycling CCs
    every call evicts the prior CC's cache before reuse and tanks hit rate
    to ≈ 0% (cf §B7). Secondary sort by note_id for stable ordering.
    """
    # Outer-join `source='llm' AND parsed_kind='aamc_topic'` rows so we can
    # filter them out (already-resolved notes).
    cc_tag = AnkiNoteTag.__table__.alias("cc_tag")
    llm_tag = AnkiNoteTag.__table__.alias("llm_tag")
    cc = ContentCategory.__table__.alias("cc")

    stmt = (
        select(AnkiNote, cc.c.code)
        .join(cc_tag, cc_tag.c.note_id == AnkiNote.note_id)
        .join(cc, cc.c.id == cc_tag.c.content_category_id)
        .outerjoin(
            llm_tag,
            (llm_tag.c.note_id == AnkiNote.note_id)
            & (llm_tag.c.source == "llm")
            & (llm_tag.c.parsed_kind == "aamc_topic")
            & (llm_tag.c.content_category_id == cc_tag.c.content_category_id),
        )
        .where(cc_tag.c.parsed_kind == "aamc_cc")
        .where(llm_tag.c.id.is_(None))
        .order_by(cc.c.code, AnkiNote.note_id)
        # Eager-load AnkiNote.tags so the per-note tag-filter (§V25) doesn't
        # trigger lazy SQL in the loop body (MissingGreenlet under async).
        .options(selectinload(AnkiNote.tags))
    )
    rows = (await session.execute(stmt)).all()
    return [(note, cc_code) for note, cc_code in rows]


async def run(
    session: AsyncSession,
    *,
    anthropic_client: AsyncAnthropic,
    cache: AnkiTopicResolverCache | None = None,
    max_cost_usd: float | None = None,
    lookup: OutlineLookup | None = None,
    extractor_version: str | None = None,
) -> ResolverSummary:
    """Drain pending aamc_cc cards into aamc_topic LLM tags.

    Stops early if accumulated cost reaches `max_cost_usd`
    (default: settings.ANKI_TOPIC_RESOLVER_PER_RUN_BUDGET_USD).
    """
    if max_cost_usd is None:
        max_cost_usd = settings.ANKI_TOPIC_RESOLVER_PER_RUN_BUDGET_USD
    if lookup is None:
        lookup = await OutlineLookup.load(session)
    if extractor_version is None:
        extractor_version = EXTRACTOR_VERSION
    threshold = settings.ANKI_TOPIC_RESOLVER_CONFIDENCE_THRESHOLD

    candidates = await _candidate_notes(session)
    logger.info(
        "anki topic resolver: %d (note, cc) candidates; budget=$%.2f",
        len(candidates),
        max_cost_usd,
    )

    processed = persisted = low_conf = declined = cache_hits = 0
    total_cost = 0.0
    total_saved = 0.0
    partial_failure = False
    partial_error_text: str | None = None

    for note, cc_code in candidates:
        if total_cost >= max_cost_usd:
            logger.info("anki topic resolver: budget hit @ $%.4f; stopping early", total_cost)
            break

        filtered_tags = _filter_anking_tags(note)
        note_text = _note_text(note)
        # §V25 (hybrid) skip rule: only when BOTH signals are absent.
        if not filtered_tags and len(note_text) < MIN_RESOLVABLE_TEXT_LEN:
            continue

        try:
            result = await resolve_topic(
                filtered_tags=filtered_tags,
                card_text=note_text,
                cc_code=cc_code,
                anthropic_client=anthropic_client,
                cache=cache,
                extractor_version=extractor_version,
            )
        except APIError as exc:
            # §V41: SDK retries already exhausted (max_retries on the client).
            # Don't crash the whole drain — log, mark partial, break, return
            # what we've accumulated. Idempotent re-runs pick up the rest via
            # _candidate_cards' outer-join filter.
            partial_failure = True
            partial_error_text = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning(
                "anki topic resolver: transient Anthropic API error after card #%d "
                "(processed=%d, persisted=%d); breaking loop, returning partial. err=%s",
                processed + 1,
                processed,
                persisted,
                partial_error_text,
            )
            break
        processed += 1
        total_cost += result.estimated_cost_usd
        total_saved += result.cost_saved_usd
        if result.cache_hit:
            cache_hits += 1
        if not result.picks:
            declined += 1
            continue

        # Split picks into above/below confidence threshold up-front so we
        # know whether to touch the DB (some picks may fall below 0.5).
        accepted_picks = [p for p in result.picks if p.confidence >= threshold]
        if not accepted_picks:
            low_conf += 1
            continue

        cc_id = lookup.content_category_id(cc_code)

        # Idempotent re-write per (note, cc): drop any prior LLM-derived
        # aamc_topic rows for THIS note + this CC, then insert the fresh set.
        # Multi-pick (§V25) means N rows per (note, cc) instead of 1.
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
                logger.warning(
                    "anki topic resolver: pick path %r did not resolve",
                    pick.topic_path,
                )
                continue
            # synthetic_raw embeds topic_path so multiple rows for the same
            # note under the same CC keep the UNIQUE(note_id, tag_raw)
            # constraint happy.
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
    return ResolverSummary(
        processed=processed,
        persisted=persisted,
        skipped_low_confidence=low_conf,
        declined_by_llm=declined,
        cache_hits=cache_hits,
        total_cost_usd=total_cost,
        total_cost_saved_usd=total_saved,
        error=partial_error_text,
        partial_failure=partial_failure,
    )


def make_summary_text(summary: ResolverSummary) -> str:
    parts = [
        f"processed={summary.processed} persisted={summary.persisted} "
        f"low_conf={summary.skipped_low_confidence} declined={summary.declined_by_llm} "
        f"cache_hits={summary.cache_hits} cost=${summary.total_cost_usd:.4f} "
        f"saved=${summary.total_cost_saved_usd:.4f}"
    ]
    if summary.partial_failure:
        parts.append(f"partial=true error={summary.error!r}")
    return " ".join(parts)


__all__ = ["ResolverSummary", "run", "make_summary_text"]
# Silence unused-import warning for `Any` and `OutlineLookup` retained for type hints.
_ = (Any,)
