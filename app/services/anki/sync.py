"""Anki deck sync service (SPEC §T3).

Pulls cards from AnkiConnect, upserts rows into `anki_cards`, and replaces
the parsed-tag set on each card to match the deck state Anki reports.

Invariants enforced here:

- §V1: upsert by (deck_name, anki_card_id). When the Anki-mirrored fields
  match what is already stored, the data fields are not rewritten. `sync_at`
  bumps on every visit because it is bookkeeping ("last contact with Anki
  about this card"), not Anki state being mirrored.
- §V3: tag classification via `tag_parser.parse_tag`; unmatched tags
  persist as `parsed_kind='unparsed'` rather than crashing the sync.
- §V4: when AnkiConnect is unreachable, return the
  `{synced_cards: 0, linked_qids: 0, error: 'anki_not_running'}` envelope.
  No retry inside the call; the scheduler runs the job again on its next
  tick.
- §V13: this service calls only the client's read methods
  (`find_cards`, `cards_info`, `notes_info`).

The `due_date` field is a best-effort approximation. AnkiConnect does not
expose the collection creation date, so the raw `card.due` integer cannot
be exactly translated. For review cards (queue=2) we set
`due_date = today + interval_days` (over-estimates when a card is already
overdue but never under-estimates); for learning/relearning cards
(queue ∈ {1, 3}) we treat the card as due today; for new (queue=0) and
suspended (queue<0) cards we leave `due_date` NULL. The T5 review-queue
endpoint can refine its ordering later if this turns out to mislead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiCardReview, AnkiNote, AnkiNoteTag
from app.services.anki.client import (
    AnkiConnectClient,
    AnkiConnectError,
    AnkiUnreachableError,
)
from app.services.anki.tag_parser import parse_tag
from app.services.categorizer.outline_lookup import OutlineLookup


logger = logging.getLogger(__name__)


@dataclass
class SyncSummary:
    synced_cards: int
    linked_qids: int
    error: Optional[str] = None
    reviews_synced: int = 0


_REVIEW_TYPE_BY_INT = {0: "learn", 1: "review", 2: "relearn", 3: "cram"}


# Card mirror = SRS state only (§V75 — content/model/fields live on AnkiNote).
_MIRRORED_FIELDS = (
    "note_id",
    "due_date",
    "interval_days",
    "ease",
    "lapses",
    "queue",
)


async def sync_deck(
    session: AsyncSession,
    client: AnkiConnectClient,
    *,
    deck_name: str,
    outline_lookup: Optional[OutlineLookup] = None,
) -> SyncSummary:
    """Sync a single Anki deck. Returns the summary callers expose to HTTP/MCP."""
    # §V20/§V21: log deck_name + AnkiConnect base URL the client will hit so
    # /admin run history can be cross-referenced with the deck the run
    # actually consulted, and so the IPv6/127.0.0.1 resolver class of bug
    # (SPEC §B1) can be spotted from logs without process introspection.
    logger.info(
        "anki sync: starting deck=%r base_url=%r",
        deck_name,
        getattr(client, "_base_url", "<unknown>"),
    )
    try:
        card_ids = await client.find_cards(f'deck:"{deck_name}"')
        if not card_ids:
            # §V20: loud-fail on empty findCards. Probe deckNames so the log
            # message tells the user exactly which decks AnkiConnect knows
            # about — the most common cause of zero results is a deck-name
            # typo or a default value that does not match the user's deck.
            try:
                available = await client.deck_names()
            except (AnkiConnectError, AnkiUnreachableError):
                available = []
            logger.warning(
                "anki sync: deck %r returned 0 cards from findCards; AnkiConnect known decks = %s",
                deck_name,
                available,
            )
            return SyncSummary(
                synced_cards=0,
                linked_qids=0,
                error="deck_empty_or_misspelled",
            )
        cards = await client.cards_info(card_ids)
        unique_note_ids = sorted({c["note"] for c in cards if c.get("note") is not None})
        notes = await client.notes_info(unique_note_ids) if unique_note_ids else []
    except AnkiUnreachableError as exc:
        logger.warning(
            "anki sync: AnkiConnect unreachable at base_url=%r (cause: %s); skipping run",
            getattr(client, "_base_url", "<unknown>"),
            exc,
        )
        return SyncSummary(synced_cards=0, linked_qids=0, error="anki_not_running")

    tags_by_note: dict[int, list[str]] = {n["noteId"]: list(n.get("tags") or []) for n in notes}

    if outline_lookup is None:
        outline_lookup = await OutlineLookup.load(session)

    now = datetime.now(timezone.utc)

    # §V75 note-as-unit order: upsert notes (content + model + deck) FIRST so
    # the anki_cards.note_id FK is satisfiable, then the note-scoped tag set,
    # then the per-card SRS rows. model_name + fields_json are note-level —
    # take them from a representative card (cardsInfo carries them; every card
    # of a note shares the same content).
    note_content: dict[int, dict[str, Any]] = {}
    for card_data in cards:
        nid = card_data.get("note")
        if nid is None or nid in note_content:
            continue
        note_content[nid] = {
            "model_name": card_data.get("modelName"),
            "fields_json": card_data.get("fields"),
        }
    for nid in set(note_content) | set(tags_by_note):
        content = note_content.get(nid, {})
        await _upsert_note(
            session,
            note_id=nid,
            deck_name=deck_name,
            model_name=content.get("model_name"),
            fields_json=content.get("fields_json"),
        )
    await session.flush()

    # Replace the regex tag set per NOTE (§V75: one set per note, ⊥ per-card
    # fan-out; §V43: source!='regex' rows untouched). linked_qids counts
    # uworld_qid rows per note (deduped), not per card.
    linked_qid_count = 0
    for nid, tag_strings in tags_by_note.items():
        linked_qid_count += await _replace_tags(
            session,
            note_id=nid,
            tag_strings=tag_strings,
            outline_lookup=outline_lookup,
        )
    await session.flush()

    # Upsert cards (SRS state only). FK to anki_notes now satisfied.
    synced_count = 0
    anki_to_pk: dict[int, int] = {}
    for card_data in cards:
        anki_card_id = card_data.get("cardId")
        if anki_card_id is None:
            continue
        card_row = await _upsert_card(
            session,
            deck_name=deck_name,
            anki_card_id=anki_card_id,
            card_data=card_data,
            now=now,
        )
        await session.flush()
        anki_to_pk[int(anki_card_id)] = card_row.id
        synced_count += 1

    reviews_synced = await _sync_reviews(
        session,
        client,
        deck_name=deck_name,
        anki_to_pk=anki_to_pk,
    )

    return SyncSummary(
        synced_cards=synced_count,
        linked_qids=linked_qid_count,
        error=None,
        reviews_synced=reviews_synced,
    )


async def _upsert_note(
    session: AsyncSession,
    *,
    note_id: int,
    deck_name: str,
    model_name: Optional[str],
    fields_json: Optional[dict[str, Any]],
) -> None:
    """Upsert one anki_notes row (§V75). ON CONFLICT refreshes content so a
    re-sync mirrors Anki's current note state; row count stays stable."""
    stmt = (
        pg_insert(AnkiNote)
        .values(
            note_id=note_id,
            deck_name=deck_name,
            model_name=model_name,
            fields_json=fields_json,
        )
        .on_conflict_do_update(
            index_elements=["note_id"],
            set_={
                "deck_name": deck_name,
                "model_name": model_name,
                "fields_json": fields_json,
            },
        )
    )
    await session.execute(stmt)


async def _sync_reviews(
    session: AsyncSession,
    client: AnkiConnectClient,
    *,
    deck_name: str,
    anki_to_pk: dict[int, int],
) -> int:
    """Append revlog rows from AnkiConnect into `anki_card_reviews` (§T36/§V26).

    `startID = MAX(review_id) + 1` for incremental sync; first run reads
    everything via `startID=0`. PK on `review_id` makes the insert
    idempotent — re-runs that observe overlapping ids no-op via
    `ON CONFLICT DO NOTHING`.
    """
    max_review_id = (await session.execute(select(func.max(AnkiCardReview.review_id)))).scalar()
    start_id = 0 if max_review_id is None else int(max_review_id) + 1

    try:
        rows = await client.card_reviews(deck_name, start_id)
    except AnkiUnreachableError as exc:
        # §V4: never raise on AnkiConnect dropping mid-sync — partial work
        # (card upserts above) commits anyway. Log and report zero appends.
        logger.warning(
            "anki sync: AnkiConnect unreachable during cardReviews at startID=%d (cause: %s)",
            start_id,
            exc,
        )
        return 0

    if not rows:
        return 0

    payload: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 9:
            continue
        review_id, anki_card_id, _usn, button, new_iv, prev_iv, _factor, dur_ms, rtype_int = row[:9]
        pk = anki_to_pk.get(int(anki_card_id))
        if pk is None:
            # Card not in the current sync (deleted, suspended-out-of-deck,
            # or filtered by deck query) — drop the row defensively rather
            # than insert an orphan card_id that violates the FK.
            continue
        rtype = _REVIEW_TYPE_BY_INT.get(int(rtype_int))
        if rtype is None:
            continue
        ease_int = int(button)
        if not 1 <= ease_int <= 4:
            continue
        payload.append(
            {
                "review_id": int(review_id),
                "card_id": pk,
                "reviewed_at": datetime.fromtimestamp(int(review_id) / 1000, tz=timezone.utc),
                "ease": ease_int,
                "type": rtype,
                "interval_before": int(prev_iv) if prev_iv is not None else None,
                "interval_after": int(new_iv) if new_iv is not None else None,
                "time_ms": int(dur_ms) if dur_ms is not None else None,
            }
        )

    if not payload:
        return 0

    stmt = (
        pg_insert(AnkiCardReview)
        .values(payload)
        .on_conflict_do_nothing(index_elements=["review_id"])
    )
    result = await session.execute(stmt)
    return int(result.rowcount or 0)


async def _upsert_card(
    session: AsyncSession,
    *,
    deck_name: str,
    anki_card_id: int,
    card_data: dict[str, Any],
    now: datetime,
) -> AnkiCard:
    existing = (
        await session.execute(
            select(AnkiCard).where(
                AnkiCard.deck_name == deck_name,
                AnkiCard.anki_card_id == anki_card_id,
            )
        )
    ).scalar_one_or_none()

    payload = _payload_from_anki(card_data)

    if existing is None:
        new_row = AnkiCard(
            deck_name=deck_name,
            anki_card_id=anki_card_id,
            sync_at=now,
            **payload,
        )
        session.add(new_row)
        return new_row

    if _payload_differs(existing, payload):
        for k, v in payload.items():
            setattr(existing, k, v)
    existing.sync_at = now
    return existing


def _payload_from_anki(card_data: dict[str, Any]) -> dict[str, Any]:
    queue = card_data.get("queue")
    interval_days = card_data.get("interval")
    return dict(
        note_id=card_data.get("note"),
        due_date=_compute_due_date(queue, interval_days),
        interval_days=interval_days,
        ease=card_data.get("factor"),
        lapses=card_data.get("lapses"),
        queue=queue,
    )


def _payload_differs(existing: AnkiCard, payload: dict[str, Any]) -> bool:
    for field in _MIRRORED_FIELDS:
        if getattr(existing, field) != payload[field]:
            return True
    return False


def _compute_due_date(queue: Optional[int], interval_days: Optional[int]) -> Optional[date]:
    if queue is None:
        return None
    if queue == 2 and interval_days is not None:
        return date.today() + timedelta(days=max(0, interval_days))
    if queue in (1, 3):
        return date.today()
    return None


async def _replace_tags(
    session: AsyncSession,
    *,
    note_id: int,
    tag_strings: list[str],
    outline_lookup: OutlineLookup,
) -> int:
    """Diff a NOTE's regex tag set vs incoming, applying deletes + inserts (§V75).

    Note-scoped per §V75: one tag set per note (`anki_note_tags`), not the
    pre-T93 per-card fan-out. Per §V43 sync still only owns rows it itself
    wrote (`source='regex'`); LLM-resolver (`source='llm'`) + future
    manual-override (`source='manual'`) rows are scoped out of both the
    deletion sweep AND the dedupe-by-raw insert guard. Their synthetic
    `tag_raw` values (e.g. `__llm_topic__::v5-…::<topic_path>`) never appear
    in real AnkiConnect tag lists, so without the source filter the diff loop
    would wipe every LLM-resolved topic row on every sync (B9).

    Returns the number of `parsed_kind='uworld_qid'` rows (regex-sourced) on
    this note — `SyncSummary.linked_qids` sums it across notes (deduped, not
    multiplied by the note's card count as the pre-T93 per-card path did).
    """
    existing_rows = (
        (
            await session.execute(
                select(AnkiNoteTag).where(
                    AnkiNoteTag.note_id == note_id,
                    AnkiNoteTag.source == "regex",
                )
            )
        )
        .scalars()
        .all()
    )
    existing_by_raw = {t.tag_raw: t for t in existing_rows}
    incoming = set(tag_strings)

    for raw, row in existing_by_raw.items():
        if raw not in incoming:
            await session.delete(row)

    qid_count = 0
    for raw in tag_strings:
        if raw in existing_by_raw:
            existing_row = existing_by_raw[raw]
            # Re-parse rows that previously landed as `unparsed`. The §V3
            # regex evolves as new tag sources (AnKing → others) are
            # supported; without this branch, rows ingested under an older
            # regex stay unparsed forever even after the parser catches up.
            if existing_row.parsed_kind == "unparsed":
                reparsed = parse_tag(raw, outline_lookup=outline_lookup)
                if reparsed.parsed_kind != "unparsed":
                    existing_row.parsed_kind = reparsed.parsed_kind
                    existing_row.topic_id = reparsed.topic_id
                    existing_row.content_category_id = reparsed.content_category_id
                    existing_row.skill_number = reparsed.skill_number
                    existing_row.question_qid = reparsed.question_qid
            if existing_row.parsed_kind == "uworld_qid":
                qid_count += 1
            continue
        parsed = parse_tag(raw, outline_lookup=outline_lookup)
        session.add(
            AnkiNoteTag(
                note_id=note_id,
                tag_raw=parsed.tag_raw,
                topic_id=parsed.topic_id,
                content_category_id=parsed.content_category_id,
                skill_number=parsed.skill_number,
                question_qid=parsed.question_qid,
                parsed_kind=parsed.parsed_kind,
                source="regex",
            )
        )
        if parsed.parsed_kind == "uworld_qid":
            qid_count += 1

    return qid_count
