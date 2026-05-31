"""Review service — V53 amended 2026-05-23 per T76.

`create_review` inserts a pending `anki_reviews` row whose `deck_name`
follows the V53-amended convention `<ANKI_DECK_PREFIX>::review::{id}`
(derived from the row's own PK, so it is unique by construction and
needs no `(date, slug)` UNIQUE constraint).

`run_review_due` is the scheduler-facing batch that fires
`AnkiConnectClient.create_filtered_deck` for every pending review whose
`review_date ≤ today`, then chains `add_tags(card_ids, [f"coach::review:{id}"])`
per V50 to leave an audit-trail tag on the source-deck cards. Both
AnkiConnect calls record their own `anki_writes` row keyed by `review_id`.
V55 retry-with-cap semantics apply to the createFilteredDeck call;
addTags failure ⊥ revert review status or bump failure_count (the tag
is write-only audit-trail, ⊥ load-bearing on review lifecycle — same
contract as T75's unlock chain).

Filtered decks self-clear in Anki once emptied; mcat-coach never
proactively deletes them (V50 forbids `deleteDecks`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.anki import AnkiCard, AnkiReview, AnkiWrite
from app.services.anki.client import (
    AnkiConnectClient,
    AnkiConnectError,
    AnkiUnreachableError,
    AnkiWriteFailed,
)

logger = logging.getLogger(__name__)


_REVIEW_FAILURE_CAP = 3
_REVIEW_RETRYABLE_ERRORS = (AnkiUnreachableError, AnkiWriteFailed, AnkiConnectError)


class ReviewError(RuntimeError):
    """Base class for review service errors."""


def _deck_name(prefix: str, review_id: int) -> str:
    """V53 amended (T76): filtered-deck name derived from the row's own
    PK. Each review row gets a fresh deck name; re-creating a review
    produces a new row + new id + new deck."""
    return f"{prefix}::review::{review_id}"


def _payload_hash_for_filtered_deck(name: str, card_ids: list[int]) -> str:
    encoded = json.dumps(
        {"name": name, "card_ids": sorted(int(c) for c in card_ids)},
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _payload_hash_for_add_tags(note_ids: list[int], tags: list[str]) -> str:
    """§V75: addTags targets notes (`notes=...`), so the hash keys on note ids."""
    encoded = json.dumps(
        {"action": "addTags", "notes": sorted(int(n) for n in note_ids), "tags": sorted(tags)},
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


async def create_review(
    session: AsyncSession,
    *,
    card_ids: list[int],
    review_date: date,
    write_deck_prefix: Optional[str] = None,
) -> AnkiReview:
    """Insert a pending `anki_reviews` row.

    Two-flush dance: first add+flush assigns the PK; second sets
    `deck_name` derived from that PK and flushes again. `deck_name` is
    NOT NULL so the dance is required — we can't insert a row without
    knowing its own id first.

    Re-creating a review (same card_ids, same date) creates a NEW row
    with a NEW id and a NEW deck name per V53 amend — tags-as-log
    accepts dup reviews; idempotency lives in UI debounce, ⊥ DB.
    """
    prefix = write_deck_prefix if write_deck_prefix is not None else settings.ANKI_DECK_PREFIX
    native_ids = [int(c) for c in card_ids]
    # §V75: derive the parent note_ids for these native card_ids so the
    # push-time addTags audit write can target notes (notes=...). Cards not in
    # anki_cards (unsynced) simply contribute no note; the filtered deck still
    # uses card_ids via the cid: query regardless.
    note_ids: list[int] = []
    if native_ids:
        rows = (
            (
                await session.execute(
                    select(AnkiCard.note_id)
                    .where(AnkiCard.anki_card_id.in_(native_ids))
                    .where(AnkiCard.note_id.is_not(None))
                    .distinct()
                )
            )
            .scalars()
            .all()
        )
        note_ids = [int(n) for n in rows if n is not None]
    review = AnkiReview(
        review_date=review_date,
        card_ids=native_ids,
        note_ids=note_ids,
        deck_name="",  # placeholder; replaced after PK assignment below
        status="pending",
    )
    session.add(review)
    await session.flush()  # assigns review.id
    review.deck_name = _deck_name(prefix, review.id)
    await session.flush()
    return review


@dataclass
class ReviewBatchSummary:
    processed: int = 0
    pushed: int = 0
    failed: int = 0
    terminal_failed: int = 0
    errors: list[str] = field(default_factory=list)


async def run_review_due(
    session: AsyncSession,
    anki_client: AnkiConnectClient,
    *,
    today: Optional[date] = None,
) -> ReviewBatchSummary:
    """T76 scheduler core per V53 amended + V55.

    Loops pending reviews WHERE `review_date ≤ today`. For each:
      * Call `anki_client.create_filtered_deck(deck_name, card_ids)`.
      * On success: insert `anki_writes(createFilteredDeck, succeeded)`,
        flip `status='pushed'`, stamp `pushed_at`. Then chain
        `anki_client.add_tags(card_ids, [f"coach::review:{review.id}"])` per
        V50; insert second `anki_writes(addTags, ...)` row. addTags
        failure ⊥ revert status or bump failure_count (audit-only).
      * On V55-retryable createFilteredDeck failure: insert
        `anki_writes(createFilteredDeck, failed)`, bump `failure_count`;
        ≥3 → terminal `status='failed'` + `error_text`. ⊥ chain addTags.
    Per-row commit so transient mid-batch failures don't roll back
    earlier successes.
    """
    today = today or datetime.now(timezone.utc).date()
    pending = (
        (
            await session.execute(
                select(AnkiReview)
                .where(AnkiReview.status == "pending")
                .where(AnkiReview.review_date <= today)
                .order_by(AnkiReview.review_date.asc(), AnkiReview.id.asc())
            )
        )
        .scalars()
        .all()
    )

    summary = ReviewBatchSummary(processed=len(pending))
    for review in pending:
        card_ids = list(review.card_ids or [])
        payload_hash = _payload_hash_for_filtered_deck(review.deck_name, card_ids)
        try:
            deck_id = await anki_client.create_filtered_deck(review.deck_name, card_ids)
        except _REVIEW_RETRYABLE_ERRORS as exc:
            err_text = str(exc)[:2000]
            session.add(
                AnkiWrite(
                    action="createFilteredDeck",
                    payload_hash=payload_hash,
                    response_json=None,
                    status="failed",
                    error_text=err_text,
                    source="scheduler",
                    review_id=review.id,
                )
            )
            review.failure_count = (review.failure_count or 0) + 1
            terminal = review.failure_count >= _REVIEW_FAILURE_CAP
            if terminal:
                review.status = "failed"
                review.error_text = err_text
                summary.terminal_failed += 1
            summary.failed += 1
            summary.errors.append(err_text)
            await session.commit()
            logger.warning(
                "anki_review id=%s failed (count=%d terminal=%s): %s",
                review.id,
                review.failure_count,
                terminal,
                err_text,
            )
            continue

        session.add(
            AnkiWrite(
                action="createFilteredDeck",
                payload_hash=payload_hash,
                response_json={"deck_id": int(deck_id)},
                status="succeeded",
                error_text=None,
                source="scheduler",
                review_id=review.id,
            )
        )
        review.status = "pushed"
        review.pushed_at = datetime.now(timezone.utc)

        # T76: chain addTags audit-trail write per V50. Failure here ⊥
        # revert status or bump failure_count — the tag is a write-only
        # audit marker (matches T75's unlock chain contract). Empty
        # note_ids → skip both the call (AnkiConnect add_tags short-
        # circuits anyway, client.py §V75) and the audit row, since an
        # AnkiWrite(succeeded) for a no-op would falsely claim a tag
        # write that never reached Anki. AnkiWrite.status CHECK only
        # allows succeeded|failed, so no "skipped" sentinel row.
        note_ids = list(review.note_ids or [])
        if note_ids:
            tag_value = f"coach::review:{review.id}"
            tag_payload_hash = _payload_hash_for_add_tags(note_ids, [tag_value])
            try:
                await anki_client.add_tags(note_ids, [tag_value])
            except _REVIEW_RETRYABLE_ERRORS as exc:
                tag_err_text = str(exc)[:2000]
                session.add(
                    AnkiWrite(
                        action="addTags",
                        payload_hash=tag_payload_hash,
                        response_json=None,
                        status="failed",
                        error_text=tag_err_text,
                        source="scheduler",
                        review_id=review.id,
                    )
                )
                logger.warning(
                    "anki_review id=%s addTags failed (status=pushed retained, ⊥ load-bearing): %s",
                    review.id,
                    tag_err_text,
                )
            else:
                session.add(
                    AnkiWrite(
                        action="addTags",
                        payload_hash=tag_payload_hash,
                        response_json={"tags": [tag_value]},
                        status="succeeded",
                        error_text=None,
                        source="scheduler",
                        review_id=review.id,
                    )
                )
        else:
            logger.info(
                "anki_review id=%s addTags skipped: empty note_ids (no audit row written)",
                review.id,
            )

        await session.commit()
        summary.pushed += 1
        logger.info(
            "anki_review id=%s pushed deck=%s cards=%d",
            review.id,
            review.deck_name,
            len(card_ids),
        )

    return summary


__all__ = [
    "ReviewBatchSummary",
    "ReviewError",
    "create_review",
    "run_review_due",
]
