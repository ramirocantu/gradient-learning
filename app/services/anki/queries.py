"""Read queries against anki_cards / anki_note_tags.

Partially FENCED (T18, V-RB2, V-O5). The topic/CC-subtree query helpers
(`list_cards_for_topic`, `list_cards_for_cc`, `list_review_queue_for_cc`,
`list_review_queue_for_topic_subtree`, `due_count_for_subtree`) are
FENCED — their consumers (`/api/v1/anki/cards?topic_id=...`, the
dashboard topics/mastery surfaces) are route-disabled per T17/T18.
Restoration is tied to the node_id subtree-set port (post-P0.5,
candidate for P3 or T34 SPA reassessment).

The outline-free helpers (`list_review_queue`, `list_cards_for_qid`,
`get_tag_parse_stats`, `get_tag_card_coverage`, `get_anki_card_total`)
remain real — they work unchanged on the canonical `anki_note_tags`
and back the still-mounted anki dashboard / API routes.

This file is partially FENCED, not a stub: behavior is deliberate, not
in-progress.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.anki import AnkiCard, AnkiNoteTag

logger = logging.getLogger(__name__)


_MIN_LIMIT = 1
_MAX_LIMIT = 200


def _clamp_limit(limit: int) -> int:
    return max(_MIN_LIMIT, min(_MAX_LIMIT, limit))


# --------------------------------------------------------------------------- #
# Subtree-scope helpers — FENCED (T18, V-RB2, V-O5).
# --------------------------------------------------------------------------- #


_FENCED_MSG = (
    "anki.queries subtree-scope helpers are FENCED (T18, V-RB2) — "
    "consumer routes unmounted; restoration pending node_id subtree port"
)


async def list_cards_for_topic(
    session: AsyncSession, *, topic_id: int, limit: int = 50
) -> list[AnkiCard]:
    """FENCED — returns empty list. See module docstring."""
    logger.warning(_FENCED_MSG)
    return []


async def list_cards_for_cc(
    session: AsyncSession, *, cc_code: str, limit: int = 20
) -> list[AnkiCard]:
    """FENCED — returns empty list. See module docstring."""
    logger.warning(_FENCED_MSG)
    return []


async def list_review_queue_for_cc(
    session: AsyncSession,
    *,
    cc_code: str,
    due_before: datetime,
    limit: int = 20,
) -> list[AnkiCard]:
    """FENCED — returns empty list. See module docstring."""
    logger.warning(_FENCED_MSG)
    return []


async def list_review_queue_for_topic_subtree(
    session: AsyncSession,
    *,
    topic_id: int,
    due_before: datetime,
    limit: int = 20,
) -> list[AnkiCard]:
    """FENCED — returns empty list. See module docstring."""
    logger.warning(_FENCED_MSG)
    return []


async def due_count_for_subtree(
    session: AsyncSession,
    *,
    topic_id: int,
    due_before: datetime,
) -> int:
    """FENCED — returns 0. See module docstring."""
    logger.warning(_FENCED_MSG)
    return 0


# --------------------------------------------------------------------------- #
# Outline-free helpers — preserved (work unchanged on canonical anki_note_tags).
# --------------------------------------------------------------------------- #


async def list_review_queue(session: AsyncSession, *, limit: int = 50) -> list[AnkiCard]:
    """Cards with a scheduled `due_date`, soonest first.

    Cards with `due_date IS NULL` (new and suspended) excluded — the queue
    surfaces material the spaced-repetition scheduler has actually scheduled.
    """
    limit = _clamp_limit(limit)
    stmt = (
        select(AnkiCard)
        .where(AnkiCard.due_date.is_not(None))
        .options(selectinload(AnkiCard.tags))
        .order_by(AnkiCard.due_date.asc(), AnkiCard.id.asc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).unique().scalars().all())


async def get_tag_parse_stats(session: AsyncSession) -> dict[str, int]:
    """Return `{parsed_kind: count}` over all `anki_note_tags` rows."""
    stmt = select(AnkiNoteTag.parsed_kind, func.count()).group_by(AnkiNoteTag.parsed_kind)
    result = await session.execute(stmt)
    return {kind: int(count) for kind, count in result.all()}


async def get_tag_card_coverage(session: AsyncSession) -> dict[str, int]:
    """Return `{parsed_kind: distinct_card_count}` joined via `note_id`."""
    stmt = (
        select(AnkiNoteTag.parsed_kind, func.count(func.distinct(AnkiCard.id)))
        .join(AnkiCard, AnkiCard.note_id == AnkiNoteTag.note_id)
        .group_by(AnkiNoteTag.parsed_kind)
    )
    result = await session.execute(stmt)
    return {kind: int(count) for kind, count in result.all()}


async def get_anki_card_total(session: AsyncSession) -> int:
    """Total count of `anki_cards` rows."""
    return int((await session.execute(select(func.count()).select_from(AnkiCard))).scalar_one())


async def list_cards_for_qid(session: AsyncSession, *, qid: str) -> list[AnkiCard]:
    stmt = (
        select(AnkiCard)
        .join(AnkiNoteTag, AnkiNoteTag.note_id == AnkiCard.note_id)
        .where(AnkiNoteTag.question_qid == qid)
        .options(selectinload(AnkiCard.tags))
        .order_by(AnkiCard.id.asc())
    )
    return list((await session.execute(stmt)).unique().scalars().all())
