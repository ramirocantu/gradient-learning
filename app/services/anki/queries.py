"""Read queries against anki_cards / anki_note_tags (SPEC §T5 + §T28; §V75).

Each `list_*` helper eager-loads `AnkiCard.tags` via `selectinload` so callers
serialise the response in one round trip — no lazy-load N+1. Post-§V75 a
card's tags live on its note (`anki_note_tags`); `AnkiCard.tags` is a viewonly
relationship through the shared `note_id`, and the topic/CC scope joins reach
the tag rows the same way (`AnkiNoteTag.note_id == AnkiCard.note_id`).

`get_tag_parse_stats` (SPEC §T28, V19) returns the `parsed_kind` distribution
across `anki_note_tags`. Surfaced on `/admin` so the unparsed-rate against the
real MileDown corpus is observable — the primary signal for whether the V3
regex/path-converter holds vs needs amend.
"""

from __future__ import annotations

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.anki import AnkiCard, AnkiNoteTag
from app.models.outline import ContentCategory, Topic


_MIN_LIMIT = 1
_MAX_LIMIT = 200


def _clamp_limit(limit: int) -> int:
    return max(_MIN_LIMIT, min(_MAX_LIMIT, limit))


async def list_cards_for_topic(
    session: AsyncSession, *, topic_id: int, limit: int = 50
) -> list[AnkiCard]:
    limit = _clamp_limit(limit)
    stmt = (
        select(AnkiCard)
        .join(AnkiNoteTag, AnkiNoteTag.note_id == AnkiCard.note_id)
        .where(AnkiNoteTag.topic_id == topic_id)
        .options(selectinload(AnkiCard.tags))
        .order_by(AnkiCard.due_date.asc().nullslast(), AnkiCard.id.asc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).unique().scalars().all())


async def list_cards_for_cc(
    session: AsyncSession, *, cc_code: str, limit: int = 20
) -> list[AnkiCard]:
    """Cards whose note's parsed tag set links to this CC directly or via topic.

    Two paths matter post-T31:
    - Direct CC link: `anki_note_tags.content_category_id = <cc.id>`
      (parsed_kind='aamc_cc'). This is where AnKing's AAMC tags land —
      AnKing tops out at CC granularity.
    - Topic→CC: `anki_note_tags.topic_id → topics.id → content_categories`
      (parsed_kind='aamc_topic'). The T32 LLM resolver writes these.

    Distinct on `AnkiCard.id` because one note can carry both kinds of link
    (an aamc_cc row + an aamc_topic row) and a note backs ≥1 card.
    """
    limit = _clamp_limit(limit)
    target_cc_id_subq = select(ContentCategory.id).where(ContentCategory.code == cc_code)
    stmt = (
        select(AnkiCard)
        .join(AnkiNoteTag, AnkiNoteTag.note_id == AnkiCard.note_id)
        .outerjoin(Topic, Topic.id == AnkiNoteTag.topic_id)
        .where(
            (AnkiNoteTag.content_category_id.in_(target_cc_id_subq))
            | (Topic.content_category_id.in_(target_cc_id_subq))
        )
        .options(selectinload(AnkiCard.tags))
        .order_by(AnkiCard.due_date.asc().nullslast(), AnkiCard.id.asc())
        .distinct()
        .limit(limit)
    )
    return list((await session.execute(stmt)).unique().scalars().all())


async def list_review_queue_for_cc(
    session: AsyncSession,
    *,
    cc_code: str,
    due_before,
    limit: int = 20,
) -> list[AnkiCard]:
    """Due-soon Anki cards scoped to a single CC (subtree + direct-CC link).

    `due_before` is an inclusive upper bound on `AnkiCard.due_date` — pass
    `datetime.now(tz=UTC) + timedelta(days=1)` for the standard "due
    today" view on `/mastery/{cc}` per §V34. Cards without a scheduled
    `due_date` (new / suspended) are excluded so the queue only shows
    material the scheduler has actually queued up.
    """
    limit = _clamp_limit(limit)
    target_cc_id_subq = select(ContentCategory.id).where(ContentCategory.code == cc_code)
    stmt = (
        select(AnkiCard)
        .join(AnkiNoteTag, AnkiNoteTag.note_id == AnkiCard.note_id)
        .outerjoin(Topic, Topic.id == AnkiNoteTag.topic_id)
        .where(
            (AnkiNoteTag.content_category_id.in_(target_cc_id_subq))
            | (Topic.content_category_id.in_(target_cc_id_subq))
        )
        .where(AnkiCard.due_date.is_not(None))
        .where(AnkiCard.due_date <= due_before)
        .options(selectinload(AnkiCard.tags))
        .order_by(AnkiCard.due_date.asc(), AnkiCard.id.asc())
        .distinct()
        .limit(limit)
    )
    return list((await session.execute(stmt)).unique().scalars().all())


async def list_review_queue(session: AsyncSession, *, limit: int = 50) -> list[AnkiCard]:
    """Cards with a scheduled `due_date`, soonest first.

    Cards with `due_date IS NULL` (new and suspended cards) are excluded —
    the "review queue" surfaces material the spaced-repetition scheduler
    has actually scheduled, so the tutor can suggest "knock these out
    before your next UWorld block".
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


async def list_review_queue_for_topic_subtree(
    session: AsyncSession,
    *,
    topic_id: int,
    due_before,
    limit: int = 20,
) -> list[AnkiCard]:
    """Due-soon Anki cards scoped to the subtree of `topic_id` (§V33).

    Subtree-membership per §V31 — recursive CTE on
    `topics.parent_topic_id`. Cards without `due_date` (new / suspended)
    excluded so the queue surfaces material the scheduler has scheduled.
    Direct-CC tags do NOT contribute — topic subtree scopes are
    topic-tag-only. A card is in scope iff its note carries a matching tag
    (§V75: `anki_note_tags.note_id = anki_cards.note_id`).
    """
    limit = _clamp_limit(limit)
    subtree_cte = text(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT id FROM topics WHERE id = :topic_id
            UNION ALL
            SELECT child.id
            FROM topics child
            JOIN subtree s ON child.parent_topic_id = s.id
        )
        SELECT DISTINCT c.id
        FROM anki_cards c
        JOIN anki_note_tags t ON t.note_id = c.note_id
        WHERE c.due_date IS NOT NULL
          AND c.due_date <= :due_before
          AND t.topic_id IN (SELECT id FROM subtree)
        ORDER BY c.id
        """
    )
    card_id_rows = (
        await session.execute(subtree_cte, {"topic_id": topic_id, "due_before": due_before})
    ).all()
    card_ids = [int(r[0]) for r in card_id_rows]
    if not card_ids:
        return []
    stmt = (
        select(AnkiCard)
        .where(AnkiCard.id.in_(card_ids))
        .options(selectinload(AnkiCard.tags))
        .order_by(AnkiCard.due_date.asc(), AnkiCard.id.asc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).unique().scalars().all())


async def due_count_for_subtree(
    session: AsyncSession,
    *,
    topic_id: int,
    due_before,
) -> int:
    """Count cards due on/before `due_before` whose note links to any topic in
    the subtree.

    Subtree-membership per §V31 — recursive CTE walks
    `topics.parent_topic_id` from `topic_id` down. Direct-CC tags do
    not count for topic scopes (no topic-level resolution).
    """
    sql = text(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT id FROM topics WHERE id = :topic_id
            UNION ALL
            SELECT child.id
            FROM topics child
            JOIN subtree s ON child.parent_topic_id = s.id
        )
        SELECT count(DISTINCT c.id)
        FROM anki_cards c
        JOIN anki_note_tags t ON t.note_id = c.note_id
        WHERE c.due_date IS NOT NULL
          AND c.due_date <= :due_before
          AND t.topic_id IN (SELECT id FROM subtree)
        """
    )
    result = await session.execute(sql, {"topic_id": topic_id, "due_before": due_before})
    return int(result.scalar_one())


async def get_tag_parse_stats(session: AsyncSession) -> dict[str, int]:
    """Return `{parsed_kind: count}` over all `anki_note_tags` rows.

    Empty table → `{}`. Per SPEC §V19, this powers the admin tag-parse
    health widget — the unparsed share is the headline signal for whether
    the V3 tag-regex + path-converter is matching real AnKing shape. Post
    §V75 the counts are per note-tag row (deduped), no longer fanned out
    across each note's cards.
    """
    stmt = select(AnkiNoteTag.parsed_kind, func.count()).group_by(AnkiNoteTag.parsed_kind)
    result = await session.execute(stmt)
    return {kind: int(count) for kind, count in result.all()}


async def get_tag_card_coverage(session: AsyncSession) -> dict[str, int]:
    """Return `{parsed_kind: distinct_card_count}` (§V23).

    Card-level coverage = distinct anki_cards whose note carries a tag of
    that parsed_kind (join `anki_note_tags.note_id = anki_cards.note_id`).
    Surfaced alongside the per-row distribution because the two diverge on
    AnKing-shape decks: a note averages ~9 tags so per-row aamc_cc share
    (~15%) looks low while per-card aamc_cc coverage is ~97%.
    """
    stmt = (
        select(AnkiNoteTag.parsed_kind, func.count(func.distinct(AnkiCard.id)))
        .join(AnkiCard, AnkiCard.note_id == AnkiNoteTag.note_id)
        .group_by(AnkiNoteTag.parsed_kind)
    )
    result = await session.execute(stmt)
    return {kind: int(count) for kind, count in result.all()}


async def get_anki_card_total(session: AsyncSession) -> int:
    """Total count of `anki_cards` rows. Used as the denominator for
    card-level parsed_kind coverage in the admin widget (§V23)."""
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
