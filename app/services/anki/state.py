"""Subtree Anki state counts per CC + per topic (SPEC §T38 / §V28 / §V31).

Reads exclusively from `anki_cards` (no live AnkiConnect call). Each
card carries a `queue` (Anki scheduler state) + `interval_days` captured
at sync time per §V2; that snapshot is the source of truth at read time.

Bucket definitions (§C + §V28 — mirror Anki desktop, do not invent
custom buckets):

| bucket    | predicate                                                |
|-----------|----------------------------------------------------------|
| suspended | `queue = -1`                                             |
| new       | `queue = 0`                                              |
| learning  | `queue IN (1, 3)`  — q=1 learning, q=3 day-learn         |
| young     | `queue = 2 AND COALESCE(interval_days, 0) <  21`         |
| mature    | `queue = 2 AND COALESCE(interval_days, 0) >= 21`         |
| assigned  | `queue >= 0`  — every non-suspended card                 |

q=3 day-learn folds into the `learning` bucket because §V28 forbids
custom buckets and Anki itself groups day-learning under learning in
its scheduler. Buried cards (queue ∈ {-2, -3}) appear in `total_cards`
but in no labelled bucket — they're transient and clear at day rollover.

`unlock_pct = assigned / total_cards`. Total = every card in scope
including suspended + buried, so unlock% is bounded ≤ 1.0 and reflects
"how much of this scope has the user actually engaged with".

Scope rollup is subtree-membership per §V31. A card is in scope iff at
least one of its `anki_card_tags` rows points into the scope. `EXISTS`
keeps each card counted once even when multiple tag rows fall in scope
(e.g. an aamc_cc row + an aamc_topic row resolving under the same CC).
Each `state_for_*` call issues one query per scope; all buckets are
computed via `count(*) FILTER (WHERE …)` in a single round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class StateCounts:
    scope: str
    total_cards: int
    assigned: int
    suspended: int
    new: int
    learning: int
    young: int
    mature: int

    @property
    def unlock_pct(self) -> float | None:
        return None if self.total_cards == 0 else self.assigned / self.total_cards


_BUCKET_PROJECTIONS = """
  count(*) AS total_cards,
  count(*) FILTER (WHERE c.queue >= 0) AS assigned,
  count(*) FILTER (WHERE c.queue = -1) AS suspended,
  count(*) FILTER (WHERE c.queue = 0) AS new_count,
  count(*) FILTER (WHERE c.queue IN (1, 3)) AS learning,
  count(*) FILTER (WHERE c.queue = 2
                   AND COALESCE(c.interval_days, 0) < 21) AS young,
  count(*) FILTER (WHERE c.queue = 2
                   AND COALESCE(c.interval_days, 0) >= 21) AS mature
"""


def _counts_from_row(row, *, scope: str) -> StateCounts:
    m = row._mapping
    return StateCounts(
        scope=scope,
        total_cards=int(m["total_cards"]),
        assigned=int(m["assigned"]),
        suspended=int(m["suspended"]),
        new=int(m["new_count"]),
        learning=int(m["learning"]),
        young=int(m["young"]),
        mature=int(m["mature"]),
    )


async def state_for_cc(session: AsyncSession, *, cc_code: str) -> StateCounts:
    """State bucket counts over cards linked to the given CC.

    A card is "in scope for CC X" when it carries either:
    - a tag row with `content_category_id = X` (parsed_kind='aamc_cc'),
      i.e. AnKing's direct CC-level tag, or
    - a tag row with `topic_id` pointing at a topic whose
      `content_category_id = X` (parsed_kind='aamc_topic'), i.e. the
      LLM topic resolver's per-CC topic assignments.

    Mirrors `retention_for_cc` + `queries.list_cards_for_cc`.
    """
    sql = text(
        f"""
        SELECT
          {_BUCKET_PROJECTIONS}
        FROM anki_cards c
        WHERE EXISTS (
            SELECT 1
            FROM anki_note_tags t
            LEFT JOIN topics tp ON tp.id = t.topic_id
            JOIN content_categories cc ON cc.code = :cc_code
            WHERE t.note_id = c.note_id
              AND (t.content_category_id = cc.id OR tp.content_category_id = cc.id)
        )
        """
    )
    row = (await session.execute(sql, {"cc_code": cc_code})).one()
    return _counts_from_row(row, scope=f"cc:{cc_code}")


async def state_for_topic(session: AsyncSession, *, topic_id: int) -> StateCounts:
    """State bucket counts over cards linked to any topic in the subtree.

    Subtree-membership per §V31: a card is in scope iff at least one of
    its `anki_card_tags` rows has `topic_id ∈ subtree(topic_id)`. The
    subtree is rendered via recursive CTE over `topics.parent_topic_id`
    so internal nodes aggregate every descendant's cards along with
    items tagged directly at the node.

    Cards tagged only at CC granularity (parsed_kind='aamc_cc',
    `topic_id IS NULL`) do not count for topic-scoped state — they have
    no topic-level resolution yet (T32 LLM resolver writes the
    aamc_topic rows that would link them in).
    """
    sql = text(
        f"""
        WITH RECURSIVE subtree(id) AS (
            SELECT id FROM topics WHERE id = :topic_id
            UNION ALL
            SELECT child.id
            FROM topics child
            JOIN subtree s ON child.parent_topic_id = s.id
        )
        SELECT
          {_BUCKET_PROJECTIONS}
        FROM anki_cards c
        WHERE EXISTS (
            SELECT 1
            FROM anki_note_tags t
            WHERE t.note_id = c.note_id
              AND t.topic_id IN (SELECT id FROM subtree)
        )
        """
    )
    row = (await session.execute(sql, {"topic_id": topic_id})).one()
    return _counts_from_row(row, scope=f"topic:{topic_id}")
