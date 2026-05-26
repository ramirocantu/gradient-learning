"""Windowed Anki "true retention" per CC + per topic (SPEC §T37 / §V27 / §V31).

Reads exclusively from the local `anki_card_reviews` table populated by
T36's incremental sync (`startID = MAX(review_id) + 1`). No live
AnkiConnect call at read time per §V27.

Definitions (mirroring Anki desktop's "true retention" stat):
- Pass = `ease ∈ {2,3,4}` (Hard / Good / Easy).
- Fail = `ease = 1` (Again).
- Excludes `type = 'learn'` from both numerator and denominator —
  initial-acquisition reviews are not a fair test of retention.
  `'review'`, `'relearn'`, `'cram'` all count.

Scope rollup is subtree-membership per §V31: a review counts iff the
reviewed card carries at least one `anki_card_tags` row pointing into
the scope. `EXISTS` keeps each review row counted once even when a
card has multiple tag rows that both fall in scope (e.g. an aamc_cc
row and an aamc_topic row resolving under the same CC).

Window semantics:
- `7` / `30` → `reviewed_at >= now() - interval 'N days'`.
- `0` → all-time (no time filter).

Each call issues one query per scope; all requested windows are computed
via `count(*) FILTER (WHERE …)` in a single round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_WINDOWS: tuple[int, ...] = (7, 30, 0)


@dataclass(frozen=True)
class RetentionWindow:
    window_days: int
    pass_count: int
    fail_count: int

    @property
    def total(self) -> int:
        return self.pass_count + self.fail_count

    @property
    def retention(self) -> float | None:
        return None if self.total == 0 else self.pass_count / self.total


@dataclass(frozen=True)
class RetentionSummary:
    scope: str
    windows: dict[int, RetentionWindow]


def _build_window_select(windows: tuple[int, ...]) -> str:
    """Render `count(*) FILTER (...)` projections for each requested window.

    Per window N: emits `pass_<N>` + `fail_<N>` columns. `N=0` means
    all-time (no `reviewed_at` predicate). Pass = ease ∈ {2,3,4};
    fail = ease = 1; the type='learn' exclusion is in the outer WHERE.
    """
    cols: list[str] = []
    for n in windows:
        if n == 0:
            cols.append("count(*) FILTER (WHERE r.ease IN (2,3,4)) AS pass_0")
            cols.append("count(*) FILTER (WHERE r.ease = 1) AS fail_0")
        else:
            cols.append(
                f"count(*) FILTER (WHERE r.ease IN (2,3,4) "
                f"AND r.reviewed_at >= now() - interval '{int(n)} days') AS pass_{int(n)}"
            )
            cols.append(
                f"count(*) FILTER (WHERE r.ease = 1 "
                f"AND r.reviewed_at >= now() - interval '{int(n)} days') AS fail_{int(n)}"
            )
    return ",\n  ".join(cols)


def _windows_from_row(row, windows: tuple[int, ...]) -> dict[int, RetentionWindow]:
    out: dict[int, RetentionWindow] = {}
    mapping = row._mapping
    for n in windows:
        out[n] = RetentionWindow(
            window_days=n,
            pass_count=int(mapping[f"pass_{int(n)}"]),
            fail_count=int(mapping[f"fail_{int(n)}"]),
        )
    return out


async def retention_for_cc(
    session: AsyncSession,
    *,
    cc_code: str,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> RetentionSummary:
    """True retention over reviews of cards linked to the given CC.

    A card is "in scope for CC X" when it carries either:
    - a tag row with `content_category_id = X` (parsed_kind='aamc_cc'),
      i.e. AnKing's direct CC-level tag, or
    - a tag row with `topic_id` pointing at a topic whose
      `content_category_id = X` (parsed_kind='aamc_topic'), i.e. the
      LLM topic resolver's per-CC topic assignments.

    Mirrors `queries.list_cards_for_cc` for consistency.
    """
    sql = text(
        f"""
        SELECT
          {_build_window_select(windows)}
        FROM anki_card_reviews r
        WHERE r.type <> 'learn'
          AND EXISTS (
            SELECT 1
            FROM anki_note_tags t
            JOIN anki_cards c ON c.note_id = t.note_id
            LEFT JOIN topics tp ON tp.id = t.topic_id
            JOIN content_categories cc ON cc.code = :cc_code
            WHERE c.id = r.card_id
              AND (t.content_category_id = cc.id OR tp.content_category_id = cc.id)
          )
        """
    )
    result = await session.execute(sql, {"cc_code": cc_code})
    row = result.one()
    return RetentionSummary(scope=f"cc:{cc_code}", windows=_windows_from_row(row, windows))


async def retention_for_topic(
    session: AsyncSession,
    *,
    topic_id: int,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> RetentionSummary:
    """True retention over reviews of cards linked to any topic in the subtree.

    Subtree-membership per §V31: a card is in scope iff at least one of
    its `anki_card_tags` rows has `topic_id ∈ subtree(topic_id)`. The
    subtree is rendered via recursive CTE over `topics.parent_topic_id`
    so internal nodes aggregate every descendant's reviews along with
    items tagged directly at the node.

    Cards tagged only at CC granularity (parsed_kind='aamc_cc',
    `topic_id IS NULL`) do not count for topic-scoped retention — they
    have no topic-level resolution yet (T32 LLM resolver writes the
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
          {_build_window_select(windows)}
        FROM anki_card_reviews r
        WHERE r.type <> 'learn'
          AND EXISTS (
            SELECT 1
            FROM anki_note_tags t
            JOIN anki_cards c ON c.note_id = t.note_id
            WHERE c.id = r.card_id
              AND t.topic_id IN (SELECT id FROM subtree)
          )
        """
    )
    result = await session.execute(sql, {"topic_id": topic_id})
    row = result.one()
    return RetentionSummary(scope=f"topic:{topic_id}", windows=_windows_from_row(row, windows))
