"""Windowed accuracy trajectory per CC + per topic (SPEC §T40 / §V36 / §V31).

Reads exclusively from `attempts` + `question_tags` + `topics` +
`content_categories`. No LLM, no external I/O.

Definition (§V36):
- `last_window` = the 10 most-recent in-scope attempts (by `attempted_at`
  DESC, breaking ties by `attempts.id` DESC for determinism).
- `prior_window` = the 10 in-scope attempts immediately preceding the
  last window (rank 11..20 under the same ordering).
- `delta = accuracy(last) − accuracy(prior)`. NULL when either window
  has fewer than 5 attempts — insufficient signal.
- Arrow rendering: `↑` iff `delta ≥ +0.10`, `↓` iff `delta ≤ −0.10`,
  else `→`. NULL when delta is NULL.
- Per §C: timing data does NOT enter the computation.

Scope rollup (§V31): each attempt is counted once per scope. A
multi-tag question reaches multiple CCs; in that case the attempt
counts once in each CC scope, but only once within a single CC scope
even when the question carries several tag rows that all resolve under
that CC (DISTINCT keyed on `attempts.id`).

For topic scopes, a recursive CTE over `topics.parent_topic_id`
renders the subtree; an attempt is in scope iff its question carries
any `question_tags` row whose `topic_id ∈ subtree`. Attempts whose
question is tagged only at CC granularity (`topic_id IS NULL`) do not
count for topic-scoped trajectory — they have no topic-level
resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

MIN_WINDOW_SIZE = 5
ARROW_THRESHOLD = 0.10


@dataclass(frozen=True)
class TrajectoryWindow:
    n: int
    correct: int

    @property
    def accuracy(self) -> float | None:
        return None if self.n == 0 else self.correct / self.n


@dataclass(frozen=True)
class TrajectorySummary:
    scope: str
    last: TrajectoryWindow
    prior: TrajectoryWindow

    @property
    def delta(self) -> float | None:
        if self.last.n < MIN_WINDOW_SIZE or self.prior.n < MIN_WINDOW_SIZE:
            return None
        last_acc = self.last.accuracy
        prior_acc = self.prior.accuracy
        assert last_acc is not None and prior_acc is not None
        return last_acc - prior_acc

    @property
    def arrow(self) -> str | None:
        d = self.delta
        if d is None:
            return None
        if d >= ARROW_THRESHOLD:
            return "↑"
        if d <= -ARROW_THRESHOLD:
            return "↓"
        return "→"


_WINDOW_PROJECTIONS = """
  count(*) FILTER (WHERE rn BETWEEN 1 AND 10) AS last_n,
  count(*) FILTER (WHERE rn BETWEEN 1 AND 10 AND is_correct) AS last_correct,
  count(*) FILTER (WHERE rn BETWEEN 11 AND 20) AS prior_n,
  count(*) FILTER (WHERE rn BETWEEN 11 AND 20 AND is_correct) AS prior_correct
"""


def _summary_from_row(row, *, scope: str) -> TrajectorySummary:
    m = row._mapping
    return TrajectorySummary(
        scope=scope,
        last=TrajectoryWindow(n=int(m["last_n"]), correct=int(m["last_correct"])),
        prior=TrajectoryWindow(n=int(m["prior_n"]), correct=int(m["prior_correct"])),
    )


async def trajectory_for_cc(session: AsyncSession, *, cc_code: str) -> TrajectorySummary:
    """last-10 vs prior-10 accuracy Δ over attempts in the given CC.

    An attempt is in scope for CC X iff its question carries either:
    - a direct `content_category_id = X` tag, or
    - a `topic_id` whose topic's `content_category_id = X`.

    Mirrors `app.services.anki.retention.retention_for_cc` so the
    UWorld trajectory shares scope semantics with the Anki retention
    rollup that lands beside it on `/mastery/{cc}`.
    """
    sql = text(
        f"""
        WITH scoped AS (
            SELECT DISTINCT a.id, a.is_correct, a.attempted_at
            FROM attempts a
            WHERE EXISTS (
                SELECT 1
                FROM question_tags qt
                LEFT JOIN topics tp ON tp.id = qt.topic_id
                JOIN content_categories cc ON cc.code = :cc_code
                WHERE qt.question_id = a.question_id
                  AND (
                    qt.content_category_id = cc.id
                    OR tp.content_category_id = cc.id
                  )
            )
        ), ranked AS (
            SELECT
              is_correct,
              row_number() OVER (ORDER BY attempted_at DESC, id DESC) AS rn
            FROM scoped
        )
        SELECT
          {_WINDOW_PROJECTIONS}
        FROM ranked
        """
    )
    row = (await session.execute(sql, {"cc_code": cc_code})).one()
    return _summary_from_row(row, scope=f"cc:{cc_code}")


async def trajectory_for_topic(session: AsyncSession, *, topic_id: int) -> TrajectorySummary:
    """last-10 vs prior-10 accuracy Δ over attempts in the topic's subtree.

    Subtree-membership per §V31: an attempt is in scope iff its
    question carries any `question_tags` row with
    `topic_id ∈ subtree(topic_id)`. Direct-CC tags do not count for
    topic scopes (they have no topic-level resolution).
    """
    sql = text(
        f"""
        WITH RECURSIVE subtree(id) AS (
            SELECT id FROM topics WHERE id = :topic_id
            UNION ALL
            SELECT child.id
            FROM topics child
            JOIN subtree s ON child.parent_topic_id = s.id
        ), scoped AS (
            SELECT DISTINCT a.id, a.is_correct, a.attempted_at
            FROM attempts a
            WHERE EXISTS (
                SELECT 1
                FROM question_tags qt
                WHERE qt.question_id = a.question_id
                  AND qt.topic_id IN (SELECT id FROM subtree)
            )
        ), ranked AS (
            SELECT
              is_correct,
              row_number() OVER (ORDER BY attempted_at DESC, id DESC) AS rn
            FROM scoped
        )
        SELECT
          {_WINDOW_PROJECTIONS}
        FROM ranked
        """
    )
    row = (await session.execute(sql, {"topic_id": topic_id})).one()
    return _summary_from_row(row, scope=f"topic:{topic_id}")
