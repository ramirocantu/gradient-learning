"""Subtree-membership query helper (SPEC §T41 / §V31).

Computes the inclusive set of topic ids in the subtree rooted at a
given topic. The recursive CTE walks `topics.parent_topic_id` from
the anchor down through all descendants, including the anchor itself
(so `metric(topic) = aggregate over { item | item.topic_id ∈ subtree(topic) }`
per §V31).

`SubtreeCache` gives a per-request memoization layer: heatmap render
(T42) issues one rollup per topic row in a CC's flat-tree view, and
without memoization each row would trigger its own recursive CTE.
`prime_cc(cc_code)` precomputes every topic's subtree under the CC in
a single round trip; subsequent `get(topic_id)` calls hit the cache.

The cache is intentionally tied to a single `AsyncSession` and meant
to be discarded at request boundaries — topic hierarchy is stable
within one request but the seed-time topic tree can change between
deployments. Do not promote to a process-wide cache without an
invalidation hook tied to `seed_outline`.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


_SUBTREE_SQL = text(
    """
    WITH RECURSIVE subtree(id) AS (
        SELECT id FROM topics WHERE id = :topic_id
        UNION ALL
        SELECT child.id
        FROM topics child
        JOIN subtree s ON child.parent_topic_id = s.id
    )
    SELECT id FROM subtree
    """
)


# One query that walks every topic under a CC and emits
# (ancestor_id, descendant_id) closure pairs. Each ancestor row's
# descendant set is the union of all descendant_id values; the anchor
# row (ancestor = descendant) keeps every topic — even leaves —
# present in the result so `prime_cc` populates a single-element
# subtree for childless nodes.
_CC_CLOSURE_SQL = text(
    """
    WITH RECURSIVE closure(ancestor_id, descendant_id) AS (
        SELECT t.id, t.id
        FROM topics t
        JOIN content_categories cc ON cc.id = t.content_category_id
        WHERE cc.code = :cc_code
        UNION ALL
        SELECT c.ancestor_id, child.id
        FROM closure c
        JOIN topics child ON child.parent_topic_id = c.descendant_id
    )
    SELECT ancestor_id, descendant_id FROM closure
    """
)


async def subtree_topic_ids(session: AsyncSession, *, topic_id: int) -> list[int]:
    """Inclusive list of topic ids in the subtree rooted at `topic_id`.

    The anchor itself is included — `subtree_topic_ids(t)` always
    contains `t`. Order is unspecified; callers that need stable
    output should sort.
    """
    rows = (await session.execute(_SUBTREE_SQL, {"topic_id": topic_id})).all()
    return [int(r[0]) for r in rows]


class SubtreeCache:
    """Per-request memoization of `subtree_topic_ids`.

    Construct one per HTTP request / job tick. Call `prime_cc(cc_code)`
    before the per-topic loop to fold N+1 recursive CTEs into one
    closure query; otherwise `get(topic_id)` falls back to issuing a
    single-anchor recursive CTE on first lookup.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._cache: dict[int, list[int]] = {}

    async def get(self, topic_id: int) -> list[int]:
        cached = self._cache.get(topic_id)
        if cached is not None:
            return cached
        descendants = await subtree_topic_ids(self._session, topic_id=topic_id)
        self._cache[topic_id] = descendants
        return descendants

    async def prime_cc(self, cc_code: str) -> None:
        """Populate the cache for every topic in `cc_code` with one query.

        After priming, `get(topic_id)` for any topic under `cc_code` —
        leaf or internal — returns immediately without further SQL.
        Topics outside the CC remain uncached and fall back to the
        single-anchor query on demand.
        """
        rows = (await self._session.execute(_CC_CLOSURE_SQL, {"cc_code": cc_code})).all()
        bucket: dict[int, list[int]] = {}
        for ancestor_id, descendant_id in rows:
            bucket.setdefault(int(ancestor_id), []).append(int(descendant_id))
        # Overwrite — re-priming the same CC must reflect any topic-tree
        # mutation since the previous prime call (rare, but cheap).
        self._cache.update(bucket)
