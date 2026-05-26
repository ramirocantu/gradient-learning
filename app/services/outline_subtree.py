"""Shared subtree-set rollup over `outline_nodes` (V-O1).

A node's subtree = itself + all descendants. Mastery/drilldown/scope readers
ask "what items live under node N?" by joining against this set rather than
walking the tree row-by-row. Recursive CTE on `outline_nodes.parent_id`.

V-O1: rollup is a *set*, not a sum — each item lives once at its most-specific
node, and a parent's set is the union of its descendants' + own direct items.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


_SUBTREE_SQL = text(
    """
    WITH RECURSIVE subtree AS (
        SELECT id FROM outline_nodes WHERE id = :root
        UNION ALL
        SELECT n.id
        FROM outline_nodes n
        JOIN subtree s ON n.parent_id = s.id
    )
    SELECT id FROM subtree
    """
)


async def subtree_node_ids(session: AsyncSession, root_id: int) -> set[int]:
    """Return `{root_id} ∪ {all descendant node ids}` (V-O1)."""
    rows = await session.execute(_SUBTREE_SQL, {"root": root_id})
    return {r[0] for r in rows}


async def subtree_node_ids_many(session: AsyncSession, root_ids: list[int]) -> set[int]:
    """Union of subtrees rooted at each id in `root_ids`. Empty list → empty set."""
    if not root_ids:
        return set()
    out: set[int] = set()
    for r in root_ids:
        out |= await subtree_node_ids(session, r)
    return out
