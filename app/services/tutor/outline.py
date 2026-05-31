"""Tutor outline helpers — node-keyed, domain-blind (T22, V-O1, V-O3, V-M1).

Replaces the FENCED AAMC-shaped section/fc/cc/topic surface from T17. The
new shape speaks `node_id` only and works for any course whose outline
was imported through `POST /api/v1/courses/{id}/outline:import` (V-O3).

Three operations:

- ``search_nodes`` — substring match on ``OutlineNode.name`` across one
  course (or all courses). Returns rows with `node_id`, full `>>`-joined
  path, `kind`, `depth`, and the owning course. ⊥ verdict / ranking
  beyond name-startswith → name-contains (V-M1: data exposure, not
  recommendation).
- ``get_outline_tree`` — flat node list for one course, ordered by
  ``(depth, position, id)``. Caller renders the tree shape it needs.
- ``get_subtree`` — `{root + descendants}` set rollup via the shared
  recursive CTE in ``app.services.outline_subtree`` (V-O1).

Consumed by ``app/api/v1/tutor.py``; the MCP server proxies the routes
HTTP-side.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import OUTLINE_PATH_DELIMITER, Course, OutlineNode
from app.services.outline_subtree import subtree_node_ids

logger = logging.getLogger(__name__)


class CourseNotFoundError(LookupError):
    """Raised when a `course_slug` filter does not match any course."""


class NodeNotFoundError(LookupError):
    """Raised when a `node_id` does not exist."""


def _path_of(node_by_id: dict[int, OutlineNode], leaf: OutlineNode) -> str:
    """Render a node's full `>>`-joined path from cached siblings."""
    segs: list[str] = [leaf.name]
    cur = leaf
    while cur.parent_id is not None:
        parent = node_by_id.get(cur.parent_id)
        if parent is None:
            break
        segs.append(parent.name)
        cur = parent
    return OUTLINE_PATH_DELIMITER.join(reversed(segs))


def _row(
    n: OutlineNode,
    path: str,
    course: Course,
) -> dict[str, Any]:
    return {
        "node_id": n.id,
        "course_id": course.id,
        "course_slug": course.slug,
        "path": path,
        "name": n.name,
        "kind": n.kind,
        "depth": n.depth,
    }


async def _load_course_by_slug(session: AsyncSession, slug: str) -> Course:
    course = (await session.execute(select(Course).where(Course.slug == slug))).scalar_one_or_none()
    if course is None:
        raise CourseNotFoundError(slug)
    return course


async def resolve_node_labels(
    session: AsyncSession, node_ids: Iterable[int]
) -> dict[int, dict[str, Any]]:
    """Resolve `node_id`s → `{node_id, name, path, kind}` (T38, V-O1, V-O5).

    Renders each node's full `>>`-joined path by walking parents. Loads every
    node in the involved courses so the parent walk resolves even when the
    tagged nodes are deep leaves; cross-course safe (V-O3). Unknown / `None`
    ids are dropped — callers surface only resolvable tags. ⊥ verdict (V-M1):
    pure data exposure.
    """
    ids = {i for i in node_ids if i is not None}
    if not ids:
        return {}
    target_rows = (
        (await session.execute(select(OutlineNode).where(OutlineNode.id.in_(ids)))).scalars().all()
    )
    if not target_rows:
        return {}

    course_ids = {n.course_id for n in target_rows}
    all_rows = (
        (await session.execute(select(OutlineNode).where(OutlineNode.course_id.in_(course_ids))))
        .scalars()
        .all()
    )
    node_by_id = {n.id: n for n in all_rows}

    return {
        n.id: {
            "node_id": n.id,
            "name": n.name,
            "path": _path_of(node_by_id, n),
            "kind": n.kind,
        }
        for n in target_rows
    }


async def search_nodes(
    session: AsyncSession,
    *,
    query: str,
    course_slug: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Substring search over ``OutlineNode.name`` (V-M1: data exposure only).

    Empty / whitespace-only query → empty list (do not return everything
    by accident). Optional `course_slug` narrows to one course; otherwise
    scans all courses (V-O3: domain-blind).

    Ranking: rows whose ``name`` starts with the query come first, then
    rows that merely contain it. Stable secondary sort on `(course_id,
    depth, id)` keeps the output deterministic across calls.
    """
    q = query.strip()
    if not q:
        return []

    needle = q.lower()
    stmt = select(OutlineNode)
    if course_slug is not None:
        course = await _load_course_by_slug(session, course_slug)
        stmt = stmt.where(OutlineNode.course_id == course.id)

    # Load matching candidates, then resolve their paths from a per-course
    # cache of all nodes in the involved courses. Two passes keep the SQL
    # simple (substring filter on lowered name) while the path walk stays
    # in-process — same pattern as `OutlineLookup.path_of`.
    candidate_rows = (await session.execute(stmt)).scalars().all()
    candidates = [n for n in candidate_rows if needle in n.name.lower()]
    if not candidates:
        return []

    course_ids = {n.course_id for n in candidates}
    courses_by_id: dict[int, Course] = {
        c.id: c
        for c in (await session.execute(select(Course).where(Course.id.in_(course_ids))))
        .scalars()
        .all()
    }
    nodes_by_course: dict[int, dict[int, OutlineNode]] = {}
    for cid in course_ids:
        rows = (
            (await session.execute(select(OutlineNode).where(OutlineNode.course_id == cid)))
            .scalars()
            .all()
        )
        nodes_by_course[cid] = {n.id: n for n in rows}

    def _rank(n: OutlineNode) -> int:
        # 0 = startswith, 1 = contains. Lower is better.
        return 0 if n.name.lower().startswith(needle) else 1

    ranked = sorted(
        candidates,
        key=lambda n: (_rank(n), n.course_id, n.depth, n.id),
    )[:limit]

    return [
        _row(
            n,
            _path_of(nodes_by_course[n.course_id], n),
            courses_by_id[n.course_id],
        )
        for n in ranked
    ]


async def get_outline_tree(
    session: AsyncSession,
    *,
    course_slug: str,
) -> dict[str, Any]:
    """Flat node list for one course ordered by ``(depth, position, id)``.

    Caller assembles the tree from `parent_id` if needed. Returning a flat
    list (rather than a nested object) keeps the seam easy to consume
    from JS / MCP and matches ``GET /api/v1/courses/{id}/outline``.
    """
    course = await _load_course_by_slug(session, course_slug)
    rows = (
        (
            await session.execute(
                select(OutlineNode)
                .where(OutlineNode.course_id == course.id)
                .order_by(OutlineNode.depth, OutlineNode.position, OutlineNode.id)
            )
        )
        .scalars()
        .all()
    )
    node_by_id = {n.id: n for n in rows}
    return {
        "course": {
            "id": course.id,
            "slug": course.slug,
            "name": course.name,
            "description": course.description,
        },
        "nodes": [
            {
                "node_id": n.id,
                "parent_id": n.parent_id,
                "kind": n.kind,
                "name": n.name,
                "depth": n.depth,
                "position": n.position,
                "path": _path_of(node_by_id, n),
            }
            for n in rows
        ],
    }


async def get_subtree(
    session: AsyncSession,
    *,
    node_id: int,
) -> dict[str, Any]:
    """Subtree rollup over the node (V-O1).

    Returns `{node_id, course_id, course_slug, path, descendants}` where
    `descendants` is the set of node ids in the subtree *excluding* the
    root (the root is named explicitly). Callers do their own membership
    checks; this surface does not aggregate metrics (V-M1).
    """
    root = await session.get(OutlineNode, node_id)
    if root is None:
        raise NodeNotFoundError(node_id)

    course = await session.get(Course, root.course_id)
    if course is None:  # defensive — FK cascade should prevent this
        raise CourseNotFoundError(f"course_id={root.course_id}")

    rows = (
        (await session.execute(select(OutlineNode).where(OutlineNode.course_id == course.id)))
        .scalars()
        .all()
    )
    node_by_id = {n.id: n for n in rows}

    ids = await subtree_node_ids(session, node_id)
    descendants = sorted(i for i in ids if i != node_id)

    return {
        "node_id": node_id,
        "course_id": course.id,
        "course_slug": course.slug,
        "path": _path_of(node_by_id, root),
        "name": root.name,
        "kind": root.kind,
        "depth": root.depth,
        "descendants": descendants,
    }
