"""Node-keyed mastery rollups (T44, V-O1, V-O5, V-D1, V-RB1).

Ported off the retired AAMC Section/FC/CC/Topic tree + the 3-target
`QuestionTag(topic_id/content_category_id/skill)` onto `OutlineNode` +
`app.services.outline_subtree`. No longer FENCED — T44 unfenced this surface
and re-exposed it on the public API (`/api/v1/outline/...`, see
`app/api/v1/outline.py`).

Mastery is a **set** rollup (V-O1): a node's stats cover the DISTINCT questions
tagged (non-overridden) to any node in its subtree (self + descendants), each
question counted once — a parent's set is the union of its descendants' + own
direct items. Reads key on `QuestionTag.node_id` only; ⊥ legacy topic_id /
cc_code joins (V-O5).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, QuestionTag
from app.models.outline import OUTLINE_PATH_DELIMITER, Course, OutlineNode
from app.services.outline_subtree import subtree_node_ids


class NodeNotFoundError(LookupError):
    """Raised when a `node_id` does not exist."""


class CourseNotFoundError(LookupError):
    """Raised when a `course_id` does not exist."""


def wilson_lower(correct: int, attempts: int, z: float = 1.96) -> float:
    """95% Wilson score lower bound on the success-rate proportion. (Pure math.)"""
    if attempts == 0:
        return 0.0
    p = correct / attempts
    denominator = 1 + z**2 / attempts
    center = p + z**2 / (2 * attempts)
    margin = sqrt(p * (1 - p) / attempts + z**2 / (4 * attempts**2)) * z
    return max(0.0, (center - margin) / denominator)


@dataclass(frozen=True)
class Rollup:
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float


def _rollup(attempts: int, correct: int) -> Rollup:
    accuracy = (correct / attempts) if attempts else 0.0
    return Rollup(
        attempts=attempts,
        correct=correct,
        accuracy=accuracy,
        wilson_lower=wilson_lower(correct, attempts),
    )


def _roll_dict(r: Rollup) -> dict[str, Any]:
    return {
        "attempts": r.attempts,
        "correct": r.correct,
        "accuracy": r.accuracy,
        "wilson_lower": r.wilson_lower,
    }


async def _subtree_rollup(session: AsyncSession, subtree_ids: set[int]) -> Rollup:
    """(attempts, correct) over the DISTINCT questions tagged (non-overridden)
    to any node in `subtree_ids` — V-O1 set rollup, each question counted once."""
    if not subtree_ids:
        return _rollup(0, 0)
    distinct_qids = (
        select(QuestionTag.question_id)
        .where(QuestionTag.node_id.in_(subtree_ids))
        .where(QuestionTag.is_overridden.is_(False))
        .distinct()
    )
    row = (
        await session.execute(
            select(
                func.count(),
                func.count().filter(Attempt.is_correct.is_(True)),
            ).where(Attempt.question_id.in_(distinct_qids))
        )
    ).one()
    return _rollup(int(row[0] or 0), int(row[1] or 0))


def _paths(nodes: list[OutlineNode]) -> dict[int, str]:
    """`node_id → ' >> '-joined path` for every node in one course (V-O4)."""
    by_id = {n.id: n for n in nodes}
    cache: dict[int, str] = {}

    def path_of(nid: int) -> str:
        if nid in cache:
            return cache[nid]
        n = by_id[nid]
        if n.parent_id is None or n.parent_id not in by_id:
            cache[nid] = n.name
        else:
            cache[nid] = path_of(n.parent_id) + OUTLINE_PATH_DELIMITER + n.name
        return cache[nid]

    return {nid: path_of(nid) for nid in by_id}


async def compute_node_mastery(session: AsyncSession, *, node_id: int) -> dict[str, Any]:
    """Subtree set-rollup for `node_id` plus a per-direct-child breakdown."""
    node = await session.get(OutlineNode, node_id)
    if node is None:
        raise NodeNotFoundError(node_id)

    course_nodes = list(
        (await session.execute(select(OutlineNode).where(OutlineNode.course_id == node.course_id)))
        .scalars()
        .all()
    )
    paths = _paths(course_nodes)

    node_roll = await _subtree_rollup(session, await subtree_node_ids(session, node_id))

    children = sorted(
        (n for n in course_nodes if n.parent_id == node_id),
        key=lambda n: (n.position, n.id),
    )
    child_payloads: list[dict[str, Any]] = []
    for c in children:
        croll = await _subtree_rollup(session, await subtree_node_ids(session, c.id))
        child_payloads.append(
            {
                "node_id": c.id,
                "name": c.name,
                "kind": c.kind,
                "path": paths.get(c.id, c.name),
                **_roll_dict(croll),
            }
        )

    return {
        "node": {
            "id": node.id,
            "name": node.name,
            "kind": node.kind,
            "depth": node.depth,
            "parent_id": node.parent_id,
            "path": paths.get(node.id, node.name),
        },
        "rollup": _roll_dict(node_roll),
        "children": child_payloads,
    }


async def compute_course_mastery(session: AsyncSession, *, course_id: int) -> dict[str, Any]:
    """Course total set-rollup plus a per-root-node breakdown."""
    course = await session.get(Course, course_id)
    if course is None:
        raise CourseNotFoundError(course_id)

    course_nodes = list(
        (await session.execute(select(OutlineNode).where(OutlineNode.course_id == course_id)))
        .scalars()
        .all()
    )
    paths = _paths(course_nodes)

    # Course total = distinct questions tagged to ANY node in the course.
    total = await _subtree_rollup(session, {n.id for n in course_nodes})

    roots = sorted(
        (n for n in course_nodes if n.parent_id is None),
        key=lambda n: (n.position, n.id),
    )
    node_payloads: list[dict[str, Any]] = []
    for r in roots:
        roll = await _subtree_rollup(session, await subtree_node_ids(session, r.id))
        node_payloads.append(
            {
                "node_id": r.id,
                "name": r.name,
                "kind": r.kind,
                "path": paths.get(r.id, r.name),
                **_roll_dict(roll),
            }
        )

    return {
        "course": {
            "id": course.id,
            "slug": course.slug,
            "name": course.name,
        },
        "total": _roll_dict(total),
        "nodes": node_payloads,
    }


__all__ = [
    "CourseNotFoundError",
    "NodeNotFoundError",
    "Rollup",
    "compute_course_mastery",
    "compute_node_mastery",
    "wilson_lower",
]
