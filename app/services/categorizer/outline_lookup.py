"""In-memory path → node_id index over the `outline_nodes` tree.

Generalized for Gradient (T12): the PoC's section/cc/topic codes are gone
(§I outline_nodes has only kind/name). Resolution is by `>>`-delimited path
(V-O4), e.g. `"CP >> FC1 >> 1A >> Amino acids"`. One lookup instance per
course; built once per request or per worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import OUTLINE_PATH_DELIMITER, Course, OutlineNode
from app.services.categorizer._text import normalize_typographic_punctuation

logger = logging.getLogger(__name__)


class OutlineNotSeededError(RuntimeError):
    """Raised when OutlineLookup loads against an unseeded outline_nodes set."""


@dataclass(frozen=True)
class _Node:
    id: int
    name: str
    kind: str
    parent_id: Optional[int]
    depth: int


class OutlineLookup:
    """Path-indexed view of one course's `outline_nodes` subtree (V-O1)."""

    def __init__(self, *, course_id: int, nodes: list[_Node]) -> None:
        self._course_id = course_id
        self._nodes = nodes
        self._by_id: dict[int, _Node] = {n.id: n for n in nodes}
        # Index roots by name for path traversal.
        self._roots_by_name: dict[str, list[_Node]] = {}
        for n in nodes:
            if n.parent_id is None:
                self._roots_by_name.setdefault(n.name, []).append(n)
        # children index for path walk.
        self._children: dict[int, dict[str, list[_Node]]] = {}
        for n in nodes:
            if n.parent_id is not None:
                self._children.setdefault(n.parent_id, {}).setdefault(n.name, []).append(n)

    @classmethod
    async def load(cls, session: AsyncSession, *, course_slug: str = "aamc") -> "OutlineLookup":
        course = (
            await session.execute(select(Course).where(Course.slug == course_slug))
        ).scalar_one_or_none()
        if course is None:
            raise OutlineNotSeededError(
                f"no course with slug {course_slug!r}; upload an outline schema "
                f"via POST /api/v1/courses/{{id}}/outline:import (T9)"
            )
        rows = (
            await session.execute(
                select(OutlineNode).where(OutlineNode.course_id == course.id)
            )
        ).scalars().all()
        if not rows:
            raise OutlineNotSeededError(
                f"course {course_slug!r} has no outline_nodes — import a schema first"
            )
        nodes = [
            _Node(
                id=n.id,
                # Normalize so path lookups match regardless of apostrophe variant.
                name=normalize_typographic_punctuation(n.name),
                kind=n.kind,
                parent_id=n.parent_id,
                depth=n.depth,
            )
            for n in rows
        ]
        return cls(course_id=course.id, nodes=nodes)

    @property
    def course_id(self) -> int:
        return self._course_id

    def node_id_by_path(self, path: str) -> int | None:
        """Resolve a `>>`-delimited path to a node_id (V-O4).

        Path walks from a root by sibling name at each depth. Returns None
        (with a warning) on missing/ambiguous segment.
        """
        parts = [
            p.strip()
            for p in normalize_typographic_punctuation(path).split(OUTLINE_PATH_DELIMITER)
        ]
        if not parts or not parts[0]:
            logger.warning("node_id_by_path: malformed path %r", path)
            return None

        # Resolve the root segment.
        root_candidates = self._roots_by_name.get(parts[0], [])
        if len(root_candidates) != 1:
            logger.warning(
                "node_id_by_path: root %r %s (path=%r)",
                parts[0],
                "ambiguous" if len(root_candidates) > 1 else "missing",
                path,
            )
            return None
        current = root_candidates[0]

        # Walk children by name.
        for i, name in enumerate(parts[1:], start=2):
            child_candidates = self._children.get(current.id, {}).get(name, [])
            if len(child_candidates) != 1:
                logger.warning(
                    "node_id_by_path: segment %d %r %s under %r (path=%r)",
                    i,
                    name,
                    "ambiguous" if len(child_candidates) > 1 else "missing",
                    current.name,
                    path,
                )
                return None
            current = child_candidates[0]

        return current.id

    def node(self, node_id: int) -> _Node | None:
        return self._by_id.get(node_id)

    def path_of(self, node_id: int) -> str | None:
        """Inverse of `node_id_by_path` — render a node's path."""
        n = self._by_id.get(node_id)
        if n is None:
            return None
        segs: list[str] = [n.name]
        while n.parent_id is not None:
            n = self._by_id[n.parent_id]
            segs.append(n.name)
        return OUTLINE_PATH_DELIMITER.join(reversed(segs))
