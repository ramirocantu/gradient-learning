"""Validate-then-materialize an uploaded outline schema (§T9, V-O2, V-O3, V-O4).

The schema is `{"course": {...}, "nodes": [{"path", "kind", "name", ...}, ...]}`
per §I. Validation is whole-upload-or-reject; on success, materialization
runs in one DB transaction so the importer can never leave a partial tree
behind. Re-uploading the AAMC seed restores the MCAT outline (V-O3) — the
materializer wipes existing nodes for the course before insert.

Validation rules (each one a §V-O2 obligation):
- `course.slug` present + non-empty.
- Every `node.path` is a non-empty list of non-empty strings.
- Every segment of every `path` is free of `OUTLINE_PATH_DELIMITER` (V-O4).
- Every `node.kind` and `node.name` is a non-empty string; `name` matches
  the last segment of `path`.
- No duplicate paths (full equality, list-of-strings compare).
- Parent chain closed: every non-root node's parent prefix appears in the
  upload (`path[:-1]` exists as some other `node.path`).
- `depth` and `position` are integers when supplied; depth must equal
  `len(path) - 1` (paths are 1-based — root is depth 0).
- No `kind` contradiction: two nodes with the same path can't disagree on
  kind (we already forbid duplicate paths, but check `kind` against an
  implicit per-depth contract — same `path` length must agree on `kind`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import OUTLINE_PATH_DELIMITER, Course, OutlineNode

logger = logging.getLogger(__name__)


class OutlineImportError(RuntimeError):
    """Raised when materialization fails after validation (e.g., DB error)."""


class OutlineSchemaValidationError(ValueError):
    """Raised on validation failure. `.errors` carries the full list so the
    API can surface every problem in one 422 instead of one-at-a-time."""

    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__("; ".join(errors[:5]) + ("…" if len(errors) > 5 else ""))
        self.errors = list(errors)


@dataclass(frozen=True)
class CourseHeader:
    slug: str
    name: str
    description: str | None


@dataclass(frozen=True)
class NodeRecord:
    path: tuple[str, ...]
    kind: str
    name: str
    position: int


@dataclass(frozen=True)
class ValidatedOutline:
    """The validator's output — a pre-flighted, fully-resolved import payload
    ready to hand to `materialize_outline`. Holds nodes in materialization
    order (parents before children) so the DB insert can assign parent_id
    sequentially without a second pass."""

    course: CourseHeader
    nodes_in_order: tuple[NodeRecord, ...]


def _isstr(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _collect_validation_errors(
    payload: Any,
) -> tuple[list[str], CourseHeader | None, list[NodeRecord]]:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ([f"top-level payload must be an object, got {type(payload).__name__}"], None, [])

    course_raw = payload.get("course")
    if not isinstance(course_raw, dict):
        errors.append("course block missing or not an object")
        course: CourseHeader | None = None
    else:
        slug = course_raw.get("slug")
        name = course_raw.get("name")
        description = course_raw.get("description")
        if not _isstr(slug):
            errors.append("course.slug missing or empty")
        if not _isstr(name):
            errors.append("course.name missing or empty")
        if description is not None and not isinstance(description, str):
            errors.append("course.description must be a string when present")
        course = (
            CourseHeader(
                slug=str(slug).strip(),
                name=str(name).strip(),
                description=str(description).strip() if isinstance(description, str) else None,
            )
            if _isstr(slug) and _isstr(name)
            else None
        )

    nodes_raw = payload.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        errors.append("nodes missing or empty — outline import requires at least one node")
        return (errors, course, [])

    seen_paths: dict[tuple[str, ...], int] = {}
    records: list[NodeRecord] = []

    for i, raw in enumerate(nodes_raw):
        prefix = f"nodes[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} not an object")
            continue
        path = raw.get("path")
        kind = raw.get("kind")
        name = raw.get("name")
        position = raw.get("position", 0)
        depth = raw.get("depth")

        if not isinstance(path, list) or not path:
            errors.append(f"{prefix}.path must be a non-empty list of strings")
            continue
        if not all(_isstr(seg) for seg in path):
            errors.append(f"{prefix}.path segments must be non-empty strings")
            continue
        bad_segments = [seg for seg in path if OUTLINE_PATH_DELIMITER in seg]
        if bad_segments:
            # V-O4: reserved delimiter — node names can't contain ` >> `.
            errors.append(
                f"{prefix}.path segments may not contain the reserved delimiter "
                f"{OUTLINE_PATH_DELIMITER!r}: {bad_segments!r}"
            )
            continue
        if not _isstr(kind):
            errors.append(f"{prefix}.kind missing or empty")
            continue
        if not _isstr(name):
            errors.append(f"{prefix}.name missing or empty")
            continue
        if name.strip() != path[-1].strip():
            errors.append(
                f"{prefix}.name {name!r} must match the last segment of path {path[-1]!r}"
            )
            continue
        if depth is not None:
            if not isinstance(depth, int) or depth < 0:
                errors.append(f"{prefix}.depth must be a non-negative integer")
                continue
            if depth != len(path) - 1:
                errors.append(
                    f"{prefix}.depth {depth} does not match path length {len(path)} (expected {len(path) - 1})"
                )
                continue
        if not isinstance(position, int) or position < 0:
            errors.append(f"{prefix}.position must be a non-negative integer")
            continue

        canon_path: tuple[str, ...] = tuple(seg.strip() for seg in path)
        if canon_path in seen_paths:
            errors.append(
                f"{prefix}.path duplicates an earlier entry (nodes[{seen_paths[canon_path]}])"
            )
            continue
        seen_paths[canon_path] = i
        records.append(
            NodeRecord(
                path=canon_path,
                kind=str(kind).strip(),
                name=str(name).strip(),
                position=int(position),
            )
        )

    # Parent-chain closure: every non-root's prefix path must be in the set.
    path_set = set(seen_paths.keys())
    for rec in records:
        if len(rec.path) > 1:
            parent_path = rec.path[:-1]
            if parent_path not in path_set:
                errors.append(
                    f"node {rec.path!r} has no parent in the upload "
                    f"(missing {parent_path!r}) — V-O2 parent chain broken"
                )

    # Kind agreement at each depth: same-depth nodes with the same kind
    # name are fine; this catches kind contradictions for sibling-set
    # consistency (e.g., a course where half the depth-1 nodes are
    # 'section' and the other half are 'unit' — likely a typo).
    kind_by_depth: dict[int, set[str]] = {}
    for rec in records:
        kind_by_depth.setdefault(len(rec.path) - 1, set()).add(rec.kind)
    for d, kinds in kind_by_depth.items():
        if len(kinds) > 1:
            errors.append(
                f"depth={d} mixes kinds {sorted(kinds)!r} — V-O2 depth/kind contradiction"
            )

    return errors, course, records


def _sort_for_materialization(records: Iterable[NodeRecord]) -> list[NodeRecord]:
    """Parents before children — sort by `(depth, path)` so the materializer
    sees a root before any of its descendants. Within a depth, lexicographic
    path order is stable and deterministic for snapshot tests."""
    return sorted(records, key=lambda r: (len(r.path), r.path))


def validate_outline_schema(payload: Any) -> ValidatedOutline:
    """Pure-function validator. Raises `OutlineSchemaValidationError` with
    the full list of problems on any failure (V-O2: whole upload rejected
    atomically). On success returns a `ValidatedOutline` ready for the
    materializer."""
    errors, course, records = _collect_validation_errors(payload)
    if errors or course is None:
        raise OutlineSchemaValidationError(errors)
    ordered = _sort_for_materialization(records)
    return ValidatedOutline(course=course, nodes_in_order=tuple(ordered))


async def materialize_outline(
    session: AsyncSession,
    validated: ValidatedOutline,
) -> Course:
    """Upsert the course, wipe its existing outline, and insert the validated
    tree in one transaction. V-O3: re-uploading the AAMC seed restores MCAT —
    a no-side-effect rebuild rather than an idempotent merge.

    Caller is responsible for `await session.commit()`. Raises
    `OutlineImportError` on DB-side failures.
    """
    course = (
        await session.execute(select(Course).where(Course.slug == validated.course.slug))
    ).scalar_one_or_none()
    if course is None:
        course = Course(
            slug=validated.course.slug,
            name=validated.course.name,
            description=validated.course.description,
        )
        session.add(course)
        await session.flush()
    else:
        # Update mutable header fields and wipe descendants.
        course.name = validated.course.name
        course.description = validated.course.description
        await session.execute(delete(OutlineNode).where(OutlineNode.course_id == course.id))
        await session.flush()

    # Insert in depth-order so parent_id is resolvable on each new row.
    id_by_path: dict[tuple[str, ...], int] = {}
    for rec in validated.nodes_in_order:
        depth = len(rec.path) - 1
        parent_id: int | None = None
        if depth > 0:
            parent_id = id_by_path.get(rec.path[:-1])
            if parent_id is None:
                # Defensive — validator should have caught this.
                raise OutlineImportError(
                    f"parent path {rec.path[:-1]!r} not yet inserted for node {rec.path!r}"
                )
        node = OutlineNode(
            course_id=course.id,
            parent_id=parent_id,
            kind=rec.kind,
            name=rec.name,
            depth=depth,
            position=rec.position,
        )
        session.add(node)
        await session.flush()
        id_by_path[rec.path] = node.id

    logger.info(
        "materialize_outline course=%r nodes=%d depths=%s",
        course.slug,
        len(validated.nodes_in_order),
        sorted({len(r.path) - 1 for r in validated.nodes_in_order}),
    )
    return course
