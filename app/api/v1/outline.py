"""Outline schema import endpoints (§T9, V-O2, V-O3, V-O4).

POST   /api/v1/courses                          — create a course
POST   /api/v1/courses/{course_id}/outline:import — validate + materialize
GET    /api/v1/courses/{course_id}/outline      — return the node tree
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.models.outline import Course, OutlineNode
from app.services.analytics import (
    CourseNotFoundError as MasteryCourseNotFound,
    NodeNotFoundError as MasteryNodeNotFound,
    compute_course_mastery,
    compute_node_mastery,
)
from app.services.outline import (
    OutlineImportError,
    OutlineSchemaValidationError,
    materialize_outline,
    validate_outline_schema,
)

# X-Coach-Token gating is enforced globally at the v1 router (app/main.py).
router = APIRouter(tags=["outline"])


class CreateCourseBody(BaseModel):
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None


def _course_payload(c: Course) -> dict[str, Any]:
    return {
        "id": c.id,
        "slug": c.slug,
        "name": c.name,
        "description": c.description,
    }


@router.get("/courses")
async def list_courses(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """List all courses (slug-ordered). The SPA course picker needs to
    enumerate courses; per V-D1 that read extends the public API rather
    than a dashboard-private route."""
    rows = (await session.execute(select(Course).order_by(Course.slug))).scalars().all()
    return [_course_payload(c) for c in rows]


@router.post("/courses", status_code=status.HTTP_201_CREATED)
async def create_course(
    body: CreateCourseBody,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    existing = (
        await session.execute(select(Course).where(Course.slug == body.slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"course slug {body.slug!r} already exists (id={existing.id})",
        )
    course = Course(slug=body.slug, name=body.name, description=body.description)
    session.add(course)
    await session.flush()
    return _course_payload(course)


@router.post(
    "/courses/{course_id}/outline:import",
    status_code=status.HTTP_200_OK,
)
async def import_outline(
    course_id: int,
    body: dict[str, Any],
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """V-O2: validate-then-materialize, atomic. The body matches the §I shape:

        { "course": {"slug","name","description?"},
          "nodes": [ {"path":[...], "kind", "name", "position?"}, ... ] }

    The validator surfaces every problem in one 422 (whole-upload rejection).
    On success, the materializer wipes any existing outline for the course
    and inserts the new tree in one transaction (V-O3: re-upload restores).
    """
    course = await session.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=404, detail=f"course id={course_id} not found")

    # The URL pins the course id; the body still carries `course.slug` for
    # the importer's contract — assert they agree to catch upload mismatch.
    body_course = body.get("course") if isinstance(body, dict) else None
    if isinstance(body_course, dict):
        body_slug = body_course.get("slug")
        if isinstance(body_slug, str) and body_slug.strip() != course.slug:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"upload course.slug {body_slug!r} does not match "
                    f"course id={course_id} (slug={course.slug!r})"
                ),
            )

    try:
        validated = validate_outline_schema(body)
    except OutlineSchemaValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": exc.errors},
        ) from exc

    if validated.course.slug != course.slug:
        # Validator accepted the body but the slug routes to a different
        # course — treat as the same mismatch case above.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"validated course.slug {validated.course.slug!r} does not match "
                f"course id={course_id} (slug={course.slug!r})"
            ),
        )

    try:
        course = await materialize_outline(session, validated)
    except OutlineImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {
        "course": _course_payload(course),
        "nodes_imported": len(validated.nodes_in_order),
    }


def _node_payload(n: OutlineNode) -> dict[str, Any]:
    return {
        "id": n.id,
        "parent_id": n.parent_id,
        "kind": n.kind,
        "name": n.name,
        "depth": n.depth,
        "position": n.position,
    }


@router.get("/courses/{course_id}/outline")
async def read_outline(
    course_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    course = await session.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=404, detail=f"course id={course_id} not found")
    rows = (
        (
            await session.execute(
                select(OutlineNode)
                .where(OutlineNode.course_id == course_id)
                .order_by(OutlineNode.depth, OutlineNode.position, OutlineNode.id)
            )
        )
        .scalars()
        .all()
    )
    return {
        "course": _course_payload(course),
        "nodes": [_node_payload(n) for n in rows],
    }


# T44 (V-O1, V-O5, V-D1, V-RB1): per-node/subtree + course mastery, ported off
# the FENCED AAMC analytics onto OutlineNode + outline_subtree set-rollup and
# re-exposed on the public API (⊥ a private/dashboard-only route).
@router.get("/outline/nodes/{node_id}/mastery")
async def node_mastery(
    node_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        return await compute_node_mastery(session, node_id=node_id)
    except MasteryNodeNotFound:
        raise HTTPException(status_code=404, detail=f"node id={node_id} not found")


@router.get("/outline/courses/{course_id}/mastery")
async def course_mastery(
    course_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        return await compute_course_mastery(session, course_id=course_id)
    except MasteryCourseNotFound:
        raise HTTPException(status_code=404, detail=f"course id={course_id} not found")
