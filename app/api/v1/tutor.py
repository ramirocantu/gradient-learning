"""Tutor-facing HTTP routes consumed by the MCP server (ticket 9.0).

Routes added here mirror the read-side of the MCP tool surface that has no
existing HTTP equivalent. Auth is X-Coach-Token (shared with ingest).

Not consumed by the dashboard. If a route ever does double-duty, move the
shared bits into a service in app.services and keep the route thin.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.services.tutor import (
    captures as captures_svc,
    flags as flags_svc,
    health as health_svc,
    outline as outline_svc,
    questions as questions_svc,
    sessions as sessions_svc,
)

router = APIRouter(prefix="/tutor", tags=["tutor"])


@router.get("/questions/by-qid/{qid}")
async def get_question_by_qid(
    qid: str,
    session: AsyncSession = Depends(get_session),) -> dict[str, Any]:
    try:
        return await questions_svc.get_question(session, qid=qid)
    except questions_svc.QuestionNotFoundError:
        raise HTTPException(404, detail={"reason": "question_not_found", "qid": qid})


@router.get("/questions/by-attempt-id/{attempt_id}")
async def get_question_by_attempt_id(
    attempt_id: int,
    session: AsyncSession = Depends(get_session),) -> dict[str, Any]:
    try:
        return await questions_svc.get_question_by_attempt_id(session, attempt_id=attempt_id)
    except questions_svc.AttemptNotFoundError:
        raise HTTPException(404, detail={"reason": "attempt_not_found", "attempt_id": attempt_id})
    except questions_svc.QuestionNotFoundError:
        raise HTTPException(
            404,
            detail={"reason": "question_not_found", "attempt_id": attempt_id},
        )


@router.get("/captures/recent")
async def get_recent_captures(
    n: Annotated[int, Query(ge=1, le=50)] = 5,
    session: AsyncSession = Depends(get_session),) -> list[dict[str, Any]]:
    return await captures_svc.get_recent_captures(session, n=n)


@router.get("/sessions/latest")
async def get_latest_session_id(
    session: AsyncSession = Depends(get_session),) -> dict[str, str | None]:
    return {"test_id": await sessions_svc.get_latest_session_id(session)}


@router.get("/sessions/recent")
async def get_recent_sessions(
    n: Annotated[int, Query(ge=1, le=50)] = 5,
    session: AsyncSession = Depends(get_session),) -> list[dict[str, Any]]:
    return await sessions_svc.get_recent_sessions(session, n=n)


@router.get("/sessions/{test_id}/summary")
async def get_session_summary(
    test_id: str,
    session: AsyncSession = Depends(get_session),) -> dict[str, Any]:
    try:
        return await sessions_svc.get_session_summary(session, test_id=test_id)
    except sessions_svc.SessionNotFoundError:
        raise HTTPException(404, detail={"reason": "session_not_found", "test_id": test_id})


@router.get("/attempts/flagged")
async def get_flagged_attempts(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_session),) -> list[dict[str, Any]]:
    return await flags_svc.get_flagged_attempts(session, limit=limit)


# T22 (V-O1, V-O3, V-D1, V-M1): node-keyed tutor outline surface.
# Domain-blind — `course` is a query/route param, not implied AAMC.
@router.get("/outline/nodes/search")
async def search_outline_nodes(
    q: Annotated[str, Query()],
    course: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_session),) -> list[dict[str, Any]]:
    try:
        return await outline_svc.search_nodes(session, query=q, course_slug=course, limit=limit)
    except outline_svc.CourseNotFoundError:
        raise HTTPException(404, detail={"reason": "course_not_found", "course_slug": course})


@router.get("/outline")
async def get_outline_tree(
    course: Annotated[str, Query()],
    session: AsyncSession = Depends(get_session),) -> dict[str, Any]:
    try:
        return await outline_svc.get_outline_tree(session, course_slug=course)
    except outline_svc.CourseNotFoundError:
        raise HTTPException(404, detail={"reason": "course_not_found", "course_slug": course})


@router.get("/outline/nodes/{node_id}/subtree")
async def get_node_subtree(
    node_id: int,
    session: AsyncSession = Depends(get_session),) -> dict[str, Any]:
    try:
        return await outline_svc.get_subtree(session, node_id=node_id)
    except outline_svc.NodeNotFoundError:
        raise HTTPException(404, detail={"reason": "node_not_found", "node_id": node_id})


@router.get("/healthz")
async def healthcheck(
    session: AsyncSession = Depends(get_session),) -> dict[str, Any]:
    return await health_svc.healthcheck(session)


__all__ = ["router"]


# Suppress unused-import lint for _date / Literal (kept for future expansion).
_ = (_date, Literal)
