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

from app.api.deps import get_session, verify_coach_token
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
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> dict[str, Any]:
    try:
        return await questions_svc.get_question(session, qid=qid)
    except questions_svc.QuestionNotFoundError:
        raise HTTPException(404, detail={"reason": "question_not_found", "qid": qid})


@router.get("/questions/by-attempt-id/{attempt_id}")
async def get_question_by_attempt_id(
    attempt_id: int,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> dict[str, Any]:
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
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> list[dict[str, Any]]:
    return await captures_svc.get_recent_captures(session, n=n)


@router.get("/sessions/latest")
async def get_latest_session_id(
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> dict[str, str | None]:
    return {"test_id": await sessions_svc.get_latest_session_id(session)}


@router.get("/sessions/recent")
async def get_recent_sessions(
    n: Annotated[int, Query(ge=1, le=50)] = 5,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> list[dict[str, Any]]:
    return await sessions_svc.get_recent_sessions(session, n=n)


@router.get("/sessions/{test_id}/summary")
async def get_session_summary(
    test_id: str,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> dict[str, Any]:
    try:
        return await sessions_svc.get_session_summary(session, test_id=test_id)
    except sessions_svc.SessionNotFoundError:
        raise HTTPException(404, detail={"reason": "session_not_found", "test_id": test_id})


@router.get("/attempts/flagged")
async def get_flagged_attempts(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> list[dict[str, Any]]:
    return await flags_svc.get_flagged_attempts(session, limit=limit)


@router.get("/outline/topics/search")
async def search_topics(
    q: Annotated[str, Query()],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> list[dict[str, Any]]:
    return await outline_svc.search_topics(session, query=q, limit=limit)


@router.get("/outline")
async def get_aamc_outline(
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> dict[str, Any]:
    return await outline_svc.get_aamc_outline(session)


@router.get("/healthz")
async def healthcheck(
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> dict[str, Any]:
    return await health_svc.healthcheck(session)


__all__ = ["router"]


# Suppress unused-import lint for _date / Literal (kept for future expansion).
_ = (_date, Literal)
