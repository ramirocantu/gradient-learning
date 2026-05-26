"""Sessions dashboard routes (Ticket 6.9d).

GET /sessions             — list recent 20 sessions + unsessioned bucket
GET /sessions/{test_id}   — single session detail
GET /sessions/unsessioned — canonical URL for the NULL-test_id bucket
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.web.dashboard.db import get_session
from app.web.dashboard.services.sessions import get_session_detail, list_sessions

router = APIRouter()


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_index(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    rows = await list_sessions(session, limit=20)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="sessions.html",
        context={"sessions": rows},
    )


@router.get("/sessions/{test_id}", response_class=HTMLResponse)
async def session_detail(
    request: Request,
    test_id: str,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    # /sessions/unsessioned routes to the NULL aggregation; real UWorld test_ids
    # are all-digits so the keyword cannot collide.
    detail = await get_session_detail(
        session,
        test_id=None if test_id == "unsessioned" else test_id,
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="session not found")
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="session_detail.html",
        context={"detail": detail},
    )
