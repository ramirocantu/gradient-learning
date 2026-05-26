"""Recommendations dashboard (Phase 6.5 recommender view).

Relocated from /study-plan/recommendations as part of T79 / V61 — the
Phase 7 study-plan layer was cut, but the recommender service (Phase
5.2) and its dashboard page survive on the standalone /recommendations
URL.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommender import recommend
from app.web.dashboard.db import get_session


router = APIRouter()


@router.get("/recommendations", response_class=HTMLResponse)
async def recommendations_page(
    request: Request,
    n: int = Query(default=10, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
):
    result = await recommend(session, n=n)
    topic_recs = [r for r in result.recommendations if r.kind == "topic_weakness"]
    pattern_recs = [r for r in result.recommendations if r.kind == "feature_pattern"]
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="recommendations.html",
        context={
            "topic_recs": topic_recs,
            "pattern_recs": pattern_recs,
            "n": n,
            "total": len(result.recommendations),
            "total_candidates_scored": result.total_candidates_scored,
        },
    )
