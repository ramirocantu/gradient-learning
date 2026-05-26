from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.recommender import recommend
from app.web.dashboard.db import get_session
from app.web.dashboard.services.mastery import build_heatmap
from app.web.dashboard.utils import get_recent_activity

router = APIRouter()


@router.get("/mastery", response_class=HTMLResponse)
async def mastery_page(request: Request, session: AsyncSession = Depends(get_session)):
    # Heatmap cells per §V29 — Wilson color / unlock ring / trajectory arrow /
    # retention badge / N<3 ghost, with CARS as a single-cell section block.
    grouped_cells = await build_heatmap(session)

    rec_result = await recommend(session, n=5)
    recommendations = rec_result.recommendations

    recent_activity = await get_recent_activity(session, limit=20)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="mastery.html",
        context={
            "grouped_cells": grouped_cells,
            "recommendations": recommendations,
            "recent_activity": recent_activity,
        },
    )
