"""Analytics endpoints (Ticket 5.1).

GET /api/v1/analytics/mastery

Localhost-only, no auth — same posture as /api/v1/admin/.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.analytics import MasteryReportOut
from app.services.analytics import compute_mastery

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/mastery", response_model=MasteryReportOut)
async def get_mastery(
    session: AsyncSession = Depends(get_session),
) -> MasteryReportOut:
    report = await compute_mastery(session)
    return MasteryReportOut.model_validate(report)
