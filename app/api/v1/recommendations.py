"""Recommendations endpoint.

GET /api/v1/recommendations/study-next

Localhost-only, no auth — same posture as analytics/analyzer endpoints.

T12 port: the previous CC-code → section-code resolution is gone — the
generalized outline_nodes schema has no codes (V-O3). When the recommender
is restored in T14, `resolved_section_code` either disappears from the
response or is resolved via node-path lookup against `OutlineLookup`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.recommendations import StudyNextResponse, StudyRecommendationOut
from app.services.recommender import MIN_ATTEMPTS, recommend

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/study-next", response_model=StudyNextResponse)
async def get_study_next(
    n: int = Query(default=5, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
) -> StudyNextResponse:
    result = await recommend(session, n=n)

    recs_out: list[StudyRecommendationOut] = [
        StudyRecommendationOut.model_validate(r) for r in result.recommendations
    ]

    return StudyNextResponse(
        recommendations=recs_out,
        total_candidates_scored=result.total_candidates_scored,
        min_attempts_threshold=MIN_ATTEMPTS,
    )
