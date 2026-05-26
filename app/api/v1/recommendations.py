"""Recommendations endpoint (Ticket 5.2).

GET /api/v1/recommendations/study-next

Localhost-only, no auth — same posture as analytics/analyzer endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.models.outline import ContentCategory, FoundationalConcept, Section
from app.schemas.recommendations import StudyNextResponse, StudyRecommendationOut
from app.services.recommender import MIN_ATTEMPTS, recommend

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/study-next", response_model=StudyNextResponse)
async def get_study_next(
    # 9.0: bumped cap from 20 to 50 to match MCP get_recommendations clamp.
    n: int = Query(default=5, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
) -> StudyNextResponse:
    result = await recommend(session, n=n)

    # 9.0: resolve CC code → section code so MCP can write plan items in one turn.
    cc_section_rows = (
        await session.execute(
            select(ContentCategory.code, Section.code)
            .join(
                FoundationalConcept,
                FoundationalConcept.id == ContentCategory.foundational_concept_id,
            )
            .join(Section, Section.id == FoundationalConcept.section_id)
        )
    ).all()
    section_by_cc_code: dict[str, str] = {
        cc_code: sec_code for cc_code, sec_code in cc_section_rows
    }

    recs_out: list[StudyRecommendationOut] = []
    for r in result.recommendations:
        out = StudyRecommendationOut.model_validate(r)
        if r.kind == "topic_weakness" and r.code:
            out.resolved_section_code = section_by_cc_code.get(r.code)
        recs_out.append(out)

    return StudyNextResponse(
        recommendations=recs_out,
        total_candidates_scored=result.total_candidates_scored,
        min_attempts_threshold=MIN_ATTEMPTS,
    )
