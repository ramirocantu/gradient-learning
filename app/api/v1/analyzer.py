"""Analyzer API — feature extraction and pattern analysis endpoints."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from anthropic import AsyncAnthropic
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.config import settings
from app.database import AsyncSessionLocal
from app.schemas.analyzer import InsightReportOut, InsightSynthesisResponse
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.patterns import AnalysisFilter, analyze
from app.services.analyzer.synthesizer import insights_for_filter
from app.services.analyzer.synthesizer_cache import SynthesizerCache
from app.services.analyzer.worker import run_extraction

router = APIRouter(prefix="/analyzer", tags=["analyzer"])


class ExtractRequest(BaseModel):
    missed_only: bool = False
    since: date | None = None
    limit: int | None = None
    max_cost_usd: float | None = None
    concurrency: int = 5


@router.post("/extract")
async def extract_features(body: ExtractRequest) -> dict[str, Any]:
    """Trigger batch feature extraction and return summary as JSON.

    Synchronous — for ~75 questions at concurrency=5 expect ~75s response time.
    No auth required (localhost-only admin endpoint).
    """
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    cache = FeatureExtractorCache(settings.FEATURE_EXTRACTOR_CACHE_PATH)
    try:
        summary = await run_extraction(
            AsyncSessionLocal,
            anthropic_client=client,
            cache=cache,
            missed_only=body.missed_only,
            since=body.since,
            limit=body.limit,
            max_cost_usd=body.max_cost_usd,
            concurrency=body.concurrency,
        )
    finally:
        cache.close()
    return summary.as_dict()


@router.get("/patterns", response_model=InsightReportOut)
async def get_patterns(
    section: Annotated[Literal["CP", "CARS", "BB", "PS"] | None, Query()] = None,
    content_category: Annotated[str | None, Query()] = None,
    topic_id: Annotated[int | None, Query()] = None,
    skill: Annotated[int | None, Query(ge=1, le=4)] = None,
    since: Annotated[date | None, Query()] = None,
    until: Annotated[date | None, Query()] = None,
    min_sample_size: Annotated[int, Query(ge=1)] = 10,
    session: AsyncSession = Depends(get_session),
) -> InsightReportOut:
    """Compute accuracy deltas per feature value for the given scope.

    No auth required (localhost-only admin endpoint).
    """
    af = AnalysisFilter(
        section_code=section,
        content_category_code=content_category,
        topic_id=topic_id,
        skill=skill,
        since=since,
        until=until,
        min_sample_size=min_sample_size,
    )
    report = await analyze(af, session)
    return InsightReportOut.model_validate(report)


@router.get("/insights", response_model=InsightSynthesisResponse)
async def get_insights(
    section: Annotated[Literal["CP", "CARS", "BB", "PS"] | None, Query()] = None,
    content_category: Annotated[str | None, Query()] = None,
    topic_id: Annotated[int | None, Query()] = None,
    skill: Annotated[int | None, Query(ge=1, le=4)] = None,
    since: Annotated[date | None, Query()] = None,
    until: Annotated[date | None, Query()] = None,
    min_sample_size: Annotated[int, Query(ge=1)] = 10,
    bust_cache: Annotated[bool, Query()] = False,
    session: AsyncSession = Depends(get_session),
) -> InsightSynthesisResponse:
    """Synthesize pattern findings into readable markdown prose.

    No auth required (localhost-only endpoint).
    Pass bust_cache=true to force a fresh LLM call even if a cached synthesis exists.
    """
    af = AnalysisFilter(
        section_code=section,
        content_category_code=content_category,
        topic_id=topic_id,
        skill=skill,
        since=since,
        until=until,
        min_sample_size=min_sample_size,
    )
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    cache = SynthesizerCache(settings.SYNTHESIZER_CACHE_PATH)
    try:
        synthesis = await insights_for_filter(
            af,
            session,
            anthropic_client=client,
            cache=cache,
            bust_cache=bust_cache,
        )
    finally:
        cache.close()
    return InsightSynthesisResponse.model_validate(synthesis)
