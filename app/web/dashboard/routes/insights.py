"""Pattern Insights page (Ticket 6.4).

Renders LLM synthesis prose plus structured FeatureFinding cards for the
current filter scope. Calls backend services directly — no HTTP hop.
"""

from __future__ import annotations

from datetime import date
from typing import Literal
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.services.llm.client import build_openai_client
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.patterns import AnalysisFilter, analyze
from app.services.analyzer.synthesizer import synthesize
from app.services.analyzer.synthesizer_cache import SynthesizerCache
from app.services.analyzer.worker import run_extraction
from app.web.dashboard.db import get_session

router = APIRouter()


@router.get("/insights", response_class=HTMLResponse)
async def insights_page(
    request: Request,
    section: Literal["CP", "CARS", "BB", "PS"] | None = Query(default=None),
    skill: int | None = Query(default=None, ge=1, le=4),
    since: date | None = Query(default=None),
    min_sample_size: int = Query(default=10, ge=1),
    bust_cache: bool = Query(default=False),
    session: AsyncSession = Depends(get_session),
):
    af = AnalysisFilter(
        section_code=section,
        skill=skill,
        since=since,
        min_sample_size=min_sample_size,
    )

    report = await analyze(af, session)

    client = build_openai_client(max_retries=5)
    cache = SynthesizerCache(settings.SYNTHESIZER_CACHE_PATH)
    try:
        synthesis = await synthesize(
            report,
            openai_client=client,
            cache=cache,
            bust_cache=bust_cache,
            run_llm=bust_cache,
        )
    finally:
        cache.close()

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="insights.html",
        context={
            "synthesis": synthesis,
            "report": report,
            "filter": af,
            "section": section,
            "skill": skill,
            "since": since,
            "min_sample_size": min_sample_size,
        },
    )


@router.post("/insights/run-extraction", response_class=RedirectResponse)
async def run_extraction_and_redirect(
    section: str | None = Query(default=None),
    skill: int | None = Query(default=None),
    since: date | None = Query(default=None),
    min_sample_size: int = Query(default=10),
):
    client = build_openai_client(max_retries=5)
    cache = FeatureExtractorCache(settings.FEATURE_EXTRACTOR_CACHE_PATH)
    try:
        await run_extraction(
            AsyncSessionLocal,
            openai_client=client,
            cache=cache,
            missed_only=False,
            limit=None,
            max_cost_usd=None,
            concurrency=5,
        )
    finally:
        cache.close()

    params: list[tuple[str, str]] = []
    if section is not None:
        params.append(("section", section))
    if skill is not None:
        params.append(("skill", str(skill)))
    if since is not None:
        params.append(("since", str(since)))
    if min_sample_size != 10:
        params.append(("min_sample_size", str(min_sample_size)))

    qs = urlencode(params)
    return RedirectResponse(url=f"/insights{'?' + qs if qs else ''}", status_code=303)
