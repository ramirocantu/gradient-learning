"""Admin page: scheduler job history and manual trigger.

GET  /admin        — job history + next-run times
POST /admin/jobs/{job_name}/trigger  — triggers a scheduler job
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.admin import list_jobs_payload, trigger_job_logic
from app.models.task_run import TaskRun
from app.services.anki.queries import (
    get_anki_card_total,
    get_tag_card_coverage,
    get_tag_parse_stats,
)
from app.web.dashboard.db import get_session

router = APIRouter()

_JOBS = [
    "run_categorizer",
    "run_feature_extraction",
    "run_anki_sync",
    "run_anki_topic_resolver",
    "run_anki_assignment_unlock",
    "run_anki_assignment_complete",
    "run_anki_review",
]
_JOB_DISPLAY = {
    "run_categorizer": "Categorizer",
    "run_feature_extraction": "Feature Extractor",
    "run_anki_sync": "Anki Sync",
    "run_anki_topic_resolver": "Anki Topic Resolver",
    "run_anki_assignment_unlock": "Anki Assignment Unlock",
    "run_anki_assignment_complete": "Anki Assignment Complete",
    "run_anki_review": "Anki Review",
}


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, session: AsyncSession = Depends(get_session)):
    templates = request.app.state.templates

    job_runs: dict[str, list[TaskRun]] = {}
    for job_name in _JOBS:
        result = await session.execute(
            select(TaskRun)
            .where(TaskRun.job_name == job_name)
            .order_by(desc(TaskRun.started_at))
            .limit(10)
        )
        job_runs[job_name] = list(result.scalars().all())

    # Direct in-process call into the same helper the JSON route uses. If
    # the scheduler isn't running (e.g. SCHEDULER_ENABLED=0 in tests), the
    # helper just returns an empty list and the template renders the
    # "Scheduler offline or disabled" copy.
    next_run_times: dict[str, str | None] = {}
    for entry in await list_jobs_payload():
        next_run_times[entry["job_id"]] = entry.get("next_run_time")

    jobs = [
        {
            "name": job_name,
            "display_name": _JOB_DISPLAY[job_name],
            "runs": job_runs[job_name],
            "next_run_time": next_run_times.get(job_name),
        }
        for job_name in _JOBS
    ]

    # SPEC §T28 / §V19: tag-parse health widget. Per §V18, the dashboard calls
    # the service helper directly (no HTTP self-call).
    tag_parse_stats = await get_tag_parse_stats(session)
    coverage = await get_tag_card_coverage(session)
    total_cards = await get_anki_card_total(session)
    anki_tag_parse = _build_anki_tag_parse_context(tag_parse_stats, coverage, total_cards)

    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"jobs": jobs, "anki_tag_parse": anki_tag_parse},
    )


def _build_anki_tag_parse_context(
    stats: dict[str, int],
    coverage: dict[str, int],
    total_cards: int,
) -> dict:
    """Shape the parsed_kind counts for the template.

    Returns:
      kinds: list of {label, tag_count, tag_pct, card_count, card_pct}
        ordered (aamc_topic, aamc_cc, aamc_skill, uworld_qid, unparsed).
        `card_count` = COUNT(DISTINCT anki_card_id) WHERE parsed_kind=X;
        `card_pct` = card_count / total_cards * 100 (0 when total_cards=0).
      total: int — total tag rows (denominator for tag_pct)
      total_cards: int — denominator for card_pct (per §V23)
      unparsed_pct: float | None — None when total=0 to drive empty-state copy
    """
    ordered = ("aamc_topic", "aamc_cc", "aamc_skill", "uworld_qid", "unparsed")
    total = sum(stats.values())
    kinds = []
    for kind in ordered:
        tag_count = stats.get(kind, 0)
        card_count = coverage.get(kind, 0)
        tag_pct = (tag_count / total * 100.0) if total else 0.0
        card_pct = (card_count / total_cards * 100.0) if total_cards else 0.0
        kinds.append(
            {
                "label": kind,
                "tag_count": tag_count,
                "tag_pct": tag_pct,
                "card_count": card_count,
                "card_pct": card_pct,
            }
        )
    unparsed_pct = (stats.get("unparsed", 0) / total * 100.0) if total else None
    return {
        "kinds": kinds,
        "total": total,
        "total_cards": total_cards,
        "unparsed_pct": unparsed_pct,
    }


@router.post("/admin/jobs/{job_name}/trigger", response_class=HTMLResponse)
async def trigger_job(job_name: str, request: Request):
    if job_name not in _JOBS:
        raise HTTPException(404, detail=f"unknown job: {job_name}")

    try:
        await trigger_job_logic(job_name)
    except HTTPException as exc:
        if exc.status_code == 409:
            return HTMLResponse("Already running")
        if exc.status_code == 503:
            return HTMLResponse("Backend unreachable")
        return HTMLResponse(f"Backend error ({exc.status_code})")
    return HTMLResponse("Triggered — refresh in a moment")
