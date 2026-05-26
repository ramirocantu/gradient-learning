"""Anki dashboard page (SPEC §T25 + §T69, P11 + P11+).

GET  /anki              — review queue + sync runs + assignments panel
                          + review-push panel + plan-adherence chip
                          + load-config form + Run-now buttons.
POST /anki/load-config  — upsert the V59 singleton + redirect back to /anki.

Per §V18, queries hit `app.services.anki.*` and SQLAlchemy directly —
no HTTP self-call to the JSON API. Templates render read-only; no
business logic in Jinja.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import (
    AnkiAssignment,
    AnkiCardReview,
    AnkiLoadConfig,
    AnkiReview,
)
from app.models.task_run import TaskRun
from app.services.anki.load_adherence import compute_load_adherence
from app.services.anki.queries import list_review_queue
from app.web.dashboard.db import get_session
from app.web.dashboard.services.anki_scope import attach_scope_labels


router = APIRouter()


_REVIEW_QUEUE_LIMIT = 20
_SYNC_HISTORY_LIMIT = 10
_JOB_NAME = "run_anki_sync"
_ASSIGN_PENDING_WINDOW_DAYS = 30
_ASSIGN_COMPLETED_WINDOW_DAYS = 7
_PUSH_RECENT_WINDOW_DAYS = 30
_BURNDOWN_PAST_DAYS = 14
_BURNDOWN_FUTURE_DAYS = 14
_DEFAULT_CARD_BUDGET = 200
_DEFAULT_MINUTES_BUDGET = Decimal("60")


async def _daily_review_counts(
    session: AsyncSession, *, now: datetime, past_days: int
) -> dict[date, int]:
    """count(*) of `anki_card_reviews` rows per UTC date over the trailing
    window. Excludes `type='learn'` to mirror V27 / load_adherence.py."""
    window_start = now - timedelta(days=past_days)
    day_expr = func.date(AnkiCardReview.reviewed_at)
    rows = (
        await session.execute(
            select(day_expr, func.count())
            .where(AnkiCardReview.reviewed_at >= window_start)
            .where(AnkiCardReview.reviewed_at <= now)
            .where(AnkiCardReview.type != "learn")
            .group_by(day_expr)
        )
    ).all()
    return {r[0]: int(r[1]) for r in rows}


def _burndown_series(
    today_: date,
    counts_by_day: dict[date, int],
    *,
    past_days: int,
    future_days: int,
    projected_daily: int,
) -> list[dict]:
    series: list[dict] = []
    for offset in range(-past_days + 1, 1):
        day = today_ + timedelta(days=offset)
        series.append({"day": day, "count": counts_by_day.get(day, 0), "kind": "actual"})
    for offset in range(1, future_days + 1):
        day = today_ + timedelta(days=offset)
        series.append({"day": day, "count": projected_daily, "kind": "projected"})
    return series


def _sparkline_svg(series: list[dict], *, width: int = 280, height: int = 60) -> dict:
    if not series:
        return {
            "width": width,
            "height": height,
            "actual_path": "",
            "projected_path": "",
            "max_count": 0,
        }
    max_count = max((p["count"] for p in series), default=0) or 1
    n = len(series)
    pad_x = 4
    pad_y = 4
    inner_w = width - 2 * pad_x
    inner_h = height - 2 * pad_y

    def _xy(i: int, count: int) -> tuple[float, float]:
        x = pad_x + (inner_w * i / max(n - 1, 1))
        y = pad_y + inner_h - (inner_h * count / max_count)
        return x, y

    actual_pts: list[str] = []
    projected_pts: list[str] = []
    boundary_index = -1
    for i, p in enumerate(series):
        if p["kind"] == "actual":
            boundary_index = i
    for i, p in enumerate(series):
        x, y = _xy(i, p["count"])
        if p["kind"] == "actual":
            actual_pts.append(f"{x:.1f},{y:.1f}")
        else:
            projected_pts.append(f"{x:.1f},{y:.1f}")
    if boundary_index >= 0 and boundary_index + 1 < n and projected_pts:
        bx, by = _xy(boundary_index, series[boundary_index]["count"])
        projected_pts.insert(0, f"{bx:.1f},{by:.1f}")
    return {
        "width": width,
        "height": height,
        "actual_path": " ".join(actual_pts),
        "projected_path": " ".join(projected_pts),
        "max_count": max_count,
    }


async def _read_or_seed_load_config(session: AsyncSession) -> AnkiLoadConfig:
    row = await session.get(AnkiLoadConfig, 1)
    if row is None:
        row = AnkiLoadConfig(
            id=1,
            daily_card_review_budget=_DEFAULT_CARD_BUDGET,
            daily_minutes_budget=_DEFAULT_MINUTES_BUDGET,
        )
        session.add(row)
        await session.flush()
    return row


@router.get("/anki", response_class=HTMLResponse)
async def anki_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    templates = request.app.state.templates
    now = datetime.now(timezone.utc)

    cards = await list_review_queue(session, limit=_REVIEW_QUEUE_LIMIT)

    sync_runs = (
        (
            await session.execute(
                select(TaskRun)
                .where(TaskRun.job_name == _JOB_NAME)
                .order_by(desc(TaskRun.started_at))
                .limit(_SYNC_HISTORY_LIMIT)
            )
        )
        .scalars()
        .all()
    )

    # V51 status × time-window grouping for the assignments panel.
    pending_assignments = (
        (
            await session.execute(
                select(AnkiAssignment)
                .where(AnkiAssignment.status == "pending")
                .where(AnkiAssignment.scheduled_unlock_at >= now)
                .where(
                    AnkiAssignment.scheduled_unlock_at
                    <= now + timedelta(days=_ASSIGN_PENDING_WINDOW_DAYS)
                )
                .order_by(AnkiAssignment.scheduled_unlock_at.asc())
            )
        )
        .scalars()
        .all()
    )
    unlocked_assignments = (
        (
            await session.execute(
                select(AnkiAssignment)
                .where(AnkiAssignment.status == "unlocked")
                .order_by(desc(AnkiAssignment.actual_unlock_at))
            )
        )
        .scalars()
        .all()
    )
    completed_assignments = (
        (
            await session.execute(
                select(AnkiAssignment)
                .where(AnkiAssignment.status == "completed")
                .where(
                    AnkiAssignment.updated_at >= now - timedelta(days=_ASSIGN_COMPLETED_WINDOW_DAYS)
                )
                .order_by(desc(AnkiAssignment.updated_at))
            )
        )
        .scalars()
        .all()
    )

    # V53 review-push panel: pending + last-N recent non-pending.
    today = now.date()
    pending_pushes = (
        (
            await session.execute(
                select(AnkiReview)
                .where(AnkiReview.status == "pending")
                .order_by(AnkiReview.review_date.asc())
            )
        )
        .scalars()
        .all()
    )
    recent_pushes = (
        (
            await session.execute(
                select(AnkiReview)
                .where(AnkiReview.status != "pending")
                .where(AnkiReview.review_date >= today - timedelta(days=_PUSH_RECENT_WINDOW_DAYS))
                .order_by(desc(AnkiReview.review_date))
            )
        )
        .scalars()
        .all()
    )

    # Enrich assignment rows with scope_label + scope_url so templates
    # can render a name (and link) instead of the raw topic id / CC code.
    pending_assignments = list(pending_assignments)
    unlocked_assignments = list(unlocked_assignments)
    completed_assignments = list(completed_assignments)
    await attach_scope_labels(
        session,
        pending_assignments + unlocked_assignments + completed_assignments,
    )

    # V59 singleton + V54 deterministic adherence.
    load_config = await _read_or_seed_load_config(session)
    adherence = await compute_load_adherence(session)

    # T70 burndown sparkline (folded from cut /study-plan per V61).
    counts_by_day = await _daily_review_counts(session, now=now, past_days=_BURNDOWN_PAST_DAYS)
    burndown_series = _burndown_series(
        today,
        counts_by_day,
        past_days=_BURNDOWN_PAST_DAYS,
        future_days=_BURNDOWN_FUTURE_DAYS,
        projected_daily=adherence.projected_daily_load,
    )
    sparkline = _sparkline_svg(burndown_series)

    return templates.TemplateResponse(
        request=request,
        name="anki.html",
        context={
            "cards": cards,
            "sync_runs": list(sync_runs),
            "pending_assignments": list(pending_assignments),
            "unlocked_assignments": list(unlocked_assignments),
            "completed_assignments": list(completed_assignments),
            "pending_pushes": list(pending_pushes),
            "recent_pushes": list(recent_pushes),
            "load_config": load_config,
            "adherence": adherence,
            "burndown_series": burndown_series,
            "sparkline": sparkline,
            "burndown_past_days": _BURNDOWN_PAST_DAYS,
            "burndown_future_days": _BURNDOWN_FUTURE_DAYS,
            "assign_pending_window_days": _ASSIGN_PENDING_WINDOW_DAYS,
            "assign_completed_window_days": _ASSIGN_COMPLETED_WINDOW_DAYS,
            "push_recent_window_days": _PUSH_RECENT_WINDOW_DAYS,
        },
    )


@router.post("/anki/load-config")
async def upsert_load_config_dashboard(
    daily_card_review_budget: int = Form(...),
    daily_minutes_budget: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    """Dashboard-side upsert. Local-only single-user app, so no
    X-Coach-Token here — the route is in-process behind the same UI
    chrome the user is already viewing.

    Validates the V59 positivity constraints in Python (mirrors the
    DB CHECK + the Pydantic API schema). Invalid input redirects back
    with an error flag in the query string so the template can render
    a banner without bouncing through 422.
    """
    try:
        minutes = Decimal(daily_minutes_budget)
    except (InvalidOperation, TypeError, ValueError):
        return RedirectResponse(
            url="/anki?load_config_error=invalid_minutes",
            status_code=303,
        )
    if daily_card_review_budget <= 0 or minutes <= 0:
        return RedirectResponse(
            url="/anki?load_config_error=non_positive",
            status_code=303,
        )

    row = await session.get(AnkiLoadConfig, 1)
    if row is None:
        row = AnkiLoadConfig(
            id=1,
            daily_card_review_budget=daily_card_review_budget,
            daily_minutes_budget=minutes,
        )
        session.add(row)
    else:
        row.daily_card_review_budget = daily_card_review_budget
        row.daily_minutes_budget = minutes
        row.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return RedirectResponse(url="/anki?load_config_saved=1", status_code=303)
