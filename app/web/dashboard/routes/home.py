from datetime import datetime, timezone
from datetime import timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question
from app.services.analytics import compute_mastery
from app.services.recommender import recommend
from app.web.dashboard.db import get_session
from app.web.dashboard.utils import get_relative_time, get_recent_activity

router = APIRouter()


def get_streak(attempts: list[Attempt]) -> int:
    """Calculate the streak based on unique calendar days of attempts in UTC."""
    if not attempts:
        return 0

    # Extract unique dates in UTC
    attempt_dates = {a.attempted_at.astimezone(timezone.utc).date() for a in attempts}
    if not attempt_dates:
        return 0

    today = datetime.now(timezone.utc).date()

    streak = 0
    current_date = today

    # If there are no attempts today, but there are yesterday, the streak can still count from yesterday.
    # The requirement: "If today has no attempts but yesterday did, streak is 0."
    # Wait, the kickoff said: "ending today where at least one attempt occurred. If today has no attempts but yesterday did, streak is 0."
    # Let me follow that strictly.
    if today not in attempt_dates:
        return 0

    while current_date in attempt_dates:
        streak += 1
        # move back one day
        current_date = (
            current_date.replace(day=current_date.day - 1)
            if current_date.day > 1
            else (
                current_date.replace(month=current_date.month - 1, day=1)
                if current_date.month > 1
                else current_date.replace(year=current_date.year - 1, month=12, day=1)
            )
        )
        # better way to subtract days:

        current_date = current_date - timedelta(days=1)

    return streak


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, session: AsyncSession = Depends(get_session)):
    # 1. Headline numbers
    mastery = await compute_mastery(session)

    # We need all attempts to calculate the streak
    attempts_query = select(Attempt).order_by(Attempt.attempted_at.desc())
    all_attempts_result = await session.execute(attempts_query)
    all_attempts = list(all_attempts_result.scalars().all())

    streak = get_streak(all_attempts)

    accuracy_str = "—"
    total_unique_attempted = sum(s.attempts for s in mastery.by_section)
    if total_unique_attempted > 0:
        overall_correct = sum(s.correct for s in mastery.by_section)
        accuracy_val = (overall_correct / total_unique_attempted) * 100
        accuracy_str = f"{accuracy_val:.1f}%"

    # 2. Study Next (top 5 recommendations)
    rec_result = await recommend(session, n=5)
    recommendations = rec_result.recommendations

    # 3. Recent activity (last 5 attempts)
    recent_activity = await get_recent_activity(session, limit=5)

    # 4. Subtle footer (last captured)
    last_captured_query = (
        select(Question.first_seen_at).order_by(Question.first_seen_at.desc()).limit(1)
    )
    last_captured_result = await session.execute(last_captured_query)
    last_captured_val = last_captured_result.scalar_one_or_none()
    last_captured_str = get_relative_time(last_captured_val) if last_captured_val else "Never"

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "mastery": mastery,
            "accuracy_str": accuracy_str,
            "streak": streak,
            "recommendations": recommendations,
            "recent_activity": recent_activity,
            "last_captured": last_captured_str,
        },
    )
