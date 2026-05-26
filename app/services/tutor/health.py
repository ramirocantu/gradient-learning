from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.captures import Attempt
from app.models.features import QuestionFeatures


async def healthcheck(session: AsyncSession) -> dict[str, Any]:
    """DB reachability + sanity counters + configured BACKEND_BASE_URL.

    `recommender_ready` is coarse: any QuestionFeatures row exists. A real
    recommender call could still fail for other reasons (no eligible topics,
    all attempts orphaned) — this flag answers "has Phase 4 batch ever run",
    not "will the recommender succeed".

    Never raises. Failures surface via `db_error` so the tutor can fail loudly
    on a misconfigured install without crashing the process.
    """
    try:
        row = (
            await session.execute(
                select(
                    func.count(Attempt.id).label("attempt_count"),
                    func.max(Attempt.attempted_at).label("latest_attempt_at"),
                )
            )
        ).one()
        attempt_count = int(row.attempt_count or 0)
        latest_attempt_at = row.latest_attempt_at.isoformat() if row.latest_attempt_at else None

        features_count = (
            await session.execute(select(func.count(QuestionFeatures.question_id)))
        ).scalar_one()
        recommender_ready = bool(features_count)

        return {
            "db_reachable": True,
            "db_error": None,
            "attempt_count": attempt_count,
            "latest_attempt_at": latest_attempt_at,
            "recommender_ready": recommender_ready,
            "backend_base_url": settings.BACKEND_BASE_URL,
        }
    except SQLAlchemyError as exc:
        return {
            "db_reachable": False,
            "db_error": str(exc),
            "attempt_count": 0,
            "latest_attempt_at": None,
            "recommender_ready": False,
            "backend_base_url": settings.BACKEND_BASE_URL,
        }
