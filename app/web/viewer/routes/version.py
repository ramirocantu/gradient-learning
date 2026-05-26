from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.web.viewer.db import get_session

router = APIRouter()


@router.get("/_version", response_class=JSONResponse)
async def version(session: AsyncSession = Depends(get_session)) -> dict[str, str]:
    """Cheap freshness probe for the auto-refresh poller.

    Returns the greatest of: questions.last_updated_at, raw_captures.captured_at,
    attempts.attempted_at, media.first_seen_at. Clients compare across polls and
    reload the page when the value changes.
    """
    sql = text(
        """
        SELECT GREATEST(
          (SELECT COALESCE(MAX(last_updated_at), 'epoch'::timestamptz) FROM questions),
          (SELECT COALESCE(MAX(captured_at),     'epoch'::timestamptz) FROM raw_captures),
          (SELECT COALESCE(MAX(attempted_at),    'epoch'::timestamptz) FROM attempts),
          (SELECT COALESCE(MAX(first_seen_at),   'epoch'::timestamptz) FROM media)
        ) AS v
        """
    )
    v = (await session.execute(sql)).scalar_one()
    return {"v": v.isoformat() if v else ""}
