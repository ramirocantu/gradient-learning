"""POST /api/v1/captures — ingest endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.captures import CapturePayload, IngestResponse
from app.services.adapters import UnknownCourseError, UnknownSourceError
from app.services.ingest import ingest_capture

router = APIRouter()


@router.post("/captures", response_model=IngestResponse)
async def post_capture(
    payload: CapturePayload,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    try:
        return await ingest_capture(payload, session)
    except (UnknownSourceError, UnknownCourseError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
