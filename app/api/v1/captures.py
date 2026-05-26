"""POST /api/v1/captures — ingest endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, verify_coach_token
from app.schemas.captures import CapturePayload, IngestResponse
from app.services.ingest import ingest_capture

router = APIRouter()


@router.post("/captures", response_model=IngestResponse)
async def post_capture(
    payload: CapturePayload,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> IngestResponse:
    return await ingest_capture(payload, session)
