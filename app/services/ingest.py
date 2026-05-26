"""Ingest entrypoint for `POST /api/v1/captures`.

Thin dispatcher (T3): routes a capture to its source adapter by
`payload.source` (§A plugin seam). The per-source normalization lives in
`app/services/adapters/` (UWorld = reference). Runs inside the caller's
transaction — the endpoint owns commit/rollback.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.captures import CapturePayload, IngestResponse
from app.services.adapters import get_adapter


async def ingest_capture(payload: CapturePayload, session: AsyncSession) -> IngestResponse:
    adapter = get_adapter(payload.source)
    return await adapter.ingest(payload, session)
