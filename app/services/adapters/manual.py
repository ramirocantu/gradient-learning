"""Manual-entry source adapter (§A).

A user-transcribed practice question plus the answer they gave, posted with
``source='manual'`` (``extension_version='manual'``, no browser scrape). The
data still fits the normalized {Question, Attempt} shape — manual entry just
omits media/passage, which the shared normalizer handles when empty. Routes
through the same source-agnostic ``extension_capture.normalize_capture`` (§A).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.captures import CapturePayload, IngestResponse
from app.services.adapters import register_adapter
from app.services.adapters.extension_capture import normalize_capture


class ManualAdapter:
    """Manual-entry capture → Question + Attempt (§A)."""

    source = "manual"

    async def ingest(self, payload: CapturePayload, session: AsyncSession) -> IngestResponse:
        return await normalize_capture(payload, session)


register_adapter(ManualAdapter())
