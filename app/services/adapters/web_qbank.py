"""Generic web-Qbank source adapter (§A).

Any browser-extension capture from a non-UWorld web question bank routes here
by ``source='web-qbank'``. The wire shape is the same extension
``CapturePayload``; normalization is the shared, source-agnostic
``extension_capture.normalize_capture`` — proving the seam takes a new source
without touching the ingest entrypoint (§A). MCAT-specific fields
(``uworld_*``) are simply left empty by a generic qbank.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.captures import CapturePayload, IngestResponse
from app.services.adapters import register_adapter
from app.services.adapters.extension_capture import normalize_capture


class WebQbankAdapter:
    """Generic web-Qbank capture → Question + Attempt (§A)."""

    source = "web-qbank"

    async def ingest(self, payload: CapturePayload, session: AsyncSession) -> IngestResponse:
        return await normalize_capture(payload, session)


register_adapter(WebQbankAdapter())
