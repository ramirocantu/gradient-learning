"""UWorld source adapter — reference implementation of the §A plugin seam.

UWorld is *one* source, not privileged: the capture→{Question, Attempt}
normalization lives in the shared, source-agnostic
``extension_capture.normalize_capture`` (§A — domain-blind core). This
adapter just registers under ``source='uworld'`` and delegates. Runs inside
the caller's transaction — the endpoint owns commit/rollback.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.captures import CapturePayload, IngestResponse
from app.services.adapters import register_adapter
from app.services.adapters.extension_capture import normalize_capture


class UWorldAdapter:
    """Reference source adapter (§A). UWorld capture → Question + Attempt."""

    source = "uworld"

    async def ingest(self, payload: CapturePayload, session: AsyncSession) -> IngestResponse:
        return await normalize_capture(payload, session)


register_adapter(UWorldAdapter())
