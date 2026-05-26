"""Source-adapter registry (§A plugin seam).

A source adapter normalizes a raw `capture → {Question, Attempt}` and is keyed
by its `source` discriminator. `/api/v1/captures` dispatches by `payload.source`
through `get_adapter`. UWorld is the reference adapter; new sources (web-Qbank,
manual, pdf-qset) register here without touching the ingest entrypoint.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.captures import CapturePayload, IngestResponse


class UnknownSourceError(ValueError):
    """Raised when a capture's `source` has no registered adapter."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(
            f"no source adapter registered for {source!r}; "
            f"known: {sorted(_REGISTRY)}"
        )


@runtime_checkable
class SourceAdapter(Protocol):
    """capture → normalized {Question, Attempt}, keyed by `source` (§A)."""

    source: str

    async def ingest(self, payload: CapturePayload, session: AsyncSession) -> IngestResponse: ...


_REGISTRY: dict[str, SourceAdapter] = {}


def register_adapter(adapter: SourceAdapter) -> None:
    _REGISTRY[adapter.source] = adapter


def get_adapter(source: str) -> SourceAdapter:
    try:
        return _REGISTRY[source]
    except KeyError as exc:
        raise UnknownSourceError(source) from exc


def registered_sources() -> list[str]:
    return sorted(_REGISTRY)


# Register built-in adapters (import for side effect). Must come after
# register_adapter is defined to avoid a circular-import failure.
from app.services.adapters import uworld as _uworld  # noqa: E402,F401
from app.services.adapters import web_qbank as _web_qbank  # noqa: E402,F401
from app.services.adapters import manual as _manual  # noqa: E402,F401

# pdf-qset adapter deferred (T33 "hardest, last"): a PDF question set needs a
# file-upload payload + a multi-question (no-Attempt) response shape, distinct
# from the single-capture CapturePayload above. Register it here once built.
