"""Central AsyncOpenAI client factory.

One place to build the client so transient-error retries (V41) and the
optional `OPENAI_BASE_URL` knob (for an OpenAI-compatible local server)
live in a single seam. Tests mock at this boundary (V16).
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings


def build_openai_client(*, max_retries: int = 5) -> AsyncOpenAI:
    """V41: SDK-level retry budget ≥5 for transient OpenAI errors. V16: this
    factory is the sole construction point that production wires through, so
    tests can patch `app.services.llm.client.build_openai_client` to return a
    fake client without touching every call site.
    """
    kwargs: dict[str, object] = {
        "api_key": settings.OPENAI_API_KEY or "test",
        "max_retries": max_retries,
    }
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return AsyncOpenAI(**kwargs)  # type: ignore[arg-type]
