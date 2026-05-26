"""Embedding write+versioning seam (T26, V-E1, V16).

Wraps the OpenAI embeddings SDK call. The client is injected so tests
mock at the SDK boundary (V16). Versioning rule: every row is stamped
with ``embedding_version``; a provider/dim/model change ⇒ bump version
+ full re-embed (V-E1, ⊥ mixed-dim vectors in one column).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.content_embedding import ContentEmbedding

_logger = logging.getLogger("app.services.kb.embeddings")


@dataclass
class EmbedResult:
    row: ContentEmbedding
    reused: bool        # True when an existing same-version row was returned
    tokens: int         # OpenAI usage.prompt_tokens, 0 if reused


def current_version() -> str:
    """V-E1 stamp. Format: ``<model>-v1``. Bump the suffix when a non-model
    change (e.g. truncation policy) invalidates prior rows but the model
    name itself stays put; otherwise the model name carries the dim
    identity (text-embedding-3-small ⇒ 1536) and a rotation triggers a
    natural version bump.
    """

    return f"{settings.EMBEDDING_MODEL}-v1"


async def embed_and_persist(
    session: AsyncSession,
    *,
    openai_client: Any,
    entity_kind: str,
    entity_id: int,
    text: str,
    version: str | None = None,
) -> EmbedResult:
    """Embed ``text`` and persist a ``content_embeddings`` row.

    Idempotent: a row matching ``(entity_kind, entity_id, version)`` is
    returned unchanged (no OpenAI call). The UQ on that triple lets
    multiple versions coexist briefly during a V-E1 re-embed sweep —
    callers pass the new version explicitly.
    """

    version = version or current_version()

    existing = (
        await session.execute(
            select(ContentEmbedding).where(
                ContentEmbedding.entity_kind == entity_kind,
                ContentEmbedding.entity_id == entity_id,
                ContentEmbedding.embedding_version == version,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return EmbedResult(row=existing, reused=True, tokens=0)

    resp = await openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    vec = list(resp.data[0].embedding)
    usage = getattr(resp, "usage", None)
    tokens = int(getattr(usage, "prompt_tokens", 0)) if usage is not None else 0

    row = ContentEmbedding(
        entity_kind=entity_kind,
        entity_id=entity_id,
        embedding=vec,
        embedding_version=version,
    )
    session.add(row)
    await session.flush()
    return EmbedResult(row=row, reused=False, tokens=tokens)
