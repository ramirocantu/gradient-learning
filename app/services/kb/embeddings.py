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


# V-E1 dim guard. The DB column is JSONB placeholder until pgvector
# lands (T24 note); until then, dim enforcement is app-level. Any
# embedding whose length does not match the expected dim for the
# configured model raises ``DimMismatchError`` before persistence —
# this keeps the "⊥ mixed-dim vectors in one column" rule meaningful
# even without a typed column to reject the bad row at the DB layer.
# A model not in this map is treated as "unknown dim" (validation
# skipped); callers who care about strict enforcement should add an
# entry or set the version explicitly.
EXPECTED_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
}


class DimMismatchError(ValueError):
    """V-E1: an embedding vector's length disagrees with EXPECTED_DIMS
    for the configured ``EMBEDDING_MODEL``. Raising before
    ``session.add`` keeps the content_embeddings column homogeneous.
    """


@dataclass
class EmbedResult:
    row: ContentEmbedding
    reused: bool        # True when an existing same-version row was returned
    tokens: int         # OpenAI usage.prompt_tokens, 0 if reused


def expected_dim(model: str | None = None) -> int | None:
    """Return the canonical dim for ``model`` (defaults to the configured
    ``EMBEDDING_MODEL``), or ``None`` when the model isn't in the map.
    """

    return EXPECTED_DIMS.get(model or settings.EMBEDDING_MODEL)


def current_version() -> str:
    """V-E1 stamp. Format: ``<model>-v2``. Bump the suffix when a non-model
    change (e.g. truncation policy, embed-input text) invalidates prior rows
    but the model name itself stays put; otherwise the model name carries the
    dim identity (text-embedding-3-small ⇒ 1536) and a rotation triggers a
    natural version bump.

    v1 → v2: outline nodes now embed their full ``>>`` path, not the bare
    leaf name (B / recall fix). Old name-vectors are not comparable to the
    new path-vectors, so the whole corpus re-embeds under v2 (V-E1: ⊥ mixed
    embed-policy in one column).
    """

    return f"{settings.EMBEDDING_MODEL}-v2"


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

    # V-E1: enforce homogeneous dim per content_embeddings column.
    exp = expected_dim()
    if exp is not None and len(vec) != exp:
        raise DimMismatchError(
            f"embedding dim {len(vec)} != expected {exp} for model "
            f"{settings.EMBEDDING_MODEL!r}; bump EMBEDDING_MODEL "
            "(and re-embed under a new version) to rotate dim."
        )

    row = ContentEmbedding(
        entity_kind=entity_kind,
        entity_id=entity_id,
        embedding=vec,
        embedding_version=version,
    )
    session.add(row)
    await session.flush()
    return EmbedResult(row=row, reused=False, tokens=tokens)
