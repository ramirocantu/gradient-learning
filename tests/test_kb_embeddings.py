"""T26 — app/services/kb/embeddings.py contract tests (V-E1, V16).

V-E1: every row stamped with ``embedding_version``; version bump
yields a new row alongside the old (V-E1 brief coexistence during a
re-embed sweep). V16: OpenAI client mocked at the SDK boundary —
``tests/_openai_mocks`` shape, plus an ``embeddings.create`` mock for
this seam specifically.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content_embedding import ContentEmbedding
from app.services.kb.embeddings import (
    EmbedResult,
    current_version,
    embed_and_persist,
)


def _make_embed_client(vec: list[float], *, prompt_tokens: int = 5) -> MagicMock:
    """Forge an AsyncOpenAI-shaped client whose ``embeddings.create`` returns
    a CreateEmbeddingResponse-shaped object."""

    resp = SimpleNamespace(
        data=[SimpleNamespace(embedding=vec, index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(return_value=resp)
    return client


# --------------------------------------------------------------------------- #
# 1. Fresh write — V-E1 version stamp present
# --------------------------------------------------------------------------- #


async def test_embed_writes_row_with_version_stamp(db_session: AsyncSession):
    client = _make_embed_client([0.1, 0.2, 0.3])
    result = await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="atomic_fact",
        entity_id=1,
        text="glycolysis converts glucose to pyruvate",
    )

    assert isinstance(result, EmbedResult)
    assert result.reused is False
    assert result.tokens == 5
    assert result.row.embedding == [0.1, 0.2, 0.3]
    assert result.row.embedding_version == current_version()
    client.embeddings.create.assert_awaited_once()


# --------------------------------------------------------------------------- #
# 2. Idempotent — same (kind, id, version) returns existing row
# --------------------------------------------------------------------------- #


async def test_embed_idempotent_same_version(db_session: AsyncSession):
    client = _make_embed_client([0.4, 0.5, 0.6])
    first = await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="atomic_fact",
        entity_id=42,
        text="x",
    )
    second = await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="atomic_fact",
        entity_id=42,
        text="x (different text but same key)",
    )

    assert first.reused is False
    assert second.reused is True
    assert second.row.id == first.row.id
    # Only the first call hit the SDK.
    assert client.embeddings.create.await_count == 1


# --------------------------------------------------------------------------- #
# 3. V-E1 version bump — both rows coexist
# --------------------------------------------------------------------------- #


async def test_embed_version_bump_coexists(db_session: AsyncSession):
    client = _make_embed_client([0.7, 0.8, 0.9])
    await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="atomic_fact",
        entity_id=7,
        text="t",
        version="v1",
    )
    await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="atomic_fact",
        entity_id=7,
        text="t",
        version="v2",
    )

    rows = (
        await db_session.execute(
            select(ContentEmbedding).where(
                ContentEmbedding.entity_kind == "atomic_fact",
                ContentEmbedding.entity_id == 7,
            )
        )
    ).scalars().all()
    assert {r.embedding_version for r in rows} == {"v1", "v2"}
    assert client.embeddings.create.await_count == 2


# --------------------------------------------------------------------------- #
# 4. V16 — no SDK call on the cached path
# --------------------------------------------------------------------------- #


async def test_no_sdk_call_when_cached(db_session: AsyncSession):
    seed_client = _make_embed_client([1.0])
    await embed_and_persist(
        db_session,
        openai_client=seed_client,
        entity_kind="question",
        entity_id=11,
        text="seed",
    )

    cold = _make_embed_client([2.0])
    result = await embed_and_persist(
        db_session,
        openai_client=cold,
        entity_kind="question",
        entity_id=11,
        text="anything",
    )
    assert result.reused is True
    cold.embeddings.create.assert_not_awaited()


# --------------------------------------------------------------------------- #
# 5. current_version derives from settings
# --------------------------------------------------------------------------- #


def test_current_version_includes_model_name():
    from app.config import settings

    assert settings.EMBEDDING_MODEL in current_version()
    assert current_version().endswith("-v1")
