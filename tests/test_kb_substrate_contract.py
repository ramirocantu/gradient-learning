"""T27 — P2 KB substrate contract tests (V-KB1, V-E1, V-N1, V-N2).

Three contracts:
1. V-KB1 — substrate migration idempotent re-run. ``alembic upgrade
   head`` against an already-head DB is a no-op; full up→base→up
   roundtrip leaves the schema clean.
2. V-E1 — dim change ⇒ version bump + re-embed. Switching
   ``EMBEDDING_MODEL`` to a dim that disagrees with the existing
   vectors must (a) trigger a ``DimMismatchError`` at the seam if the
   SDK returns a dim that doesn't match the configured model, and
   (b) coexist with old rows under a new ``embedding_version``.
3. V-N1 / V-N2 — Notion write idempotent at the page level. Re-sync
   over the same node returns the same ``notion_page_id``; ⊥ a new
   row in ``notion_pages``.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.atomic_fact import AtomicFact
from app.models.content_embedding import ContentEmbedding
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.embeddings import (
    DimMismatchError,
    EXPECTED_DIMS,
    embed_and_persist,
    expected_dim,
)
from app.services.kb.notion import sync_node_to_notion


_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"
_BACKEND_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _embed_client(vec: list[float], *, prompt_tokens: int = 5) -> MagicMock:
    resp = SimpleNamespace(
        data=[SimpleNamespace(embedding=vec, index=0, object="embedding")],
        model=settings.EMBEDDING_MODEL,
        object="list",
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(return_value=resp)
    return client


def _notion_client(page_id: str = "page-xyz") -> MagicMock:
    c = MagicMock()
    c.pages = MagicMock()
    c.pages.create = AsyncMock(
        return_value={
            "id": page_id,
            "url": f"https://notion.so/{page_id}",
            "object": "page",
        }
    )
    c.blocks = MagicMock()
    c.blocks.children = MagicMock()
    c.blocks.children.append = AsyncMock(return_value={"object": "list", "results": []})
    return c


async def _make_course_node(session: AsyncSession) -> OutlineNode:
    course = Course(slug=f"c-{uuid.uuid4().hex[:8]}", name="C")
    session.add(course)
    await session.flush()
    node = OutlineNode(
        course_id=course.id,
        parent_id=None,
        kind="concept",
        name=f"Node-{uuid.uuid4().hex[:6]}",
        depth=0,
        position=0,
    )
    session.add(node)
    await session.flush()
    return node


# --------------------------------------------------------------------------- #
# 1. V-KB1 — migration idempotent re-run
# --------------------------------------------------------------------------- #


async def _run_alembic(args: list[str], *, db_url: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ALEMBIC_DATABASE_URL": db_url}
    return subprocess.run(
        ["alembic", *args],
        cwd=_BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture
async def ephemeral_db():
    """Per-test scratch DB; dropped on teardown."""
    db_name = f"gradient_t27_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(_ADMIN_DSN)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()
    yield (
        db_name,
        f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/{db_name}",
    )
    admin = await asyncpg.connect(_ADMIN_DSN)
    try:
        await admin.execute(
            f"""
            SELECT pg_terminate_backend(pid)
              FROM pg_stat_activity
             WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await admin.close()


async def test_v_kb1_migration_upgrade_head_twice_is_noop(ephemeral_db):
    """V-KB1: re-running ``alembic upgrade head`` on an already-head
    DB must be a no-op (return code 0, no DDL change)."""

    _, url = ephemeral_db

    r1 = await _run_alembic(["upgrade", "head"], db_url=url)
    assert r1.returncode == 0, r1.stderr

    r2 = await _run_alembic(["upgrade", "head"], db_url=url)
    assert r2.returncode == 0, r2.stderr

    # Second run logs "Already at head" in stdout (or stderr depending
    # on alembic version) and emits no CREATE TABLE / ALTER TABLE.
    combined = (r2.stdout + r2.stderr).lower()
    assert "create table" not in combined
    assert "alter table" not in combined


async def test_v_kb1_migration_full_roundtrip(ephemeral_db):
    """V-KB1: up→base→up roundtrip succeeds — every CREATE in 0003
    has a matching DROP in the downgrade and every UP path is fresh."""

    _, url = ephemeral_db

    for args in (
        ["upgrade", "head"],
        ["downgrade", "base"],
        ["upgrade", "head"],
    ):
        r = await _run_alembic(args, db_url=url)
        assert r.returncode == 0, (
            f"alembic {' '.join(args)} failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )


# --------------------------------------------------------------------------- #
# 2. V-E1 — dim change ⇒ version bump + re-embed
# --------------------------------------------------------------------------- #


def test_v_e1_expected_dim_map_covers_known_models():
    """Sanity: the EXPECTED_DIMS map must at least cover the configured
    model so the guard can fire. New models added to settings must add
    here in the same change."""
    assert settings.EMBEDDING_MODEL in EXPECTED_DIMS
    assert EXPECTED_DIMS[settings.EMBEDDING_MODEL] > 0


def test_v_e1_expected_dim_unknown_model_returns_none():
    assert expected_dim("nonexistent-model") is None


async def test_v_e1_dim_mismatch_raises(db_session: AsyncSession):
    """V-E1: an SDK response whose embedding length disagrees with
    EXPECTED_DIMS[EMBEDDING_MODEL] must raise before the row is
    persisted — ⊥ mixed-dim vectors in one column."""

    wrong_dim_vec = [0.1] * (EXPECTED_DIMS[settings.EMBEDDING_MODEL] - 1)
    client = _embed_client(wrong_dim_vec)

    with pytest.raises(DimMismatchError):
        await embed_and_persist(
            db_session,
            openai_client=client,
            entity_kind="atomic_fact",
            entity_id=1,
            text="dim mismatch should reject",
        )

    # Nothing persisted.
    rows = (
        (
            await db_session.execute(
                select(ContentEmbedding).where(
                    ContentEmbedding.entity_kind == "atomic_fact",
                    ContentEmbedding.entity_id == 1,
                )
            )
        )
        .scalars()
        .all()
    )
    assert rows == []


async def test_v_e1_correct_dim_persists(db_session: AsyncSession):
    """Right dim → row persisted, no raise. Mirror of the negative case."""

    right_dim = EXPECTED_DIMS[settings.EMBEDDING_MODEL]
    client = _embed_client([0.0] * right_dim)

    result = await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="atomic_fact",
        entity_id=2,
        text="dim matches",
    )
    assert result.reused is False
    assert len(result.row.embedding) == right_dim


async def test_v_e1_version_bump_triggers_reembed(db_session: AsyncSession):
    """V-E1 + V-KB1: under a NEW version the seam writes a NEW row even
    when the same entity already has an old-version row. Both coexist
    until the old version is pruned (which is a separate cleanup pass,
    out of this seam's scope)."""

    dim = EXPECTED_DIMS[settings.EMBEDDING_MODEL]
    client = _embed_client([0.1] * dim)

    r_old = await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="question",
        entity_id=99,
        text="t",
        version="text-embedding-3-small-v1",
    )
    r_new = await embed_and_persist(
        db_session,
        openai_client=client,
        entity_kind="question",
        entity_id=99,
        text="t",
        version="text-embedding-3-small-v2",
    )

    assert r_old.row.id != r_new.row.id
    rows = (
        (
            await db_session.execute(
                select(ContentEmbedding).where(
                    ContentEmbedding.entity_kind == "question",
                    ContentEmbedding.entity_id == 99,
                )
            )
        )
        .scalars()
        .all()
    )
    versions = {r.embedding_version for r in rows}
    assert versions == {
        "text-embedding-3-small-v1",
        "text-embedding-3-small-v2",
    }


# --------------------------------------------------------------------------- #
# 3. V-N1 / V-N2 — Notion write idempotent (page level)
# --------------------------------------------------------------------------- #


async def test_v_n2_resync_returns_same_notion_page_id(db_session: AsyncSession):
    """V-N2: one Notion page per outline node. Re-sync over the same
    node must NOT create a second ``notion_pages`` row and MUST return
    the same ``notion_page_id``."""

    node = await _make_course_node(db_session)
    pdf = PdfSource(
        course_id=node.course_id,
        filename="x.pdf",
        sha256=uuid.uuid4().hex,
        status="ingested",
    )
    db_session.add(pdf)
    await db_session.flush()
    facts = [
        AtomicFact(
            course_id=node.course_id,
            pdf_source_id=pdf.id,
            text="A fact long enough to ingest.",
            content_hash=uuid.uuid4().hex,
        )
    ]
    db_session.add_all(facts)
    await db_session.flush()

    client = _notion_client(page_id="page-idem")

    r1 = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts,
    )
    r2 = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=[],
    )

    assert r1.notion_page_id == r2.notion_page_id == "page-idem"
    assert r1.notion_page_row_id == r2.notion_page_row_id
    client.pages.create.assert_awaited_once()

    pointers = (
        (await db_session.execute(select(NotionPage).where(NotionPage.node_id == node.id)))
        .scalars()
        .all()
    )
    assert len(pointers) == 1


async def test_v_n1_resync_no_read_endpoint_called(db_session: AsyncSession):
    """V-N1: re-sync ⊥ read Notion. The mock has no read endpoint
    wired; if the seam touched one, ``AttributeError`` would surface."""

    node = await _make_course_node(db_session)
    client = _notion_client(page_id="page-readonly-check")

    # First sync creates; second sync re-enters the upsert path.
    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=[],
    )
    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=[],
    )

    # Both create + (empty) append paths fired without raising — no
    # read endpoint was needed.
    assert client.pages.create.await_count == 1
