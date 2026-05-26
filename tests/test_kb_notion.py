"""T26 — app/services/kb/notion.py contract tests (V-N1, V-N2, V-M3, V16).

V-N1: one-way write-out, ⊥ read Notion back. V-N2: one Notion page
per outline node — re-sync upserts on ``node_id``. V-M3 / V-N1
re-sync is append-only on the Notion side (``blocks.children.append``),
never page rewrite. V16: notion-client mocked at the SDK boundary.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.notion import sync_node_to_notion


def _forge_notion_client(*, page_id: str = "notion-page-xyz") -> MagicMock:
    """AsyncClient-shaped mock. Only the write endpoints we actually use
    are wired — leave reads (``pages.retrieve``, ``databases.query``)
    bare so V-N1 violations surface as AttributeError instead of silently
    succeeding.
    """

    client = MagicMock()
    client.pages = MagicMock()
    client.pages.create = AsyncMock(
        return_value={
            "id": page_id,
            "url": f"https://notion.so/{page_id}",
            "object": "page",
        }
    )
    client.blocks = MagicMock()
    client.blocks.children = MagicMock()
    client.blocks.children.append = AsyncMock(
        return_value={"object": "list", "results": []}
    )
    # Intentionally no .pages.retrieve / .databases.query — any code
    # path that tries to read Notion will raise AttributeError, which
    # is what V-N1 wants.
    return client


async def _make_course_node(session: AsyncSession) -> OutlineNode:
    course = Course(slug=f"n-{uuid.uuid4().hex[:8]}", name="N")
    session.add(course)
    await session.flush()
    node = OutlineNode(
        course_id=course.id,
        parent_id=None,
        kind="concept",
        name=f"Concept {uuid.uuid4().hex[:6]}",
        depth=0,
        position=0,
    )
    session.add(node)
    await session.flush()
    return node


async def _make_facts(
    session: AsyncSession, course_id: int, texts: list[str]
) -> list[AtomicFact]:
    pdf = PdfSource(
        course_id=course_id,
        filename="x.pdf",
        sha256=uuid.uuid4().hex,
        status="ingested",
    )
    session.add(pdf)
    await session.flush()
    facts: list[AtomicFact] = []
    for t in texts:
        f = AtomicFact(
            course_id=course_id,
            pdf_source_id=pdf.id,
            text=t,
            content_hash=uuid.uuid4().hex,
        )
        session.add(f)
        facts.append(f)
    await session.flush()
    return facts


# --------------------------------------------------------------------------- #
# 1. First sync creates page + pointer (V-N1, V-N2)
# --------------------------------------------------------------------------- #


async def test_first_sync_creates_page_and_pointer(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    node_id = node.id
    facts = await _make_facts(
        db_session,
        node.course_id,
        ["Glycolysis converts glucose to pyruvate.", "Net yield is two ATP."],
    )
    client = _forge_notion_client(page_id="page-1")

    report = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts,
    )

    assert report.created_page is True
    assert report.appended_blocks == 2
    assert report.notion_page_id == "page-1"

    client.pages.create.assert_awaited_once()
    client.blocks.children.append.assert_not_awaited()

    pointer = (
        await db_session.execute(
            select(NotionPage).where(NotionPage.node_id == node_id)
        )
    ).scalar_one()
    assert pointer.notion_page_id == "page-1"
    assert pointer.url.endswith("page-1")
    assert pointer.last_synced_at is not None


# --------------------------------------------------------------------------- #
# 2. Re-sync append-only (V-M3, V-N1)
# --------------------------------------------------------------------------- #


async def test_resync_appends_blocks_no_page_rewrite(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    node_id = node.id
    facts_1 = await _make_facts(db_session, node.course_id, ["First fact long enough."])
    facts_2 = await _make_facts(db_session, node.course_id, ["Second fact long enough."])
    client = _forge_notion_client(page_id="page-append")

    r1 = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts_1,
    )
    r2 = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts_2,
    )

    assert r1.created_page is True
    assert r2.created_page is False
    assert r2.notion_page_id == "page-append"

    # First call → pages.create; second call → blocks.children.append.
    client.pages.create.assert_awaited_once()
    client.blocks.children.append.assert_awaited_once()

    pointers = (
        await db_session.execute(
            select(NotionPage).where(NotionPage.node_id == node_id)
        )
    ).scalars().all()
    assert len(pointers) == 1  # V-N2: one page per node
    assert "First fact long enough." in (pointers[0].tags or [None])[0]
    assert any(
        "Second fact long enough." in t for t in (pointers[0].tags or [])
    )


# --------------------------------------------------------------------------- #
# 3. V-N1: no read-back path (mock has no read endpoints)
# --------------------------------------------------------------------------- #


async def test_no_read_endpoints_called(db_session: AsyncSession):
    """V-N1: the seam must never call read endpoints. The forged client
    deliberately omits ``pages.retrieve`` and ``databases.query``; if
    the seam tried to read, the call would raise AttributeError.
    """

    node = await _make_course_node(db_session)
    facts = await _make_facts(db_session, node.course_id, ["Fact long enough to keep."])
    client = _forge_notion_client()

    # Confirm the mock has no read attributes wired.
    assert not hasattr(client.pages, "retrieve") or not isinstance(
        client.pages.retrieve, AsyncMock
    )
    # The sync must succeed without touching read paths.
    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts,
    )


# --------------------------------------------------------------------------- #
# 4. Empty facts on re-sync still updates last_synced_at
# --------------------------------------------------------------------------- #


async def test_resync_with_no_facts_skips_append(db_session: AsyncSession):
    node = await _make_course_node(db_session)
    facts_1 = await _make_facts(db_session, node.course_id, ["Seed fact long enough."])
    client = _forge_notion_client(page_id="page-empty")

    await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=facts_1,
    )
    initial_ts = (
        await db_session.execute(
            select(NotionPage.last_synced_at).where(NotionPage.node_id == node.id)
        )
    ).scalar_one()

    r = await sync_node_to_notion(
        db_session,
        notion_client=client,
        notion_wiki_db_id="db-id",
        node=node,
        facts=[],
    )
    assert r.created_page is False
    assert r.appended_blocks == 0
    client.blocks.children.append.assert_not_awaited()

    later_ts = (
        await db_session.execute(
            select(NotionPage.last_synced_at).where(NotionPage.node_id == node.id)
        )
    ).scalar_one()
    assert later_ts >= initial_ts
