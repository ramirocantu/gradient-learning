"""T51 — sync_pending_nodes orchestrator tests (V-N1, V-N2, V41).

Only atomic facts the categorizer (T50) has tagged (``node_id`` set) own a
node page and get mirrored; untagged facts are invisible to the sync. First
sync creates the page; a re-sync with no newer facts is a no-op (append-only,
⊥ rewrite — V-N1). notion-client mocked at the SDK boundary (V16).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.notion import sync_pending_nodes


def _forge_notion_client(*, page_id: str = "page-1") -> MagicMock:
    client = MagicMock()
    client.pages = MagicMock()
    client.pages.create = AsyncMock(
        return_value={"id": page_id, "url": f"https://notion.so/{page_id}", "object": "page"}
    )
    client.blocks = MagicMock()
    client.blocks.children = MagicMock()
    client.blocks.children.append = AsyncMock(return_value={"object": "list", "results": []})
    return client


async def _make_node(session: AsyncSession) -> OutlineNode:
    course = Course(slug=f"ns-{uuid.uuid4().hex[:8]}", name="NS")
    session.add(course)
    await session.flush()
    node = OutlineNode(
        course_id=course.id, parent_id=None, kind="concept",
        name=f"Concept {uuid.uuid4().hex[:6]}", depth=0, position=0,
    )
    session.add(node)
    await session.flush()
    return node


async def _add_facts(
    session: AsyncSession, *, course_id: int, node_id: int | None,
    texts: list[str], created_at: datetime | None = None,
) -> None:
    pdf = PdfSource(
        course_id=course_id, filename="x.pdf", sha256=uuid.uuid4().hex, status="ingested"
    )
    session.add(pdf)
    await session.flush()
    for t in texts:
        kw = {} if created_at is None else {"created_at": created_at}
        session.add(
            AtomicFact(
                course_id=course_id, pdf_source_id=pdf.id, node_id=node_id,
                text=t, content_hash=uuid.uuid4().hex, **kw,
            )
        )
    await session.flush()


async def test_sync_creates_page_for_tagged_facts(db_session: AsyncSession):
    node = await _make_node(db_session)
    await _add_facts(
        db_session, course_id=node.course_id, node_id=node.id,
        texts=["Glycolysis yields pyruvate.", "Net two ATP per glucose."],
    )
    client = _forge_notion_client(page_id="pg-create")

    report = await sync_pending_nodes(
        db_session, notion_client=client, notion_wiki_db_id="db-1"
    )

    assert report.nodes_synced == 1
    assert report.pages_created == 1
    assert report.blocks_appended == 2
    assert not report.partial_failure
    client.pages.create.assert_awaited_once()

    pointer = (
        await db_session.execute(select(NotionPage).where(NotionPage.node_id == node.id))
    ).scalar_one()
    assert pointer.notion_page_id == "pg-create"


async def test_sync_ignores_untagged_facts(db_session: AsyncSession):
    node = await _make_node(db_session)
    # node_id NULL → not yet categorized → no node page to mirror onto.
    await _add_facts(
        db_session, course_id=node.course_id, node_id=None,
        texts=["Untagged fact floating free."],
    )
    client = _forge_notion_client()

    report = await sync_pending_nodes(
        db_session, notion_client=client, notion_wiki_db_id="db-1"
    )
    assert report.nodes_synced == 0
    client.pages.create.assert_not_called()


async def test_resync_appends_only_newer_facts(db_session: AsyncSession):
    node = await _make_node(db_session)
    await _add_facts(
        db_session, course_id=node.course_id, node_id=node.id,
        texts=["Original fact one.", "Original fact two."],
    )
    client = _forge_notion_client(page_id="pg-resync")

    first = await sync_pending_nodes(
        db_session, notion_client=client, notion_wiki_db_id="db-1"
    )
    assert first.pages_created == 1

    # No new facts → re-sync is a no-op (existing facts' created_at predates
    # the pointer's last_synced_at).
    again = await sync_pending_nodes(
        db_session, notion_client=client, notion_wiki_db_id="db-1"
    )
    assert again.nodes_synced == 0
    assert again.blocks_appended == 0

    # A genuinely newer fact (explicit future created_at) → appended, ⊥ a new page.
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    await _add_facts(
        db_session, course_id=node.course_id, node_id=node.id,
        texts=["A freshly tagged fact."], created_at=future,
    )
    third = await sync_pending_nodes(
        db_session, notion_client=client, notion_wiki_db_id="db-1"
    )
    assert third.nodes_synced == 1
    assert third.pages_created == 0
    assert third.blocks_appended == 1
    client.blocks.children.append.assert_awaited()
