"""Notion write-out seam (T26, V-N1, V-N2, V-M3, V16).

One-way mirror: Postgres → Notion. Never reads Notion content back
(V-N1); the only stored Notion state is the ``notion_pages`` pointer
(page_id, url, tags snapshot, node_id) used for backlinks. One Notion
page per outline node (V-N2) — re-sync upserts on ``node_id``.

The notion-client ``AsyncClient`` is injected so tests mock at the
SDK boundary (V16). All write operations only — ``pages.create``,
``blocks.children.append``; we never call read endpoints
(``pages.retrieve`` / ``databases.query``) here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.notion_page import NotionPage
from app.models.outline import OutlineNode

_logger = logging.getLogger("app.services.kb.notion")


@dataclass
class SyncReport:
    notion_page_row_id: int
    notion_page_id: str
    created_page: bool      # True on first sync, False on subsequent
    appended_blocks: int


def _fact_blocks(facts: Iterable[AtomicFact]) -> list[dict[str, Any]]:
    """Render atomic facts as Notion bulleted_list_item blocks. Append-only
    on re-sync (V-M3: ⊥ rewrite). Each fact's text becomes a single
    bullet — chunking finer than the sentence is the LLM4Tag job (T29),
    not this seam.
    """

    blocks: list[dict[str, Any]] = []
    for f in facts:
        blocks.append(
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": f.text}}
                    ],
                },
            }
        )
    return blocks


def _tag_snapshot(facts: Iterable[AtomicFact], *, prev: list[Any] | None) -> list[str]:
    snapshot = list(prev or [])
    for f in facts:
        head = f.text[:50]
        if head not in snapshot:
            snapshot.append(head)
    return snapshot


async def sync_node_to_notion(
    session: AsyncSession,
    *,
    notion_client: Any,
    notion_wiki_db_id: str,
    node: OutlineNode,
    facts: list[AtomicFact],
) -> SyncReport:
    """Upsert a Notion page for ``node`` + append fact blocks.

    First sync (no ``notion_pages`` row yet): ``pages.create`` with the
    initial bulleted facts; persist a pointer row with page id/url.
    Subsequent sync: ``blocks.children.append`` adds new fact blocks
    underneath the same page — V-M3 append-only, V-N1 ⊥ rewrite.

    Caller owns dedupe: ``facts`` should be the slice the caller wants
    appended this run; passing the full set on every re-sync would
    duplicate blocks. (Atomic-fact dedupe lives one layer up, on the
    Postgres side via ``UQ(course_id, content_hash)``.)
    """

    pointer = (
        await session.execute(
            select(NotionPage).where(NotionPage.node_id == node.id)
        )
    ).scalar_one_or_none()

    blocks = _fact_blocks(facts)

    if pointer is None:
        result = await notion_client.pages.create(
            parent={"database_id": notion_wiki_db_id},
            properties={
                "Name": {
                    "title": [
                        {"type": "text", "text": {"content": node.name}}
                    ]
                },
            },
            children=blocks,
        )
        page_id = result["id"]
        url = result.get("url", "")
        pointer = NotionPage(
            node_id=node.id,
            notion_page_id=page_id,
            url=url,
            tags=_tag_snapshot(facts, prev=None),
            last_synced_at=datetime.now(timezone.utc),
        )
        session.add(pointer)
        await session.flush()
        return SyncReport(
            notion_page_row_id=pointer.id,
            notion_page_id=page_id,
            created_page=True,
            appended_blocks=len(blocks),
        )

    if blocks:
        await notion_client.blocks.children.append(
            block_id=pointer.notion_page_id,
            children=blocks,
        )

    pointer.tags = _tag_snapshot(facts, prev=pointer.tags)
    pointer.last_synced_at = datetime.now(timezone.utc)
    await session.flush()
    return SyncReport(
        notion_page_row_id=pointer.id,
        notion_page_id=pointer.notion_page_id,
        created_page=False,
        appended_blocks=len(blocks),
    )
