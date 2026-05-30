"""Knowledge-base substrate read endpoints (T45–T48, desktop ¶T8–¶T11).

Public, X-Coach-Token-gated GET reads over the P2 substrate tables, so the
native client can render its connections / facts / Notion-index / inbox
views. Per V-D1 these extend the public `/api/v1/*` contract — not a
dashboard-private seam.

    GET /api/v1/concept-edges     — concept_edges feed         (T45, V-E2)
    GET /api/v1/atomic-facts      — atomic_facts by node/pdf   (T46, V-KB1)
    GET /api/v1/notion/pages      — notion_pages pointer index (T47, V-N1/V-N2)
    GET /api/v1/pdf-sources       — pdf_sources inbox          (T48, V-KB1)

These are read-only projections; each returns `[]` until its substrate is
populated by the matching pipeline (similarity derivation, PDF ingest,
Notion write-out). V-N1 note: the notion read exposes OUR pointer rows only
— never a read-back of Notion content.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_session
from app.models.atomic_fact import AtomicFact
from app.models.concept_edge import ConceptEdge
from app.models.notion_page import NotionPage
from app.models.outline import OutlineNode
from app.models.pdf_source import PdfSource

router = APIRouter(tags=["kb-reads"])


async def _node_names(session: AsyncSession, node_ids: set[int]) -> dict[int, str]:
    """Map node_id → name for the given ids (one query, empty-safe)."""
    if not node_ids:
        return {}
    rows = (
        await session.execute(
            select(OutlineNode.id, OutlineNode.name).where(OutlineNode.id.in_(node_ids))
        )
    ).all()
    return {nid: name for nid, name in rows}


# ---------- T45: concept_edges (V-E2, V-O1, V-D1) ----------


@router.get("/concept-edges")
async def list_concept_edges(
    session: AsyncSession = Depends(get_session),
    node_id: int | None = Query(
        default=None, description="filter to edges touching this node (src or dst)"
    ),
    kind: str | None = Query(default=None, description="'similarity' | 'manual'"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Concept-edge feed for the connections view. `node_id` matches either
    endpoint. Derived (similarity) and manual edges are both surfaced as-is —
    the read does not weight or recompute (V-E2)."""
    stmt = select(ConceptEdge).order_by(ConceptEdge.created_at.desc(), ConceptEdge.id.desc())
    if node_id is not None:
        stmt = stmt.where(
            (ConceptEdge.src_node_id == node_id) | (ConceptEdge.dst_node_id == node_id)
        )
    if kind is not None:
        stmt = stmt.where(ConceptEdge.kind == kind)
    edges = (await session.execute(stmt.limit(limit))).scalars().all()

    names = await _node_names(
        session, {e.src_node_id for e in edges} | {e.dst_node_id for e in edges}
    )
    return [
        {
            "id": e.id,
            "from": {"node_id": e.src_node_id, "name": names.get(e.src_node_id)},
            "to": {"node_id": e.dst_node_id, "name": names.get(e.dst_node_id)},
            "kind": e.kind,
            "score": float(e.score) if e.score is not None else None,
            "created_at": e.created_at,
        }
        for e in edges
    ]


# ---------- T46: atomic_facts (V-KB1, V-T1, V-D1) ----------


@router.get("/atomic-facts")
async def list_atomic_facts(
    session: AsyncSession = Depends(get_session),
    node_id: int | None = Query(default=None),
    pdf_source_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Grounded facts by outline node or PDF source. (`extractor_version` is
    stamped on atomic_fact_tags per §I, not on the fact row — surface it via
    the tag read if a client needs it.)"""
    stmt = (
        select(AtomicFact)
        .options(selectinload(AtomicFact.pdf_source))
        .order_by(AtomicFact.id.desc())
    )
    if node_id is not None:
        stmt = stmt.where(AtomicFact.node_id == node_id)
    if pdf_source_id is not None:
        stmt = stmt.where(AtomicFact.pdf_source_id == pdf_source_id)
    facts = (await session.execute(stmt.limit(limit))).scalars().all()
    return [
        {
            "id": f.id,
            "text": f.text,
            "node_id": f.node_id,
            "page": f.page,
            "pdf_source": {
                "id": f.pdf_source_id,
                "filename": f.pdf_source.filename if f.pdf_source else None,
            },
            "created_at": f.created_at,
        }
        for f in facts
    ]


# ---------- T47: notion_pages pointer index (V-N1, V-N2, V-D1) ----------


@router.get("/notion/pages")
async def list_notion_pages(
    session: AsyncSession = Depends(get_session),
    node_id: int | None = Query(default=None),
) -> list[dict[str, Any]]:
    """Pointer index for the Notion page-index view. V-N1: reads OUR
    `notion_pages` rows only — ⊥ a read-back of Notion content. One page per
    node (V-N2). `title` is the outline node's name (we keep no local Notion
    content copy)."""
    stmt = select(NotionPage).order_by(
        NotionPage.last_synced_at.desc().nullslast(), NotionPage.id.desc()
    )
    if node_id is not None:
        stmt = stmt.where(NotionPage.node_id == node_id)
    pages = (await session.execute(stmt)).scalars().all()

    names = await _node_names(session, {p.node_id for p in pages})
    return [
        {
            "node_id": p.node_id,
            "title": names.get(p.node_id),
            "url": p.url,
            "notion_page_id": p.notion_page_id,
            "tags": p.tags,
            "last_synced_at": p.last_synced_at,
        }
        for p in pages
    ]


# ---------- T48: pdf_sources inbox (V-KB1, V-D1) ----------


@router.get("/pdf-sources")
async def list_pdf_sources(
    session: AsyncSession = Depends(get_session),
    course_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Ingested-PDF inbox. `facts_count` rolls up atomic_facts per source."""
    facts_count = func.count(AtomicFact.id).label("facts_count")
    stmt = (
        select(PdfSource, facts_count)
        .outerjoin(AtomicFact, AtomicFact.pdf_source_id == PdfSource.id)
        .group_by(PdfSource.id)
        .order_by(PdfSource.id.desc())
    )
    if course_id is not None:
        stmt = stmt.where(PdfSource.course_id == course_id)
    if status is not None:
        stmt = stmt.where(PdfSource.status == status)
    rows = (await session.execute(stmt.limit(limit))).all()
    return [
        {
            "id": p.id,
            "filename": p.filename,
            "sha256": p.sha256,
            "status": p.status,
            "facts_count": count,
            "course_id": p.course_id,
            "ingested_at": p.ingested_at,
            "created_at": p.created_at,
        }
        for p, count in rows
    ]
