"""KB-substrate read endpoints (T45–T48, desktop ¶T8–¶T11).

Public, token-gated GET reads over the P2 substrate (V-D1). Each returns []
until populated. V-N1: the notion read exposes OUR pointer rows, ⊥ Notion
content read-back.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.atomic_fact import AtomicFact
from app.models.concept_edge import ConceptEdge
from app.models.notion_page import NotionPage
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _seed(db: AsyncSession) -> dict[str, int]:
    """course → 2 nodes (A,B); 1 similarity edge A→B; 1 pdf w/ 2 facts
    (one tagged to A); 1 notion page for A."""
    course = Course(slug="biochem", name="Biochem")
    db.add(course)
    await db.flush()
    ids: dict[str, int] = {"course": course.id}

    a = OutlineNode(
        course_id=course.id, parent_id=None, kind="topic", name="Amino acids", depth=0, position=0
    )
    b = OutlineNode(
        course_id=course.id, parent_id=None, kind="topic", name="Enzymes", depth=0, position=1
    )
    db.add_all([a, b])
    await db.flush()
    ids["A"], ids["B"] = a.id, b.id

    db.add(ConceptEdge(src_node_id=a.id, dst_node_id=b.id, kind="similarity", score=0.87))
    await db.flush()

    pdf = PdfSource(
        course_id=course.id, filename="lecture1.pdf", sha256="abc123", status="ingested"
    )
    db.add(pdf)
    await db.flush()
    ids["pdf"] = pdf.id

    db.add_all(
        [
            AtomicFact(
                course_id=course.id,
                pdf_source_id=pdf.id,
                page=1,
                text="fact one",
                node_id=a.id,
                content_hash="h1",
            ),
            AtomicFact(
                course_id=course.id,
                pdf_source_id=pdf.id,
                page=2,
                text="fact two",
                node_id=None,
                content_hash="h2",
            ),
        ]
    )
    db.add(
        NotionPage(
            node_id=a.id, notion_page_id="np-1", url="https://notion.so/np-1", tags=["amino"]
        )
    )
    await db.commit()
    return ids


# ---------- T45: concept-edges ----------


@pytest.mark.asyncio
async def test_concept_edges_feed(client: AsyncClient, db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    r = await client.get("/api/v1/concept-edges", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    e = body[0]
    assert e["from"] == {"node_id": ids["A"], "name": "Amino acids"}
    assert e["to"] == {"node_id": ids["B"], "name": "Enzymes"}
    assert e["kind"] == "similarity"
    assert e["score"] == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_concept_edges_filter_by_node_and_kind(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    # node filter matches either endpoint
    r = await client.get(f"/api/v1/concept-edges?node_id={ids['B']}", headers=_AUTH)
    assert len(r.json()) == 1
    r = await client.get("/api/v1/concept-edges?node_id=999999", headers=_AUTH)
    assert r.json() == []
    r = await client.get("/api/v1/concept-edges?kind=manual", headers=_AUTH)
    assert r.json() == []


# ---------- T46: atomic-facts ----------


@pytest.mark.asyncio
async def test_atomic_facts_by_node_and_pdf(client: AsyncClient, db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    r = await client.get(f"/api/v1/atomic-facts?node_id={ids['A']}", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["text"] == "fact one"
    assert body[0]["node_id"] == ids["A"]
    assert body[0]["pdf_source"] == {"id": ids["pdf"], "filename": "lecture1.pdf"}

    r = await client.get(f"/api/v1/atomic-facts?pdf_source_id={ids['pdf']}", headers=_AUTH)
    assert len(r.json()) == 2
    r = await client.get("/api/v1/atomic-facts?node_id=999999", headers=_AUTH)
    assert r.json() == []


# ---------- T47: notion pages pointer index ----------


@pytest.mark.asyncio
async def test_notion_pages_pointer_index(client: AsyncClient, db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    r = await client.get("/api/v1/notion/pages", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    p = body[0]
    assert p["node_id"] == ids["A"]
    assert p["title"] == "Amino acids"  # title = outline node name (V-N1: no local content copy)
    assert p["url"] == "https://notion.so/np-1"
    assert p["tags"] == ["amino"]


# ---------- T48: pdf-sources inbox ----------


@pytest.mark.asyncio
async def test_pdf_sources_inbox_with_facts_count(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed(db_session)
    r = await client.get("/api/v1/pdf-sources", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    s = body[0]
    assert s["filename"] == "lecture1.pdf"
    assert s["status"] == "ingested"
    assert s["facts_count"] == 2
    assert s["sha256"] == "abc123"

    r = await client.get("/api/v1/pdf-sources?status=pending", headers=_AUTH)
    assert r.json() == []


# ---------- empty substrate → [] (V-D1, returns-empty contract) ----------


@pytest.mark.asyncio
async def test_all_routes_empty_when_unpopulated(client: AsyncClient) -> None:
    for path in (
        "/api/v1/concept-edges",
        "/api/v1/atomic-facts",
        "/api/v1/notion/pages",
        "/api/v1/pdf-sources",
    ):
        r = await client.get(path, headers=_AUTH)
        assert r.status_code == 200, r.text
        assert r.json() == []


# ---------- auth seam (V-D1, token-gated) ----------


@pytest.mark.asyncio
async def test_routes_require_coach_token(client: AsyncClient) -> None:
    for path in (
        "/api/v1/concept-edges",
        "/api/v1/atomic-facts",
        "/api/v1/notion/pages",
        "/api/v1/pdf-sources",
    ):
        r = await client.get(path)
        assert r.status_code in (401, 403), f"{path}: {r.status_code}"
