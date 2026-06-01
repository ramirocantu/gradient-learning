"""RCA-10 §T5 — E2E: PDF → Grounded Tag (full arc).

Drives the real HTTP ingest route (POST /api/v1/pdf/ingest) + the shared
grounded arc (embed_pending → recall → tag_pending) in-process, codifying the
T4 manual pass:

- ingest persists pdf_sources + atomic_facts (node_id NULL pre-tag), readable
  via GET /pdf-sources + GET /atomic-facts (V1, I);
- V7: a fact runs the grounded arc → atomic_fact_tags persisted (source='llm',
  node_id, confidence, manual_review per V-T3) + AtomicFact.node_id
  denormalized; the read surface then exposes the node_id;
- V8: embed_pending precedes tag_pending — empty recall (no embedded nodes)
  persists NO tag and leaves node_id NULL (assertion would be vacuous otherwise).

Mock altitude (V2): vision + extract patched at app.api.v1.pdf.build_openai_client
(make_client, scripted vision→extraction); embeddings via make_embeddings_client;
the grounded LLM pass patched at the jobs layer (its SDK internals + V12
calibrator owned by test_grounded / test_calibrator). A real tiny PDF is
rendered by PyMuPDF (no committed file). Self-seeded per test (V3).
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.outline import OutlineNode
from app.services.kb.jobs import embed_pending, tag_pending
from app.services.llm.grounded import GroundedResult, GroundedTag
from tests._openai_mocks import make_client, make_completion, make_embeddings_client

_PDF_INGEST = "/api/v1/pdf/ingest"


def _facts_completion(*facts: str):
    return make_completion(content=json.dumps({"facts": [{"text": f} for f in facts]}))


def _mini_pdf() -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 750), "Lecture page one.")
    data = doc.tobytes()
    doc.close()
    return data


def _grounded_for(recall, *, conf: float = 0.8) -> GroundedResult:
    """Stand-in for the LLM pass: tag the top recall candidate, or empty when
    recall surfaced nothing (mirrors generate_grounded_tags' real contract)."""
    tags = []
    if recall.candidates:
        c = recall.candidates[0]
        tags = [
            GroundedTag(
                node_id=c.node_id,
                path=c.path,
                candidate_index=1,
                via="embedding",
                rationale="matches the fact",
                calibrated_confidence=conf,
                manual_review=conf < 0.5,
            )
        ]
    return GroundedResult(
        tags=tags,
        extractor_version="grounded-v1",
        model="m",
        calibrator_model="c",
        input_tokens=1,
        output_tokens=1,
        cached_tokens=0,
    )


def _grounded_patch():
    """Patch generate_grounded_tags with a recall-aware stand-in."""

    async def _fake(*, entity_text: str, recall_result: Any, **_kw: Any) -> GroundedResult:
        return _grounded_for(recall_result)

    return patch("app.services.kb.jobs.generate_grounded_tags", new=AsyncMock(side_effect=_fake))


async def _node(session: AsyncSession, course_id: int, name: str) -> OutlineNode:
    n = OutlineNode(
        course_id=course_id, parent_id=None, kind="concept", name=name, depth=0, position=0
    )
    session.add(n)
    await session.flush()
    return n


async def _ingest(client: AsyncClient, headers, course_id: int, *fact_texts: str):
    """POST a 1-page PDF; mock vision transcription + structured extraction."""
    fake = make_client(
        make_completion(content="Lecture page one."),
        _facts_completion(*fact_texts),
    )
    with patch("app.api.v1.pdf.build_openai_client", return_value=fake):
        resp = await client.post(
            _PDF_INGEST,
            headers=headers,
            data={"course_id": str(course_id)},
            files={"file": ("lecture.pdf", _mini_pdf(), "application/pdf")},
        )
    return resp


# --------------------------------------------------------------------------- #
# ingest → atomic_facts (node_id NULL), readable via HTTP
# --------------------------------------------------------------------------- #


async def test_pdf_ingest_persists_facts_readable(
    client: AsyncClient, coach_headers, make_course
):
    course = await make_course(slug=f"pdf-{uuid.uuid4().hex[:6]}", name="PDF E2E")
    resp = await _ingest(client, coach_headers, course.id, "Glucose has six carbons.")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pages"] == 1
    assert body["new_facts"] == 1
    assert body["extractor_version"] == "pdf-vision-v1"

    # GET /pdf-sources → status ingested
    src = await client.get("/api/v1/pdf-sources", headers=coach_headers)
    assert src.status_code == 200
    mine = [s for s in src.json() if s["course_id"] == course.id]
    assert len(mine) == 1 and mine[0]["status"] == "ingested"

    # GET /atomic-facts → fact present, node_id NULL (pre-tag)
    facts = await client.get(
        f"/api/v1/atomic-facts?pdf_source_id={mine[0]['id']}", headers=coach_headers
    )
    rows = facts.json()
    assert len(rows) == 1
    assert rows[0]["node_id"] is None


# --------------------------------------------------------------------------- #
# V7 — fact runs the grounded arc → atomic_fact_tags persisted + node_id set
# --------------------------------------------------------------------------- #


async def test_pdf_fact_grounded_tagged_and_readable(
    client: AsyncClient, coach_headers, make_course, db_session: AsyncSession
):
    course = await make_course(slug=f"pdf-{uuid.uuid4().hex[:6]}", name="PDF Tag E2E")
    node = await _node(db_session, course.id, "Carbohydrate structure")
    await db_session.commit()

    resp = await _ingest(client, coach_headers, course.id, "Glucose has six carbons.")
    assert resp.status_code == 200, resp.text

    fact = (
        await db_session.execute(select(AtomicFact).where(AtomicFact.course_id == course.id))
    ).scalar_one()
    assert fact.node_id is None  # V8: not yet tagged

    # embed (node + fact same mocked vector → recall ranks the node) then tag
    await embed_pending(db_session, openai_client=make_embeddings_client())
    with _grounded_patch():
        report = await tag_pending(
            db_session, tagging_client=MagicMock(), calibrator_client=MagicMock()
        )
    await db_session.commit()

    assert report.facts_tagged == 1
    assert report.tags_persisted == 1

    tags = (
        (await db_session.execute(select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id == fact.id)))
        .scalars()
        .all()
    )
    assert len(tags) == 1
    t = tags[0]
    assert t.source == "llm"
    assert t.node_id == node.id
    assert t.confidence is not None
    assert t.manual_review is False  # conf 0.8 ≥ 0.5 (V-T3)

    await db_session.refresh(fact)
    assert fact.node_id == node.id  # denormalized primary

    # read surface now exposes the node_id
    facts = await client.get(
        f"/api/v1/atomic-facts?node_id={node.id}", headers=coach_headers
    )
    ids = [f["id"] for f in facts.json()]
    assert fact.id in ids


# --------------------------------------------------------------------------- #
# V8 — empty recall (no embedded nodes) → no tag, node_id stays NULL
# --------------------------------------------------------------------------- #


async def test_pdf_fact_empty_recall_no_tag(
    client: AsyncClient, coach_headers, make_course, db_session: AsyncSession
):
    course = await make_course(slug=f"pdf-{uuid.uuid4().hex[:6]}", name="PDF NoNode E2E")
    resp = await _ingest(client, coach_headers, course.id, "Orphan fact, no outline.")
    assert resp.status_code == 200, resp.text

    fact = (
        await db_session.execute(select(AtomicFact).where(AtomicFact.course_id == course.id))
    ).scalar_one()

    await embed_pending(db_session, openai_client=make_embeddings_client())  # no nodes to recall
    with _grounded_patch():
        report = await tag_pending(
            db_session, tagging_client=MagicMock(), calibrator_client=MagicMock()
        )
    await db_session.commit()

    assert report.tags_persisted == 0
    tags = (
        (await db_session.execute(select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id == fact.id)))
        .scalars()
        .all()
    )
    assert tags == []
    await db_session.refresh(fact)
    assert fact.node_id is None
