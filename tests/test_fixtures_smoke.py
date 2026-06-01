"""RCA-10 §T1 — smoke coverage for the shared workflow-test fixtures.

One assertion per fixture/helper added in T1, so a drift in any building
block fails here (fast, isolated) rather than inside a workflow E2E test.

Invariants exercised:
- V2: OpenAI mocked at the SDK boundary; embeddings return the configured dim.
- V3: factories seed per-test state (no shared singleton, rolls back).
- I:  `CapturePayload`, the `Renderer` seam, the embeddings response shape.
"""

from __future__ import annotations

from pathlib import Path

from httpx import AsyncClient

from app.schemas.captures import CapturePayload
from app.services.kb.embeddings import expected_dim
from app.services.kb.pdf_ingest import RenderedPage
from tests._openai_mocks import make_embeddings_client, unit_vector


def test_uworld_capture_payload_round_trips(uworld_capture_payload):
    body = uworld_capture_payload(qid="q-x", course_slug="biochem")
    # The wire dict re-validates through the strict schema (the route's path).
    p = CapturePayload.model_validate(body)
    assert p.source == "uworld"
    assert p.qid == "q-x"
    assert p.course_slug == "biochem"
    assert p.parsed.correct_choice == "A"


def test_uworld_capture_payload_defaults_course_slug_none(uworld_capture_payload):
    # The single-course fallback path (V6) — no slug supplied.
    body = uworld_capture_payload()
    assert body["course_slug"] is None


def test_fake_renderer_yields_stub_pages(fake_renderer):
    pages = fake_renderer(3)(Path("/does-not-exist.pdf"))
    assert len(pages) == 3
    assert all(isinstance(pg, RenderedPage) for pg in pages)
    assert pages[0].page == 1 and pages[0].image_png


def test_unit_vector_dim_and_orthogonality():
    assert len(unit_vector()) == expected_dim()  # 1536 for text-embedding-3-small
    assert unit_vector(hot=0) != unit_vector(hot=1)


async def test_make_embeddings_client_returns_configured_dim():
    client = make_embeddings_client()
    resp = await client.embeddings.create(model="text-embedding-3-small", input="hi")
    assert len(resp.data[0].embedding) == expected_dim()
    assert resp.usage.prompt_tokens > 0


async def test_make_embeddings_client_maps_per_input():
    client = make_embeddings_client(
        vector_for=lambda t: unit_vector(hot=0 if t == "a" else 1)
    )
    va = (await client.embeddings.create(model="m", input="a")).data[0].embedding
    vb = (await client.embeddings.create(model="m", input="b")).data[0].embedding
    assert va != vb


async def test_coach_headers_unlock_gated_route(
    client: AsyncClient, coach_headers: dict[str, str]
):
    # Gated GET /courses: 401 without the token, non-401 with it.
    assert (await client.get("/api/v1/courses")).status_code == 401
    assert (await client.get("/api/v1/courses", headers=coach_headers)).status_code != 401


async def test_make_course_inserts_row(make_course):
    course = await make_course(slug="anat", name="Anatomy")
    assert course.id is not None
    assert course.slug == "anat"
