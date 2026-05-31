"""HTTP-level coverage for `/api/v1/courses/*` and outline import (§T21).

Exercises the route surface end-to-end through the ASGI test client. The
validator + materializer have unit coverage in ``test_outline_importer``;
this module verifies the HTTP wiring, status codes, and the V-O2 / V-O3
contract at the route boundary.

Invariants:
- V-O2: validate-then-materialize is whole-upload-or-reject; a validation
  failure ⊥ touch the DB.
- V-O3: re-uploading the same schema restores the outline — wipe and
  reinsert in one transaction, no leftover nodes.
- I.outline-import: the route shape and payload contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.outline import Course, OutlineNode

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


@pytest.fixture(autouse=True)
def _send_coach_token(client: AsyncClient) -> None:
    """The whole `/api/v1/courses` + outline router is X-Coach-Token gated;
    attach the token to every request in this module."""
    client.headers.update(_AUTH)


def _payload(slug: str = "t21-course", name: str = "T21 Course") -> dict[str, Any]:
    """A small but multi-depth schema: one root section, two topic leaves."""
    return {
        "course": {"slug": slug, "name": name},
        "nodes": [
            {"path": ["Root"], "kind": "section", "name": "Root", "position": 1},
            {
                "path": ["Root", "LeafA"],
                "kind": "topic",
                "name": "LeafA",
                "position": 1,
            },
            {
                "path": ["Root", "LeafB"],
                "kind": "topic",
                "name": "LeafB",
                "position": 2,
            },
        ],
    }


# ---------- POST /api/v1/courses ----------


@pytest.mark.asyncio
async def test_create_course_returns_201_with_payload(client: AsyncClient) -> None:
    r = await client.post(
        "/api/v1/courses",
        json={"slug": "biochem", "name": "Biochemistry", "description": "intro"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["slug"] == "biochem"
    assert body["name"] == "Biochemistry"
    assert body["description"] == "intro"
    assert isinstance(body["id"], int)


@pytest.mark.asyncio
async def test_create_course_duplicate_slug_returns_409(client: AsyncClient) -> None:
    body = {"slug": "dup", "name": "Dup"}
    r1 = await client.post("/api/v1/courses", json=body)
    assert r1.status_code == 201
    r2 = await client.post("/api/v1/courses", json=body)
    assert r2.status_code == 409
    assert "already exists" in r2.json()["detail"]


# ---------- GET /api/v1/courses ----------


@pytest.mark.asyncio
async def test_list_courses_empty_then_populated(client: AsyncClient) -> None:
    # I.api: public courses-list seam the SPA picker consumes (V-D1).
    r0 = await client.get("/api/v1/courses")
    assert r0.status_code == 200
    assert r0.json() == []

    # Insert out of slug order to assert the endpoint sorts.
    await client.post("/api/v1/courses", json={"slug": "zeta", "name": "Zeta"})
    await client.post(
        "/api/v1/courses", json={"slug": "alpha", "name": "Alpha", "description": "first"}
    )

    r = await client.get("/api/v1/courses")
    assert r.status_code == 200
    body = r.json()
    assert [c["slug"] for c in body] == ["alpha", "zeta"]  # slug-ordered
    alpha = body[0]
    assert set(alpha) == {"id", "slug", "name", "description"}
    assert alpha["name"] == "Alpha"
    assert alpha["description"] == "first"


# ---------- POST /api/v1/courses/{id}/outline:import ----------


@pytest.mark.asyncio
async def test_import_happy_path_inserts_tree(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    create = await client.post("/api/v1/courses", json={"slug": "t21-course", "name": "T21 Course"})
    course_id = create.json()["id"]

    r = await client.post(f"/api/v1/courses/{course_id}/outline:import", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["nodes_imported"] == 3
    assert body["course"]["slug"] == "t21-course"

    # DB-side: 3 rows with the expected parent/depth shape.
    rows = (
        (
            await db_session.execute(
                select(OutlineNode)
                .where(OutlineNode.course_id == course_id)
                .order_by(OutlineNode.depth, OutlineNode.position)
            )
        )
        .scalars()
        .all()
    )
    assert [(n.depth, n.name) for n in rows] == [
        (0, "Root"),
        (1, "LeafA"),
        (1, "LeafB"),
    ]
    root = next(n for n in rows if n.depth == 0)
    assert all(n.parent_id == root.id for n in rows if n.depth == 1)


@pytest.mark.asyncio
async def test_import_reupload_is_idempotent_v_o3(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """V-O3: re-upload wipes the prior outline and reinserts."""
    create = await client.post("/api/v1/courses", json={"slug": "t21-course", "name": "T21 Course"})
    course_id = create.json()["id"]

    r1 = await client.post(f"/api/v1/courses/{course_id}/outline:import", json=_payload())
    assert r1.status_code == 200

    first_ids = {
        n.id
        for n in (
            await db_session.execute(select(OutlineNode).where(OutlineNode.course_id == course_id))
        )
        .scalars()
        .all()
    }
    assert len(first_ids) == 3

    r2 = await client.post(f"/api/v1/courses/{course_id}/outline:import", json=_payload())
    assert r2.status_code == 200
    assert r2.json()["nodes_imported"] == 3

    second_rows = (
        (await db_session.execute(select(OutlineNode).where(OutlineNode.course_id == course_id)))
        .scalars()
        .all()
    )
    # Same count, but every row is freshly inserted (wipe-then-insert, not merge).
    assert len(second_rows) == 3
    assert {n.id for n in second_rows}.isdisjoint(first_ids)


@pytest.mark.asyncio
async def test_import_validation_failure_returns_422_and_leaves_db_clean_v_o2(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """V-O2: whole-upload rejection on validation; no partial materialization."""
    create = await client.post("/api/v1/courses", json={"slug": "t21-course", "name": "T21 Course"})
    course_id = create.json()["id"]

    bad = _payload()
    # Break the parent chain: a depth-2 node whose parent isn't in the upload.
    bad["nodes"].append(
        {
            "path": ["Root", "Missing", "Orphan"],
            "kind": "topic",
            "name": "Orphan",
            "position": 1,
        }
    )

    r = await client.post(f"/api/v1/courses/{course_id}/outline:import", json=bad)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "errors" in detail
    assert any("parent" in e for e in detail["errors"])

    # DB untouched.
    rows = (
        (await db_session.execute(select(OutlineNode).where(OutlineNode.course_id == course_id)))
        .scalars()
        .all()
    )
    assert rows == []


@pytest.mark.asyncio
async def test_import_collects_multiple_errors_in_one_response(
    client: AsyncClient,
) -> None:
    create = await client.post("/api/v1/courses", json={"slug": "t21-course", "name": "T21 Course"})
    course_id = create.json()["id"]

    # Body slug matches course; multiple node-level problems.
    bad = {
        "course": {"slug": "t21-course", "name": "T21 Course"},
        "nodes": [
            {"path": [], "kind": "x", "name": ""},  # empty path + empty name
            {
                "path": ["X"],
                "kind": "topic",
                "name": "Y",  # name/path mismatch
                "position": 1,
            },
        ],
    }
    r = await client.post(f"/api/v1/courses/{course_id}/outline:import", json=bad)
    assert r.status_code == 422
    assert len(r.json()["detail"]["errors"]) >= 2


@pytest.mark.asyncio
async def test_import_slug_mismatch_returns_409(client: AsyncClient) -> None:
    create = await client.post("/api/v1/courses", json={"slug": "t21-course", "name": "T21 Course"})
    course_id = create.json()["id"]

    body = _payload(slug="wrong-slug", name="Wrong")
    r = await client.post(f"/api/v1/courses/{course_id}/outline:import", json=body)
    assert r.status_code == 409
    assert "does not match" in r.json()["detail"]


@pytest.mark.asyncio
async def test_import_unknown_course_returns_404(client: AsyncClient) -> None:
    r = await client.post("/api/v1/courses/999999/outline:import", json=_payload())
    assert r.status_code == 404


# ---------- GET /api/v1/courses/{id}/outline ----------


@pytest.mark.asyncio
async def test_read_outline_empty_before_import(client: AsyncClient) -> None:
    create = await client.post("/api/v1/courses", json={"slug": "t21-empty", "name": "Empty"})
    course_id = create.json()["id"]

    r = await client.get(f"/api/v1/courses/{course_id}/outline")
    assert r.status_code == 200
    body = r.json()
    assert body["course"]["slug"] == "t21-empty"
    assert body["nodes"] == []


@pytest.mark.asyncio
async def test_read_outline_returns_full_tree_in_depth_order(
    client: AsyncClient,
) -> None:
    create = await client.post("/api/v1/courses", json={"slug": "t21-course", "name": "T21 Course"})
    course_id = create.json()["id"]
    await client.post(f"/api/v1/courses/{course_id}/outline:import", json=_payload())

    r = await client.get(f"/api/v1/courses/{course_id}/outline")
    assert r.status_code == 200
    nodes = r.json()["nodes"]
    assert [(n["depth"], n["name"]) for n in nodes] == [
        (0, "Root"),
        (1, "LeafA"),
        (1, "LeafB"),
    ]


@pytest.mark.asyncio
async def test_read_outline_unknown_course_returns_404(client: AsyncClient) -> None:
    r = await client.get("/api/v1/courses/999999/outline")
    assert r.status_code == 404


# ---------- auth gate (whole router) ----------


@pytest.mark.asyncio
async def test_outline_router_requires_coach_token(client: AsyncClient) -> None:
    """No token → 401/403 across the router (read + write)."""
    client.headers.pop("X-Coach-Token", None)
    assert (await client.get("/api/v1/courses")).status_code in (401, 403)
    assert (await client.post("/api/v1/courses", json={"slug": "x", "name": "X"})).status_code in (
        401,
        403,
    )
    assert (await client.get("/api/v1/courses/1/outline")).status_code in (401, 403)
