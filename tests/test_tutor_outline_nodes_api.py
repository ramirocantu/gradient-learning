"""Node-keyed tutor outline surface (§T22, V-O1, V-O3, V-D1, V-M1).

The fenced AAMC-shaped routes from T17 are replaced with a domain-blind
node surface keyed on `node_id`. Tests cover service + HTTP wiring:

- search_nodes: substring + ranking + course filter; ⊥ verdict (V-M1).
- get_outline_tree: flat depth-ordered list for any course (V-O3).
- get_subtree: rollup set via `subtree_node_ids` (V-O1).
- Two-course coexistence: search across both, filter to one (V-O3).
- X-Coach-Token gate.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.outline import Course, OutlineNode
from app.services.tutor import outline as outline_svc


_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _seed_course(
    db: AsyncSession,
    *,
    slug: str,
    name: str,
    shape: list[tuple[str | None, str, str, int, int]],
) -> dict[str, int]:
    """Materialize a small outline directly. Returns name → node_id (per course).

    `shape` items: (parent_local_name, name, kind, depth, position). The
    first item must have `parent_local_name=None` (a root).
    """
    course = Course(slug=slug, name=name)
    db.add(course)
    await db.flush()
    ids: dict[str, int] = {}
    for parent_name, node_name, kind, depth, pos in shape:
        parent_id = ids[parent_name] if parent_name is not None else None
        n = OutlineNode(
            course_id=course.id,
            parent_id=parent_id,
            kind=kind,
            name=node_name,
            depth=depth,
            position=pos,
        )
        db.add(n)
        await db.flush()
        ids[node_name] = n.id
    await db.commit()
    return {"_course_id": course.id, **ids}


@pytest.fixture
async def two_courses(db_session: AsyncSession) -> dict[str, dict[str, int]]:
    """Two coexisting courses (V-O3 domain-blind)."""
    biochem = await _seed_course(
        db_session,
        slug="biochem",
        name="Biochem",
        shape=[
            (None, "Proteins", "section", 0, 0),
            ("Proteins", "Amino acids", "topic", 1, 0),
            ("Proteins", "Enzymes", "topic", 1, 1),
        ],
    )
    anatomy = await _seed_course(
        db_session,
        slug="anatomy",
        name="Anatomy",
        shape=[
            (None, "Skeletal", "section", 0, 0),
            ("Skeletal", "Bones of the hand", "topic", 1, 0),
        ],
    )
    return {"biochem": biochem, "anatomy": anatomy}


# ---------- service-level (V-O1, V-O3, V-M1) ----------


@pytest.mark.asyncio
async def test_search_nodes_substring_case_insensitive(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    rows = await outline_svc.search_nodes(db_session, query="amino")
    assert any(r["name"] == "Amino acids" for r in rows)

    rows_upper = await outline_svc.search_nodes(db_session, query="AMINO")
    assert {r["node_id"] for r in rows_upper} == {
        r["node_id"] for r in rows
    }


@pytest.mark.asyncio
async def test_search_nodes_returns_path_kind_depth_course(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    [row] = await outline_svc.search_nodes(
        db_session, query="Amino", course_slug="biochem"
    )
    assert row["path"] == "Proteins >> Amino acids"
    assert row["kind"] == "topic"
    assert row["depth"] == 1
    assert row["course_slug"] == "biochem"
    assert row["course_id"] == two_courses["biochem"]["_course_id"]


@pytest.mark.asyncio
async def test_search_nodes_course_filter_narrows_to_one(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    """V-O3: course_slug filter; without it, both courses appear."""
    all_rows = await outline_svc.search_nodes(db_session, query="s")
    slugs = {r["course_slug"] for r in all_rows}
    assert slugs == {"biochem", "anatomy"}

    one_only = await outline_svc.search_nodes(
        db_session, query="s", course_slug="biochem"
    )
    assert {r["course_slug"] for r in one_only} == {"biochem"}


@pytest.mark.asyncio
async def test_search_nodes_ranks_startswith_before_contains(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    """V-M1 stays satisfied: ranking is deterministic + obvious, ⊥ a verdict."""
    rows = await outline_svc.search_nodes(
        db_session, query="bones", course_slug="anatomy"
    )
    # "Bones of the hand" begins with the query → first.
    assert rows[0]["name"] == "Bones of the hand"


@pytest.mark.asyncio
async def test_search_nodes_empty_query_returns_empty(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    """⊥ accidentally return the entire outline when the query is empty."""
    assert await outline_svc.search_nodes(db_session, query="") == []
    assert await outline_svc.search_nodes(db_session, query="   ") == []


@pytest.mark.asyncio
async def test_search_nodes_unknown_course_raises(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    with pytest.raises(outline_svc.CourseNotFoundError):
        await outline_svc.search_nodes(
            db_session, query="x", course_slug="no-such-course"
        )


@pytest.mark.asyncio
async def test_get_outline_tree_returns_flat_depth_ordered(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    tree = await outline_svc.get_outline_tree(db_session, course_slug="biochem")
    assert tree["course"]["slug"] == "biochem"
    names = [n["name"] for n in tree["nodes"]]
    # depth-then-position order:
    assert names == ["Proteins", "Amino acids", "Enzymes"]
    paths = {n["name"]: n["path"] for n in tree["nodes"]}
    assert paths["Amino acids"] == "Proteins >> Amino acids"


@pytest.mark.asyncio
async def test_get_subtree_rolls_up_descendants_v_o1(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    biochem = two_courses["biochem"]
    sub = await outline_svc.get_subtree(db_session, node_id=biochem["Proteins"])
    assert sub["node_id"] == biochem["Proteins"]
    assert sub["course_slug"] == "biochem"
    assert sub["path"] == "Proteins"
    assert set(sub["descendants"]) == {
        biochem["Amino acids"],
        biochem["Enzymes"],
    }


@pytest.mark.asyncio
async def test_get_subtree_leaf_has_no_descendants(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    biochem = two_courses["biochem"]
    sub = await outline_svc.get_subtree(db_session, node_id=biochem["Enzymes"])
    assert sub["descendants"] == []


@pytest.mark.asyncio
async def test_get_subtree_unknown_node_raises(
    db_session: AsyncSession, two_courses: dict[str, dict[str, int]]
) -> None:
    with pytest.raises(outline_svc.NodeNotFoundError):
        await outline_svc.get_subtree(db_session, node_id=9_999_999)


# ---------- HTTP-level (V-D1) ----------


async def _seed_via_client(client: AsyncClient) -> dict[str, int]:
    """Onboard a course + outline through the public API (V-D1)."""
    create = await client.post(
        "/api/v1/courses", json={"slug": "t22", "name": "T22 Course"}, headers=_AUTH
    )
    assert create.status_code == 201, create.text
    course_id = create.json()["id"]

    payload: dict[str, Any] = {
        "course": {"slug": "t22", "name": "T22 Course"},
        "nodes": [
            {"path": ["Root"], "kind": "section", "name": "Root", "position": 1},
            {
                "path": ["Root", "Alpha"],
                "kind": "topic",
                "name": "Alpha",
                "position": 1,
            },
            {
                "path": ["Root", "Beta"],
                "kind": "topic",
                "name": "Beta",
                "position": 2,
            },
        ],
    }
    imp = await client.post(
        f"/api/v1/courses/{course_id}/outline:import", json=payload, headers=_AUTH
    )
    assert imp.status_code == 200, imp.text
    return {"course_id": course_id}


@pytest.mark.asyncio
async def test_route_search_returns_rows(client: AsyncClient) -> None:
    await _seed_via_client(client)
    r = await client.get(
        "/api/v1/tutor/outline/nodes/search",
        params={"q": "alpha", "course": "t22"},
        headers=_AUTH,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["name"] == "Alpha"
    assert body[0]["path"] == "Root >> Alpha"


@pytest.mark.asyncio
async def test_route_search_unknown_course_returns_404(client: AsyncClient) -> None:
    r = await client.get(
        "/api/v1/tutor/outline/nodes/search",
        params={"q": "x", "course": "nope"},
        headers=_AUTH,
    )
    assert r.status_code == 404
    assert r.json()["detail"]["reason"] == "course_not_found"


@pytest.mark.asyncio
async def test_route_search_requires_coach_token(client: AsyncClient) -> None:
    r = await client.get(
        "/api/v1/tutor/outline/nodes/search",
        params={"q": "alpha"},
    )
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_route_outline_returns_tree(client: AsyncClient) -> None:
    await _seed_via_client(client)
    r = await client.get(
        "/api/v1/tutor/outline",
        params={"course": "t22"},
        headers=_AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["course"]["slug"] == "t22"
    assert [n["name"] for n in body["nodes"]] == ["Root", "Alpha", "Beta"]


@pytest.mark.asyncio
async def test_route_outline_unknown_course_returns_404(client: AsyncClient) -> None:
    r = await client.get(
        "/api/v1/tutor/outline",
        params={"course": "nope"},
        headers=_AUTH,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_route_subtree_returns_descendants(client: AsyncClient) -> None:
    await _seed_via_client(client)
    # find the root node_id via the outline endpoint
    outline = (
        await client.get(
            "/api/v1/tutor/outline",
            params={"course": "t22"},
            headers=_AUTH,
        )
    ).json()
    root = next(n for n in outline["nodes"] if n["name"] == "Root")
    alpha = next(n for n in outline["nodes"] if n["name"] == "Alpha")
    beta = next(n for n in outline["nodes"] if n["name"] == "Beta")

    r = await client.get(
        f"/api/v1/tutor/outline/nodes/{root['node_id']}/subtree",
        headers=_AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == root["node_id"]
    assert set(body["descendants"]) == {alpha["node_id"], beta["node_id"]}


@pytest.mark.asyncio
async def test_route_subtree_unknown_node_returns_404(client: AsyncClient) -> None:
    r = await client.get(
        "/api/v1/tutor/outline/nodes/999999/subtree", headers=_AUTH
    )
    assert r.status_code == 404
    assert r.json()["detail"]["reason"] == "node_not_found"
