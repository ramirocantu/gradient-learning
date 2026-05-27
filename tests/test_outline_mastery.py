"""Node/subtree + course mastery rollup (T44, desktop ¶T7).

V-O1 set rollup: a node's stats cover the DISTINCT questions tagged to any
node in its subtree (self + descendants), each question counted once; sibling
branches are excluded. Reads key on QuestionTag.node_id + outline_subtree
(V-O5, ⊥ legacy topic/cc joins). Exposed on the public API (V-D1).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Course, OutlineNode
from app.services import analytics


async def _seed(db: AsyncSession) -> dict[str, int]:
    """
    Proteins (section)            Lipids (section)
      ├── Amino acids (topic)       └── Fatty acids (topic)
      └── Enzymes (topic)

    q1 → Amino acids   (2 attempts: 1 correct, 1 wrong)
    q2 → Enzymes       (1 wrong)
    q3 → Fatty acids   (1 correct)   [Lipids branch — excluded from Proteins]
    q4 → Amino acids + Enzymes       (1 correct) [both under Proteins → once]
    """
    course = Course(slug="biochem", name="Biochem")
    db.add(course)
    await db.flush()
    ids: dict[str, int] = {"_course_id": course.id}
    shape = [
        (None, "Proteins", "section", 0, 0),
        ("Proteins", "Amino acids", "topic", 1, 0),
        ("Proteins", "Enzymes", "topic", 1, 1),
        (None, "Lipids", "section", 0, 1),
        ("Lipids", "Fatty acids", "topic", 1, 0),
    ]
    for parent, name, kind, depth, pos in shape:
        n = OutlineNode(
            course_id=course.id,
            parent_id=ids[parent] if parent else None,
            kind=kind,
            name=name,
            depth=depth,
            position=pos,
        )
        db.add(n)
        await db.flush()
        ids[name] = n.id

    async def q(qid: str) -> int:
        row = Question(
            source="uworld", qid=qid, stem_html="<p>q</p>", stem_plain="q",
            choices=[{"label": "A", "text": "a"}], correct_choice="A",
        )
        db.add(row)
        await db.flush()
        return row.id

    def tag(qid: int, node_id: int) -> None:
        db.add(QuestionTag(question_id=qid, node_id=node_id, source="manual", confidence=None))

    def attempt(qid: int, correct: bool, minute: int) -> None:
        db.add(Attempt(
            question_id=qid, source="uworld",
            attempted_at=datetime(2026, 5, 27, 12, minute, tzinfo=timezone.utc),
            selected_choice="A", is_correct=correct,
        ))

    q1, q2, q3, q4 = await q("Q1"), await q("Q2"), await q("Q3"), await q("Q4")
    tag(q1, ids["Amino acids"])
    tag(q2, ids["Enzymes"])
    tag(q3, ids["Fatty acids"])
    tag(q4, ids["Amino acids"])
    tag(q4, ids["Enzymes"])
    attempt(q1, True, 1)
    attempt(q1, False, 2)
    attempt(q2, False, 3)
    attempt(q3, True, 4)
    attempt(q4, True, 5)
    await db.commit()
    return ids


# ---------- service: set rollup (V-O1, V-O5) ----------


@pytest.mark.asyncio
async def test_node_rollup_unions_subtree_and_dedups_questions(db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    out = await analytics.compute_node_mastery(db_session, node_id=ids["Proteins"])
    # Distinct questions under Proteins = {q1, q2, q4}; q4 (tagged to two
    # child nodes) counts ONCE. Attempts: q1=2, q2=1, q4=1 → 4 total, 2 correct.
    assert out["rollup"]["attempts"] == 4
    assert out["rollup"]["correct"] == 2
    assert out["rollup"]["accuracy"] == pytest.approx(0.5)
    assert out["node"]["path"] == "Proteins"


@pytest.mark.asyncio
async def test_node_rollup_excludes_sibling_branch(db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    out = await analytics.compute_node_mastery(db_session, node_id=ids["Proteins"])
    # q3 lives under Lipids — must not leak into Proteins (would make it 5).
    assert out["rollup"]["attempts"] == 4


@pytest.mark.asyncio
async def test_node_children_breakdown(db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    out = await analytics.compute_node_mastery(db_session, node_id=ids["Proteins"])
    by_node = {c["node_id"]: c for c in out["children"]}
    # Amino acids: q1 (2 att) + q4 (1 att) = 3 attempts, 2 correct.
    assert by_node[ids["Amino acids"]]["attempts"] == 3
    assert by_node[ids["Amino acids"]]["correct"] == 2
    assert by_node[ids["Amino acids"]]["path"] == "Proteins >> Amino acids"
    # Enzymes: q2 (1 wrong) + q4 (1 correct) = 2 attempts, 1 correct.
    assert by_node[ids["Enzymes"]]["attempts"] == 2
    assert by_node[ids["Enzymes"]]["correct"] == 1


@pytest.mark.asyncio
async def test_leaf_node_rollup(db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    out = await analytics.compute_node_mastery(db_session, node_id=ids["Amino acids"])
    assert out["rollup"]["attempts"] == 3
    assert out["rollup"]["correct"] == 2
    assert out["children"] == []


@pytest.mark.asyncio
async def test_node_not_found_raises(db_session: AsyncSession) -> None:
    await _seed(db_session)
    with pytest.raises(analytics.NodeNotFoundError):
        await analytics.compute_node_mastery(db_session, node_id=999_999)


# ---------- service: course rollup ----------


@pytest.mark.asyncio
async def test_course_total_and_per_root(db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    out = await analytics.compute_course_mastery(db_session, course_id=ids["_course_id"])
    # Total distinct questions {q1,q2,q3,q4}; attempts 2+1+1+1 = 5, 3 correct.
    assert out["total"]["attempts"] == 5
    assert out["total"]["correct"] == 3
    by_node = {n["node_id"]: n for n in out["nodes"]}
    assert by_node[ids["Proteins"]]["attempts"] == 4
    assert by_node[ids["Lipids"]]["attempts"] == 1
    assert by_node[ids["Lipids"]]["correct"] == 1


@pytest.mark.asyncio
async def test_course_not_found_raises(db_session: AsyncSession) -> None:
    with pytest.raises(analytics.CourseNotFoundError):
        await analytics.compute_course_mastery(db_session, course_id=999_999)


# ---------- HTTP (V-D1, public API) ----------


@pytest.mark.asyncio
async def test_route_node_mastery(client: AsyncClient, db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    r = await client.get(f"/api/v1/outline/nodes/{ids['Proteins']}/mastery")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["node"]["name"] == "Proteins"
    assert body["rollup"]["attempts"] == 4
    assert len(body["children"]) == 2


@pytest.mark.asyncio
async def test_route_course_mastery(client: AsyncClient, db_session: AsyncSession) -> None:
    ids = await _seed(db_session)
    r = await client.get(f"/api/v1/outline/courses/{ids['_course_id']}/mastery")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"]["attempts"] == 5
    assert {n["name"] for n in body["nodes"]} == {"Proteins", "Lipids"}


@pytest.mark.asyncio
async def test_route_node_mastery_404(client: AsyncClient) -> None:
    r = await client.get("/api/v1/outline/nodes/999999/mastery")
    assert r.status_code == 404
