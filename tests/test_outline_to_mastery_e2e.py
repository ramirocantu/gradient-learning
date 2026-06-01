"""RCA-10 §T7 — E2E: Outline → Mastery.

Drives the real HTTP surface (POST /courses, POST outline:import, GET tree,
GET mastery) end to end, codifying the T6 manual pass:

- import validates-then-materializes atomically; a broken upload is rejected
  whole with 422 and leaves the existing tree untouched (V-O2);
- mastery rolls up over the subtree as a SET — a question tagged to multiple
  nodes is counted ONCE (V-O1). This is the load-bearing assertion: course
  total = distinct tagged questions' attempts, not per-tag double counting.

Mastery inputs (Question/QuestionTag/Attempt) are seeded directly (source=
'manual', no LLM) so the rollup arithmetic is deterministic. Self-seeded per
test against gradient_test; the per-test savepoint rolls back (V1, V3).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag


def _schema(slug: str) -> dict[str, Any]:
    """Physics → {Kinematics, Energy} — a 3-node, 2-deep outline."""
    return {
        "course": {"slug": slug, "name": "Mastery E2E"},
        "nodes": [
            {"path": ["Physics"], "kind": "section", "name": "Physics", "position": 1},
            {"path": ["Physics", "Kinematics"], "kind": "topic", "name": "Kinematics", "position": 1},
            {"path": ["Physics", "Energy"], "kind": "topic", "name": "Energy", "position": 2},
        ],
    }


async def _create_course(client: AsyncClient, headers, slug: str) -> int:
    r = await client.post("/api/v1/courses", json={"slug": slug, "name": "Mastery E2E"}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def _tree_by_name(client: AsyncClient, headers, course_id: int) -> dict[str, int]:
    r = await client.get(f"/api/v1/courses/{course_id}/outline", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    nodes = body["nodes"] if isinstance(body, dict) else body
    return {n["name"]: n["id"] for n in nodes}


async def _seed_question(session: AsyncSession, course_id: int) -> Question:
    q = Question(
        source="manual",
        qid=f"q-{uuid.uuid4().hex[:10]}",
        course_id=course_id,
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[{"key": "A", "html": "a", "plain": "a", "media_ids": []}],
        correct_choice="A",
        needs_categorization=False,
    )
    session.add(q)
    await session.flush()
    return q


async def _tag(session: AsyncSession, question_id: int, node_id: int) -> None:
    # source='manual' → confidence NULL (V-T3 check constraint).
    session.add(QuestionTag(question_id=question_id, node_id=node_id, source="manual"))
    await session.flush()


async def _attempt(session: AsyncSession, question_id: int, *, correct: bool) -> None:
    session.add(
        Attempt(
            question_id=question_id,
            source="manual",
            attempted_at=datetime.now(timezone.utc),
            selected_choice="A",
            is_correct=correct,
        )
    )
    await session.flush()


# --------------------------------------------------------------------------- #
# import — validate-then-materialize, atomic reject (V-O2)
# --------------------------------------------------------------------------- #


async def test_import_materializes_tree_and_rejects_invalid(client: AsyncClient, coach_headers):
    slug = f"m-{uuid.uuid4().hex[:6]}"
    cid = await _create_course(client, coach_headers, slug)

    ok = await client.post(
        f"/api/v1/courses/{cid}/outline:import", json=_schema(slug), headers=coach_headers
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["nodes_imported"] == 3
    assert set(await _tree_by_name(client, coach_headers, cid)) == {"Physics", "Kinematics", "Energy"}

    # broken parent chain → whole-upload 422, tree unchanged (V-O2 atomic)
    bad = _schema(slug)
    bad["nodes"].append(
        {"path": ["Physics", "Ghost", "Orphan"], "kind": "topic", "name": "Orphan", "position": 3}
    )
    rej = await client.post(
        f"/api/v1/courses/{cid}/outline:import", json=bad, headers=coach_headers
    )
    assert rej.status_code == 422
    assert len(await _tree_by_name(client, coach_headers, cid)) == 3  # untouched


# --------------------------------------------------------------------------- #
# mastery — subtree set rollup + multi-tag dedup (V-O1)
# --------------------------------------------------------------------------- #


async def test_mastery_rollup_dedups_multi_tagged_question(
    client: AsyncClient, coach_headers, db_session: AsyncSession
):
    slug = f"m-{uuid.uuid4().hex[:6]}"
    cid = await _create_course(client, coach_headers, slug)
    await client.post(
        f"/api/v1/courses/{cid}/outline:import", json=_schema(slug), headers=coach_headers
    )
    ids = await _tree_by_name(client, coach_headers, cid)

    # q_a: tagged Kinematics only, 1 correct attempt.
    q_a = await _seed_question(db_session, cid)
    await _tag(db_session, q_a.id, ids["Kinematics"])
    await _attempt(db_session, q_a.id, correct=True)

    # q_b: tagged BOTH Kinematics AND Energy (2 tags), 1 incorrect attempt.
    q_b = await _seed_question(db_session, cid)
    await _tag(db_session, q_b.id, ids["Kinematics"])
    await _tag(db_session, q_b.id, ids["Energy"])
    await _attempt(db_session, q_b.id, correct=False)
    await db_session.commit()

    async def mastery(path: str) -> dict[str, Any]:
        r = await client.get(path, headers=coach_headers)
        assert r.status_code == 200, r.text
        return r.json()

    # Kinematics subtree: {q_a, q_b} → 2 attempts, 1 correct.
    kin = await mastery(f"/api/v1/outline/nodes/{ids['Kinematics']}/mastery")
    assert (kin["rollup"]["attempts"], kin["rollup"]["correct"]) == (2, 1)

    # Energy subtree: {q_b} → 1 attempt, 0 correct.
    eng = await mastery(f"/api/v1/outline/nodes/{ids['Energy']}/mastery")
    assert (eng["rollup"]["attempts"], eng["rollup"]["correct"]) == (1, 0)

    # Course total: distinct {q_a, q_b} → 2 attempts (q_b's 2 tags counted
    # ONCE — V-O1), 1 correct. NOT 3.
    course = await mastery(f"/api/v1/outline/courses/{cid}/mastery")
    assert (course["total"]["attempts"], course["total"]["correct"]) == (2, 1)

    # Physics root subtree = union(Kinematics, Energy) with q_b deduped = 2.
    phys = await mastery(f"/api/v1/outline/nodes/{ids['Physics']}/mastery")
    assert phys["rollup"]["attempts"] == 2


async def test_mastery_empty_course_is_zero(client: AsyncClient, coach_headers):
    slug = f"m-{uuid.uuid4().hex[:6]}"
    cid = await _create_course(client, coach_headers, slug)
    await client.post(
        f"/api/v1/courses/{cid}/outline:import", json=_schema(slug), headers=coach_headers
    )
    r = await client.get(f"/api/v1/outline/courses/{cid}/mastery", headers=coach_headers)
    assert r.status_code == 200
    assert r.json()["total"] == {"attempts": 0, "correct": 0, "accuracy": 0.0, "wilson_lower": 0.0}
