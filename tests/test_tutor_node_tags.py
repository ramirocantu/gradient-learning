"""Node tags surfaced on tutor reads (§T38, V-O1, V-O5, V-T1, V-D1, V-M1).

`tutor/captures/recent` and `tutor/sessions/{id}/summary` previously
returned empty `topics` / `by_topic` stubs. T38 resolves each question's
canonical `QuestionTag.node_id` → `{node_id, name, path, kind}` and rolls
session attempts up per node (set membership, V-O1).

Coverage:
- captures: topics resolve node_id + `>>` path (V-T1, V-O5); overridden
  tag excluded; multi-tag question surfaces multiple nodes.
- sessions: by_topic per-node counts + accuracy; a question in two nodes
  counts in both (V-O1 set rollup); top_topics deterministic (V-M1).
- HTTP: populated payload through the public API + X-Coach-Token gate.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Course, OutlineNode
from app.services.tutor import captures as captures_svc
from app.services.tutor import sessions as sessions_svc

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _seed_outline(db: AsyncSession) -> dict[str, int]:
    """Two-deep outline: Proteins >> {Amino acids, Enzymes}."""
    course = Course(slug="biochem", name="Biochem")
    db.add(course)
    await db.flush()
    ids: dict[str, int] = {"_course_id": course.id}
    shape = [
        (None, "Proteins", "section", 0, 0),
        ("Proteins", "Amino acids", "topic", 1, 0),
        ("Proteins", "Enzymes", "topic", 1, 1),
    ]
    for parent_name, name, kind, depth, pos in shape:
        n = OutlineNode(
            course_id=course.id,
            parent_id=ids[parent_name] if parent_name else None,
            kind=kind,
            name=name,
            depth=depth,
            position=pos,
        )
        db.add(n)
        await db.flush()
        ids[name] = n.id
    return ids


async def _add_question(db: AsyncSession, qid: str) -> Question:
    q = Question(
        source="uworld",
        qid=qid,
        stem_html=f"<p>{qid}</p>",
        stem_plain=f"stem {qid}",
        choices=[{"label": "A", "text": "a"}],
        correct_choice="A",
    )
    db.add(q)
    await db.flush()
    return q


def _tag(question_id: int, node_id: int, *, overridden: bool = False) -> QuestionTag:
    return QuestionTag(
        question_id=question_id,
        node_id=node_id,
        source="manual",
        confidence=None,
        is_overridden=overridden,
    )


def _attempt(question_id: int, *, test_id: str, correct: bool, when: int) -> Attempt:
    return Attempt(
        question_id=question_id,
        source="uworld",
        attempted_at=datetime(2026, 5, 27, 12, when, tzinfo=timezone.utc),
        selected_choice="A",
        is_correct=correct,
        uworld_test_id=test_id,
    )


# ---------- captures (V-T1, V-O5) ----------


@pytest.mark.asyncio
async def test_captures_topics_resolve_node_id_and_path(db_session: AsyncSession) -> None:
    ids = await _seed_outline(db_session)
    q = await _add_question(db_session, "Q1")
    db_session.add(_tag(q.id, ids["Amino acids"]))
    db_session.add(_attempt(q.id, test_id="T1", correct=True, when=1))
    await db_session.commit()

    [row] = await captures_svc.get_recent_captures(db_session, n=5)
    assert row["topics"] == [
        {
            "node_id": ids["Amino acids"],
            "name": "Amino acids",
            "path": "Proteins >> Amino acids",
            "kind": "topic",
        }
    ]


@pytest.mark.asyncio
async def test_captures_excludes_overridden_tag(db_session: AsyncSession) -> None:
    ids = await _seed_outline(db_session)
    q = await _add_question(db_session, "Q1")
    db_session.add(_tag(q.id, ids["Amino acids"], overridden=True))
    db_session.add(_attempt(q.id, test_id="T1", correct=True, when=1))
    await db_session.commit()

    [row] = await captures_svc.get_recent_captures(db_session, n=5)
    assert row["topics"] == []


@pytest.mark.asyncio
async def test_captures_multi_tag_question_surfaces_both_nodes(
    db_session: AsyncSession,
) -> None:
    ids = await _seed_outline(db_session)
    q = await _add_question(db_session, "Q1")
    db_session.add(_tag(q.id, ids["Amino acids"]))
    db_session.add(_tag(q.id, ids["Enzymes"]))
    db_session.add(_attempt(q.id, test_id="T1", correct=True, when=1))
    await db_session.commit()

    [row] = await captures_svc.get_recent_captures(db_session, n=5)
    # deterministic order: sorted by node_id.
    assert [t["node_id"] for t in row["topics"]] == sorted(
        [ids["Amino acids"], ids["Enzymes"]]
    )
    assert {t["name"] for t in row["topics"]} == {"Amino acids", "Enzymes"}


# ---------- sessions (V-O1 set rollup, V-M1) ----------


@pytest.mark.asyncio
async def test_session_by_topic_counts_and_accuracy(db_session: AsyncSession) -> None:
    ids = await _seed_outline(db_session)
    q1 = await _add_question(db_session, "Q1")
    q2 = await _add_question(db_session, "Q2")
    db_session.add(_tag(q1.id, ids["Amino acids"]))
    db_session.add(_tag(q2.id, ids["Amino acids"]))
    db_session.add(_attempt(q1.id, test_id="S1", correct=True, when=1))
    db_session.add(_attempt(q2.id, test_id="S1", correct=False, when=2))
    await db_session.commit()

    summary = await sessions_svc.get_session_summary(db_session, test_id="S1")
    [node] = summary["by_topic"]
    assert node["node_id"] == ids["Amino acids"]
    assert node["path"] == "Proteins >> Amino acids"
    assert node["attempt_count"] == 2
    assert node["correct_count"] == 1
    assert node["accuracy"] == 0.5


@pytest.mark.asyncio
async def test_session_question_in_two_nodes_counts_in_both(
    db_session: AsyncSession,
) -> None:
    """V-O1: set membership — a question tagged to N nodes counts in each."""
    ids = await _seed_outline(db_session)
    q = await _add_question(db_session, "Q1")
    db_session.add(_tag(q.id, ids["Amino acids"]))
    db_session.add(_tag(q.id, ids["Enzymes"]))
    db_session.add(_attempt(q.id, test_id="S1", correct=True, when=1))
    await db_session.commit()

    summary = await sessions_svc.get_session_summary(db_session, test_id="S1")
    by_node = {n["node_id"]: n for n in summary["by_topic"]}
    assert by_node[ids["Amino acids"]]["attempt_count"] == 1
    assert by_node[ids["Enzymes"]]["attempt_count"] == 1


@pytest.mark.asyncio
async def test_session_top_topics_ranked_by_volume(db_session: AsyncSession) -> None:
    ids = await _seed_outline(db_session)
    q1 = await _add_question(db_session, "Q1")
    q2 = await _add_question(db_session, "Q2")
    db_session.add(_tag(q1.id, ids["Enzymes"]))
    db_session.add(_tag(q2.id, ids["Amino acids"]))
    # Amino acids gets two attempts, Enzymes one → Amino acids ranks first.
    db_session.add(_attempt(q1.id, test_id="S1", correct=True, when=1))
    db_session.add(_attempt(q2.id, test_id="S1", correct=True, when=2))
    db_session.add(_attempt(q2.id, test_id="S1", correct=False, when=3))
    await db_session.commit()

    summary = await sessions_svc.get_session_summary(db_session, test_id="S1")
    assert [t["node_id"] for t in summary["top_topics"]] == [
        ids["Amino acids"],
        ids["Enzymes"],
    ]


# ---------- HTTP (V-D1) ----------


@pytest.mark.asyncio
async def test_route_captures_recent_populates_topics(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed_outline(db_session)
    q = await _add_question(db_session, "Q1")
    db_session.add(_tag(q.id, ids["Enzymes"]))
    db_session.add(_attempt(q.id, test_id="T1", correct=True, when=1))
    await db_session.commit()

    r = await client.get("/api/v1/tutor/captures/recent", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body[0]["topics"][0]["path"] == "Proteins >> Enzymes"


@pytest.mark.asyncio
async def test_route_session_summary_populates_by_topic(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    ids = await _seed_outline(db_session)
    q = await _add_question(db_session, "Q1")
    db_session.add(_tag(q.id, ids["Amino acids"]))
    db_session.add(_attempt(q.id, test_id="S1", correct=True, when=1))
    await db_session.commit()

    r = await client.get("/api/v1/tutor/sessions/S1/summary", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["by_topic"][0]["node_id"] == ids["Amino acids"]
    assert body["by_topic"][0]["path"] == "Proteins >> Amino acids"


@pytest.mark.asyncio
async def test_route_captures_requires_coach_token(client: AsyncClient) -> None:
    r = await client.get("/api/v1/tutor/captures/recent")
    assert r.status_code in (401, 403)
