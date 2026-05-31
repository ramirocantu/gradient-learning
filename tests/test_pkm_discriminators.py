"""T31 — POST /api/v1/pkm/discriminators contract tests (V-M1, V-M3).

V-M1: persist-only — the payload carries data, no verdict / grade /
heuristic; X-Coach-Token gates the write.
V-M3: append-only — dedupe by (question_id, factor_text); re-writing the
same factor is idempotent (⊥ duplicate); distinct factors on one question
are all preserved.
"""

from __future__ import annotations

import uuid as _uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.captures import Question
from app.models.discriminator_factor import DiscriminatorFactor
from app.models.outline import Course, OutlineNode
from app.schemas.pkm import DiscriminatorIn

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _make_question(db: AsyncSession) -> Question:
    q = Question(
        source="uworld",
        qid=f"q-{_uuid.uuid4().hex[:10]}",
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[{"id": "A", "text": "x"}],
        correct_choice="A",
    )
    db.add(q)
    await db.flush()
    await db.commit()
    return q


async def _make_node(db: AsyncSession) -> OutlineNode:
    course = Course(slug=f"c-{_uuid.uuid4().hex[:8]}", name="C")
    db.add(course)
    await db.flush()
    node = OutlineNode(
        course_id=course.id,
        parent_id=None,
        kind="concept",
        name=f"n-{_uuid.uuid4().hex[:4]}",
        depth=0,
        position=0,
    )
    db.add(node)
    await db.flush()
    await db.commit()
    return node


async def _count(db: AsyncSession, question_id: int) -> int:
    return int(
        (
            await db.execute(
                select(func.count())
                .select_from(DiscriminatorFactor)
                .where(DiscriminatorFactor.question_id == question_id)
            )
        ).scalar_one()
    )


# --------------------------------------------------------------------------- #
# persist (V-M1)
# --------------------------------------------------------------------------- #


async def test_persist_creates_factor(client: AsyncClient, db_session: AsyncSession):
    q = await _make_question(db_session)
    r = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": q.id, "factor_text": "confuses Km with Vmax"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["question_id"] == q.id
    assert body["factor_text"] == "confuses Km with Vmax"
    assert body["node_id"] is None
    assert body["notion_block_id"] is None  # Notion mirror = T32
    assert body["id"] > 0


async def test_node_id_persisted(client: AsyncClient, db_session: AsyncSession):
    q = await _make_question(db_session)
    node = await _make_node(db_session)
    r = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": q.id, "factor_text": "f", "node_id": node.id},
    )
    assert r.status_code == 200
    assert r.json()["node_id"] == node.id


# --------------------------------------------------------------------------- #
# append-only dedupe (V-M3)
# --------------------------------------------------------------------------- #


async def test_dedupe_same_factor_is_idempotent(client: AsyncClient, db_session: AsyncSession):
    """V-M3: re-writing the same (question, factor) returns the same row,
    ⊥ a duplicate."""
    q = await _make_question(db_session)
    payload = {"question_id": q.id, "factor_text": "same factor"}

    first = await client.post("/api/v1/pkm/discriminators", headers=_AUTH, json=payload)
    second = await client.post("/api/v1/pkm/discriminators", headers=_AUTH, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert await _count(db_session, q.id) == 1


async def test_distinct_factors_all_preserved(client: AsyncClient, db_session: AsyncSession):
    """V-M3: distinct factors on one question are all kept (links preserved)."""
    q = await _make_question(db_session)
    for txt in ("factor a", "factor b", "factor c"):
        r = await client.post(
            "/api/v1/pkm/discriminators",
            headers=_AUTH,
            json={"question_id": q.id, "factor_text": txt},
        )
        assert r.status_code == 200
    assert await _count(db_session, q.id) == 3


async def test_factor_text_trimmed_before_dedupe(client: AsyncClient, db_session: AsyncSession):
    q = await _make_question(db_session)
    a = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": q.id, "factor_text": "trimmed"},
    )
    b = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": q.id, "factor_text": "  trimmed  "},
    )
    assert a.json()["id"] == b.json()["id"]
    assert await _count(db_session, q.id) == 1


# --------------------------------------------------------------------------- #
# auth + validation guards
# --------------------------------------------------------------------------- #


async def test_missing_token_401(client: AsyncClient, db_session: AsyncSession):
    q = await _make_question(db_session)
    r = await client.post(
        "/api/v1/pkm/discriminators",
        json={"question_id": q.id, "factor_text": "x"},
    )
    assert r.status_code == 401


async def test_unknown_question_404(client: AsyncClient):
    r = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": 999_999, "factor_text": "x"},
    )
    assert r.status_code == 404
    assert r.json()["detail"]["reason"] == "question_not_found"


async def test_blank_factor_422(client: AsyncClient, db_session: AsyncSession):
    q = await _make_question(db_session)
    r = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": q.id, "factor_text": ""},
    )
    assert r.status_code == 422


async def test_whitespace_only_factor_422(client: AsyncClient, db_session: AsyncSession):
    q = await _make_question(db_session)
    r = await client.post(
        "/api/v1/pkm/discriminators",
        headers=_AUTH,
        json={"question_id": q.id, "factor_text": "   "},
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# V-M1 — no verdict in the signature
# --------------------------------------------------------------------------- #


def test_payload_carries_data_only_no_verdict():
    """V-M1: the write payload exposes data fields only — no verdict /
    grade / score / heuristic the host's reasoning would belong in."""
    assert set(DiscriminatorIn.model_fields) == {
        "question_id",
        "factor_text",
        "node_id",
    }
