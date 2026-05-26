"""HTTP tests for the /api/v1/tutor/* surface added in ticket 9.0.

Integration-style: each test exercises the FastAPI route via ASGITransport
against the seeded outline + a per-test rolled-back transaction. Auth uses
the shared COACH_TOKEN.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.config import settings
from app.main import app
from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question
from app.models.features import QuestionFeatures
from app.models.outline import Topic

HEADERS = {"X-Coach-Token": settings.COACH_TOKEN}


@pytest.fixture
async def env(seeded_report, test_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MEDIA_ROOT", tmp_path)
    conn = await test_engine.connect()
    await conn.begin()

    def make_session() -> AsyncSession:
        return AsyncSession(
            bind=conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    async def _override_session():
        session = make_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    app.dependency_overrides[get_session] = _override_session
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, make_session
    finally:
        app.dependency_overrides.pop(get_session, None)
        await conn.rollback()
        await conn.close()


def _new_qid() -> str:
    return f"q-{uuid.uuid4().hex[:10]}"


def _make_question(qid: str | None = None) -> Question:
    return Question(
        qid=qid or _new_qid(),
        stem_html="<p>What is acceleration?</p>",
        stem_plain="What is acceleration?",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a"},
            {"key": "B", "html": "<p>b</p>", "plain": "b"},
        ],
        correct_choice="A",
        explanation_html=None,
        explanation_plain="because",
        uworld_aamc_tags=[],
        needs_categorization=False,
    )


def _make_attempt(
    *, question_id: int, when: datetime | None = None, uworld_test_id: str | None = None
) -> Attempt:
    return Attempt(
        question_id=question_id,
        attempted_at=when or datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        selected_choice="A",
        is_correct=True,
        time_seconds=None,
        flagged=False,
        uworld_test_id=uworld_test_id,
    )


# ---------- auth ----------


@pytest.mark.asyncio
async def test_tutor_route_requires_coach_token(env):
    client, _ = env
    r = await client.get("/api/v1/tutor/healthz")
    assert r.status_code == 401
    r2 = await client.get("/api/v1/tutor/healthz", headers={"X-Coach-Token": "wrong"})
    assert r2.status_code == 401
    r3 = await client.get("/api/v1/tutor/healthz", headers=HEADERS)
    assert r3.status_code == 200


# ---------- healthcheck ----------


@pytest.mark.asyncio
async def test_healthcheck_db_reachable_when_db_up(env):
    client, _ = env
    r = await client.get("/api/v1/tutor/healthz", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["db_reachable"] is True
    assert body["db_error"] is None
    assert "attempt_count" in body
    assert "latest_attempt_at" in body
    assert "recommender_ready" in body
    assert "backend_base_url" in body


@pytest.mark.asyncio
async def test_healthcheck_attempt_count_matches(env, monkeypatch):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        for i in range(3):
            s.add(
                _make_attempt(
                    question_id=q.id,
                    when=datetime(2026, 5, 18, 10 + i, 0, tzinfo=timezone.utc),
                )
            )
        await s.commit()

    r = await client.get("/api/v1/tutor/healthz", headers=HEADERS)
    assert r.json()["attempt_count"] == 3
    assert r.json()["latest_attempt_at"] == "2026-05-18T12:00:00+00:00"


@pytest.mark.asyncio
async def test_healthcheck_recommender_ready_flag(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        await s.commit()
        qid_int = q.id

    r = await client.get("/api/v1/tutor/healthz", headers=HEADERS)
    assert r.json()["recommender_ready"] is False

    async with make_session() as s:
        s.add(
            QuestionFeatures(
                question_id=qid_int,
                question_format="discrete",
                reasoning_type="application",
                requires_calculation=False,
                calculation_steps=0,
                involves_graph_or_figure=False,
                involves_data_table=False,
                has_negative_phrasing=False,
                distractor_difficulty="medium",
                trap_distractor_present=False,
                jargon_density="low",
                key_concept_summary="x",
                extractor_version="v1",
            )
        )
        await s.commit()

    r2 = await client.get("/api/v1/tutor/healthz", headers=HEADERS)
    assert r2.json()["recommender_ready"] is True


# ---------- captures ----------


@pytest.mark.asyncio
async def test_recent_captures_returns_newest_first_and_excludes_time(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        s.add(_make_attempt(question_id=q.id, when=datetime(2026, 5, 1, tzinfo=timezone.utc)))
        s.add(_make_attempt(question_id=q.id, when=datetime(2026, 5, 10, tzinfo=timezone.utc)))
        s.add(_make_attempt(question_id=q.id, when=datetime(2026, 5, 5, tzinfo=timezone.utc)))
        await s.commit()

    r = await client.get("/api/v1/tutor/captures/recent?n=5", headers=HEADERS)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 3
    # newest first
    assert rows[0]["attempted_at"].startswith("2026-05-10")
    # exclude time_seconds per CLAUDE.md
    assert "time_seconds" not in rows[0]


@pytest.mark.asyncio
async def test_recent_captures_clamps_n(env):
    client, _ = env
    r = await client.get("/api/v1/tutor/captures/recent?n=9999", headers=HEADERS)
    assert r.status_code == 422  # ge=1, le=50 query validator


# ---------- questions ----------


@pytest.mark.asyncio
async def test_get_question_by_qid_ok_and_404(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question(qid="QQQ1")
        s.add(q)
        await s.commit()

    r = await client.get("/api/v1/tutor/questions/by-qid/QQQ1", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["qid"] == "QQQ1"
    assert body["correct_choice"] == "A"

    r404 = await client.get("/api/v1/tutor/questions/by-qid/does-not-exist", headers=HEADERS)
    assert r404.status_code == 404
    assert r404.json()["detail"]["reason"] == "question_not_found"


@pytest.mark.asyncio
async def test_get_question_by_attempt_id_ok_and_404(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        a = _make_attempt(question_id=q.id)
        s.add(a)
        await s.flush()
        attempt_id = a.id
        await s.commit()

    r = await client.get(f"/api/v1/tutor/questions/by-attempt-id/{attempt_id}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["question_id"] == q.id

    r404 = await client.get("/api/v1/tutor/questions/by-attempt-id/999999", headers=HEADERS)
    assert r404.status_code == 404
    assert r404.json()["detail"]["reason"] == "attempt_not_found"


# ---------- sessions ----------


@pytest.mark.asyncio
async def test_latest_session_id_or_null(env):
    client, _ = env
    r = await client.get("/api/v1/tutor/sessions/latest", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["test_id"] is None


@pytest.mark.asyncio
async def test_recent_sessions_groups_by_test_id(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        s.add(
            _make_attempt(
                question_id=q.id,
                when=datetime(2026, 5, 17, 10, tzinfo=timezone.utc),
                uworld_test_id="TID-A",
            )
        )
        s.add(
            _make_attempt(
                question_id=q.id,
                when=datetime(2026, 5, 18, 10, tzinfo=timezone.utc),
                uworld_test_id="TID-B",
            )
        )
        s.add(
            _make_attempt(
                question_id=q.id,
                when=datetime(2026, 5, 18, 11, tzinfo=timezone.utc),
                uworld_test_id="TID-B",
            )
        )
        await s.commit()

    r = await client.get("/api/v1/tutor/sessions/recent?n=5", headers=HEADERS)
    rows = r.json()
    assert {row["test_id"] for row in rows} == {"TID-A", "TID-B"}
    tid_b = next(r for r in rows if r["test_id"] == "TID-B")
    assert tid_b["attempt_count"] == 2

    rlatest = await client.get("/api/v1/tutor/sessions/latest", headers=HEADERS)
    assert rlatest.json()["test_id"] == "TID-B"

    rsum = await client.get("/api/v1/tutor/sessions/TID-B/summary", headers=HEADERS)
    assert rsum.status_code == 200
    assert rsum.json()["attempt_count"] == 2

    r404 = await client.get("/api/v1/tutor/sessions/MISSING/summary", headers=HEADERS)
    assert r404.status_code == 404
    assert r404.json()["detail"]["reason"] == "session_not_found"


# ---------- flagged attempts ----------


@pytest.mark.asyncio
async def test_flagged_attempts_includes_dashboard_url(env, monkeypatch):
    client, make_session = env
    monkeypatch.setattr(settings, "BACKEND_BASE_URL", "http://override.example.com")
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        a = _make_attempt(question_id=q.id)
        s.add(a)
        await s.flush()
        s.add(
            AttemptNote(
                attempt_id=a.id,
                note_text="flag-me",
                source="mcp",
                flag_for_review=True,
                created_at=datetime(2026, 5, 18, 12, tzinfo=timezone.utc),
            )
        )
        await s.commit()
        qid_int = q.id

    r = await client.get("/api/v1/tutor/attempts/flagged?limit=10", headers=HEADERS)
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["dashboard_url"] == f"http://override.example.com/questions/{qid_int}"
    assert rows[0]["dashboard_path"] == f"/questions/{qid_int}"


@pytest.mark.asyncio
async def test_flagged_attempts_relative_url_when_base_unset(env, monkeypatch):
    client, make_session = env
    # BACKEND_BASE_URL has a non-None default, so simulate the "unset" branch
    # with an empty string — flags.py treats falsy bases as "no override".
    monkeypatch.setattr(settings, "BACKEND_BASE_URL", "")
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        a = _make_attempt(question_id=q.id)
        s.add(a)
        await s.flush()
        s.add(
            AttemptNote(
                attempt_id=a.id,
                note_text="x",
                source="mcp",
                flag_for_review=True,
                created_at=datetime(2026, 5, 18, 12, tzinfo=timezone.utc),
            )
        )
        await s.commit()
        qid_int = q.id

    r = await client.get("/api/v1/tutor/attempts/flagged?limit=10", headers=HEADERS)
    rows = r.json()
    assert rows[0]["dashboard_url"] == f"/questions/{qid_int}"


# ---------- outline ----------


@pytest.mark.asyncio
async def test_topics_search_ilike_match(env, make_session=None):
    client, make_session = env
    async with make_session() as s:
        # seeded outline already has plenty of topics; pick one and search by substring
        row = (await s.execute(select(Topic).limit(1))).scalar_one()
        substr = row.name[:4].lower()

    r = await client.get(
        f"/api/v1/tutor/outline/topics/search?q={substr}&limit=5",
        headers=HEADERS,
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 1
    assert all(substr in row["name"].lower() for row in rows)
    assert "section_code" in rows[0]


@pytest.mark.asyncio
async def test_aamc_outline_returns_tree(env):
    client, _ = env
    r = await client.get("/api/v1/tutor/outline", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    section_codes = {s["code"] for s in body["sections"]}
    assert {"CP", "BB", "PS", "CARS"}.issubset(section_codes)


# ---------- notes (existing route extension: source='mcp') ----------


@pytest.mark.asyncio
async def test_add_note_with_source_mcp_via_existing_route(env):
    client, make_session = env
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        a = _make_attempt(question_id=q.id)
        s.add(a)
        await s.flush()
        await s.commit()
        attempt_id = a.id

    r = await client.post(
        f"/api/v1/attempts/{attempt_id}/notes",
        json={"note_text": "from mcp", "flag_for_review": True, "source": "mcp"},
        headers=HEADERS,
    )
    assert r.status_code == 201
    assert r.json()["source"] == "mcp"
    assert r.json()["flag_for_review"] is True


# ---------- recommender (existing route extension: bumped cap + resolved_section_code) ----------


@pytest.mark.asyncio
async def test_recommendations_accepts_n_up_to_50(env):
    """9.0 bumped the n cap from 20 to 50 to match MCP get_recommendations."""
    client, _ = env
    r = await client.get("/api/v1/recommendations/study-next?n=30", headers=HEADERS)
    assert r.status_code == 200
    # n=51 still rejected
    r2 = await client.get("/api/v1/recommendations/study-next?n=51", headers=HEADERS)
    assert r2.status_code == 422


@pytest.mark.asyncio
async def test_recommendations_response_includes_resolved_section_code_field(env):
    client, _ = env
    r = await client.get("/api/v1/recommendations/study-next?n=5", headers=HEADERS)
    body = r.json()
    # whether or not there are recs, the field shape is now part of the response model
    for rec in body["recommendations"]:
        assert "resolved_section_code" in rec
