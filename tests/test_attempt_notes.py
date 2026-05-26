"""Per-attempt notes — backend API + service + cascade (Ticket 6.9c)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.main import app
from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Question

HEADERS = {"X-Coach-Token": "change_me_before_use"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def env(seeded_report, test_engine):
    """ASGI client + session factory under one rolled-back outer transaction."""
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


# --------------------------------------------------------------------------- #
# Seed helpers
# --------------------------------------------------------------------------- #


def _new_qid() -> str:
    return f"q-{uuid.uuid4().hex[:10]}"


def _make_question() -> Question:
    return Question(
        qid=_new_qid(),
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_content_hashes": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_content_hashes": []},
        ],
        correct_choice="A",
        explanation_html=None,
        explanation_plain="why",
        uworld_aamc_tags=[],
        needs_categorization=False,
    )


def _make_attempt(*, question_id: int, is_correct: bool = True) -> Attempt:
    return Attempt(
        question_id=question_id,
        attempted_at=datetime.now(timezone.utc),
        selected_choice="A" if is_correct else "B",
        is_correct=is_correct,
        time_seconds=30,
        flagged=False,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_note_persists_to_db(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        attempt_id = a.id
        await session.commit()

    r = await client.post(
        f"/api/v1/attempts/{attempt_id}/notes",
        json={"note_text": "good note", "flag_for_review": False},
        headers=HEADERS,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["note_text"] == "good note"
    assert body["source"] == "user"
    assert body["flag_for_review"] is False

    async with make_session() as session:
        row = (
            await session.execute(select(AttemptNote).where(AttemptNote.attempt_id == attempt_id))
        ).scalar_one_or_none()
    assert row is not None
    assert row.source == "user"
    assert row.flag_for_review is False


@pytest.mark.asyncio
async def test_create_note_with_flag(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        attempt_id = a.id
        await session.commit()

    r = await client.post(
        f"/api/v1/attempts/{attempt_id}/notes",
        json={"note_text": "flagged note", "flag_for_review": True},
        headers=HEADERS,
    )
    assert r.status_code == 201
    assert r.json()["flag_for_review"] is True

    async with make_session() as session:
        row = (
            await session.execute(select(AttemptNote).where(AttemptNote.attempt_id == attempt_id))
        ).scalar_one_or_none()
    assert row is not None
    assert row.flag_for_review is True


@pytest.mark.asyncio
async def test_create_note_blank_text_returns_422(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        attempt_id = a.id
        await session.commit()

    r = await client.post(
        f"/api/v1/attempts/{attempt_id}/notes",
        json={"note_text": "   "},
        headers=HEADERS,
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_note_unknown_attempt_returns_404(env):
    client, _ = env
    r = await client.post(
        "/api/v1/attempts/999999/notes",
        json={"note_text": "hello"},
        headers=HEADERS,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_notes_returns_newest_first(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        attempt_id = a.id
        older = AttemptNote(
            attempt_id=attempt_id,
            note_text="older note",
            flag_for_review=False,
            source="user",
            created_at=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
        newer = AttemptNote(
            attempt_id=attempt_id,
            note_text="newer note",
            flag_for_review=False,
            source="user",
            created_at=datetime(2026, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
        session.add(older)
        session.add(newer)
        await session.commit()

    r = await client.get(f"/api/v1/attempts/{attempt_id}/notes", headers=HEADERS)
    assert r.status_code == 200
    texts = [n["note_text"] for n in r.json()]
    assert texts[0] == "newer note"
    assert texts[1] == "older note"


@pytest.mark.asyncio
async def test_list_notes_requires_auth(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        attempt_id = a.id
        await session.commit()

    r = await client.get(f"/api/v1/attempts/{attempt_id}/notes")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_delete_note_removes_row(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        note = AttemptNote(
            attempt_id=a.id,
            note_text="to be deleted",
            flag_for_review=False,
            source="user",
        )
        session.add(note)
        await session.flush()
        note_id = note.id
        await session.commit()

    r = await client.delete(f"/api/v1/attempts/notes/{note_id}", headers=HEADERS)
    assert r.status_code == 204

    async with make_session() as session:
        row = await session.get(AttemptNote, note_id)
    assert row is None


@pytest.mark.asyncio
async def test_delete_note_unknown_returns_404(env):
    client, _ = env
    r = await client.delete("/api/v1/attempts/notes/999999", headers=HEADERS)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_note_requires_auth(env):
    client, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        note = AttemptNote(
            attempt_id=a.id,
            note_text="auth test",
            flag_for_review=False,
            source="user",
        )
        session.add(note)
        await session.flush()
        note_id = note.id
        await session.commit()

    r = await client.delete(f"/api/v1/attempts/notes/{note_id}")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_cascade_on_attempt_delete(env):
    _, make_session = env
    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        note = AttemptNote(
            attempt_id=a.id,
            note_text="cascade test",
            flag_for_review=False,
            source="user",
        )
        session.add(note)
        await session.flush()
        attempt_id = a.id
        note_id = note.id
        await session.commit()

    async with make_session() as session:
        attempt = await session.get(Attempt, attempt_id)
        await session.delete(attempt)
        await session.commit()

    async with make_session() as session:
        row = await session.get(AttemptNote, note_id)
    assert row is None


@pytest.mark.asyncio
async def test_mcp_source_path_works(env):
    """create_note accepts source='mcp' — MCP write path stays open without HTTP endpoint."""
    _, make_session = env
    from app.services.attempt_notes import create_note

    async with make_session() as session:
        q = _make_question()
        session.add(q)
        await session.flush()
        a = _make_attempt(question_id=q.id)
        session.add(a)
        await session.flush()
        note = await create_note(
            session,
            attempt_id=a.id,
            note_text="mcp note",
            flag_for_review=False,
            source="mcp",
        )
        await session.flush()
        note_id = note.id
        await session.commit()

    async with make_session() as session:
        row = await session.get(AttemptNote, note_id)
    assert row is not None
    assert row.source == "mcp"
