"""End-to-end ingest tests for Ticket 2.2.

Hits the real test Postgres DB and writes real files to a per-test tmp
MEDIA_ROOT. Each test rolls back at teardown.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.config import settings
from app.main import app
from app.models.captures import Attempt, Passage, Question, RawCapture
from app.models.media import Media
from app.models.outline import Course


_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")
_PNG_HASH = hashlib.sha256(_PNG_BYTES).hexdigest()


def _png_variant(seed: int) -> tuple[bytes, str, str]:
    raw = _PNG_BYTES + seed.to_bytes(4, "big")
    return raw, base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _base_payload(**top: Any) -> dict[str, Any]:
    return {
        "qid": top.get("qid", f"q-{uuid.uuid4().hex[:10]}"),
        "captured_at": top.get("captured_at", _now_iso()),
        "html": top.get("html", "<html>raw</html>"),
        "parsed": {
            "passage": None,
            "stem_html": "<p>What is 2+2?</p>",
            "stem_plain": "What is 2+2?",
            "choices": [
                {
                    "key": "A",
                    "html": "<p>3</p>",
                    "plain": "3",
                    "media_content_hashes": [],
                },
                {
                    "key": "B",
                    "html": "<p>4</p>",
                    "plain": "4",
                    "media_content_hashes": [],
                },
            ],
            "correct_choice": "B",
            "explanation_html": "<p>obvious</p>",
            "explanation_plain": "obvious",
            "uworld_aamc_tags": ["Math — Arithmetic"],
            "selected_choice": "B",
            "is_correct": True,
            "time_seconds": 30,
            "flagged": False,
        },
        "media": [],
        "parse_warnings": [],
        "extension_version": "0.1.0",
    }


@pytest.fixture
async def ingest_env(seeded_report, test_engine, tmp_path, monkeypatch):
    """Yield (client, make_session, media_root).

    All work shares one DB connection wrapped in an outer transaction that
    is rolled back at teardown. Each endpoint call and each verification
    session uses `join_transaction_mode='create_savepoint'`.
    """
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
            yield client, make_session, tmp_path
    finally:
        app.dependency_overrides.pop(get_session, None)
        await conn.rollback()
        await conn.close()


def _auth_headers() -> dict[str, str]:
    return {"X-Coach-Token": settings.COACH_TOKEN}


# --------------------------------------------------------------------------- #
# 1. Happy path
# --------------------------------------------------------------------------- #


async def test_happy_path_single_capture(ingest_env):
    client, make_session, _ = ingest_env
    payload = _base_payload()

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {
        "capture_id",
        "question_id",
        "attempt_id",
        "passage_id",
        "media_ids",
    }
    assert body["passage_id"] is None
    assert body["media_ids"] == []

    async with make_session() as s:
        captures = (await s.execute(select(RawCapture))).scalars().all()
        questions = (await s.execute(select(Question))).scalars().all()
        attempts = (await s.execute(select(Attempt))).scalars().all()
        assert len(captures) == 1
        assert len(questions) == 1
        assert len(attempts) == 1


# --------------------------------------------------------------------------- #
# 2. Same qid, two attempts
# --------------------------------------------------------------------------- #


async def test_same_qid_two_attempts(ingest_env):
    client, make_session, _ = ingest_env
    qid = f"q-{uuid.uuid4().hex[:10]}"
    p1 = _base_payload(qid=qid)
    p2 = copy.deepcopy(p1)
    p2["parsed"]["selected_choice"] = "A"
    p2["parsed"]["is_correct"] = False
    p2["captured_at"] = _now_iso()

    r1 = await client.post("/api/v1/captures", json=p1, headers=_auth_headers())
    assert r1.status_code == 200
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        first_updated = q.last_updated_at

    await asyncio.sleep(0.02)

    r2 = await client.post("/api/v1/captures", json=p2, headers=_auth_headers())
    assert r2.status_code == 200

    async with make_session() as s:
        questions = (await s.execute(select(Question).where(Question.qid == qid))).scalars().all()
        attempts = (await s.execute(select(Attempt))).scalars().all()
        captures = (await s.execute(select(RawCapture))).scalars().all()
        assert len(questions) == 1
        assert len(attempts) == 2
        assert len(captures) == 2
        assert questions[0].last_updated_at == first_updated


# --------------------------------------------------------------------------- #
# 3. Passage dedupe by uworld_passage_id
# --------------------------------------------------------------------------- #


async def test_passage_dedupe_by_uworld_id(ingest_env):
    client, make_session, _ = ingest_env
    uw_id = f"uw-{uuid.uuid4().hex[:8]}"

    p1 = _base_payload(qid=f"q-{uuid.uuid4().hex[:10]}")
    p1["parsed"]["passage"] = {
        "uworld_passage_id": uw_id,
        "html": "<p>Passage body</p>",
        "plain": "Passage body",
    }
    p2 = copy.deepcopy(p1)
    p2["qid"] = f"q-{uuid.uuid4().hex[:10]}"
    p2["parsed"]["passage"]["html"] = "<p>Passage  body</p>"  # whitespace diff
    p2["parsed"]["passage"]["plain"] = "Passage  body"

    r1 = await client.post("/api/v1/captures", json=p1, headers=_auth_headers())
    r2 = await client.post("/api/v1/captures", json=p2, headers=_auth_headers())
    assert r1.status_code == 200
    assert r2.status_code == 200

    async with make_session() as s:
        passages = (await s.execute(select(Passage))).scalars().all()
        questions = (await s.execute(select(Question))).scalars().all()
        assert len(passages) == 1
        assert len(questions) == 2
        assert {q.passage_id for q in questions} == {passages[0].id}
        assert r1.json()["passage_id"] == r2.json()["passage_id"] == passages[0].id


# --------------------------------------------------------------------------- #
# 4. Passage dedupe by content_hash
# --------------------------------------------------------------------------- #


async def test_passage_dedupe_by_content_hash(ingest_env):
    client, make_session, _ = ingest_env
    html = "<p>Identical passage</p>"

    p1 = _base_payload(qid=f"q-{uuid.uuid4().hex[:10]}")
    p1["parsed"]["passage"] = {
        "uworld_passage_id": None,
        "html": html,
        "plain": "Identical passage",
    }
    p2 = copy.deepcopy(p1)
    p2["qid"] = f"q-{uuid.uuid4().hex[:10]}"

    r1 = await client.post("/api/v1/captures", json=p1, headers=_auth_headers())
    r2 = await client.post("/api/v1/captures", json=p2, headers=_auth_headers())
    assert r1.status_code == 200
    assert r2.status_code == 200

    async with make_session() as s:
        passages = (await s.execute(select(Passage).where(Passage.html == html))).scalars().all()
        questions = (await s.execute(select(Question))).scalars().all()
        assert len(passages) == 1
        assert len(questions) == 2
        assert {q.passage_id for q in questions} == {passages[0].id}


# --------------------------------------------------------------------------- #
# 5. Media written + deduped by content_hash
# --------------------------------------------------------------------------- #


async def test_media_written_to_disk_and_deduped(ingest_env):
    client, make_session, media_root = ingest_env
    payload = _base_payload(qid=f"q-{uuid.uuid4().hex[:10]}")
    payload["parsed"]["choices"][0]["media_content_hashes"] = [_PNG_HASH]
    payload["parsed"]["choices"][1]["media_content_hashes"] = [_PNG_HASH]
    payload["media"] = [
        {
            "content_hash": _PNG_HASH,
            "mime_type": "image/png",
            "bytes_b64": _PNG_B64,
        }
    ]

    r1 = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r1.status_code == 200, r1.text
    expected_file = media_root / _PNG_HASH[:2] / f"{_PNG_HASH}.png"
    assert expected_file.exists()
    assert expected_file.read_bytes() == _PNG_BYTES
    mtime_before = os.path.getmtime(expected_file)

    payload2 = _base_payload(qid=f"q-{uuid.uuid4().hex[:10]}")
    payload2["parsed"]["choices"][0]["media_content_hashes"] = [_PNG_HASH]
    payload2["media"] = [
        {
            "content_hash": _PNG_HASH,
            "mime_type": "image/png",
            "bytes_b64": _PNG_B64,
        }
    ]
    await asyncio.sleep(0.02)
    r2 = await client.post("/api/v1/captures", json=payload2, headers=_auth_headers())
    assert r2.status_code == 200

    async with make_session() as s:
        media_rows = (
            (await s.execute(select(Media).where(Media.content_hash == _PNG_HASH))).scalars().all()
        )
        assert len(media_rows) == 1
    assert os.path.getmtime(expected_file) == mtime_before


# --------------------------------------------------------------------------- #
# 6. Choices.media_ids resolved correctly
# --------------------------------------------------------------------------- #


async def test_choices_media_ids_resolved(ingest_env):
    client, make_session, _ = ingest_env
    _, b64_a, hash_a = _png_variant(1)
    _, b64_b, hash_b = _png_variant(2)

    payload = _base_payload(qid=f"q-{uuid.uuid4().hex[:10]}")
    payload["parsed"]["choices"][0]["media_content_hashes"] = [hash_a]
    payload["parsed"]["choices"][1]["media_content_hashes"] = [hash_b]
    payload["media"] = [
        {"content_hash": hash_a, "mime_type": "image/png", "bytes_b64": b64_a},
        {"content_hash": hash_b, "mime_type": "image/png", "bytes_b64": b64_b},
    ]

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text

    async with make_session() as s:
        rows = (
            (await s.execute(select(Media).where(Media.content_hash.in_([hash_a, hash_b]))))
            .scalars()
            .all()
        )
        ids = {m.content_hash: m.id for m in rows}
        assert set(ids.keys()) == {hash_a, hash_b}

        q = (await s.execute(select(Question).where(Question.qid == payload["qid"]))).scalar_one()
        choices_by_key = {c["key"]: c for c in q.choices}
        assert choices_by_key["A"]["media_ids"] == [ids[hash_a]]
        assert choices_by_key["B"]["media_ids"] == [ids[hash_b]]


# --------------------------------------------------------------------------- #
# 7. needs_categorization lifecycle
# --------------------------------------------------------------------------- #


async def test_needs_categorization_lifecycle(ingest_env):
    client, make_session, _ = ingest_env
    qid = f"q-{uuid.uuid4().hex[:10]}"
    p = _base_payload(qid=qid)

    r = await client.post("/api/v1/captures", json=p, headers=_auth_headers())
    assert r.status_code == 200
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        assert q.needs_categorization is True

    # categorizer flips to False
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        q.needs_categorization = False
        await s.commit()

    # re-post identical
    r = await client.post("/api/v1/captures", json=p, headers=_auth_headers())
    assert r.status_code == 200
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        assert q.needs_categorization is False

    # re-post with changed tags
    p2 = copy.deepcopy(p)
    p2["captured_at"] = _now_iso()
    p2["parsed"]["uworld_aamc_tags"] = ["Math — Algebra"]
    r = await client.post("/api/v1/captures", json=p2, headers=_auth_headers())
    assert r.status_code == 200
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        assert q.needs_categorization is True


# --------------------------------------------------------------------------- #
# 8. last_updated_at bumps only on content change
# --------------------------------------------------------------------------- #


async def test_last_updated_at_bumps_on_content_change(ingest_env):
    client, make_session, _ = ingest_env
    qid = f"q-{uuid.uuid4().hex[:10]}"
    p = _base_payload(qid=qid)

    await client.post("/api/v1/captures", json=p, headers=_auth_headers())
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        ts0 = q.last_updated_at

    await asyncio.sleep(0.02)
    await client.post("/api/v1/captures", json=p, headers=_auth_headers())
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        assert q.last_updated_at == ts0

    p2 = copy.deepcopy(p)
    p2["parsed"]["stem_html"] = "<p>Changed stem</p>"
    p2["parsed"]["stem_plain"] = "Changed stem"
    await asyncio.sleep(0.02)
    await client.post("/api/v1/captures", json=p2, headers=_auth_headers())
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        assert q.last_updated_at > ts0


# --------------------------------------------------------------------------- #
# 9. Missing or wrong token => 401
# --------------------------------------------------------------------------- #


async def test_missing_or_wrong_token_returns_401(ingest_env):
    client, make_session, _ = ingest_env
    payload = _base_payload()

    r = await client.post("/api/v1/captures", json=payload)
    assert r.status_code == 401
    r = await client.post("/api/v1/captures", json=payload, headers={"X-Coach-Token": "wrong"})
    assert r.status_code == 401

    async with make_session() as s:
        assert (await s.execute(select(func.count()).select_from(Question))).scalar() == 0
        assert (await s.execute(select(func.count()).select_from(RawCapture))).scalar() == 0


# --------------------------------------------------------------------------- #
# 10. Invalid payload => 422 + WARNING log
# --------------------------------------------------------------------------- #


async def test_invalid_payload_returns_422_and_logs_warning(ingest_env, caplog):
    client, _, _ = ingest_env
    payload = _base_payload()
    payload["haxxor"] = True  # extra='forbid' rejects

    with caplog.at_level(logging.WARNING, logger="app.ingest.validation"):
        r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 422
    matched = [
        rec
        for rec in caplog.records
        if rec.name == "app.ingest.validation" and rec.levelno == logging.WARNING
    ]
    assert matched, f"expected WARNING log on app.ingest.validation, got {caplog.records}"


# --------------------------------------------------------------------------- #
# 11. parse_warnings round-trip
# --------------------------------------------------------------------------- #


async def test_parse_warnings_round_trip(ingest_env):
    client, make_session, _ = ingest_env
    warnings = [
        {"code": "missing_selector", "message": "no stem", "selector": ".q-stem"},
        {
            "code": "explanation_absent",
            "message": "no explanation block",
            "selector": None,
        },
    ]
    payload = _base_payload()
    payload["parse_warnings"] = warnings

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text
    capture_id = r.json()["capture_id"]
    async with make_session() as s:
        rc = (await s.execute(select(RawCapture).where(RawCapture.id == capture_id))).scalar_one()
        stored = rc.parse_warnings or []
        # the first two entries are the wire warnings, in order, before any sanity additions
        assert stored[: len(warnings)] == warnings


# --------------------------------------------------------------------------- #
# 12. Discrete question, no passage
# --------------------------------------------------------------------------- #


async def test_discrete_question_no_passage(ingest_env):
    client, make_session, _ = ingest_env
    payload = _base_payload()
    payload["parsed"]["passage"] = None

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["passage_id"] is None
    async with make_session() as s:
        questions = (await s.execute(select(Question))).scalars().all()
        passages = (await s.execute(select(Passage))).scalars().all()
        assert len(questions) == 1
        assert questions[0].passage_id is None
        assert passages == []


# --------------------------------------------------------------------------- #
# 13. CARS-shaped capture inserts cleanly
# --------------------------------------------------------------------------- #


async def test_cars_capture_with_only_cc_tag(ingest_env):
    client, make_session, _ = ingest_env
    qid = f"q-{uuid.uuid4().hex[:10]}"
    payload = _base_payload(qid=qid)
    payload["parsed"]["passage"] = {
        "uworld_passage_id": f"uw-{uuid.uuid4().hex[:8]}",
        "html": "<p>A long-ish CARS passage.</p>",
        "plain": "A long-ish CARS passage.",
    }
    payload["parsed"]["uworld_aamc_tags"] = ["CARS — Passage"]

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text
    async with make_session() as s:
        q = (await s.execute(select(Question).where(Question.qid == qid))).scalar_one()
        assert q.uworld_aamc_tags == ["CARS — Passage"]
        assert q.needs_categorization is True


# --------------------------------------------------------------------------- #
# 14. uworld_test_id persisted to RawCapture and Attempt
# --------------------------------------------------------------------------- #


async def test_ingest_persists_uworld_test_id(ingest_env):
    client, make_session, _ = ingest_env
    qid = f"q-{uuid.uuid4().hex[:10]}"
    payload = _base_payload(qid=qid)
    payload["uworld_test_id"] = "7392051"

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text
    body = r.json()

    async with make_session() as s:
        rc = (
            await s.execute(select(RawCapture).where(RawCapture.id == body["capture_id"]))
        ).scalar_one()
        attempt = (
            await s.execute(select(Attempt).where(Attempt.id == body["attempt_id"]))
        ).scalar_one()
        assert rc.uworld_test_id == "7392051"
        assert attempt.uworld_test_id == "7392051"


# --------------------------------------------------------------------------- #
# 15. uworld_test_id omitted / null persists as NULL
# --------------------------------------------------------------------------- #


async def test_ingest_null_test_id_persists_as_null(ingest_env):
    client, make_session, _ = ingest_env

    # Case A: field omitted entirely (relies on Pydantic default of None).
    qid_a = f"q-{uuid.uuid4().hex[:10]}"
    payload_a = _base_payload(qid=qid_a)
    r_a = await client.post("/api/v1/captures", json=payload_a, headers=_auth_headers())
    assert r_a.status_code == 200, r_a.text

    # Case B: field present but explicitly null.
    qid_b = f"q-{uuid.uuid4().hex[:10]}"
    payload_b = _base_payload(qid=qid_b)
    payload_b["uworld_test_id"] = None
    r_b = await client.post("/api/v1/captures", json=payload_b, headers=_auth_headers())
    assert r_b.status_code == 200, r_b.text

    async with make_session() as s:
        rc_a = (
            await s.execute(select(RawCapture).where(RawCapture.id == r_a.json()["capture_id"]))
        ).scalar_one()
        rc_b = (
            await s.execute(select(RawCapture).where(RawCapture.id == r_b.json()["capture_id"]))
        ).scalar_one()
        att_a = (
            await s.execute(select(Attempt).where(Attempt.id == r_a.json()["attempt_id"]))
        ).scalar_one()
        att_b = (
            await s.execute(select(Attempt).where(Attempt.id == r_b.json()["attempt_id"]))
        ).scalar_one()
        assert rc_a.uworld_test_id is None
        assert rc_b.uworld_test_id is None
        assert att_a.uworld_test_id is None
        assert att_b.uworld_test_id is None


# --------------------------------------------------------------------------- #
# 16. ix_attempts_uworld_test_id index exists after migration
# --------------------------------------------------------------------------- #


async def test_attempts_uworld_test_id_index_exists(ingest_env):
    _, make_session, _ = ingest_env
    from sqlalchemy import text as _sql

    async with make_session() as s:
        rows = (
            await s.execute(
                _sql(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE tablename = 'attempts' AND indexname = 'ix_attempts_uworld_test_id'"
                )
            )
        ).all()
        assert len(rows) == 1, f"expected ix_attempts_uworld_test_id, got {rows!r}"


# --------------------------------------------------------------------------- #
# 17. Course-scoped capture (V-CAP2, T56)
# --------------------------------------------------------------------------- #


async def _make_course(make_session, slug: str) -> int:
    async with make_session() as s:
        c = Course(slug=slug, name=slug.upper())
        s.add(c)
        await s.flush()
        cid = c.id
        await s.commit()
        return cid


async def test_capture_course_slug_stamps_course_id(ingest_env):
    """V-CAP2: course_slug resolves to course_id, stamped on Question + RawCapture."""
    client, make_session, _ = ingest_env
    slug = f"course-{uuid.uuid4().hex[:8]}"
    course_id = await _make_course(make_session, slug)

    payload = _base_payload()
    payload["course_slug"] = slug
    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text

    async with make_session() as s:
        question = (await s.execute(select(Question))).scalar_one()
        capture = (await s.execute(select(RawCapture))).scalar_one()
        assert question.course_id == course_id
        assert capture.course_id == course_id


async def test_capture_unknown_course_slug_422_persists_nothing(ingest_env):
    """V-CAP2 / I.api: an unknown course_slug aborts ingest (422), nothing persisted."""
    client, make_session, _ = ingest_env
    payload = _base_payload()
    payload["course_slug"] = f"nope-{uuid.uuid4().hex[:8]}"

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 422, r.text

    async with make_session() as s:
        assert (await s.execute(select(RawCapture))).scalars().all() == []
        assert (await s.execute(select(Question))).scalars().all() == []
        assert (await s.execute(select(Attempt))).scalars().all() == []


async def test_capture_without_course_slug_backcompat(ingest_env):
    """V-CAP2: omitting course_slug stays valid (200) with course_id NULL."""
    client, make_session, _ = ingest_env
    payload = _base_payload()  # no course_slug key

    r = await client.post("/api/v1/captures", json=payload, headers=_auth_headers())
    assert r.status_code == 200, r.text

    async with make_session() as s:
        question = (await s.execute(select(Question))).scalar_one()
        capture = (await s.execute(select(RawCapture))).scalar_one()
        assert question.course_id is None
        assert capture.course_id is None
