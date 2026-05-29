"""T3 — source discriminator + source-adapter registry (§A plugin seam).

Registry dispatch by `source`, unknown-source rejection, `source` stamped on
question/attempt/raw_capture through the uworld reference adapter, and the
now-open raw_captures enum (any source may write a capture).

Self-contained: creates only the ingest tables on its own engine.
"""

import os
from datetime import datetime, timezone

import asyncpg
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import Base
from app.models.captures import Attempt, Passage, Question, RawCapture
from app.models.outline import Course
from app.schemas.captures import CapturePayload, ChoiceItem, ParsedCapture
from app.services.adapters import UnknownSourceError, get_adapter, registered_sources
from app.services.ingest import ingest_capture

_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_DB_URL = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"

_TABLES = [
    # Course first — Question + RawCapture FK→courses.id (V-CAP2, T56).
    Course.__table__,
    Passage.__table__,
    Question.__table__,
    Attempt.__table__,
    RawCapture.__table__,
]


@pytest.fixture
async def engine():
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute("CREATE DATABASE gradient_test")
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

    eng = create_async_engine(_DB_URL)
    async with eng.begin() as c:
        await c.execute(text("DROP SCHEMA public CASCADE"))
        await c.execute(text("CREATE SCHEMA public"))
        await c.run_sync(Base.metadata.create_all, tables=_TABLES)
    yield eng
    await eng.dispose()


def _payload(**over) -> CapturePayload:
    return CapturePayload(
        source=over.get("source", "uworld"),
        qid=over.get("qid", "q-src-1"),
        captured_at=datetime.now(timezone.utc),
        html="<p>raw</p>",
        parsed=ParsedCapture(
            stem_html="<p>stem</p>",
            stem_plain="stem",
            choices=[
                ChoiceItem(key="A", html="<p>a</p>", plain="a"),
                ChoiceItem(key="B", html="<p>b</p>", plain="b"),
            ],
            correct_choice="A",
            selected_choice="A",
            is_correct=True,
        ),
        extension_version=over.get("extension_version", "0.1.0"),
    )


# ── registry (no DB) ────────────────────────────────────────────────────────


def test_registry_has_uworld_reference_adapter():
    assert "uworld" in registered_sources()
    assert get_adapter("uworld").source == "uworld"


def test_get_adapter_unknown_source_raises():
    with pytest.raises(UnknownSourceError):
        get_adapter("nope")


def test_registry_has_manual_and_web_qbank_adapters():
    # T33 (§A): manual + web-Qbank register without touching the dispatcher.
    sources = registered_sources()
    assert "manual" in sources
    assert "web-qbank" in sources
    assert get_adapter("manual").source == "manual"
    assert get_adapter("web-qbank").source == "web-qbank"


def test_pdf_qset_deferred_not_registered():
    # T33 deferred the PDF question-set parser ("hardest, last").
    assert "pdf-qset" not in registered_sources()


# ── dispatch + source stamping (DB) ─────────────────────────────────────────


async def test_ingest_routes_uworld_and_stamps_source(engine):
    async with AsyncSession(engine) as s:
        resp = await ingest_capture(_payload(), s)
        await s.commit()
    async with AsyncSession(engine) as s:
        q = (
            await s.execute(select(Question).where(Question.id == resp.question_id))
        ).scalar_one()
        a = (
            await s.execute(select(Attempt).where(Attempt.id == resp.attempt_id))
        ).scalar_one()
        rc = (
            await s.execute(select(RawCapture).where(RawCapture.id == resp.capture_id))
        ).scalar_one()
    assert q.source == "uworld"
    assert a.source == "uworld"
    assert rc.source == "uworld"


async def test_ingest_routes_web_qbank_and_stamps_source(engine):
    # §A: a new source dispatches through the same seam, stamping its source.
    async with AsyncSession(engine) as s:
        resp = await ingest_capture(_payload(source="web-qbank", qid="q-wq-1"), s)
        await s.commit()
    async with AsyncSession(engine) as s:
        q = (
            await s.execute(select(Question).where(Question.id == resp.question_id))
        ).scalar_one()
        a = (
            await s.execute(select(Attempt).where(Attempt.id == resp.attempt_id))
        ).scalar_one()
        rc = (
            await s.execute(select(RawCapture).where(RawCapture.id == resp.capture_id))
        ).scalar_one()
    assert q.source == "web-qbank"
    assert a.source == "web-qbank"
    assert rc.source == "web-qbank"


async def test_ingest_routes_manual_and_stamps_source(engine):
    # Manual entry: extension_version='manual', same normalized shape.
    async with AsyncSession(engine) as s:
        resp = await ingest_capture(
            _payload(source="manual", qid="q-man-1", extension_version="manual"), s
        )
        await s.commit()
    async with AsyncSession(engine) as s:
        q = (
            await s.execute(select(Question).where(Question.id == resp.question_id))
        ).scalar_one()
        a = (
            await s.execute(select(Attempt).where(Attempt.id == resp.attempt_id))
        ).scalar_one()
    assert q.source == "manual"
    assert a.source == "manual"


async def test_ingest_unknown_source_raises_before_writes(engine):
    with pytest.raises(UnknownSourceError):
        async with AsyncSession(engine) as s:
            await ingest_capture(_payload(source="mystery"), s)
    # Nothing persisted.
    async with AsyncSession(engine) as s:
        n = (await s.execute(select(RawCapture.id))).scalars().all()
    assert n == []


async def test_raw_captures_source_enum_is_open(engine):
    # No closed CHECK now — a non-uworld source row inserts fine.
    async with AsyncSession(engine) as s:
        s.add(
            RawCapture(
                source="anki",
                qid="x1",
                captured_at=datetime.now(timezone.utc),
                raw_html="<p>x</p>",
                raw_json={},
                extension_version="0.1.0",
            )
        )
        await s.flush()
        await s.commit()
    async with AsyncSession(engine) as s:
        rc = (await s.execute(select(RawCapture).where(RawCapture.qid == "x1"))).scalar_one()
    assert rc.source == "anki"
