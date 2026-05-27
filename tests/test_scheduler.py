"""Tests for scheduler TaskRun model and admin job endpoints (Ticket 6.9b)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.main import app
from app.models.task_run import TaskRun, TaskRunStatus

COACH_TOKEN = "change_me_before_use"
AUTH = {"X-Coach-Token": COACH_TOKEN}


# --------------------------------------------------------------------------- #
# ORM round-trip tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_task_run_insert_and_query(test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    Sm = async_sessionmaker(test_engine, expire_on_commit=False)
    async with Sm() as session:
        now = datetime.now(timezone.utc)
        row = TaskRun(
            job_name="run_anki_sync",
            started_at=now,
            status=TaskRunStatus.running,
            items_processed=0,
        )
        session.add(row)
        await session.flush()

        result = await session.execute(select(TaskRun).where(TaskRun.job_name == "run_anki_sync"))
        fetched = result.scalar_one()
        assert fetched.job_name == "run_anki_sync"
        assert fetched.status == TaskRunStatus.running
        assert fetched.items_processed == 0
        assert fetched.cost_usd is None
        assert fetched.error_text is None


@pytest.mark.asyncio
async def test_task_run_status_transition(test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    Sm = async_sessionmaker(test_engine, expire_on_commit=False)
    async with Sm() as session:
        now = datetime.now(timezone.utc)
        row = TaskRun(
            job_name="run_anki_sync",
            started_at=now,
            status=TaskRunStatus.running,
            items_processed=0,
        )
        session.add(row)
        await session.flush()

        row.status = TaskRunStatus.succeeded
        row.finished_at = datetime.now(timezone.utc)
        row.items_processed = 42
        await session.flush()

        result = await session.execute(select(TaskRun).where(TaskRun.id == row.id))
        fetched = result.scalar_one()
        assert fetched.status == TaskRunStatus.succeeded
        assert fetched.finished_at is not None
        assert fetched.items_processed == 42


# --------------------------------------------------------------------------- #
# API endpoint tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_trigger_endpoint_404_unknown_job():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/admin/jobs/not_a_job/trigger", headers=AUTH)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_trigger_endpoint_401_no_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/admin/jobs/run_anki_sync/trigger")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_trigger_endpoint_409_when_inflight():
    import app.scheduler as sched_mod

    fake_job = MagicMock()
    fake_job.id = "run_anki_sync"

    with (
        patch.object(sched_mod, "_inflight", {"run_anki_sync"}),
        patch.object(sched_mod.scheduler, "get_job", return_value=fake_job),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/admin/jobs/run_anki_sync/trigger", headers=AUTH)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_list_jobs_endpoint():
    import app.scheduler as sched_mod

    fake_job_1 = MagicMock()
    fake_job_1.id = "run_anki_sync"
    fake_job_1.next_run_time = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)

    fake_job_2 = MagicMock()
    fake_job_2.id = "run_anki_review"
    fake_job_2.next_run_time = datetime(2026, 5, 16, 13, 0, 0, tzinfo=timezone.utc)

    with patch.object(sched_mod.scheduler, "get_jobs", return_value=[fake_job_1, fake_job_2]):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/v1/admin/jobs", headers=AUTH)

    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    job_ids = {entry["job_id"] for entry in data}
    assert job_ids == {"run_anki_sync", "run_anki_review"}
