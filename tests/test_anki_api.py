"""HTTP + scheduler integration tests for SPEC §T4.

The route is auth-gated. AnkiConnect itself is stubbed via the
dependency-override hook (`_anki_client`) so tests inject a
MockTransport-backed client and never touch a real port.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.v1.anki import _anki_client
from app.main import app
from app.services.anki.client import AnkiConnectClient


_AUTH = {"X-Coach-Token": "change_me_before_use"}


def _ok(result: Any) -> bytes:
    return json.dumps({"result": result, "error": None}).encode()


def _client_with(handler):
    return AnkiConnectClient("http://localhost:8765", transport=httpx.MockTransport(handler))


def _override(handler):
    """Override the route's anki-client dep with one bound to a handler."""
    overrides = []

    def _factory() -> AnkiConnectClient:
        client = _client_with(handler)
        overrides.append(client)
        return client

    return _factory, overrides


@pytest.mark.asyncio
async def test_post_sync_requires_auth(client: AsyncClient) -> None:
    r = await client.post("/api/v1/anki/sync")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_post_sync_happy_path(client: AsyncClient) -> None:
    """§V4 happy path: route returns the SyncSummary fields verbatim."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        action = body.get("action")
        if action == "findCards":
            return httpx.Response(200, content=_ok([5001]))
        if action == "cardsInfo":
            return httpx.Response(
                200,
                content=_ok(
                    [
                        {
                            "cardId": 5001,
                            "note": 9001,
                            "modelName": "MileDown",
                            "fields": {"Front": {"value": "f", "order": 0}},
                            "queue": 2,
                            "interval": 7,
                            "factor": 2500,
                            "lapses": 0,
                            "due": 1,
                        }
                    ]
                ),
            )
        if action == "notesInfo":
            return httpx.Response(
                200,
                content=_ok([{"noteId": 9001, "tags": ["#AK_MCAT_v2::#UWorld::55555"]}]),
            )
        if action == "cardReviews":
            return httpx.Response(200, content=_ok([]))
        return httpx.Response(200, content=_ok(None))

    factory, _ = _override(handler)
    app.dependency_overrides[_anki_client] = factory
    try:
        r = await client.post("/api/v1/anki/sync", headers=_AUTH)
    finally:
        app.dependency_overrides.pop(_anki_client, None)

    assert r.status_code == 200
    body = r.json()
    assert body == {
        "synced_cards": 1,
        "linked_qids": 1,
        "reviews_synced": 0,
        "error": None,
    }


@pytest.mark.asyncio
async def test_post_sync_returns_error_envelope_when_anki_down(
    client: AsyncClient,
) -> None:
    """§V4: AnkiConnect unreachable must NOT raise to a 500; the route returns
    200 with the error envelope so the tutor sees the structured failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    factory, _ = _override(handler)
    app.dependency_overrides[_anki_client] = factory
    try:
        r = await client.post("/api/v1/anki/sync", headers=_AUTH)
    finally:
        app.dependency_overrides.pop(_anki_client, None)

    assert r.status_code == 200
    assert r.json() == {
        "synced_cards": 0,
        "linked_qids": 0,
        "reviews_synced": 0,
        "error": "anki_not_running",
    }


# --------------------------------------------------------------------------- #
# Scheduler job
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_anki_sync_job_records_deck_empty_envelope(test_engine) -> None:
    """§V20: empty findCards → status=succeeded, items_processed=0,
    error_text='deck_empty_or_misspelled' in the task_runs row so the
    /admin and /anki pages can surface the loud-fail copy."""
    from sqlalchemy import select

    from app.models.task_run import TaskRun, TaskRunStatus
    from app.scheduler import _do_run_anki_sync

    Sm = async_sessionmaker(test_engine, expire_on_commit=False)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        action = body.get("action")
        if action == "findCards":
            return httpx.Response(200, content=_ok([]))
        if action == "deckNames":
            return httpx.Response(200, content=_ok(["AnKing MCAT Deck"]))
        return httpx.Response(200, content=_ok(None))

    fake_client = _client_with(handler)

    with (
        patch("app.scheduler.AsyncSessionLocal", Sm),
        patch("app.scheduler.AnkiConnectClient", return_value=fake_client),
    ):
        await _do_run_anki_sync()

    async with Sm() as session:
        row = (
            (
                await session.execute(
                    select(TaskRun)
                    .where(TaskRun.job_name == "run_anki_sync")
                    .order_by(TaskRun.started_at.desc())
                )
            )
            .scalars()
            .first()
        )
        assert row is not None
        assert row.status == TaskRunStatus.succeeded
        assert row.items_processed == 0
        assert row.error_text == "deck_empty_or_misspelled"


@pytest.mark.asyncio
async def test_run_anki_sync_job_records_anki_not_running(test_engine) -> None:
    """§V4: TaskRun row records `error_text='anki_not_running'` when
    AnkiConnect is down, with status=succeeded (no exception was raised)."""
    from sqlalchemy import select

    from app.models.task_run import TaskRun, TaskRunStatus
    from app.scheduler import _do_run_anki_sync

    Sm = async_sessionmaker(test_engine, expire_on_commit=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    fake_client = _client_with(handler)

    with (
        patch("app.scheduler.AsyncSessionLocal", Sm),
        patch("app.scheduler.AnkiConnectClient", return_value=fake_client),
    ):
        await _do_run_anki_sync()

    async with Sm() as session:
        row = (
            (
                await session.execute(
                    select(TaskRun)
                    .where(TaskRun.job_name == "run_anki_sync")
                    .order_by(TaskRun.started_at.desc())
                )
            )
            .scalars()
            .first()
        )
        assert row is not None
        assert row.status == TaskRunStatus.succeeded
        assert row.error_text == "anki_not_running"


@pytest.mark.asyncio
async def test_run_anki_sync_job_inflight_guard() -> None:
    """Re-entry while a sync is in-flight is skipped (mirrors categorizer)."""
    import app.scheduler as sched_mod

    inner = AsyncMock()
    with (
        patch.object(sched_mod, "_inflight", {"run_anki_sync"}),
        patch("app.scheduler._do_run_anki_sync", inner),
    ):
        await sched_mod.run_anki_sync_job()
    inner.assert_not_called()


@pytest.mark.asyncio
async def test_start_scheduler_registers_anki_sync_job() -> None:
    """§I.anki: `run_anki_sync_job` is registered on scheduler startup."""
    import app.scheduler as sched_mod
    from unittest.mock import MagicMock

    fake_scheduler = MagicMock()
    with (
        patch.object(sched_mod, "scheduler", fake_scheduler),
        patch.object(sched_mod.settings, "SCHEDULER_ENABLED", True),
    ):
        sched_mod.start_scheduler()

    registered_ids = {call.kwargs.get("id") for call in fake_scheduler.add_job.call_args_list}
    assert "run_anki_sync" in registered_ids
