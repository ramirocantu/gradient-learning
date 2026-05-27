"""T39: GET /api/v1/admin/status — system health probe.

V16: OpenAI + Notion + Anki are mocked at the SDK boundary via dependency
overrides; no real network calls. The scheduler is not started under the
test ASGITransport, so ``jobs`` is empty in the route tests — the last-run
rollup is exercised directly against the service in
``test_job_last_runs_*``.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1 import admin
from app.config import settings
from app.main import app
from app.models.task_run import TaskRun, TaskRunStatus
from app.services.anki.client import AnkiUnreachableError
from app.services.system_status import build_jobs_health, job_last_runs


@contextmanager
def _override(deps):
    """Temporarily install dependency overrides (a {dep_fn: value} mapping)
    on the shared app, popping them (and only them) on exit so tests don't
    leak overrides into each other."""
    for dep, value in deps.items():
        app.dependency_overrides[dep] = lambda v=value: v
    try:
        yield
    finally:
        for dep in deps:
            app.dependency_overrides.pop(dep, None)


def _token_headers() -> dict[str, str]:
    return {"X-Coach-Token": settings.COACH_TOKEN}


def _anki_ok() -> MagicMock:
    m = MagicMock()
    m.version = AsyncMock(return_value=6)
    return m


def _openai_ok() -> MagicMock:
    m = MagicMock()
    m.models.retrieve = AsyncMock(return_value=SimpleNamespace(id="gpt-4.1-mini"))
    return m


def _notion_ok() -> MagicMock:
    m = MagicMock()
    m.users.me = AsyncMock(return_value={"object": "user", "type": "bot"})
    return m


# --------------------------------------------------------------------------- #
# Route tests
# --------------------------------------------------------------------------- #


async def test_status_all_reachable(client, monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "NOTION_API_TOKEN", "ntn-test")

    with _override(
        {
            admin._anki_status_client: _anki_ok(),
            admin._openai_status_client: _openai_ok(),
            admin._notion_status_client: _notion_ok(),
        }
    ):
        resp = await client.get("/api/v1/admin/status", headers=_token_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["anki"] == {"configured": True, "reachable": True, "detail": "AnkiConnect v6"}
    assert body["openai"] == {"configured": True, "reachable": True, "detail": None}
    assert body["notion"] == {"configured": True, "reachable": True, "detail": None}
    # Scheduler not started under ASGITransport → no jobs to fold last-runs into.
    assert body["jobs"] == []


async def test_status_anki_unreachable_surfaces_detail(client, monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "NOTION_API_TOKEN", "ntn-test")

    anki = MagicMock()
    anki.version = AsyncMock(side_effect=AnkiUnreachableError("connection refused"))

    with _override(
        {
            admin._anki_status_client: anki,
            admin._openai_status_client: _openai_ok(),
            admin._notion_status_client: _notion_ok(),
        }
    ):
        resp = await client.get("/api/v1/admin/status", headers=_token_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["anki"]["reachable"] is False
    assert "connection refused" in body["anki"]["detail"]
    # An unreachable Anki must not poison the other probes.
    assert body["openai"]["reachable"] is True


async def test_status_notion_unconfigured_is_not_probed(client, monkeypatch):
    # Token unset → real _notion_status_client yields None → reported
    # unconfigured with no SDK call (and no notion-client import).
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "NOTION_API_TOKEN", "")

    with _override(
        {
            admin._anki_status_client: _anki_ok(),
            admin._openai_status_client: _openai_ok(),
        }
    ):
        resp = await client.get("/api/v1/admin/status", headers=_token_headers())

    assert resp.status_code == 200
    assert resp.json()["notion"] == {
        "configured": False,
        "reachable": False,
        "detail": "NOTION_API_TOKEN unset",
    }


async def test_status_openai_unconfigured(client, monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "NOTION_API_TOKEN", "")

    # OpenAI dep still constructs a client, but `configured=False` short-
    # circuits the probe before any call. Override anyway to assert no call.
    openai = MagicMock()
    openai.models.retrieve = AsyncMock()

    with _override(
        {
            admin._anki_status_client: _anki_ok(),
            admin._openai_status_client: openai,
        }
    ):
        resp = await client.get("/api/v1/admin/status", headers=_token_headers())

    assert resp.status_code == 200
    assert resp.json()["openai"] == {
        "configured": False,
        "reachable": False,
        "detail": "OPENAI_API_KEY unset",
    }
    openai.models.retrieve.assert_not_awaited()


async def test_status_requires_coach_token(client):
    resp = await client.get("/api/v1/admin/status")
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# Service-level: last-run rollup
# --------------------------------------------------------------------------- #


async def test_job_last_runs_picks_latest_per_job(db_session):
    now = datetime.now(timezone.utc)
    db_session.add_all(
        [
            TaskRun(
                job_name="run_anki_sync",
                started_at=now - timedelta(minutes=10),
                finished_at=now - timedelta(minutes=9),
                status=TaskRunStatus.succeeded,
                items_processed=4,
            ),
            TaskRun(
                job_name="run_anki_sync",
                started_at=now - timedelta(minutes=1),
                finished_at=now,
                status=TaskRunStatus.failed,
                items_processed=0,
                error_text="boom",
            ),
        ]
    )
    await db_session.commit()

    last = await job_last_runs(db_session)
    assert last["run_anki_sync"].status is TaskRunStatus.failed

    merged = build_jobs_health(
        [
            {"job_id": "run_anki_sync", "next_run_time": "2026-01-01T00:00:00+00:00"},
            {"job_id": "run_categorizer", "next_run_time": None},
        ],
        last,
    )
    assert merged[0]["next_run_time"] == "2026-01-01T00:00:00+00:00"
    assert merged[0]["last_run"]["status"] == "failed"
    assert merged[0]["last_run"]["error_text"] == "boom"
    # A scheduled job with no recorded run → last_run is null, not omitted.
    assert merged[1]["last_run"] is None
