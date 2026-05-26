"""Tests for dashboard admin page (Ticket 6.9b, rewired in Ticket R.2a).

R.2a removed the httpx proxy from `app.web.dashboard.routes.admin` and
replaced it with direct in-process calls into
`app.api.v1.admin.list_jobs_payload` / `trigger_job_logic`. Tests target
those helpers and the underlying APScheduler instance, not an HTTP hop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.models.task_run import TaskRun, TaskRunStatus


@pytest.fixture(autouse=True)
async def clean_task_runs(session):
    await session.execute(TaskRun.__table__.delete())
    await session.commit()


async def _seed_run(session, job_name: str, status: TaskRunStatus) -> TaskRun:
    row = TaskRun(
        job_name=job_name,
        started_at=datetime.now(timezone.utc),
        status=status,
        items_processed=3,
        cost_usd=0.0012,
    )
    session.add(row)
    await session.commit()
    return row


# --------------------------------------------------------------------------- #
# R.2a new cases
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_admin_page_renders_with_no_jobs(client, monkeypatch):
    """Empty scheduler → page still renders with the offline copy.

    Asserts the direct call replaced the httpx round-trip without behavior
    change for the empty-scheduler case (this was previously the
    "backend unreachable" path; now it is the no-jobs-registered path).
    """

    async def _empty_payload() -> list[dict]:
        return []

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.list_jobs_payload",
        _empty_payload,
    )

    r = await client.get("/admin")

    assert r.status_code == 200
    # Both jobs render with the "Scheduler offline or disabled" copy
    # because next_run_time is None for both.
    assert "Categorizer" in r.text
    assert "Feature Extractor" in r.text
    assert "Scheduler offline or disabled" in r.text


@pytest.mark.asyncio
async def test_admin_page_renders_with_jobs(client, monkeypatch):
    """Scheduler returns synthetic job payload → next_run_time renders."""

    async def _payload() -> list[dict]:
        return [
            {
                "job_id": "run_categorizer",
                "next_run_time": "2026-05-18T12:34:56+00:00",
            },
            {
                "job_id": "run_feature_extraction",
                "next_run_time": "2026-05-18T13:00:00+00:00",
            },
            {
                "job_id": "run_anki_sync",
                "next_run_time": "2026-05-18T13:30:00+00:00",
            },
            {
                "job_id": "run_anki_topic_resolver",
                "next_run_time": "2026-05-18T13:45:00+00:00",
            },
            {
                "job_id": "run_anki_assignment_unlock",
                "next_run_time": "2026-05-18T14:00:00+00:00",
            },
            {
                "job_id": "run_anki_assignment_complete",
                "next_run_time": "2026-05-19T05:15:00+00:00",
            },
            {
                "job_id": "run_anki_review",
                "next_run_time": "2026-05-18T15:00:00+00:00",
            },
        ]

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.list_jobs_payload",
        _payload,
    )

    r = await client.get("/admin")

    assert r.status_code == 200
    assert "Categorizer" in r.text
    assert "Feature Extractor" in r.text
    assert "Anki Sync" in r.text
    assert "2026-05-18T12:34:56+00:00" in r.text
    assert "Scheduler offline or disabled" not in r.text


@pytest.mark.asyncio
async def test_trigger_job_calls_scheduler_helper(client, monkeypatch):
    """POST /admin/jobs/.../trigger calls trigger_job_logic exactly once."""

    calls: list[str] = []

    async def _fake_trigger(job_name: str) -> dict:
        calls.append(job_name)
        return {"status": "triggered", "job": job_name}

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.trigger_job_logic",
        _fake_trigger,
    )

    r = await client.post("/admin/jobs/run_categorizer/trigger")

    assert r.status_code == 200
    assert "Triggered" in r.text
    assert calls == ["run_categorizer"]


@pytest.mark.asyncio
async def test_no_httpx_in_dashboard_admin():
    """Regression guard: the rewritten module must not reintroduce the proxy."""
    src = (
        Path(__file__).resolve().parents[3] / "app" / "web" / "dashboard" / "routes" / "admin.py"
    ).read_text()
    assert "httpx" not in src
    assert "_BACKEND" not in src


# --------------------------------------------------------------------------- #
# SPEC §T28 — Anki tag-parse health widget
# --------------------------------------------------------------------------- #


async def _seed_anki_tag(session, *, anki_card_id: int, parsed_kind: str, tag_raw: str) -> None:
    # §V75: tags live on the note. Seed note (note_id == anki_card_id) +
    # link the card; attach the tag to the note.
    from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag

    session.add(AnkiNote(note_id=anki_card_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="MileDown",
        note_id=anki_card_id,
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=anki_card_id,
            tag_raw=tag_raw,
            parsed_kind=parsed_kind,
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_admin_page_renders_empty_anki_widget_when_no_tags(client, monkeypatch):
    """T28 / V19: with zero `anki_card_tags` rows the widget renders its
    empty-state copy rather than a divide-by-zero table."""

    async def _empty_payload() -> list[dict]:
        return []

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.list_jobs_payload",
        _empty_payload,
    )

    r = await client.get("/admin")

    assert r.status_code == 200
    assert "Anki tag-parse health" in r.text
    assert "No Anki tags ingested yet" in r.text


@pytest.mark.asyncio
async def test_admin_page_renders_anki_widget_with_parsed_kind_breakdown(
    client, session, monkeypatch
):
    """T28 / V19: parsed_kind counts + percentages surface on the page."""

    await _seed_anki_tag(
        session, anki_card_id=9001, parsed_kind="aamc_topic", tag_raw="topic-tag::A"
    )
    await _seed_anki_tag(
        session, anki_card_id=9002, parsed_kind="uworld_qid", tag_raw="#AK_MCAT_v2::#UWorld::1"
    )
    await _seed_anki_tag(
        session, anki_card_id=9003, parsed_kind="uworld_qid", tag_raw="#AK_MCAT_v2::#UWorld::2"
    )
    await _seed_anki_tag(
        session, anki_card_id=9004, parsed_kind="unparsed", tag_raw="Legacy::Weird"
    )
    await _seed_anki_tag(
        session,
        anki_card_id=9005,
        parsed_kind="aamc_cc",
        tag_raw="#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4E-Atoms",
    )

    async def _empty_payload() -> list[dict]:
        return []

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.list_jobs_payload",
        _empty_payload,
    )

    r = await client.get("/admin")

    assert r.status_code == 200
    assert "Anki tag-parse health" in r.text
    # Total tag rows + per-card coverage denominator both rendered
    assert "5 tag rows" in r.text
    assert "5 cards" in r.text  # 5 distinct anki_cards seeded above
    # Unparsed share = 1/5 = 20.0% per row (also 1/5 cards = 20.0%)
    assert "unparsed share: 20.0%" in r.text
    # Per-kind counts present (2 uworld_qid rows)
    assert ">2<" in r.text
    # Card coverage column rendered ("X / 5")
    assert "/ 5" in r.text
    # Labels rendered — all four parsed_kinds covered
    assert "aamc_topic" in r.text
    assert "aamc_cc" in r.text
    assert "uworld_qid" in r.text
    assert "unparsed" in r.text


@pytest.mark.asyncio
async def test_admin_anki_widget_card_coverage_diverges_from_row_share(
    client, session, monkeypatch
):
    """T33 / §V23: card-level coverage line surfaces alongside per-tag-row table.

    Seed scenario mimics the AnKing case: one card carries 5 unparsed tags and
    1 aamc_cc tag, so per-row aamc_cc share = 1/6 = 16.7% but per-card coverage
    = 1/1 = 100%. The widget must surface both signals.
    """
    from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag

    # §V75: one note carries the 6 tags; one card backs that note. Per-row
    # aamc_cc share = 1/6, per-card coverage = 1/1 — the divergence the widget
    # surfaces. Tags live on the note now, not the card.
    note = AnkiNote(note_id=12345, deck_name="AnKing MCAT Deck")
    session.add(note)
    await session.flush()
    card = AnkiCard(anki_card_id=12345, deck_name="AnKing MCAT Deck", note_id=note.note_id)
    session.add(card)
    await session.flush()
    session.add_all(
        [
            AnkiNoteTag(note_id=note.note_id, tag_raw="aamc::1", parsed_kind="aamc_cc"),
            AnkiNoteTag(note_id=note.note_id, tag_raw="noise::1", parsed_kind="unparsed"),
            AnkiNoteTag(note_id=note.note_id, tag_raw="noise::2", parsed_kind="unparsed"),
            AnkiNoteTag(note_id=note.note_id, tag_raw="noise::3", parsed_kind="unparsed"),
            AnkiNoteTag(note_id=note.note_id, tag_raw="noise::4", parsed_kind="unparsed"),
            AnkiNoteTag(note_id=note.note_id, tag_raw="noise::5", parsed_kind="unparsed"),
        ]
    )
    await session.commit()

    async def _empty_payload() -> list[dict]:
        return []

    monkeypatch.setattr("app.web.dashboard.routes.admin.list_jobs_payload", _empty_payload)

    r = await client.get("/admin")
    assert r.status_code == 200
    # 6 tag rows, 1 distinct card
    assert "6 tag rows over 1 cards" in r.text
    # Per-row aamc_cc share: 1/6 = 16.7%
    assert "16.7%" in r.text
    # Per-card aamc_cc coverage: 1/1 = 100.0% — divergence proves both metrics surfaced
    assert "100.0%" in r.text
    # "X / 1" pattern present in the cards-covered column
    assert "1 / 1" in r.text


@pytest.mark.asyncio
async def test_admin_anki_widget_uses_service_helper_in_process(monkeypatch):
    """V18: dashboard calls `app.services.anki.queries.get_tag_parse_stats`
    directly (in-process), not over httpx — the regression guard above covers
    the broader file, but this asserts the specific helper is wired in."""
    from app.web.dashboard.routes import admin as admin_module

    called: list[bool] = []

    async def _fake(_session):
        called.append(True)
        return {"aamc_topic": 1}

    monkeypatch.setattr(admin_module, "get_tag_parse_stats", _fake)

    async def _empty_payload() -> list[dict]:
        return []

    monkeypatch.setattr(admin_module, "list_jobs_payload", _empty_payload)

    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/admin")

    assert r.status_code == 200
    assert called == [True]


# --------------------------------------------------------------------------- #
# Pre-existing coverage, ported off the httpx mock
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_admin_page_renders_with_runs(client, session, monkeypatch):
    await _seed_run(session, "run_categorizer", TaskRunStatus.succeeded)
    await _seed_run(session, "run_feature_extraction", TaskRunStatus.running)

    async def _payload() -> list[dict]:
        return [
            {"job_id": "run_categorizer", "next_run_time": "2026-05-16T12:00:00+00:00"},
            {
                "job_id": "run_feature_extraction",
                "next_run_time": "2026-05-16T13:00:00+00:00",
            },
        ]

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.list_jobs_payload",
        _payload,
    )

    r = await client.get("/admin")

    assert r.status_code == 200
    assert "Categorizer" in r.text
    assert "Feature Extractor" in r.text


@pytest.mark.asyncio
async def test_trigger_409_returns_already_running_text(client, monkeypatch):
    async def _fake_trigger(job_name: str) -> dict:
        raise HTTPException(409, detail=f"{job_name} already running")

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.trigger_job_logic",
        _fake_trigger,
    )

    r = await client.post("/admin/jobs/run_categorizer/trigger")

    assert r.status_code == 200
    assert "Already running" in r.text


@pytest.mark.asyncio
async def test_trigger_503_returns_backend_unreachable_text(client, monkeypatch):
    async def _fake_trigger(job_name: str) -> dict:
        raise HTTPException(503, detail="scheduler not running")

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.trigger_job_logic",
        _fake_trigger,
    )

    r = await client.post("/admin/jobs/run_categorizer/trigger")

    assert r.status_code == 200
    assert "Backend unreachable" in r.text


@pytest.mark.asyncio
async def test_trigger_unknown_job_404(client):
    r = await client.post("/admin/jobs/bogus/trigger")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# SPEC §T29 — run_anki_sync on /admin
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_admin_page_renders_anki_sync_section(client, monkeypatch):
    """T29: /admin lists 'Anki Sync' alongside Categorizer + Feature Extractor."""

    async def _empty_payload() -> list[dict]:
        return []

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.list_jobs_payload",
        _empty_payload,
    )

    r = await client.get("/admin")
    assert r.status_code == 200
    assert "Anki Sync" in r.text
    # Run-now button targets the new job endpoint
    assert 'hx-post="/admin/jobs/run_anki_sync/trigger"' in r.text


@pytest.mark.asyncio
async def test_admin_trigger_run_anki_sync_invokes_logic_once(client, monkeypatch):
    """T29: POST /admin/jobs/run_anki_sync/trigger flows through trigger_job_logic."""
    calls: list[str] = []

    async def _fake_trigger(job_name: str) -> dict:
        calls.append(job_name)
        return {"status": "triggered", "job": job_name}

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.trigger_job_logic",
        _fake_trigger,
    )

    r = await client.post("/admin/jobs/run_anki_sync/trigger")
    assert r.status_code == 200
    assert "Triggered" in r.text
    assert calls == ["run_anki_sync"]


@pytest.mark.asyncio
async def test_admin_trigger_run_anki_sync_404_when_dropped_from_valid_jobs(client, monkeypatch):
    """T29 regression guard: if `_VALID_JOBS` in the JSON-side helper ever drops
    run_anki_sync, the dashboard trigger should fall through cleanly to 404."""

    async def _fake_trigger(job_name: str) -> dict:
        raise HTTPException(404, detail=f"unknown job: {job_name}")

    monkeypatch.setattr(
        "app.web.dashboard.routes.admin.trigger_job_logic",
        _fake_trigger,
    )

    r = await client.post("/admin/jobs/run_anki_sync/trigger")
    # Dashboard route turns 404 into Backend error (404) per existing pattern.
    assert r.status_code == 200
    assert "Backend error (404)" in r.text or "Triggered" not in r.text


@pytest.mark.asyncio
async def test_admin_valid_jobs_includes_run_anki_sync():
    """T29: the JSON-side allow-list must include run_anki_sync so the dashboard
    trigger doesn't 404 out at the validator boundary."""
    from app.api.v1.admin import _VALID_JOBS

    assert "run_anki_sync" in _VALID_JOBS
