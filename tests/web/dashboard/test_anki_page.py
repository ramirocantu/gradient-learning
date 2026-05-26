"""SPEC §T25 dashboard /anki page tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.task_run import TaskRun, TaskRunStatus


@pytest.fixture(autouse=True)
async def clean_task_runs(session):
    """test_anki_api.py's scheduler tests use AsyncSessionLocal directly +
    commit() to drive the prod scheduler hook, which bypasses the per-test
    savepoint and leaks `run_anki_sync` task_runs rows into later tests.
    Mirror the test_admin.py pattern: wipe task_runs at the start of each
    test in this module so empty-state assertions hold."""
    await session.execute(TaskRun.__table__.delete())
    await session.commit()


async def _seed_card(
    session,
    *,
    anki_card_id: int,
    due_date: date | None,
    tags: list[tuple[str, str, int | None, str | None]] = (),
) -> AnkiCard:
    # §V75: a card's tags live on its note. Seed note (note_id == anki_card_id)
    # before the FK, link the card, attach tags to the note.
    session.add(AnkiNote(note_id=anki_card_id, deck_name="MileDown"))
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="MileDown",
        note_id=anki_card_id,
        due_date=due_date,
        interval_days=14,
        queue=2,
    )
    session.add(card)
    await session.flush()
    for tag_raw, parsed_kind, topic_id, qid in tags:
        session.add(
            AnkiNoteTag(
                note_id=anki_card_id,
                tag_raw=tag_raw,
                parsed_kind=parsed_kind,
                topic_id=topic_id,
                question_qid=qid,
            )
        )
    await session.commit()
    return card


async def _seed_sync_run(
    session, *, status: TaskRunStatus, error_text: str | None = None
) -> TaskRun:
    row = TaskRun(
        job_name="run_anki_sync",
        started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        finished_at=datetime.now(timezone.utc),
        status=status,
        items_processed=4,
        error_text=error_text,
    )
    session.add(row)
    await session.commit()
    return row


@pytest.mark.asyncio
async def test_anki_page_empty_state(client) -> None:
    """No sync runs + no cards → page renders w/ empty-state copy."""
    r = await client.get("/anki")
    assert r.status_code == 200
    assert "Anki" in r.text
    assert "No sync runs recorded yet" in r.text
    assert "No cards have a scheduled" in r.text


@pytest.mark.asyncio
async def test_anki_page_renders_sync_runs_and_cards(client, session) -> None:
    await _seed_sync_run(session, status=TaskRunStatus.succeeded)
    await _seed_sync_run(session, status=TaskRunStatus.succeeded, error_text="anki_not_running")
    today = date.today()
    await _seed_card(
        session,
        anki_card_id=20001,
        due_date=today - timedelta(days=1),
        tags=[("#AK_MCAT_v2::#UWorld::402391", "uworld_qid", None, "402391")],
    )
    await _seed_card(
        session,
        anki_card_id=20002,
        due_date=today + timedelta(days=3),
        tags=[
            (
                "#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::4E-Atoms_Nuclear_Decay_Electronic_Structure_and_Behavior",
                "aamc_topic",
                1,
                None,
            )
        ],
    )

    r = await client.get("/anki")
    assert r.status_code == 200
    # Sync history section
    assert "Recent syncs" in r.text
    assert "succeeded" in r.text
    assert "AnkiConnect unreachable" in r.text  # friendlier copy for anki_not_running
    # Review queue section
    assert "Review queue" in r.text
    assert "20001" in r.text
    assert "20002" in r.text
    # Tag chips
    assert "qid 402391" in r.text
    assert "#AK_MCAT_v2::#AAMC::Concepts::C/P" in r.text


@pytest.mark.asyncio
async def test_anki_nav_link_present_on_other_pages(client) -> None:
    """Nav exposed on every dashboard page → can navigate to /anki from anywhere."""
    r = await client.get("/")
    assert r.status_code == 200
    assert 'href="/anki"' in r.text


@pytest.mark.asyncio
async def test_anki_page_uses_service_helper_in_process(client, monkeypatch) -> None:
    """V18: route calls `app.services.anki.queries.list_review_queue` directly."""
    from app.web.dashboard.routes import anki as anki_module

    called: list[bool] = []

    async def _fake_list_review_queue(_session, *, limit):
        called.append(True)
        assert limit == 20
        return []

    monkeypatch.setattr(anki_module, "list_review_queue", _fake_list_review_queue)

    r = await client.get("/anki")
    assert r.status_code == 200
    assert called == [True]


@pytest.mark.asyncio
async def test_anki_page_route_module_does_not_use_http_self_call() -> None:
    """V18 regression guard: dashboard /anki route must not import httpx."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[3] / "app" / "web" / "dashboard" / "routes" / "anki.py"
    ).read_text()
    assert "httpx" not in src


@pytest.mark.asyncio
async def test_anki_page_renders_deck_empty_or_misspelled_copy(client, session) -> None:
    """§V20: friendlier copy when a sync row carries the deck-empty envelope."""
    await _seed_sync_run(
        session,
        status=TaskRunStatus.succeeded,
        error_text="deck_empty_or_misspelled",
    )
    r = await client.get("/anki")
    assert r.status_code == 200
    assert "ANKI_DECK_NAME" in r.text
    assert "verify" in r.text.lower()


def test_env_example_documents_anki_deck_name() -> None:
    """§V20: `.env.example` must document ANKI_DECK_NAME so the deck is set
    explicitly per machine, not silently defaulted to "MileDown"."""
    from pathlib import Path

    env_example = (Path(__file__).resolve().parents[4] / ".env.example").read_text()
    assert "ANKI_DECK_NAME" in env_example


# ----------------------- T69 dashboard extensions ----------------------- #


from decimal import Decimal  # noqa: E402 — grouped with T69 additions below

from app.models.anki import (  # noqa: E402
    AnkiAssignment,
    AnkiLoadConfig,
    AnkiReview,
)


async def _seed_assignment(
    session,
    *,
    scope_value: str,
    status: str,
    scheduled_unlock_at: datetime,
    actual_unlock_at: datetime | None = None,
    updated_at: datetime | None = None,
    card_ids: list[int] | None = None,
) -> AnkiAssignment:
    a = AnkiAssignment(
        scope_kind="cc",
        scope_value=scope_value,
        scheduled_unlock_at=scheduled_unlock_at,
        actual_unlock_at=actual_unlock_at,
        card_ids=card_ids or [1735689600001, 1735689600002],
        status=status,
    )
    session.add(a)
    await session.commit()
    if updated_at is not None:
        a.updated_at = updated_at
        await session.commit()
    return a


async def _seed_push(
    session,
    *,
    push_date_,
    scope_slug: str,
    status: str,
    pushed_at: datetime | None = None,
) -> AnkiReview:
    p = AnkiReview(
        review_date=push_date_,
        card_ids=[1, 2],
        deck_name=f"mcat-coach::review::{scope_slug}-{push_date_}",
        status=status,
        pushed_at=pushed_at,
    )
    session.add(p)
    await session.commit()
    return p


@pytest.mark.asyncio
async def test_anki_page_renders_adherence_chip(client, session) -> None:
    """V54 deterministic chip: status_label + headrooms + projected vs budget."""
    r = await client.get("/anki")
    assert r.status_code == 200
    # Empty state → feasible (no reviews, no upcoming).
    assert "Plan adherence" in r.text
    assert "feasible" in r.text
    # Headroom labels render.
    assert "Cards / day" in r.text
    assert "Minutes / day" in r.text
    # V60: no "recommended_changes" wording in chip copy.
    assert "recommended_changes" not in r.text


@pytest.mark.asyncio
async def test_anki_page_renders_load_config_form_with_current_values(client, session) -> None:
    session.add(
        AnkiLoadConfig(
            id=1,
            daily_card_review_budget=250,
            daily_minutes_budget=Decimal("75"),
        )
    )
    await session.commit()
    r = await client.get("/anki")
    assert r.status_code == 200
    assert 'data-test="load-config-form"' in r.text
    assert 'name="daily_card_review_budget"' in r.text
    assert 'value="250"' in r.text
    assert 'name="daily_minutes_budget"' in r.text


@pytest.mark.asyncio
async def test_post_anki_load_config_upserts_and_redirects(client, session) -> None:
    r = await client.post(
        "/anki/load-config",
        data={
            "daily_card_review_budget": "320",
            "daily_minutes_budget": "90.5",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/anki?load_config_saved=1"

    # Singleton actually updated.
    row = await session.get(AnkiLoadConfig, 1)
    assert row is not None
    assert row.daily_card_review_budget == 320
    assert row.daily_minutes_budget == Decimal("90.5")


@pytest.mark.asyncio
async def test_post_anki_load_config_rejects_non_positive(client, session) -> None:
    r = await client.post(
        "/anki/load-config",
        data={
            "daily_card_review_budget": "0",
            "daily_minutes_budget": "30",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "load_config_error=non_positive" in r.headers["location"]


@pytest.mark.asyncio
async def test_post_anki_load_config_rejects_invalid_minutes(client, session) -> None:
    r = await client.post(
        "/anki/load-config",
        data={
            "daily_card_review_budget": "200",
            "daily_minutes_budget": "not-a-number",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "load_config_error=invalid_minutes" in r.headers["location"]


@pytest.mark.asyncio
async def test_anki_page_renders_assignments_status_groups(client, session) -> None:
    now = datetime.now(timezone.utc)
    await _seed_assignment(
        session,
        scope_value="cc-pending",
        status="pending",
        scheduled_unlock_at=now + timedelta(days=2),
    )
    await _seed_assignment(
        session,
        scope_value="cc-unlocked",
        status="unlocked",
        scheduled_unlock_at=now - timedelta(days=1),
        actual_unlock_at=now - timedelta(hours=2),
    )
    await _seed_assignment(
        session,
        scope_value="cc-completed",
        status="completed",
        scheduled_unlock_at=now - timedelta(days=3),
        actual_unlock_at=now - timedelta(days=2),
        updated_at=now - timedelta(days=1),
    )

    r = await client.get("/anki")
    assert r.status_code == 200
    assert 'data-test="assignments-pending"' in r.text
    assert 'data-test="assignments-unlocked"' in r.text
    assert 'data-test="assignments-completed"' in r.text
    assert "cc-pending" in r.text
    assert "cc-unlocked" in r.text
    assert "cc-completed" in r.text


@pytest.mark.asyncio
async def test_anki_page_renders_review_pushes_panels(client, session) -> None:
    today = date.today()
    await _seed_push(
        session,
        push_date_=today + timedelta(days=1),
        scope_slug="upcoming",
        status="pending",
    )
    await _seed_push(
        session,
        push_date_=today - timedelta(days=2),
        scope_slug="done",
        status="pushed",
        pushed_at=datetime.now(timezone.utc) - timedelta(days=2, hours=1),
    )

    r = await client.get("/anki")
    assert r.status_code == 200
    assert 'data-test="review-pushes-pending"' in r.text
    assert 'data-test="review-pushes-recent"' in r.text
    # T76: pending section shows deck_name → "upcoming" substring (in
    # seeded deck_name) appears. Recent section dropped scope_slug col;
    # it renders status badge instead → "pushed" appears.
    assert "upcoming" in r.text
    assert "pushed" in r.text


@pytest.mark.asyncio
async def test_anki_page_carries_run_now_buttons_for_three_new_jobs(client, session) -> None:
    r = await client.get("/anki")
    assert r.status_code == 200
    assert 'data-test="run-now-buttons"' in r.text
    assert "/admin/jobs/run_anki_assignment_unlock/trigger" in r.text
    assert "/admin/jobs/run_anki_assignment_complete/trigger" in r.text
    assert "/admin/jobs/run_anki_review/trigger" in r.text


def test_anki_page_route_module_uses_in_process_services_only() -> None:
    """V18: dashboard route module must not import httpx (no
    self-call to the JSON API). Mirror the existing assertion shape."""
    import inspect

    from app.web.dashboard.routes import anki as anki_routes

    src = inspect.getsource(anki_routes)
    assert "httpx" not in src
    assert "import httpx" not in src
