"""Question review detail on tutor reads (§T42, desktop ¶T5, V-M1, I.api).

`GET /tutor/questions/by-qid/{qid}` (and the by-attempt-id sibling) now carry
the review-detail block: the per-choice answer distribution over all attempts
of the qid, the user's most-recent `picked` choice, and the newest-first
attempt history. Data-only — no verdict / heuristic (V-M1).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.captures import Attempt, Question
from app.services.tutor import questions as questions_svc

_AUTH = {"X-Coach-Token": settings.COACH_TOKEN}


async def _add_question(db: AsyncSession, qid: str) -> Question:
    q = Question(
        source="uworld",
        qid=qid,
        stem_html=f"<p>{qid}</p>",
        stem_plain=f"stem {qid}",
        choices=[{"label": c, "text": c} for c in ("A", "B", "C", "D")],
        correct_choice="A",
    )
    db.add(q)
    await db.flush()
    return q


def _attempt(question_id: int, *, choice: str, correct: bool, minute: int, secs: int | None) -> Attempt:
    return Attempt(
        question_id=question_id,
        source="uworld",
        attempted_at=datetime(2026, 5, 27, 12, minute, tzinfo=timezone.utc),
        selected_choice=choice,
        is_correct=correct,
        time_seconds=secs,
    )


# ---------- service (V-M1 data-only) ----------


@pytest.mark.asyncio
async def test_answer_distribution_counts_per_choice(db_session: AsyncSession) -> None:
    q = await _add_question(db_session, "Q1")
    db_session.add(_attempt(q.id, choice="A", correct=True, minute=1, secs=40))
    db_session.add(_attempt(q.id, choice="B", correct=False, minute=2, secs=55))
    db_session.add(_attempt(q.id, choice="B", correct=False, minute=3, secs=30))
    await db_session.commit()

    payload = await questions_svc.get_question(db_session, qid="Q1")
    assert payload["answer_distribution"] == {"A": 1, "B": 2}


@pytest.mark.asyncio
async def test_picked_is_most_recent_choice(db_session: AsyncSession) -> None:
    q = await _add_question(db_session, "Q1")
    db_session.add(_attempt(q.id, choice="A", correct=True, minute=1, secs=40))
    db_session.add(_attempt(q.id, choice="C", correct=False, minute=9, secs=20))
    await db_session.commit()

    payload = await questions_svc.get_question(db_session, qid="Q1")
    assert payload["picked"] == "C"


@pytest.mark.asyncio
async def test_attempt_history_newest_first_with_fields(db_session: AsyncSession) -> None:
    q = await _add_question(db_session, "Q1")
    db_session.add(_attempt(q.id, choice="A", correct=True, minute=1, secs=40))
    db_session.add(_attempt(q.id, choice="B", correct=False, minute=5, secs=None))
    await db_session.commit()

    payload = await questions_svc.get_question(db_session, qid="Q1")
    hist = payload["attempt_history"]
    assert [h["selected_choice"] for h in hist] == ["B", "A"]  # newest first
    assert hist[0]["is_correct"] is False
    assert hist[0]["time_seconds"] is None
    assert hist[1]["time_seconds"] == 40
    assert hist[0]["attempted_at"].startswith("2026-05-27T12:05")


@pytest.mark.asyncio
async def test_no_attempts_yields_empty_detail(db_session: AsyncSession) -> None:
    await _add_question(db_session, "Q1")
    payload = await questions_svc.get_question(db_session, qid="Q1")
    assert payload["picked"] is None
    assert payload["answer_distribution"] == {}
    assert payload["attempt_history"] == []


# ---------- HTTP (I.api, V-D1) ----------


@pytest.mark.asyncio
async def test_route_by_qid_returns_review_detail(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    q = await _add_question(db_session, "Q1")
    db_session.add(_attempt(q.id, choice="A", correct=True, minute=1, secs=40))
    db_session.add(_attempt(q.id, choice="D", correct=False, minute=2, secs=25))
    await db_session.commit()

    r = await client.get("/api/v1/tutor/questions/by-qid/Q1", headers=_AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer_distribution"] == {"A": 1, "D": 1}
    assert body["picked"] == "D"
    assert len(body["attempt_history"]) == 2


@pytest.mark.asyncio
async def test_route_by_qid_requires_coach_token(client: AsyncClient) -> None:
    r = await client.get("/api/v1/tutor/questions/by-qid/Q1")
    assert r.status_code in (401, 403)
