"""Tests for the batch feature-extraction worker (Ticket 4.3).

Uses a real Postgres test DB (via test_engine fixture) with committed data
so that worker sessions can see seeded questions. Each test cleans up after
itself via the `worker_db` fixture.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.captures import Attempt, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory
from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION
from app.services.analyzer.worker import run_extraction


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def worker_db(
    seeded_report, test_engine
) -> AsyncIterator[tuple[AsyncSession, async_sessionmaker]]:
    """Yields (setup_session, session_factory).

    setup_session: use to seed data and commit.
    session_factory: pass to run_extraction.
    Cleans up questions/features/attempts/tags after each test.
    """
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as setup_session:
        yield setup_session, factory
    # Cleanup in a fresh session after the test.
    async with factory() as cleanup:
        await cleanup.execute(text("DELETE FROM question_features"))
        await cleanup.execute(text("DELETE FROM attempts"))
        await cleanup.execute(text("DELETE FROM question_tags"))
        await cleanup.execute(text("DELETE FROM questions"))
        await cleanup.commit()


def _rand_qid() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


async def _make_question(
    session: AsyncSession,
    *,
    stem: str = "stem text",
    stem_html: str | None = None,
    first_seen_at: datetime | None = None,
) -> Question:
    q = Question(
        qid=_rand_qid(),
        stem_html=stem_html or f"<p>{stem}</p>",
        stem_plain=stem,
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_ids": []},
        ],
        correct_choice="A",
        explanation_html="<p>because</p>",
        explanation_plain="because",
    )
    if first_seen_at is not None:
        q.first_seen_at = first_seen_at
    session.add(q)
    await session.flush()
    return q


async def _make_features_row(
    session: AsyncSession,
    question_id: int,
    *,
    extractor_version: str = EXTRACTOR_VERSION,
) -> QuestionFeatures:
    row = QuestionFeatures(
        question_id=question_id,
        question_format="discrete",
        reasoning_type="application",
        requires_calculation=False,
        calculation_steps=0,
        involves_graph_or_figure=False,
        involves_data_table=False,
        has_negative_phrasing=False,
        passage_length_bucket=None,
        passage_type=None,
        distractor_difficulty="medium",
        trap_distractor_present=False,
        common_misconception=None,
        jargon_density="medium",
        key_concept_summary="Tests something.",
        extractor_version=extractor_version,
    )
    session.add(row)
    await session.flush()
    return row


def _tool_use_block(**input_data):
    from anthropic.types import ToolUseBlock

    return ToolUseBlock(
        id="toolu_x",
        name="submit_question_features",
        input=input_data,
        type="tool_use",
    )


def _forge_message(
    *,
    reasoning_type: str = "application",
    distractor_difficulty: str = "medium",
    involves_graph: bool = False,
    involves_table: bool = False,
    input_tokens: int = 500,
    output_tokens: int = 100,
    cache_read: int = 0,
    cache_create: int = 0,
):
    tool_input = {
        "reasoning_type": reasoning_type,
        "requires_calculation": False,
        "calculation_steps": 0,
        "passage_type": "",
        "distractor_difficulty": distractor_difficulty,
        "trap_distractor_present": False,
        "common_misconception": "",
        "jargon_density": "medium",
        "key_concept_summary": "Tests recall of basic facts.",
        "involves_graph_or_figure": involves_graph,
        "involves_data_table": involves_table,
    }
    content = [_tool_use_block(**tool_input)]
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )
    return SimpleNamespace(content=content, usage=usage)


def _make_client(message=None):
    if message is None:
        message = _forge_message()
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return client


# --------------------------------------------------------------------------- #
# Test 4: worker processes all pending questions
# --------------------------------------------------------------------------- #


async def test_worker_processes_all_pending(worker_db):
    setup, factory = worker_db
    q1 = await _make_question(setup, stem="Q1")
    q2 = await _make_question(setup, stem="Q2")
    q3 = await _make_question(setup, stem="Q3")
    await setup.commit()

    client = _make_client()
    summary = await run_extraction(factory, anthropic_client=client, cache=None)

    assert summary.processed == 3
    assert summary.succeeded == 3
    assert summary.failed == 0

    async with factory() as s:
        for q in (q1, q2, q3):
            row = (
                await s.execute(
                    select(QuestionFeatures).where(QuestionFeatures.question_id == q.id)
                )
            ).scalar_one_or_none()
            assert row is not None, f"Missing features row for question_id={q.id}"


# --------------------------------------------------------------------------- #
# Test 5: worker respects extractor_version — only re-extracts stale rows
# --------------------------------------------------------------------------- #


async def test_worker_respects_extractor_version(worker_db):
    setup, factory = worker_db
    q_old = await _make_question(setup, stem="old version")
    await _make_features_row(setup, q_old.id, extractor_version="features-v0-stale")
    q_current = await _make_question(setup, stem="current version")
    await _make_features_row(setup, q_current.id, extractor_version=EXTRACTOR_VERSION)
    await setup.commit()

    client = _make_client()
    summary = await run_extraction(factory, anthropic_client=client, cache=None)

    # Only the stale-version question should be (re-)processed.
    assert summary.processed == 1
    assert summary.succeeded == 1
    assert client.messages.create.await_count == 1


# --------------------------------------------------------------------------- #
# Test 6: concurrency bounded — peak in-flight LLM calls ≤ concurrency
# --------------------------------------------------------------------------- #


async def test_worker_concurrency_bounded(worker_db):
    setup, factory = worker_db
    for i in range(10):
        await _make_question(setup, stem=f"Q concurrency {i}")
    await setup.commit()

    in_flight = [0]
    peak = [0]
    lock = asyncio.Lock()

    async def _gated_create(*args, **kwargs):
        async with lock:
            in_flight[0] += 1
            if in_flight[0] > peak[0]:
                peak[0] = in_flight[0]
        await asyncio.sleep(0)  # yield to other coroutines
        async with lock:
            in_flight[0] -= 1
        return _forge_message()

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = _gated_create

    await run_extraction(factory, anthropic_client=client, cache=None, concurrency=3)

    assert peak[0] <= 3, f"Peak concurrent LLM calls was {peak[0]}, expected ≤ 3"


# --------------------------------------------------------------------------- #
# Test 7: retry on failure — retries once, then fails; no features row
# --------------------------------------------------------------------------- #


async def test_worker_retries_on_failure_once(worker_db):
    setup, factory = worker_db
    q = await _make_question(setup, stem="will fail")
    await setup.commit()

    call_count = [0]

    async def _always_fail(*args, **kwargs):
        call_count[0] += 1
        raise RuntimeError("LLM error simulated")

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = _always_fail

    summary = await run_extraction(factory, anthropic_client=client, cache=None)

    assert call_count[0] == 2, f"Expected 2 attempts, got {call_count[0]}"
    assert summary.processed == 1
    assert summary.failed == 1
    assert summary.retried == 1
    assert summary.succeeded == 0

    async with factory() as s:
        row = (
            await s.execute(select(QuestionFeatures).where(QuestionFeatures.question_id == q.id))
        ).scalar_one_or_none()
        assert row is None


# --------------------------------------------------------------------------- #
# Test 8: cost cap — stops after expected number of calls
# --------------------------------------------------------------------------- #


async def test_worker_respects_max_cost_usd(worker_db):
    setup, factory = worker_db
    for i in range(10):
        await _make_question(setup, stem=f"Q cost {i}")
    await setup.commit()

    # claude-sonnet-4-6 pricing: input=$3/M, output=$15/M
    # 1000 input tokens = $0.003, 100 output tokens = $0.0015 → ~$0.0045/call
    # Cap at $0.01 → should stop after ~2 calls (2 * $0.0045 = $0.009 < $0.01;
    # 3rd call would push over). Exact cutoff depends on order, but definitely < 10.
    msg = _forge_message(input_tokens=1000, output_tokens=100)
    client = _make_client(msg)

    summary = await run_extraction(
        factory,
        anthropic_client=client,
        cache=None,
        concurrency=1,
        max_cost_usd=0.01,
    )

    assert summary.cost_limit_hit is True
    assert summary.succeeded < 10
    assert summary.total_cost_usd >= 0.01


# --------------------------------------------------------------------------- #
# Test 9: --missed-only filter
# --------------------------------------------------------------------------- #


async def test_worker_filter_missed_only(worker_db):
    setup, factory = worker_db
    q_missed = await _make_question(setup, stem="missed")
    q_correct = await _make_question(setup, stem="always correct")

    now = datetime.now(tz=timezone.utc)
    setup.add(
        Attempt(
            question_id=q_missed.id,
            attempted_at=now,
            selected_choice="B",
            is_correct=False,
        )
    )
    setup.add(
        Attempt(
            question_id=q_correct.id,
            attempted_at=now,
            selected_choice="A",
            is_correct=True,
        )
    )
    await setup.commit()

    client = _make_client()
    summary = await run_extraction(factory, anthropic_client=client, cache=None, missed_only=True)

    assert summary.processed == 1
    assert summary.succeeded == 1
    assert client.messages.create.await_count == 1

    async with factory() as s:
        assert (
            await s.execute(
                select(QuestionFeatures).where(QuestionFeatures.question_id == q_missed.id)
            )
        ).scalar_one_or_none() is not None
        assert (
            await s.execute(
                select(QuestionFeatures).where(QuestionFeatures.question_id == q_correct.id)
            )
        ).scalar_one_or_none() is None


# --------------------------------------------------------------------------- #
# Test 10: --since filter
# --------------------------------------------------------------------------- #


async def test_worker_filter_since_date(worker_db):
    setup, factory = worker_db
    old_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new_dt = datetime(2026, 5, 1, tzinfo=timezone.utc)
    q_old = await _make_question(setup, stem="old question", first_seen_at=old_dt)
    q_new = await _make_question(setup, stem="new question", first_seen_at=new_dt)
    await setup.commit()

    client = _make_client()
    summary = await run_extraction(
        factory,
        anthropic_client=client,
        cache=None,
        since=date(2026, 3, 1),
    )

    assert summary.processed == 1
    assert summary.succeeded == 1

    async with factory() as s:
        assert (
            await s.execute(
                select(QuestionFeatures).where(QuestionFeatures.question_id == q_new.id)
            )
        ).scalar_one_or_none() is not None
        assert (
            await s.execute(
                select(QuestionFeatures).where(QuestionFeatures.question_id == q_old.id)
            )
        ).scalar_one_or_none() is None


# --------------------------------------------------------------------------- #
# Test 11: --limit filter
# --------------------------------------------------------------------------- #


async def test_worker_filter_limit(worker_db):
    setup, factory = worker_db
    for i in range(10):
        await _make_question(setup, stem=f"Q limit {i}")
    await setup.commit()

    client = _make_client()
    summary = await run_extraction(factory, anthropic_client=client, cache=None, limit=3)

    assert summary.processed == 3
    assert summary.succeeded == 3
    assert client.messages.create.await_count == 3


# --------------------------------------------------------------------------- #
# Test 12: summary distributions are correct
# --------------------------------------------------------------------------- #


async def test_worker_summary_distributions_correct(worker_db):
    setup, factory = worker_db
    for i in range(4):
        await _make_question(setup, stem=f"Q dist {i}")
    await setup.commit()

    # All 4 return application / medium / False / False
    msg = _forge_message(
        reasoning_type="application",
        distractor_difficulty="medium",
        involves_graph=False,
        involves_table=False,
    )
    client = _make_client(msg)
    summary = await run_extraction(factory, anthropic_client=client, cache=None, concurrency=1)

    assert summary.succeeded == 4
    dists = summary.distributions
    assert dists.get("reasoning_type", {}).get("application") == 4
    assert dists.get("distractor_difficulty", {}).get("medium") == 4
    assert dists.get("involves_graph_or_figure", {}).get("False") == 4
    assert dists.get("involves_data_table", {}).get("False") == 4


# --------------------------------------------------------------------------- #
# Test 13: CARS questions are skipped
# --------------------------------------------------------------------------- #


async def test_worker_skips_cars(worker_db):
    setup, factory = worker_db
    q = await _make_question(setup, stem="CARS reading question")

    cars_cc = (
        await setup.execute(select(ContentCategory).where(ContentCategory.code == "CARS"))
    ).scalar_one()
    setup.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cars_cc.id,
            confidence=Decimal("1.00"),
            source="manual",
        )
    )
    await setup.commit()

    client = _make_client()
    summary = await run_extraction(factory, anthropic_client=client, cache=None)

    assert summary.skipped_cars == 1
    assert summary.succeeded == 0
    assert summary.processed == 1
    client.messages.create.assert_not_called()

    async with factory() as s:
        row = (
            await s.execute(select(QuestionFeatures).where(QuestionFeatures.question_id == q.id))
        ).scalar_one_or_none()
        assert row is None
