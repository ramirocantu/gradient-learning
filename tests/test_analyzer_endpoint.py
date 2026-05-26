"""Tests for POST /api/v1/analyzer/extract endpoint (Ticket 4.3)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.main import app
from app.models.captures import Question


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def endpoint_db(
    seeded_report, test_engine
) -> AsyncIterator[tuple[AsyncSession, async_sessionmaker]]:
    """Seed data session + cleanup."""
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as setup:
        yield setup, factory
    async with factory() as cleanup:
        await cleanup.execute(text("DELETE FROM question_features"))
        await cleanup.execute(text("DELETE FROM attempts"))
        await cleanup.execute(text("DELETE FROM question_tags"))
        await cleanup.execute(text("DELETE FROM questions"))
        await cleanup.commit()


def _rand_qid() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


async def _make_question(session: AsyncSession, *, stem: str = "stem") -> Question:
    q = Question(
        qid=_rand_qid(),
        stem_html=f"<p>{stem}</p>",
        stem_plain=stem,
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_ids": []},
        ],
        correct_choice="A",
        explanation_html="<p>because</p>",
        explanation_plain="because",
    )
    session.add(q)
    await session.flush()
    return q


def _tool_use_block(**input_data):
    from anthropic.types import ToolUseBlock

    return ToolUseBlock(
        id="toolu_ep",
        name="submit_question_features",
        input=input_data,
        type="tool_use",
    )


def _forge_message():
    tool_input = {
        "reasoning_type": "application",
        "requires_calculation": False,
        "calculation_steps": 0,
        "passage_type": "",
        "distractor_difficulty": "medium",
        "trap_distractor_present": False,
        "common_misconception": "",
        "jargon_density": "medium",
        "key_concept_summary": "Tests recall of a basic concept.",
        "involves_graph_or_figure": False,
        "involves_data_table": False,
    }
    content = [_tool_use_block(**tool_input)]
    usage = SimpleNamespace(
        input_tokens=300,
        output_tokens=80,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=content, usage=usage)


# --------------------------------------------------------------------------- #
# Test 14: endpoint returns summary shape
# --------------------------------------------------------------------------- #


def _null_cache() -> MagicMock:
    """Cache mock that always misses — avoids reading/writing real disk cache."""
    m = MagicMock()
    m.get = MagicMock(return_value=None)
    m.lookup_cost = MagicMock(return_value=0.0)
    m.put = MagicMock()
    m.close = MagicMock()
    return m


async def test_extract_endpoint_returns_summary(endpoint_db):
    setup, factory = endpoint_db
    await _make_question(setup, stem="endpoint test Q1")
    await _make_question(setup, stem="endpoint test Q2")
    await setup.commit()

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_forge_message())

    with (
        patch("app.api.v1.analyzer.AsyncAnthropic", return_value=mock_client),
        patch("app.api.v1.analyzer.AsyncSessionLocal", factory),
        patch("app.api.v1.analyzer.FeatureExtractorCache", return_value=_null_cache()),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/v1/analyzer/extract", json={})

    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "model",
        "processed",
        "succeeded",
        "failed",
        "retried",
        "skipped_cars",
        "cache_hits",
        "cache_misses",
        "total_cost_usd",
        "total_cost_saved_usd",
        "cost_limit_hit",
        "dry_run",
        "distributions",
    ):
        assert key in data, f"Missing key {key!r} in response"
    assert data["succeeded"] == 2
    assert data["failed"] == 0


# --------------------------------------------------------------------------- #
# Test 15: endpoint respects filter body
# --------------------------------------------------------------------------- #


async def test_extract_endpoint_respects_filter_body(endpoint_db):
    setup, factory = endpoint_db
    for i in range(5):
        await _make_question(setup, stem=f"endpoint filter Q{i}")
    await setup.commit()

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_forge_message())

    with (
        patch("app.api.v1.analyzer.AsyncAnthropic", return_value=mock_client),
        patch("app.api.v1.analyzer.AsyncSessionLocal", factory),
        patch("app.api.v1.analyzer.FeatureExtractorCache", return_value=_null_cache()),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/v1/analyzer/extract", json={"limit": 2})

    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 2
    assert data["succeeded"] == 2
    assert mock_client.messages.create.await_count == 2
