"""Insight synthesizer tests — Ticket 4.5.

Anthropic SDK mocked at boundary. Tests 1-8 are pure unit tests (no DB).
Tests 9-11 use the live test DB with the outer-transaction rollback pattern.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.main import app
from app.models.captures import Attempt, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION as FEAT_VERSION
from app.services.analyzer.patterns import (
    AnalysisFilter,
    CoverageStats,
    FeatureFinding,
    InsightReport,
    analyze,
)
from app.services.analyzer.synthesizer import (
    InsightSynthesis,
    insights_for_filter,
    synthesize,
)
from app.services.analyzer.synthesizer_cache import SynthesizerCache


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

_SAMPLE_MARKDOWN = """\
## Quick read

You're at 47% overall across 75 attempts. The single loudest signal is negative
phrasing — when the stem says NOT or EXCEPT, your accuracy drops from 74% to 25%.

## What's hurting you

### Negative phrasing: 25% vs 74% without

When the question asks "which of the following is NOT true," your accuracy tanks.
This is a pacing and attention issue, not a content gap. See qids q-abc, q-def.

## What's working

Calculation-heavy questions (3+ steps) are actually a slight strength at 62%.

## Caveats

- 5 questions lacked feature coverage (no extractor run yet).\
"""


def _make_finding(
    *,
    feature_name: str = "has_negative_phrasing",
    feature_value: str = "True",
    accuracy_with: float = 0.25,
    accuracy_without: float = 0.74,
    attempts_with: int = 20,
    attempts_without: int = 50,
    correct_with: int = 5,
    correct_without: int = 37,
    confident_delta: float = -0.55,
    missed_qids: list[str] | None = None,
) -> FeatureFinding:
    delta = accuracy_with - accuracy_without
    return FeatureFinding(
        feature_name=feature_name,
        feature_value=feature_value,
        accuracy_with=accuracy_with,
        accuracy_without=accuracy_without,
        attempts_with=attempts_with,
        attempts_without=attempts_without,
        correct_with=correct_with,
        correct_without=correct_without,
        accuracy_delta=delta,
        wilson_lower_with=0.10,
        wilson_lower_without=0.60,
        confident_delta=confident_delta,
        representative_missed_qids=missed_qids or ["q-abc123", "q-def456"],
    )


def _make_report(
    *,
    findings: list[FeatureFinding] | None = None,
    baseline_accuracy: float = 0.47,
    total_attempts: int = 75,
    skill: int | None = None,
) -> InsightReport:
    if findings is None:
        findings = [
            _make_finding(
                feature_name="has_negative_phrasing",
                confident_delta=-0.55,
            ),
            _make_finding(
                feature_name="involves_graph_or_figure",
                feature_value="True",
                accuracy_with=0.22,
                accuracy_without=0.60,
                confident_delta=-0.45,
            ),
            _make_finding(
                feature_name="jargon_density",
                feature_value="high",
                accuracy_with=0.30,
                accuracy_without=0.58,
                confident_delta=-0.35,
            ),
            _make_finding(
                feature_name="calculation_steps",
                feature_value="3+",
                accuracy_with=0.62,
                accuracy_without=0.45,
                confident_delta=0.10,
            ),
        ]
    return InsightReport(
        filter_applied=AnalysisFilter(skill=skill),
        total_attempts_in_scope=total_attempts,
        total_questions_in_scope=70,
        baseline_accuracy=baseline_accuracy,
        baseline_wilson_lower=0.36,
        findings=findings,
        coverage=CoverageStats(
            questions_with_features=65,
            questions_without_features=5,
            feature_extractor_version=FEAT_VERSION,
        ),
    )


def _forge_response(
    *,
    text: str = _SAMPLE_MARKDOWN,
    input_tokens: int = 500,
    output_tokens: int = 300,
    cache_read: int = 0,
    cache_create: int = 0,
) -> SimpleNamespace:
    content_block = SimpleNamespace(text=text)
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )
    return SimpleNamespace(content=[content_block], usage=usage)


def _make_client(response: SimpleNamespace | None = None) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response or _forge_response())
    return client


def _null_cache(tmp_path: Path) -> SynthesizerCache:
    return SynthesizerCache(tmp_path / "test-synth.db")


# --------------------------------------------------------------------------- #
# Test 1: prompt shape
# --------------------------------------------------------------------------- #


async def test_synthesize_calls_anthropic_with_expected_prompt_shape(tmp_path):
    report = _make_report()
    client = _make_client()
    cache = _null_cache(tmp_path)

    await synthesize(report, anthropic_client=client, cache=cache)

    assert client.messages.create.await_count == 1
    call_kwargs = client.messages.create.call_args.kwargs

    # System prompt checks
    system = call_kwargs["system"]
    assert isinstance(system, list)
    system_text = system[0]["text"]
    assert "MCAT tutor" in system_text
    assert "## Quick read" in system_text
    assert "## What's hurting you" in system_text
    assert "## Caveats" in system_text

    # User message checks
    messages = call_kwargs["messages"]
    user_content = messages[0]["content"]
    assert "has_negative_phrasing" in user_content
    assert "involves_graph_or_figure" in user_content
    assert "47.0%" in user_content  # baseline_accuracy

    cache.close()


# --------------------------------------------------------------------------- #
# Test 2: parseable markdown returned
# --------------------------------------------------------------------------- #


async def test_synthesize_returns_parseable_markdown(tmp_path):
    report = _make_report()
    client = _make_client(_forge_response(text=_SAMPLE_MARKDOWN))
    cache = _null_cache(tmp_path)

    result = await synthesize(report, anthropic_client=client, cache=cache)

    assert isinstance(result, InsightSynthesis)
    assert "## Quick read" in result.markdown
    assert "## What's hurting you" in result.markdown
    assert not result.cache_hit
    cache.close()


# --------------------------------------------------------------------------- #
# Test 3: caches by report hash — second call is cache hit
# --------------------------------------------------------------------------- #


async def test_synthesize_caches_by_report_hash(tmp_path):
    report = _make_report()
    client = _make_client()
    cache = _null_cache(tmp_path)

    r1 = await synthesize(report, anthropic_client=client, cache=cache)
    r2 = await synthesize(report, anthropic_client=client, cache=cache)

    assert not r1.cache_hit
    assert r2.cache_hit
    assert client.messages.create.await_count == 1
    assert r2.markdown == r1.markdown
    cache.close()


# --------------------------------------------------------------------------- #
# Test 4: version bump invalidates cache
# --------------------------------------------------------------------------- #


async def test_synthesize_invalidates_on_version_bump(tmp_path):
    report = _make_report()
    client = _make_client()
    cache = _null_cache(tmp_path)

    r1 = await synthesize(report, anthropic_client=client, cache=cache, extractor_version="v1")
    r2 = await synthesize(
        report, anthropic_client=client, cache=cache, extractor_version="v2-bumped"
    )

    assert not r1.cache_hit
    assert not r2.cache_hit
    assert client.messages.create.await_count == 2
    cache.close()


# --------------------------------------------------------------------------- #
# Test 5: different model = cache miss
# --------------------------------------------------------------------------- #


async def test_synthesize_caches_by_model(tmp_path):
    report = _make_report()
    client = _make_client()
    cache = _null_cache(tmp_path)

    r1 = await synthesize(report, anthropic_client=client, cache=cache, model="claude-sonnet-4-6")
    r2 = await synthesize(
        report, anthropic_client=client, cache=cache, model="claude-haiku-4-5-20251001"
    )

    assert not r1.cache_hit
    assert not r2.cache_hit
    assert client.messages.create.await_count == 2
    cache.close()


# --------------------------------------------------------------------------- #
# Test 6: empty findings — graceful no-LLM fallback
# --------------------------------------------------------------------------- #


async def test_synthesize_handles_empty_findings_report(tmp_path):
    report = _make_report(findings=[])
    client = _make_client()
    cache = _null_cache(tmp_path)

    result = await synthesize(report, anthropic_client=client, cache=cache)

    assert client.messages.create.await_count == 0
    assert "not enough data" in result.markdown.lower()
    assert result.input_tokens == 0
    assert result.estimated_cost_usd == 0.0
    cache.close()


# --------------------------------------------------------------------------- #
# Test 7: malformed/empty LLM response — fallback returned
# --------------------------------------------------------------------------- #


async def test_synthesize_falls_back_on_malformed_response(tmp_path):
    report = _make_report()
    client = _make_client(_forge_response(text=""))
    cache = _null_cache(tmp_path)

    result = await synthesize(report, anthropic_client=client, cache=cache)

    assert "## Quick read" in result.markdown
    assert "parseable response" in result.markdown
    cache.close()


# --------------------------------------------------------------------------- #
# Test 8: token usage and cost logged correctly
# --------------------------------------------------------------------------- #


async def test_synthesize_logs_token_usage(tmp_path):
    report = _make_report()
    response = _forge_response(
        text=_SAMPLE_MARKDOWN,
        input_tokens=400,
        output_tokens=250,
        cache_read=100,
        cache_create=50,
    )
    client = _make_client(response)
    cache = _null_cache(tmp_path)

    result = await synthesize(report, anthropic_client=client, cache=cache)

    # input_tokens stored = input + cache_create + cache_read = 400+50+100 = 550
    assert result.input_tokens == 550
    assert result.output_tokens == 250
    # cost = (400+50)/1M * 3.0 + 100/1M * 0.30 + 250/1M * 15.0
    expected_cost = (450 / 1_000_000) * 3.0 + (100 / 1_000_000) * 0.30 + (250 / 1_000_000) * 15.0
    assert abs(result.estimated_cost_usd - expected_cost) < 1e-9
    cache.close()


# --------------------------------------------------------------------------- #
# DB fixtures (tests 9-11)
# --------------------------------------------------------------------------- #


def _qid() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


def _make_question() -> Question:
    return Question(
        qid=_qid(),
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_content_hashes": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_content_hashes": []},
        ],
        correct_choice="A",
        explanation_html=None,
        explanation_plain="why",
        uworld_aamc_tags=[],
        needs_categorization=False,
    )


def _attempt(*, question_id: int, is_correct: bool) -> Attempt:
    return Attempt(
        question_id=question_id,
        attempted_at=datetime.now(tz=timezone.utc),
        selected_choice="A" if is_correct else "B",
        is_correct=is_correct,
        flagged=False,
    )


def _features(question_id: int, **kwargs) -> QuestionFeatures:
    defaults: dict = {
        "question_format": "discrete",
        "reasoning_type": "application",
        "requires_calculation": False,
        "calculation_steps": 0,
        "involves_graph_or_figure": False,
        "involves_data_table": False,
        "has_negative_phrasing": False,
        "distractor_difficulty": "medium",
        "trap_distractor_present": False,
        "common_misconception": None,
        "jargon_density": "medium",
        "key_concept_summary": "summary",
        "passage_length_bucket": None,
        "passage_type": None,
        "extractor_version": FEAT_VERSION,
    }
    defaults.update(kwargs)
    return QuestionFeatures(question_id=question_id, **defaults)


def _skill_tag(question_id: int, skill: int) -> QuestionTag:
    return QuestionTag(question_id=question_id, skill=skill, confidence=1.0, source="llm")


@pytest.fixture
async def env(seeded_report, test_engine) -> AsyncIterator[tuple[AsyncClient, any]]:
    """ASGI client + session factory, rolled-back outer transaction."""
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
            yield client, make_session
    finally:
        app.dependency_overrides.pop(get_session, None)
        await conn.rollback()
        await conn.close()


# --------------------------------------------------------------------------- #
# Test 9: insights_for_filter composes analyze + synthesize
# --------------------------------------------------------------------------- #


async def test_insights_for_filter_composes_analyze_and_synthesize(env, tmp_path):
    _, make_session = env
    # Seed 12 Skill-3 questions with features so findings meet min_sample_size=3
    async with make_session() as s:
        for i in range(12):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            s.add(_features(q.id, has_negative_phrasing=(i < 6)))
            s.add(_attempt(question_id=q.id, is_correct=(i < 5)))
        await s.commit()

    af = AnalysisFilter(skill=3, min_sample_size=3)
    client = _make_client()
    cache = _null_cache(tmp_path)

    async with make_session() as s:
        synthesis = await insights_for_filter(af, s, anthropic_client=client, cache=cache)
        direct_report = await analyze(af, s)

    assert synthesis.report.total_attempts_in_scope == direct_report.total_attempts_in_scope
    assert synthesis.report.baseline_accuracy == pytest.approx(direct_report.baseline_accuracy)
    assert len(synthesis.markdown) > 0
    cache.close()


# --------------------------------------------------------------------------- #
# Test 10: endpoint returns synthesis JSON
# --------------------------------------------------------------------------- #


async def test_endpoint_returns_synthesis_json(env, tmp_path):
    client_http, make_session = env
    async with make_session() as s:
        for i in range(12):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            # Vary has_negative_phrasing so findings are produced (6 with, 6 without)
            s.add(_features(q.id, has_negative_phrasing=(i < 6)))
            s.add(_attempt(question_id=q.id, is_correct=(i < 4)))
        await s.commit()

    mock_client = _make_client()
    mock_cache = MagicMock()
    mock_cache.get = MagicMock(return_value=None)
    mock_cache.lookup_cost = MagicMock(return_value=0.0)
    mock_cache.put = MagicMock()
    mock_cache.close = MagicMock()

    with (
        patch("app.api.v1.analyzer.AsyncAnthropic", return_value=mock_client),
        patch("app.api.v1.analyzer.SynthesizerCache", return_value=mock_cache),
    ):
        resp = await client_http.get("/api/v1/analyzer/insights?skill=3&min_sample_size=3")

    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "markdown",
        "report",
        "cache_hit",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "cost_saved_usd",
        "extractor_version",
        "model",
    ):
        assert key in data, f"Missing key {key!r}"
    assert data["report"]["filter_applied"]["skill"] == 3
    assert isinstance(data["markdown"], str)


# --------------------------------------------------------------------------- #
# Test 11: bust_cache forces fresh SDK call
# --------------------------------------------------------------------------- #


async def test_endpoint_bust_cache_param_forces_fresh_call(env, tmp_path):
    client_http, make_session = env
    async with make_session() as s:
        for i in range(12):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            # Vary has_negative_phrasing so findings are produced (6 with, 6 without)
            s.add(_features(q.id, has_negative_phrasing=(i < 6)))
            s.add(_attempt(question_id=q.id, is_correct=(i < 4)))
        await s.commit()

    mock_client = _make_client()

    # Cache mock that always returns a hit (simulates warm cache)
    cached_result = MagicMock()
    cached_result.markdown = _SAMPLE_MARKDOWN
    cached_result.input_tokens = 500
    cached_result.output_tokens = 300
    cached_result.cost_estimate_usd = 0.005
    mock_cache = MagicMock()
    mock_cache.get = MagicMock(return_value=cached_result)
    mock_cache.lookup_cost = MagicMock(return_value=0.005)
    mock_cache.put = MagicMock()
    mock_cache.close = MagicMock()

    with (
        patch("app.api.v1.analyzer.AsyncAnthropic", return_value=mock_client),
        patch("app.api.v1.analyzer.SynthesizerCache", return_value=mock_cache),
    ):
        # Without bust_cache — should hit cache, SDK NOT called
        resp_cached = await client_http.get("/api/v1/analyzer/insights?skill=3&min_sample_size=3")
        assert resp_cached.status_code == 200
        assert mock_client.messages.create.await_count == 0

        # With bust_cache=true — SDK must be called
        resp_bust = await client_http.get(
            "/api/v1/analyzer/insights?skill=3&min_sample_size=3&bust_cache=true"
        )
        assert resp_bust.status_code == 200
        assert mock_client.messages.create.await_count == 1


# --------------------------------------------------------------------------- #
# Tests 12–15: run_llm flag (Ticket 6.7 / Bug #19)
# --------------------------------------------------------------------------- #


async def test_synthesize_returns_none_when_run_llm_false_and_cache_miss(tmp_path):
    """Cache miss + run_llm=False returns None without calling the LLM."""
    report = _make_report()
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=AssertionError("LLM must not be called"))
    cache = _null_cache(tmp_path)

    result = await synthesize(report, anthropic_client=client, cache=cache, run_llm=False)

    assert result is None
    client.messages.create.assert_not_awaited()
    cache.close()


async def test_synthesize_returns_cached_when_run_llm_false_and_cache_hit(tmp_path):
    """Cache hit + run_llm=False returns the cached result without calling the LLM."""
    report = _make_report()
    real_client = _make_client()
    cache = _null_cache(tmp_path)

    r1 = await synthesize(report, anthropic_client=real_client, cache=cache, run_llm=True)
    assert r1 is not None
    assert not r1.cache_hit

    blocking_client = MagicMock()
    blocking_client.messages = MagicMock()
    blocking_client.messages.create = AsyncMock(
        side_effect=AssertionError("LLM must not be called on cache hit")
    )

    r2 = await synthesize(report, anthropic_client=blocking_client, cache=cache, run_llm=False)

    assert r2 is not None
    assert r2.cache_hit
    assert r2.markdown == r1.markdown
    blocking_client.messages.create.assert_not_awaited()
    cache.close()


async def test_synthesize_calls_llm_when_run_llm_true_and_cache_miss(tmp_path):
    """Cache miss + run_llm=True calls the LLM and returns a result."""
    report = _make_report()
    client = _make_client()
    cache = _null_cache(tmp_path)

    result = await synthesize(report, anthropic_client=client, cache=cache, run_llm=True)

    assert result is not None
    assert not result.cache_hit
    assert len(result.markdown) > 0
    assert client.messages.create.await_count == 1
    cache.close()


async def test_synthesize_empty_findings_returns_no_data_marker_regardless_of_run_llm(
    tmp_path,
):
    """Empty findings → no-data short-circuit regardless of run_llm; LLM never called."""
    report = _make_report(findings=[])
    blocking_client = MagicMock()
    blocking_client.messages = MagicMock()
    blocking_client.messages.create = AsyncMock(
        side_effect=AssertionError("LLM must not be called for empty findings")
    )
    cache = _null_cache(tmp_path)

    r_false = await synthesize(report, anthropic_client=blocking_client, cache=cache, run_llm=False)
    r_true = await synthesize(report, anthropic_client=blocking_client, cache=cache, run_llm=True)

    assert r_false is not None
    assert r_true is not None
    assert "not enough data" in r_false.markdown.lower()
    assert "not enough data" in r_true.markdown.lower()
    assert r_false.markdown == r_true.markdown
    blocking_client.messages.create.assert_not_awaited()
    cache.close()
