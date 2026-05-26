"""Pattern Insights page tests (Ticket 6.4).

Boundary-mocked — `insights_for_filter` and `run_extraction` are patched at
the dashboard.routes.insights module level so no LLM or DB extraction runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.analyzer.patterns import (
    AnalysisFilter,
    CoverageStats,
    FeatureFinding,
    InsightReport,
)
from app.services.analyzer.synthesizer import InsightSynthesis


def _make_synthesis(
    *,
    markdown: str = "## Quick read\n\nYou are doing alright.\n",
    findings: list[FeatureFinding] | None = None,
    questions_without_features: int = 0,
    filter_applied: AnalysisFilter | None = None,
) -> InsightSynthesis:
    af = filter_applied or AnalysisFilter()
    report = InsightReport(
        filter_applied=af,
        total_attempts_in_scope=50,
        total_questions_in_scope=50,
        baseline_accuracy=0.70,
        baseline_wilson_lower=0.60,
        findings=findings or [],
        coverage=CoverageStats(
            questions_with_features=50 - questions_without_features,
            questions_without_features=questions_without_features,
            feature_extractor_version="test-v1",
        ),
    )
    return InsightSynthesis(
        markdown=markdown,
        report=report,
        cache_hit=False,
        input_tokens=100,
        output_tokens=200,
        estimated_cost_usd=0.0015,
        cost_saved_usd=0.0,
        extractor_version="synth-test",
        model="claude-sonnet-4-6",
    )


def _make_finding(
    *,
    name: str = "has_negative_phrasing",
    value: str = "True",
    accuracy_with: float = 0.30,
    accuracy_without: float = 0.80,
    confident_delta: float = -0.40,
) -> FeatureFinding:
    return FeatureFinding(
        feature_name=name,
        feature_value=value,
        accuracy_with=accuracy_with,
        accuracy_without=accuracy_without,
        attempts_with=20,
        attempts_without=30,
        correct_with=int(round(accuracy_with * 20)),
        correct_without=int(round(accuracy_without * 30)),
        accuracy_delta=accuracy_with - accuracy_without,
        wilson_lower_with=max(0.0, accuracy_with - 0.10),
        wilson_lower_without=max(0.0, accuracy_without - 0.10),
        confident_delta=confident_delta,
        representative_missed_qids=["Q-001", "Q-002", "Q-003"],
    )


@pytest.mark.asyncio
async def test_insights_page_loads(client, session):
    synthesis = _make_synthesis(
        markdown="## Quick read\n\nLooking solid.\n",
        findings=[_make_finding(), _make_finding(name="requires_calculation")],
    )
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=synthesis),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert "Pattern Insights" in response.text


@pytest.mark.asyncio
async def test_insights_filter_params_populate_form(client, session):
    synthesis = _make_synthesis(filter_applied=AnalysisFilter(section_code="CP", skill=3))
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=synthesis),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights?section=CP&skill=3")
    assert response.status_code == 200
    assert 'data-section="CP"' in response.text
    assert 'data-skill="3"' in response.text
    # Active state markers
    assert 'data-section="CP" data-active="true"' in response.text
    assert 'data-skill="3" data-active="true"' in response.text


@pytest.mark.asyncio
async def test_insights_renders_synthesis_markdown(client, session):
    synthesis = _make_synthesis(
        markdown="## Quick read\n\nYou missed several negation questions.\n"
    )
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=synthesis),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert "<h2>Quick read</h2>" in response.text
    # Raw markdown header should not appear unrendered
    assert "## Quick read" not in response.text


@pytest.mark.asyncio
async def test_insights_renders_finding_cards(client, session):
    findings = [
        _make_finding(name="has_negative_phrasing", value="True"),
        _make_finding(name="requires_calculation", value="True"),
    ]
    synthesis = _make_synthesis(findings=findings)
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=synthesis),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert response.text.count("finding-card") == 2
    assert "Has Negative Phrasing" in response.text
    assert "Requires Calculation" in response.text


@pytest.mark.asyncio
async def test_insights_empty_findings_shows_message(client, session):
    synthesis = _make_synthesis(findings=[])
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=synthesis),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert "No findings meet the minimum sample size" in response.text


@pytest.mark.asyncio
async def test_insights_coverage_warning_shown_when_gaps(client, session):
    synthesis = _make_synthesis(questions_without_features=5)
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=synthesis),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert "5 question(s) are missing feature data" in response.text


@pytest.mark.asyncio
async def test_run_extraction_redirects_back(client, session):
    fake_cache = MagicMock()
    with (
        patch(
            "app.web.dashboard.routes.insights.run_extraction",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.web.dashboard.routes.insights.FeatureExtractorCache",
            return_value=fake_cache,
        ),
    ):
        response = await client.post("/insights/run-extraction", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/insights"


# --------------------------------------------------------------------------- #
# Ticket 6.7 — Bug #19: run_llm gate + empty state
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_insights_empty_state_when_synthesis_none(client, session):
    """When synthesize returns None the template renders the Generate Insights CTA."""
    synthesis = _make_synthesis(findings=[_make_finding()])
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert "Generate Insights" in response.text
    assert "bust_cache=true" in response.text
    assert "## Quick read" not in response.text


@pytest.mark.asyncio
async def test_insights_passes_run_llm_false_on_default_load(client, session):
    """Default GET /insights passes run_llm=False to synthesize."""
    synthesis = _make_synthesis()
    mock_synth = AsyncMock(return_value=None)
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch("app.web.dashboard.routes.insights.synthesize", new=mock_synth),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights")
    assert response.status_code == 200
    assert mock_synth.await_count == 1
    call_kwargs = mock_synth.call_args.kwargs
    assert call_kwargs["run_llm"] is False


@pytest.mark.asyncio
async def test_insights_passes_run_llm_true_when_bust_cache(client, session):
    """GET /insights?bust_cache=true passes run_llm=True and bust_cache=True."""
    synthesis = _make_synthesis()
    mock_synth = AsyncMock(return_value=synthesis)
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch("app.web.dashboard.routes.insights.synthesize", new=mock_synth),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights?bust_cache=true")
    assert response.status_code == 200
    assert mock_synth.await_count == 1
    call_kwargs = mock_synth.call_args.kwargs
    assert call_kwargs["run_llm"] is True
    assert call_kwargs["bust_cache"] is True


@pytest.mark.asyncio
async def test_insights_generate_button_href_preserves_filters(client, session):
    """Empty state 'Generate Insights' href includes active filter params + bust_cache."""
    synthesis = _make_synthesis()
    with (
        patch(
            "app.web.dashboard.routes.insights.analyze",
            new=AsyncMock(return_value=synthesis.report),
        ),
        patch(
            "app.web.dashboard.routes.insights.synthesize",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.web.dashboard.routes.insights.SynthesizerCache",
            return_value=MagicMock(),
        ),
        patch("app.web.dashboard.routes.insights.AsyncAnthropic", return_value=MagicMock()),
    ):
        response = await client.get("/insights?section=CP&skill=3&min_sample_size=5")
    assert response.status_code == 200
    body = response.text
    assert "Generate Insights" in body
    assert "bust_cache=true" in body
    assert "section=CP" in body
    assert "skill=3" in body
    assert "min_sample_size=5" in body


@pytest.mark.asyncio
async def test_run_extraction_preserves_filter_params(client, session):
    fake_cache = MagicMock()
    with (
        patch(
            "app.web.dashboard.routes.insights.run_extraction",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.web.dashboard.routes.insights.FeatureExtractorCache",
            return_value=fake_cache,
        ),
    ):
        response = await client.post("/insights/run-extraction?section=BB", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/insights?section=BB"
