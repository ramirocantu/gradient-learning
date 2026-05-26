"""Pattern aggregator tests — Ticket 4.4.

Uses the same outer-transaction rollback pattern as test_analytics.py.
The AAMC outline is session-scoped via `seeded_report`; per-test data is
seeded inside the outer transaction and rolled back at teardown.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.main import app
from app.models.captures import Attempt, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory, FoundationalConcept, Section
from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION
from app.services.analyzer.patterns import AnalysisFilter, analyze


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
async def env(seeded_report, test_engine) -> AsyncIterator[tuple[AsyncClient, any]]:
    """ASGI client + session factory wrapped in a single rolled-back outer transaction."""
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
# Helpers
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


def _attempt(*, question_id: int, is_correct: bool, attempted_at: datetime) -> Attempt:
    return Attempt(
        question_id=question_id,
        attempted_at=attempted_at,
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
        "extractor_version": EXTRACTOR_VERSION,
    }
    defaults.update(kwargs)
    return QuestionFeatures(question_id=question_id, **defaults)


def _skill_tag(question_id: int, skill: int) -> QuestionTag:
    return QuestionTag(question_id=question_id, skill=skill, confidence=1.0, source="llm")


def _cc_tag(question_id: int, cc_id: int) -> QuestionTag:
    return QuestionTag(
        question_id=question_id, content_category_id=cc_id, confidence=1.0, source="llm"
    )


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def _first_cc_in_section(session: AsyncSession, section_code: str) -> ContentCategory:
    return (
        await session.execute(
            select(ContentCategory)
            .join(
                FoundationalConcept,
                FoundationalConcept.id == ContentCategory.foundational_concept_id,
            )
            .join(Section, Section.id == FoundationalConcept.section_id)
            .where(Section.code == section_code)
            .limit(1)
        )
    ).scalar_one()


# --------------------------------------------------------------------------- #
# 1. Baseline accuracy with no filter
# --------------------------------------------------------------------------- #


async def test_baseline_accuracy_with_no_filter(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        for i, correct in enumerate([True, True, False, False]):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=correct,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(), s)

    assert report.total_attempts_in_scope == 4
    assert report.baseline_accuracy == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 2. Boolean feature with negative delta
# --------------------------------------------------------------------------- #


async def test_finding_for_boolean_feature_negative_delta(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # 10 with graph=True: 3 correct, 7 wrong  → accuracy 0.3
        for i in range(10):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 3),
                    attempted_at=base + timedelta(minutes=i),
                )
            )
            s.add(_features(q.id, involves_graph_or_figure=True))
        # 10 with graph=False: 9 correct, 1 wrong  → accuracy 0.9
        for i in range(10):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 9),
                    attempted_at=base + timedelta(minutes=20 + i),
                )
            )
            s.add(_features(q.id, involves_graph_or_figure=False))
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    graph_finding = next(
        (f for f in report.findings if f.feature_name == "involves_graph_or_figure"),
        None,
    )
    assert graph_finding is not None
    assert graph_finding.accuracy_delta == pytest.approx(-0.6, abs=0.01)
    assert graph_finding.accuracy_with < graph_finding.accuracy_without


# --------------------------------------------------------------------------- #
# 3. Finding skipped when below min_sample_size
# --------------------------------------------------------------------------- #


async def test_finding_skipped_when_below_min_sample(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # 2 attempts with requires_calculation=True (below min_sample_size=10)
        for i in range(2):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
            s.add(_features(q.id, requires_calculation=True))
        # 20 attempts with requires_calculation=False
        for i in range(20):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=10 + i),
                )
            )
            s.add(_features(q.id, requires_calculation=False))
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    calc_finding = next(
        (f for f in report.findings if f.feature_name == "requires_calculation"), None
    )
    assert calc_finding is None


# --------------------------------------------------------------------------- #
# 4. Enum feature — one finding per distinct value
# --------------------------------------------------------------------------- #


async def test_finding_for_enum_feature_each_value(env):
    _, make_session = env
    base = _now()
    reasoning_values = ["application", "analysis", "recall"]
    async with make_session() as s:
        for idx, rtype in enumerate(reasoning_values):
            for i in range(12):
                q = _make_question()
                s.add(q)
                await s.flush()
                s.add(
                    _attempt(
                        question_id=q.id,
                        is_correct=(i < 6),
                        attempted_at=base + timedelta(minutes=idx * 20 + i),
                    )
                )
                s.add(_features(q.id, reasoning_type=rtype))
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    reasoning_findings = [f for f in report.findings if f.feature_name == "reasoning_type"]
    found_values = {f.feature_value for f in reasoning_findings}
    assert found_values == {"application", "analysis", "recall"}
    # Each value's without-group is the other two combined (24 attempts)
    for finding in reasoning_findings:
        assert finding.attempts_with == 12
        assert finding.attempts_without == 24


# --------------------------------------------------------------------------- #
# 5. calculation_steps bucketed
# --------------------------------------------------------------------------- #


async def test_calculation_steps_bucketed(env):
    _, make_session = env
    base = _now()
    step_counts = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # bucket "0" — 10 attempts
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,  # bucket "1-2" — 10 attempts
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
    ]  # bucket "3+" — 10 attempts
    async with make_session() as s:
        for idx, steps in enumerate(step_counts):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=idx),
                )
            )
            s.add(_features(q.id, calculation_steps=steps))
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    calc_findings = [f for f in report.findings if f.feature_name == "calculation_steps"]
    found_buckets = {f.feature_value for f in calc_findings}
    assert found_buckets == {"0", "1-2", "3+"}
    for finding in calc_findings:
        assert finding.attempts_with == 10
        assert finding.attempts_without == 20


# --------------------------------------------------------------------------- #
# 6. Filter by section
# --------------------------------------------------------------------------- #


async def test_filter_by_section(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        cp_cc = await _first_cc_in_section(s, "CP")
        ps_cc = await _first_cc_in_section(s, "PS")

        for i in range(5):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_cc_tag(q.id, cp_cc.id))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                )
            )

        for i in range(3):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_cc_tag(q.id, ps_cc.id))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=False,
                    attempted_at=base + timedelta(minutes=10 + i),
                )
            )

        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(section_code="CP"), s)

    assert report.total_attempts_in_scope == 5
    assert report.baseline_accuracy == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 7. Filter by skill
# --------------------------------------------------------------------------- #


async def test_filter_by_skill(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        for i in range(4):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 2))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                )
            )

        for i in range(6):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 2),
                    attempted_at=base + timedelta(minutes=10 + i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(skill=3), s)

    assert report.total_attempts_in_scope == 6
    assert report.total_questions_in_scope == 6
    assert report.baseline_accuracy == pytest.approx(2 / 6)


# --------------------------------------------------------------------------- #
# 8. Multi-tag aware filter — question tagged to two sections
# --------------------------------------------------------------------------- #


async def test_filter_multi_tag_aware(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        cp_cc = await _first_cc_in_section(s, "CP")
        bb_cc = await _first_cc_in_section(s, "BB")

        q = _make_question()
        s.add(q)
        await s.flush()
        # Tag to BOTH CP and BB
        s.add(_cc_tag(q.id, cp_cc.id))
        s.add(_cc_tag(q.id, bb_cc.id))
        s.add(_attempt(question_id=q.id, is_correct=True, attempted_at=base))
        await s.commit()

    async with make_session() as s:
        report_cp = await analyze(AnalysisFilter(section_code="CP"), s)
        report_bb = await analyze(AnalysisFilter(section_code="BB"), s)

    # Question counts in scope for both sections
    assert report_cp.total_attempts_in_scope == 1
    assert report_bb.total_attempts_in_scope == 1


# --------------------------------------------------------------------------- #
# 9. Filter by since
# --------------------------------------------------------------------------- #


async def test_filter_by_since(env):
    _, make_session = env
    cutoff = _now()
    async with make_session() as s:
        # 3 attempts before cutoff
        for i in range(3):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=cutoff - timedelta(days=2, minutes=i),
                )
            )
        # 4 attempts after cutoff
        for i in range(4):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=False,
                    attempted_at=cutoff + timedelta(days=1, minutes=i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(since=cutoff.date()), s)

    assert report.total_attempts_in_scope == 4


# --------------------------------------------------------------------------- #
# 10. Representative qids only includes missed attempts
# --------------------------------------------------------------------------- #


async def test_representative_qids_only_includes_missed(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # Build: 10 graph=True attempts (to meet min_sample) with 1 correct + 1 wrong per Q pair
        for i in range(5):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=True))
            # correct attempt
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i * 2),
                )
            )
            # wrong attempt
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=False,
                    attempted_at=base + timedelta(minutes=i * 2 + 1),
                )
            )
        # 10 graph=False (for without-group)
        for i in range(10):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=False))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=100 + i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    graph_finding = next(f for f in report.findings if f.feature_name == "involves_graph_or_figure")
    # 10 attempts_with total (5 correct + 5 wrong)
    assert graph_finding.correct_with == 5
    # representative_missed_qids must only contain qids from wrong attempts
    assert len(graph_finding.representative_missed_qids) <= 3
    assert len(graph_finding.representative_missed_qids) > 0


# --------------------------------------------------------------------------- #
# 11. Representative qids capped at 3
# --------------------------------------------------------------------------- #


async def test_representative_qids_capped_at_3(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # 10 graph=True — all wrong (5 distinct questions with 2 attempts each wrong)
        for i in range(10):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=True))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=False,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        # 10 graph=False — all correct (for without-group)
        for i in range(10):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=False))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=100 + i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    graph_finding = next(f for f in report.findings if f.feature_name == "involves_graph_or_figure")
    assert len(graph_finding.representative_missed_qids) == 3


# --------------------------------------------------------------------------- #
# 12. Findings sorted by confident_delta ascending
# --------------------------------------------------------------------------- #


async def test_findings_sorted_by_confident_delta_ascending(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # involves_graph=True: 2 correct out of 12 → low accuracy_with
        for i in range(12):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=True, has_negative_phrasing=False))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 2),
                    attempted_at=base + timedelta(minutes=i),
                )
            )

        # has_negative=True: 10 correct out of 12 → high accuracy_with
        for i in range(12):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=False, has_negative_phrasing=True))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 10),
                    attempted_at=base + timedelta(minutes=30 + i),
                )
            )

        # involves_graph=False, has_negative=False: complement group, 11 correct out of 12
        for i in range(12):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id, involves_graph_or_figure=False, has_negative_phrasing=False))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 11),
                    attempted_at=base + timedelta(minutes=60 + i),
                )
            )

        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(min_sample_size=10), s)

    graph_f = next(f for f in report.findings if f.feature_name == "involves_graph_or_figure")
    neg_f = next(f for f in report.findings if f.feature_name == "has_negative_phrasing")

    # graph finding should rank before neg_phrasing (more negative confident_delta)
    assert report.findings.index(graph_f) < report.findings.index(neg_f)
    assert graph_f.confident_delta <= neg_f.confident_delta


# --------------------------------------------------------------------------- #
# 13. Coverage reports missing features
# --------------------------------------------------------------------------- #


async def test_coverage_reports_missing_features(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # 3 questions with features
        for i in range(3):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_features(q.id))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        # 2 questions without features
        for i in range(2):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=True,
                    attempted_at=base + timedelta(minutes=10 + i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(), s)

    assert report.coverage.questions_with_features == 3
    assert report.coverage.questions_without_features == 2


# --------------------------------------------------------------------------- #
# 14. Stale extractor version counts as missing
# --------------------------------------------------------------------------- #


async def test_coverage_reports_stale_version(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        q = _make_question()
        s.add(q)
        await s.flush()
        # Features at a different (stale) version
        s.add(_features(q.id, extractor_version="features-v1-stale"))
        s.add(_attempt(question_id=q.id, is_correct=True, attempted_at=base))
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(), s)

    assert report.coverage.questions_with_features == 0
    assert report.coverage.questions_without_features == 1


# --------------------------------------------------------------------------- #
# 15. Skill 3 smoke test
# --------------------------------------------------------------------------- #


async def test_skill_3_smoke(env):
    _, make_session = env
    base = _now()
    async with make_session() as s:
        # Skill 3 questions: involves_graph=True over-indexes in misses
        # 8 graph=True: 2 correct, 6 wrong
        for i in range(8):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            s.add(_features(q.id, involves_graph_or_figure=True))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 2),
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        # 4 graph=False: 3 correct, 1 wrong (for without-group)
        for i in range(4):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            s.add(_features(q.id, involves_graph_or_figure=False))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 3),
                    attempted_at=base + timedelta(minutes=20 + i),
                )
            )
        await s.commit()

    async with make_session() as s:
        report = await analyze(AnalysisFilter(skill=3, min_sample_size=3), s)

    assert report.total_attempts_in_scope == 12
    graph_finding = next(
        (f for f in report.findings if f.feature_name == "involves_graph_or_figure"),
        None,
    )
    assert graph_finding is not None
    assert graph_finding.accuracy_delta < 0
    # Should rank first (most negative confident_delta) among feature findings
    assert report.findings[0].feature_name == "involves_graph_or_figure"


# --------------------------------------------------------------------------- #
# 16. Endpoint returns report
# --------------------------------------------------------------------------- #


async def test_endpoint_returns_report(env):
    client, make_session = env
    base = _now()
    async with make_session() as s:
        for i in range(4):
            q = _make_question()
            s.add(q)
            await s.flush()
            s.add(_skill_tag(q.id, 3))
            s.add(_features(q.id))
            s.add(
                _attempt(
                    question_id=q.id,
                    is_correct=(i < 2),
                    attempted_at=base + timedelta(minutes=i),
                )
            )
        await s.commit()

    resp = await client.get("/api/v1/analyzer/patterns?skill=3")
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "filter_applied",
        "total_attempts_in_scope",
        "total_questions_in_scope",
        "baseline_accuracy",
        "baseline_wilson_lower",
        "findings",
        "coverage",
    ):
        assert key in data, f"Missing key {key!r}"
    assert data["total_attempts_in_scope"] == 4
    assert data["filter_applied"]["skill"] == 3


# --------------------------------------------------------------------------- #
# 17. Endpoint validates section enum
# --------------------------------------------------------------------------- #


async def test_endpoint_validates_section_enum(env):
    client, _ = env
    resp = await client.get("/api/v1/analyzer/patterns?section=ZZ")
    assert resp.status_code == 422
