"""Tests for the LLM feature extractor (Ticket 4.2).

Anthropic SDK is mocked at the boundary. The orchestrator tests use the real
test Postgres DB (transactional savepoint fixture) but with a mocked SDK.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory
from app.services.analyzer import (
    CARS_SKIPPED_REASON,
    extract_features_for_question,
)
from app.services.analyzer.cache import FeatureExtractorCache
from app.services.analyzer.feature_extractor import (
    EXTRACTOR_VERSION,
    extract_judgment_features,
    make_features_cache_key,
)
from app.services.analyzer.mechanical_features import (
    MechanicalFeatures,
    compute_mechanical_features,
)


# --------------------------------------------------------------------------- #
# Helpers — forge SDK responses and minimal Question stand-ins
# --------------------------------------------------------------------------- #


def _make_question_stub(
    *,
    qid: str = "Q1",
    stem: str = "When a 5 kg box slides 2 m...",
    explanation: str | None = "Work = force * distance.",
    passage_id: int | None = None,
    choices: list | None = None,
    correct: str = "B",
):
    return SimpleNamespace(
        qid=qid,
        passage_id=passage_id,
        stem_plain=stem,
        stem_html=f"<p>{stem}</p>",
        explanation_plain=explanation,
        explanation_html=f"<p>{explanation}</p>" if explanation else None,
        choices=choices
        or [
            {"key": "A", "html": "<p>5J</p>", "plain": "5J", "media_ids": []},
            {"key": "B", "html": "<p>10J</p>", "plain": "10J", "media_ids": []},
            {"key": "C", "html": "<p>15J</p>", "plain": "15J", "media_ids": []},
            {"key": "D", "html": "<p>20J</p>", "plain": "20J", "media_ids": []},
        ],
        correct_choice=correct,
    )


def _tool_use_block(**input_data):
    from anthropic.types import ToolUseBlock

    return ToolUseBlock(
        id="toolu_1",
        name="submit_question_features",
        input=input_data,
        type="tool_use",
    )


def _forge_message(
    *,
    tool_input: dict,
    input_tokens: int = 1200,
    output_tokens: int = 180,
    cache_read: int = 0,
    cache_create: int = 0,
):
    content = [_tool_use_block(**tool_input)]
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )
    return SimpleNamespace(content=content, usage=usage)


def _make_client(message):
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return client


def _good_tool_input(**overrides):
    base = {
        "reasoning_type": "application",
        "requires_calculation": True,
        "calculation_steps": 2,
        "passage_type": "",  # discrete by default
        "distractor_difficulty": "medium",
        "trap_distractor_present": True,
        "common_misconception": "confuses work with kinetic energy",
        "jargon_density": "low",
        "key_concept_summary": "Tests application of W=Fd in a sliding-box scenario.",
        "involves_graph_or_figure": False,
        "involves_data_table": False,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# 9. Happy path: LLM returns good features
# --------------------------------------------------------------------------- #


async def test_extract_judgment_features_with_mocked_llm(tmp_path: Path):
    q = _make_question_stub()
    mech = compute_mechanical_features(q, passage=None)
    client = _make_client(_forge_message(tool_input=_good_tool_input()))
    cache = FeatureExtractorCache(tmp_path / "fc.db")

    try:
        result = await extract_judgment_features(
            q, passage=None, mechanical=mech, anthropic_client=client, cache=cache
        )
    finally:
        cache.close()

    assert result.cache_hit is False
    assert result.features.reasoning_type == "application"
    assert result.features.requires_calculation is True
    assert result.features.calculation_steps == 2
    assert result.features.passage_type is None  # discrete forces None
    assert result.features.distractor_difficulty == "medium"
    assert result.features.trap_distractor_present is True
    assert result.features.common_misconception == "confuses work with kinetic energy"
    assert result.features.jargon_density == "low"
    assert "Tests application of W=Fd" in result.features.key_concept_summary
    assert result.estimated_cost_usd > 0
    assert result.input_tokens == 1200
    assert result.output_tokens == 180


# --------------------------------------------------------------------------- #
# 10. Cache hit returns immediately, SDK not re-called
# --------------------------------------------------------------------------- #


async def test_extract_judgment_features_caches_results(tmp_path: Path):
    q = _make_question_stub()
    mech = compute_mechanical_features(q, passage=None)
    client = _make_client(_forge_message(tool_input=_good_tool_input()))
    cache = FeatureExtractorCache(tmp_path / "fc.db")

    try:
        a = await extract_judgment_features(
            q, passage=None, mechanical=mech, anthropic_client=client, cache=cache
        )
        b = await extract_judgment_features(
            q, passage=None, mechanical=mech, anthropic_client=client, cache=cache
        )
    finally:
        cache.close()

    assert a.cache_hit is False
    assert b.cache_hit is True
    assert b.estimated_cost_usd == 0.0
    assert b.cost_saved_usd > 0
    assert client.messages.create.await_count == 1


# --------------------------------------------------------------------------- #
# 11. Version bump invalidates cache entry
# --------------------------------------------------------------------------- #


async def test_extract_judgment_features_invalidates_on_version_bump(tmp_path: Path):
    q = _make_question_stub()
    mech = compute_mechanical_features(q, passage=None)
    client = _make_client(_forge_message(tool_input=_good_tool_input()))
    cache = FeatureExtractorCache(tmp_path / "fc.db")

    try:
        await extract_judgment_features(
            q,
            passage=None,
            mechanical=mech,
            anthropic_client=client,
            cache=cache,
            extractor_version="features-v1",
        )
        miss = await extract_judgment_features(
            q,
            passage=None,
            mechanical=mech,
            anthropic_client=client,
            cache=cache,
            extractor_version="features-v2",
        )
    finally:
        cache.close()

    assert miss.cache_hit is False
    assert client.messages.create.await_count == 2


# --------------------------------------------------------------------------- #
# 12. passage_type forced to None when discrete
# --------------------------------------------------------------------------- #


async def test_passage_type_forced_null_on_discrete(tmp_path: Path):
    q = _make_question_stub(passage_id=None)
    mech = compute_mechanical_features(q, passage=None)
    client = _make_client(_forge_message(tool_input=_good_tool_input(passage_type="experimental")))

    result = await extract_judgment_features(
        q, passage=None, mechanical=mech, anthropic_client=client, cache=None
    )
    assert mech.question_format == "discrete"
    assert result.features.passage_type is None
    assert any("forced to None" in w for w in result.parse_warnings)


# --------------------------------------------------------------------------- #
# 13. calculation_steps clamped when requires_calculation=False
# --------------------------------------------------------------------------- #


async def test_calculation_steps_clamped_when_requires_calc_false(tmp_path: Path):
    q = _make_question_stub()
    mech = compute_mechanical_features(q, passage=None)
    client = _make_client(
        _forge_message(tool_input=_good_tool_input(requires_calculation=False, calculation_steps=4))
    )

    result = await extract_judgment_features(
        q, passage=None, mechanical=mech, anthropic_client=client, cache=None
    )
    assert result.features.requires_calculation is False
    assert result.features.calculation_steps == 0
    assert any("clamped to 0" in w for w in result.parse_warnings)


# --------------------------------------------------------------------------- #
# 14. common_misconception "" becomes None
# --------------------------------------------------------------------------- #


async def test_common_misconception_empty_string_becomes_none(tmp_path: Path):
    q = _make_question_stub()
    mech = compute_mechanical_features(q, passage=None)
    client = _make_client(_forge_message(tool_input=_good_tool_input(common_misconception="")))

    result = await extract_judgment_features(
        q, passage=None, mechanical=mech, anthropic_client=client, cache=None
    )
    assert result.features.common_misconception is None


# --------------------------------------------------------------------------- #
# Orchestrator (real DB, mocked SDK)
# --------------------------------------------------------------------------- #


@pytest.fixture
async def tx_session(seeded_report, test_engine):
    conn = await test_engine.connect()
    await conn.begin()
    session = AsyncSession(
        bind=conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        await session.close()
        await conn.rollback()
        await conn.close()


def _rand_qid() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


async def _make_db_question(
    session: AsyncSession, *, stem: str = "stem", passage_id: int | None = None
) -> Question:
    q = Question(
        qid=_rand_qid(),
        passage_id=passage_id,
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


# --------------------------------------------------------------------------- #
# 15. Orchestrator UPSERT semantics
# --------------------------------------------------------------------------- #


async def test_extract_features_for_question_upserts_row(tx_session: AsyncSession):
    q = await _make_db_question(tx_session, stem="Sliding box")
    q_id = q.id

    client_a = _make_client(
        _forge_message(
            tool_input=_good_tool_input(
                reasoning_type="application",
                distractor_difficulty="low",
                key_concept_summary="Tests application of W=Fd v1.",
            )
        )
    )
    res_a = await extract_features_for_question(
        q_id, tx_session, anthropic_client=client_a, cache=None
    )
    assert res_a.persisted is True
    assert res_a.skipped_reason is None

    row_a = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q_id)
        )
    ).scalar_one()
    assert row_a.reasoning_type == "application"
    assert row_a.distractor_difficulty == "low"
    first_row_id = row_a.id

    client_b = _make_client(
        _forge_message(
            tool_input=_good_tool_input(
                reasoning_type="analysis",
                distractor_difficulty="high",
                key_concept_summary="Tests analysis of energy transfer v2.",
            )
        )
    )
    res_b = await extract_features_for_question(
        q_id, tx_session, anthropic_client=client_b, cache=None
    )
    assert res_b.persisted is True

    rows = (
        (
            await tx_session.execute(
                select(QuestionFeatures).where(QuestionFeatures.question_id == q_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].id == first_row_id  # same row, UPSERT in place
    tx_session.expire_all()
    fresh = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q_id)
        )
    ).scalar_one()
    assert fresh.reasoning_type == "analysis"
    assert fresh.distractor_difficulty == "high"
    assert "v2" in fresh.key_concept_summary


# --------------------------------------------------------------------------- #
# 16. Orchestrator skips CARS questions
# --------------------------------------------------------------------------- #


async def _get_cars_cc_id(session: AsyncSession) -> int:
    row = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == "CARS"))
    ).scalar_one()
    return row.id


async def test_extract_features_for_question_skips_cars(tx_session: AsyncSession):
    from decimal import Decimal

    q = await _make_db_question(tx_session, stem="A reading passage discussion")
    cars_cc_id = await _get_cars_cc_id(tx_session)

    tx_session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cars_cc_id,
            confidence=Decimal("1.00"),
            source="manual",
        )
    )
    await tx_session.flush()

    client = _make_client(_forge_message(tool_input=_good_tool_input()))
    result = await extract_features_for_question(
        q.id, tx_session, anthropic_client=client, cache=None
    )

    assert result.persisted is False
    assert result.skipped_reason == CARS_SKIPPED_REASON
    assert result.mechanical is None
    assert result.features is None
    client.messages.create.assert_not_called()

    row = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q.id)
        )
    ).scalar_one_or_none()
    assert row is None


# --------------------------------------------------------------------------- #
# 17. Orchestrator persists every column
# --------------------------------------------------------------------------- #


async def test_extract_features_for_question_persists_all_columns(
    tx_session: AsyncSession,
):
    q = await _make_db_question(tx_session, stem="A box NOT in motion EXCEPT...")
    q_id = q.id

    client = _make_client(
        _forge_message(
            tool_input=_good_tool_input(
                reasoning_type="inference",
                requires_calculation=False,
                calculation_steps=0,
                distractor_difficulty="high",
                trap_distractor_present=True,
                common_misconception="confuses static and kinetic friction",
                jargon_density="high",
                key_concept_summary="Tests inference about frictional forces in a scenario with a static box.",
            )
        )
    )

    res = await extract_features_for_question(q_id, tx_session, anthropic_client=client, cache=None)
    assert res.persisted is True

    tx_session.expire_all()
    row = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q_id)
        )
    ).scalar_one()

    # Mechanical columns
    assert row.question_format == "discrete"
    assert row.has_negative_phrasing is True
    assert row.involves_graph_or_figure is False
    assert row.involves_data_table is False
    assert row.passage_length_bucket is None

    # Judgment columns
    assert row.reasoning_type == "inference"
    assert row.requires_calculation is False
    assert row.calculation_steps == 0
    assert row.passage_type is None
    assert row.distractor_difficulty == "high"
    assert row.trap_distractor_present is True
    assert row.common_misconception == "confuses static and kinetic friction"
    assert row.jargon_density == "high"
    assert row.key_concept_summary.startswith("Tests inference")

    # Metadata
    assert row.extractor_version == EXTRACTOR_VERSION
    assert row.extracted_at is not None


# --------------------------------------------------------------------------- #
# Cache key sanity (cheap)
# --------------------------------------------------------------------------- #


def test_cache_key_changes_when_mechanical_changes():
    q = _make_question_stub()
    mech_a = compute_mechanical_features(q, passage=None)
    mech_b = MechanicalFeatures(
        question_format=mech_a.question_format,
        has_negative_phrasing=not mech_a.has_negative_phrasing,
        passage_length_bucket=mech_a.passage_length_bucket,
    )
    a = make_features_cache_key(q.stem_plain, q.explanation_plain, None, mech_a, "m1")
    b = make_features_cache_key(q.stem_plain, q.explanation_plain, None, mech_b, "m1")
    assert a != b


# --------------------------------------------------------------------------- #
# 18. LLM judgment overrides naive HTML scan for involves_graph_or_figure
# --------------------------------------------------------------------------- #


async def test_involves_graph_or_figure_from_llm_judgment_persists(
    tx_session: AsyncSession,
):
    # Question stem has an <img> tag — old mechanical regex would have flagged True.
    # LLM mock returns False — prove the persisted row honours LLM judgment.
    q = await _make_db_question(
        tx_session,
        stem="Which enzyme is depicted <img src='icon.svg'>?",
    )
    q_id = q.id

    client = _make_client(
        _forge_message(
            tool_input=_good_tool_input(
                involves_graph_or_figure=False,
                involves_data_table=False,
            )
        )
    )
    res = await extract_features_for_question(q_id, tx_session, anthropic_client=client, cache=None)

    assert res.persisted is True
    tx_session.expire_all()
    row = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q_id)
        )
    ).scalar_one()
    assert row.involves_graph_or_figure is False
