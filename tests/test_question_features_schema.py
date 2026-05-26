"""Schema-level tests for Ticket 4.1 question_features table.

Tests run against the real test Postgres DB (see conftest). Each test uses a
nested-savepoint fixture so its writes are rolled back at teardown.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question
from app.models.features import QuestionFeatures


_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"
_BACKEND_DIR = Path(__file__).resolve().parent.parent

_EXTRACTOR_VERSION = "v1-claude-haiku-4-5-test"


@pytest.fixture
async def tx_session(seeded_report, test_engine):
    """Per-test session bound to a connection with an outer transaction
    that is always rolled back at teardown."""
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


async def _make_question(session: AsyncSession) -> Question:
    q = Question(
        qid=_rand_qid(),
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[
            {"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": []},
            {"key": "B", "html": "<p>b</p>", "plain": "b", "media_ids": []},
        ],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()
    return q


def _features_kwargs(question_id: int, **overrides) -> dict:
    base = dict(
        question_id=question_id,
        question_format="passage_based",
        reasoning_type="application",
        requires_calculation=True,
        calculation_steps=3,
        involves_graph_or_figure=True,
        involves_data_table=False,
        has_negative_phrasing=False,
        passage_length_bucket="medium",
        passage_type="experimental",
        distractor_difficulty="high",
        trap_distractor_present=True,
        common_misconception="confuses Km with Vmax",
        jargon_density="medium",
        key_concept_summary="enzyme kinetics interpretation from a Lineweaver-Burk plot",
        extractor_version=_EXTRACTOR_VERSION,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# 1. Migration roundtrip
# --------------------------------------------------------------------------- #


async def test_migration_apply_and_rollback():
    db_name = "gradient_4_1_migrate_test"
    admin = await asyncpg.connect(_ADMIN_DSN)
    try:
        await admin.execute(
            f"""
            SELECT pg_terminate_backend(pid)
              FROM pg_stat_activity
             WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
            """
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    target_url = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/{db_name}"
    env = {**os.environ, "ALEMBIC_DATABASE_URL": target_url}

    try:
        for args in (
            ["alembic", "upgrade", "head"],
            ["alembic", "downgrade", "base"],
            ["alembic", "upgrade", "head"],
        ):
            r = subprocess.run(
                args,
                cwd=_BACKEND_DIR,
                env=env,
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, (
                f"alembic {' '.join(args[1:])} failed\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
    finally:
        admin = await asyncpg.connect(_ADMIN_DSN)
        try:
            await admin.execute(
                f"""
                SELECT pg_terminate_backend(pid)
                  FROM pg_stat_activity
                 WHERE datname = '{db_name}' AND pid <> pg_backend_pid();
                """
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        finally:
            await admin.close()


# --------------------------------------------------------------------------- #
# 2. Round-trip insert
# --------------------------------------------------------------------------- #


async def test_insert_round_trip(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    q_id = q.id
    qf = QuestionFeatures(**_features_kwargs(q_id))
    tx_session.add(qf)
    await tx_session.flush()
    qf_id = qf.id

    tx_session.expire_all()
    fetched = (
        await tx_session.execute(select(QuestionFeatures).where(QuestionFeatures.id == qf_id))
    ).scalar_one()
    assert fetched.question_id == q_id
    assert fetched.question_format == "passage_based"
    assert fetched.reasoning_type == "application"
    assert fetched.requires_calculation is True
    assert fetched.calculation_steps == 3
    assert fetched.involves_graph_or_figure is True
    assert fetched.involves_data_table is False
    assert fetched.has_negative_phrasing is False
    assert fetched.passage_length_bucket == "medium"
    assert fetched.passage_type == "experimental"
    assert fetched.distractor_difficulty == "high"
    assert fetched.trap_distractor_present is True
    assert fetched.common_misconception == "confuses Km with Vmax"
    assert fetched.jargon_density == "medium"
    assert (
        fetched.key_concept_summary == "enzyme kinetics interpretation from a Lineweaver-Burk plot"
    )
    assert fetched.extractor_version == _EXTRACTOR_VERSION
    assert fetched.extracted_at is not None


# --------------------------------------------------------------------------- #
# 3. UNIQUE on question_id
# --------------------------------------------------------------------------- #


async def test_question_id_unique_constraint(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    tx_session.add(QuestionFeatures(**_features_kwargs(q.id)))
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(QuestionFeatures(**_features_kwargs(q.id)))
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 4. ON DELETE CASCADE
# --------------------------------------------------------------------------- #


async def test_fk_cascade_on_question_delete(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    qf = QuestionFeatures(**_features_kwargs(q.id))
    tx_session.add(qf)
    await tx_session.flush()
    qf_id = qf.id
    q_id = q.id

    await tx_session.delete(q)
    await tx_session.flush()

    remaining = (
        await tx_session.execute(select(QuestionFeatures).where(QuestionFeatures.id == qf_id))
    ).scalar_one_or_none()
    assert remaining is None

    leftover = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q_id)
        )
    ).scalar_one_or_none()
    assert leftover is None


# --------------------------------------------------------------------------- #
# 5. Enum-shaped CHECK constraints
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("question_format", "invalid"),
        ("reasoning_type", "guessing"),
        ("passage_length_bucket", "huge"),
        ("passage_type", "narrative"),
        ("distractor_difficulty", "impossible"),
        ("jargon_density", "extreme"),
    ],
)
async def test_question_format_enum_check(tx_session: AsyncSession, field: str, bad_value: str):
    q = await _make_question(tx_session)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(QuestionFeatures(**_features_kwargs(q.id, **{field: bad_value})))
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 6. calculation_steps >= 0
# --------------------------------------------------------------------------- #


async def test_calculation_steps_nonneg_check(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            QuestionFeatures(
                **_features_kwargs(q.id, requires_calculation=False, calculation_steps=-1)
            )
        )
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 7. Passage fields NULL when discrete
# --------------------------------------------------------------------------- #


async def test_passage_fields_nullable_for_discrete(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    tx_session.add(
        QuestionFeatures(
            **_features_kwargs(
                q.id,
                question_format="discrete",
                passage_length_bucket=None,
                passage_type=None,
            )
        )
    )
    await tx_session.flush()


# --------------------------------------------------------------------------- #
# 8. Passage fields populated when passage_based
# --------------------------------------------------------------------------- #


async def test_passage_fields_populated_for_passage_based(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    tx_session.add(
        QuestionFeatures(
            **_features_kwargs(
                q.id,
                question_format="passage_based",
                passage_length_bucket="medium",
                passage_type="experimental",
            )
        )
    )
    await tx_session.flush()


# --------------------------------------------------------------------------- #
# 9. extractor_version NOT NULL
# --------------------------------------------------------------------------- #


async def test_extractor_version_required(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(QuestionFeatures(**_features_kwargs(q.id, extractor_version=None)))
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 10. common_misconception nullable
# --------------------------------------------------------------------------- #


async def test_common_misconception_nullable(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    tx_session.add(QuestionFeatures(**_features_kwargs(q.id, common_misconception=None)))
    await tx_session.flush()

    fetched = (
        await tx_session.execute(
            select(QuestionFeatures).where(QuestionFeatures.question_id == q.id)
        )
    ).scalar_one()
    assert fetched.common_misconception is None
