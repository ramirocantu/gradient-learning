"""Schema-level tests for Ticket 2.1 captures tables.

Tests run against the real test Postgres DB (see conftest). Each test uses a
nested-savepoint fixture so its writes are rolled back at teardown — keeps
tests independent of one another and of the seeded outline.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import (
    Attempt,
    Passage,
    Question,
    QuestionTag,
    RawCapture,
)
from app.models.media import Media
from app.models.outline import ContentCategory


_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"
_BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
async def tx_session(seeded_report, test_engine):
    """Per-test session bound to a connection with an outer transaction
    that is always rolled back at teardown. Session commits become savepoint
    releases via join_transaction_mode='create_savepoint'.
    """
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


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _rand_qid() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


def _rand_hash() -> str:
    return uuid.uuid4().hex


async def _make_question(session: AsyncSession, **overrides) -> Question:
    q = Question(
        qid=overrides.get("qid", _rand_qid()),
        passage_id=overrides.get("passage_id"),
        stem_html=overrides.get("stem_html", "<p>stem</p>"),
        stem_plain=overrides.get("stem_plain", "stem"),
        choices=overrides.get(
            "choices",
            [
                {"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": []},
                {"key": "B", "html": "<p>b</p>", "plain": "b", "media_ids": []},
            ],
        ),
        correct_choice=overrides.get("correct_choice", "A"),
        explanation_html=overrides.get("explanation_html"),
        explanation_plain=overrides.get("explanation_plain"),
        uworld_aamc_tags=overrides.get("uworld_aamc_tags"),
    )
    session.add(q)
    await session.flush()
    return q


async def _get_cars_cc_id(session: AsyncSession) -> int:
    row = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == "CARS"))
    ).scalar_one()
    return row.id


# --------------------------------------------------------------------------- #
# 1. Migrations
# --------------------------------------------------------------------------- #


async def test_migrations_apply_and_rollback():
    db_name = "gradient_migrate_test"
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
# 2. raw_captures
# --------------------------------------------------------------------------- #


async def test_raw_capture_round_trip(tx_session: AsyncSession):
    warnings = [
        {
            "code": "missing_selector",
            "message": "stem not found",
            "selector": ".q-stem",
        },
        {"code": "no_explanation", "message": "explanation block absent"},
    ]
    rc = RawCapture(
        qid="q123",
        captured_at=_now(),
        raw_html="<html>...</html>",
        raw_json={"foo": "bar"},
        parse_warnings=warnings,
        extension_version="0.1.0",
    )
    tx_session.add(rc)
    await tx_session.flush()

    fetched = (
        await tx_session.execute(select(RawCapture).where(RawCapture.id == rc.id))
    ).scalar_one()
    assert fetched.parse_warnings == warnings
    assert fetched.source == "uworld"
    assert fetched.raw_json == {"foo": "bar"}


# --------------------------------------------------------------------------- #
# 3. questions.qid uniqueness
# --------------------------------------------------------------------------- #


async def test_question_qid_is_unique(tx_session: AsyncSession):
    qid = _rand_qid()
    await _make_question(tx_session, qid=qid)
    await tx_session.commit()  # savepoint release

    with pytest.raises(IntegrityError):
        await _make_question(tx_session, qid=qid)


# --------------------------------------------------------------------------- #
# 4. passages content_hash dedupe
# --------------------------------------------------------------------------- #


async def test_passage_content_hash_dedupe(tx_session: AsyncSession):
    h = _rand_hash()
    tx_session.add(Passage(content_hash=h, html="<p>a</p>", plain_text="a"))
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(Passage(content_hash=h, html="<p>b</p>", plain_text="b"))
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 5. passages partial unique on uworld_passage_id
# --------------------------------------------------------------------------- #


async def test_passage_uworld_id_dedupe(tx_session: AsyncSession):
    uw_id = f"uw-{uuid.uuid4().hex[:8]}"
    tx_session.add(
        Passage(
            uworld_passage_id=uw_id,
            content_hash=_rand_hash(),
            html="<p>1</p>",
            plain_text="1",
        )
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            Passage(
                uworld_passage_id=uw_id,
                content_hash=_rand_hash(),
                html="<p>2</p>",
                plain_text="2",
            )
        )
        await tx_session.flush()
    await tx_session.rollback()

    # Multiple NULLs allowed.
    for _ in range(3):
        tx_session.add(
            Passage(
                uworld_passage_id=None,
                content_hash=_rand_hash(),
                html="<p>x</p>",
                plain_text="x",
            )
        )
    await tx_session.flush()


# --------------------------------------------------------------------------- #
# 6. ON DELETE SET NULL on questions.passage_id
# --------------------------------------------------------------------------- #


async def test_question_passage_fk_set_null_on_delete(tx_session: AsyncSession):
    p = Passage(content_hash=_rand_hash(), html="<p>p</p>", plain_text="p")
    tx_session.add(p)
    await tx_session.flush()

    q = await _make_question(tx_session, passage_id=p.id)
    assert q.passage_id == p.id

    await tx_session.delete(p)
    await tx_session.flush()
    await tx_session.refresh(q)
    assert q.passage_id is None


# --------------------------------------------------------------------------- #
# 7. choices JSONB round-trip with media_ids
# --------------------------------------------------------------------------- #


async def test_choices_round_trip_with_media_ids(tx_session: AsyncSession):
    choices = [
        {"key": "A", "html": "<p>a</p>", "plain": "a", "media_ids": [1, 2]},
        {"key": "B", "html": "<p>b</p>", "plain": "b", "media_ids": []},
        {"key": "C", "html": "<p>c</p>", "plain": "c", "media_ids": [3]},
    ]
    q = await _make_question(tx_session, choices=choices, correct_choice="B")
    qid = q.qid
    tx_session.expire_all()
    fetched = (await tx_session.execute(select(Question).where(Question.qid == qid))).scalar_one()
    assert fetched.choices == choices
    assert [c["key"] for c in fetched.choices] == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# 8. attempts CASCADE on question delete
# --------------------------------------------------------------------------- #


async def test_attempts_cascade_on_question_delete(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    for _ in range(3):
        tx_session.add(
            Attempt(
                question_id=q.id,
                attempted_at=_now(),
                selected_choice="A",
                is_correct=True,
            )
        )
    await tx_session.flush()

    q_id = q.id
    await tx_session.delete(q)
    await tx_session.flush()

    remaining = (
        (await tx_session.execute(select(Attempt).where(Attempt.question_id == q_id)))
        .scalars()
        .all()
    )
    assert remaining == []


# --------------------------------------------------------------------------- #
# 9. question_tags exactly-one-target CHECK
# --------------------------------------------------------------------------- #


async def test_question_tag_exactly_one_target(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    cars_cc_id = await _get_cars_cc_id(tx_session)
    q_id = q.id
    await tx_session.commit()

    from app.models.outline import Topic

    a_topic_id = (await tx_session.execute(select(Topic.id).limit(1))).scalar_one()

    # (a) all-NULL fails
    with pytest.raises(IntegrityError):
        tx_session.add(
            QuestionTag(
                question_id=q_id,
                topic_id=None,
                content_category_id=None,
                skill=None,
                confidence=Decimal("1.00"),
                source="manual",
            )
        )
        await tx_session.flush()
    await tx_session.rollback()

    # (b) two non-null fails
    with pytest.raises(IntegrityError):
        tx_session.add(
            QuestionTag(
                question_id=q_id,
                topic_id=None,
                content_category_id=cars_cc_id,
                skill=2,
                confidence=Decimal("1.00"),
                source="manual",
            )
        )
        await tx_session.flush()
    await tx_session.rollback()

    # (c) each single-target form succeeds
    tx_session.add(
        QuestionTag(
            question_id=q_id,
            skill=3,
            confidence=Decimal("1.00"),
            source="manual",
        )
    )
    await tx_session.flush()

    tx_session.add(
        QuestionTag(
            question_id=q_id,
            content_category_id=cars_cc_id,
            confidence=Decimal("0.90"),
            source="manual",
        )
    )
    await tx_session.flush()

    tx_session.add(
        QuestionTag(
            question_id=q_id,
            topic_id=a_topic_id,
            confidence=Decimal("0.80"),
            source="manual",
        )
    )
    await tx_session.flush()


# --------------------------------------------------------------------------- #
# 10. question_tags UNIQUE NULLS NOT DISTINCT
# --------------------------------------------------------------------------- #


async def test_question_tag_unique_nulls_not_distinct(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    cars_cc_id = await _get_cars_cc_id(tx_session)

    tx_session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cars_cc_id,
            confidence=Decimal("1.00"),
            source="uworld_map",
        )
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cars_cc_id,
                confidence=Decimal("0.50"),
                source="uworld_map",
            )
        )
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 11. question_tags skill range CHECK
# --------------------------------------------------------------------------- #


async def test_question_tag_skill_range(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    q_id = q.id
    await tx_session.commit()

    for bad in (0, 5):
        with pytest.raises(IntegrityError):
            tx_session.add(
                QuestionTag(
                    question_id=q_id,
                    skill=bad,
                    confidence=Decimal("1.00"),
                    source="manual",
                )
            )
            await tx_session.flush()
        await tx_session.rollback()

    for good in (1, 2, 3, 4):
        tx_session.add(
            QuestionTag(
                question_id=q_id,
                skill=good,
                confidence=Decimal("1.00"),
                source="manual",
            )
        )
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 12. three sources, same target — all succeed
# --------------------------------------------------------------------------- #


async def test_question_tag_three_sources_same_target(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    cars_cc_id = await _get_cars_cc_id(tx_session)

    for src in ("uworld_map", "llm", "manual"):
        tx_session.add(
            QuestionTag(
                question_id=q.id,
                content_category_id=cars_cc_id,
                confidence=Decimal("1.00"),
                source=src,
            )
        )
    await tx_session.flush()

    rows = (
        (await tx_session.execute(select(QuestionTag).where(QuestionTag.question_id == q.id)))
        .scalars()
        .all()
    )
    assert {r.source for r in rows} == {"uworld_map", "llm", "manual"}


# --------------------------------------------------------------------------- #
# 13. CARS tag against the synthetic CC works
# --------------------------------------------------------------------------- #


async def test_cars_tag_targets_synthetic_cc(tx_session: AsyncSession):
    q = await _make_question(tx_session)
    cars_cc_id = await _get_cars_cc_id(tx_session)

    tx_session.add(
        QuestionTag(
            question_id=q.id,
            content_category_id=cars_cc_id,
            confidence=Decimal("1.00"),
            source="uworld_map",
        )
    )
    await tx_session.flush()

    row = (
        await tx_session.execute(select(QuestionTag).where(QuestionTag.question_id == q.id))
    ).scalar_one()
    assert row.content_category_id == cars_cc_id


# --------------------------------------------------------------------------- #
# 14. media content_hash unique
# --------------------------------------------------------------------------- #


async def test_media_content_hash_unique(tx_session: AsyncSession):
    h = _rand_hash()
    tx_session.add(
        Media(
            content_hash=h,
            local_path=f"{h[:2]}/{h}.png",
            mime_type="image/png",
            byte_size=1024,
        )
    )
    await tx_session.flush()
    await tx_session.commit()

    with pytest.raises(IntegrityError):
        tx_session.add(
            Media(
                content_hash=h,
                local_path=f"{h[:2]}/{h}-dup.png",
                mime_type="image/png",
                byte_size=2048,
            )
        )
        await tx_session.flush()


# --------------------------------------------------------------------------- #
# 15. media description columns default NULL
# --------------------------------------------------------------------------- #


async def test_media_description_columns_default_null(tx_session: AsyncSession):
    h = _rand_hash()
    m = Media(
        content_hash=h,
        local_path=f"{h[:2]}/{h}.png",
        mime_type="image/png",
        byte_size=512,
    )
    tx_session.add(m)
    await tx_session.flush()

    fetched = (await tx_session.execute(select(Media).where(Media.id == m.id))).scalar_one()
    assert fetched.description is None
    assert fetched.description_model is None
    assert fetched.described_at is None


# --------------------------------------------------------------------------- #
# 16. MEDIA_ROOT setting present, writeable
# --------------------------------------------------------------------------- #


def test_media_root_setting_exists_and_is_path(tmp_path):
    from app.config import ensure_media_root, settings

    assert isinstance(settings.MEDIA_ROOT, Path)
    ensure_media_root()
    assert settings.MEDIA_ROOT.exists()
    assert settings.MEDIA_ROOT.is_dir()
    probe = settings.MEDIA_ROOT / ".write_probe"
    try:
        probe.write_text("ok")
        assert probe.read_text() == "ok"
    finally:
        if probe.exists():
            probe.unlink()
