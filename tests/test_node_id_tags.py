"""T2 — canonical node_id tag shape on question_tags + anki_note_tags.

V-T1 (sole target = node_id), V-T2 (source enum + llm re-run preserves
manual/schema_map), V-T3 (confidence required iff llm; <0.5 ⇒ manual_review).

Self-contained: creates only the FK-related tables on its own engine, so the
suite-wide create_all (still referencing pre-T12 outline FKs elsewhere) is
not needed.
"""

import os

import asyncpg
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import Base
from app.models.anki import AnkiNote, AnkiNoteTag
from app.models.captures import Passage, Question, QuestionTag
from app.models.outline import Course, OutlineNode

_HOST_PORT = os.environ.get("HOST_POSTGRES_PORT", "5432")
_DB_URL = f"postgresql+asyncpg://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient_test"
_ADMIN_DSN = f"postgresql://gradient:gradient_secret@localhost:{_HOST_PORT}/gradient"

_TABLES = [
    Course.__table__,
    OutlineNode.__table__,
    Passage.__table__,
    Question.__table__,
    QuestionTag.__table__,
    AnkiNote.__table__,
    AnkiNoteTag.__table__,
]


@pytest.fixture
async def engine():
    conn = await asyncpg.connect(_ADMIN_DSN)
    try:
        await conn.execute("CREATE DATABASE gradient_test")
    except asyncpg.exceptions.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()

    eng = create_async_engine(_DB_URL)
    # Clean slate (cf conftest): the shared test DB may carry tables from a
    # prior full create_all whose FKs block a subset drop.
    async with eng.begin() as c:
        await c.execute(text("DROP SCHEMA public CASCADE"))
        await c.execute(text("CREATE SCHEMA public"))
        await c.run_sync(Base.metadata.create_all, tables=_TABLES)
    yield eng
    await eng.dispose()


async def _seed(eng) -> tuple[int, int, int]:
    """One course+node, one question, one anki note. Returns (q_id, node_id, note_id)."""
    async with AsyncSession(eng) as s:
        course = Course(slug="c", name="C")
        s.add(course)
        await s.flush()
        node = OutlineNode(
            course_id=course.id, parent_id=None, kind="cc", name="N", depth=0, position=0
        )
        s.add(node)
        await s.flush()
        q = Question(qid="q1", stem_html="<p>q</p>", stem_plain="q", choices=[], correct_choice="A")
        s.add(q)
        note = AnkiNote(note_id=1234567890123)
        s.add(note)
        await s.flush()
        ids = (q.id, node.id, note.note_id)
        await s.commit()
        return ids


# ── V-T1: node_id is the sole tag target ────────────────────────────────────


async def test_question_tag_targets_node_id(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(QuestionTag(question_id=qid, node_id=nid, source="manual", confidence=None))
        await s.flush()
        await s.commit()
    # The retired 3-target columns no longer exist on the table.
    async with AsyncSession(engine) as s:
        cols = (
            (
                await s.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'question_tags'"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert "node_id" in cols
    assert {"topic_id", "content_category_id", "skill"}.isdisjoint(cols)


# ── V-T2: source enum + llm re-run preserves manual / schema_map ─────────────


async def test_bad_source_rejected(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(QuestionTag(question_id=qid, node_id=nid, source="uworld_map", confidence=None))
        with pytest.raises(IntegrityError):
            await s.flush()


async def test_llm_rerun_preserves_manual_and_schema_map(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add_all(
            [
                QuestionTag(question_id=qid, node_id=nid, source="schema_map", confidence=None),
                QuestionTag(question_id=qid, node_id=nid, source="manual", confidence=None),
                QuestionTag(question_id=qid, node_id=nid, source="llm", confidence=0.9),
            ]
        )
        await s.commit()
    # Re-run pattern (V-T2): DELETE source='llm'; INSERT new.
    async with AsyncSession(engine) as s:
        await s.execute(
            text("DELETE FROM question_tags WHERE question_id = :q AND source = 'llm'"),
            {"q": qid},
        )
        await s.commit()
    async with AsyncSession(engine) as s:
        rows = (
            (
                await s.execute(
                    text("SELECT source FROM question_tags WHERE question_id = :q ORDER BY source"),
                    {"q": qid},
                )
            )
            .scalars()
            .all()
        )
    assert rows == ["manual", "schema_map"]


async def test_uq_question_node_source(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(QuestionTag(question_id=qid, node_id=nid, source="manual", confidence=None))
        await s.flush()
        s.add(QuestionTag(question_id=qid, node_id=nid, source="manual", confidence=None))
        with pytest.raises(IntegrityError):
            await s.flush()


# ── V-T3: confidence required iff llm; <0.5 ⇒ manual_review ──────────────────


async def test_llm_requires_confidence(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(QuestionTag(question_id=qid, node_id=nid, source="llm", confidence=None))
        with pytest.raises(IntegrityError):
            await s.flush()


async def test_schema_map_confidence_must_be_null(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(QuestionTag(question_id=qid, node_id=nid, source="schema_map", confidence=0.9))
        with pytest.raises(IntegrityError):
            await s.flush()


async def test_low_confidence_requires_manual_review(engine):
    qid, nid, _ = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(
            QuestionTag(
                question_id=qid, node_id=nid, source="llm", confidence=0.3, manual_review=False
            )
        )
        with pytest.raises(IntegrityError):
            await s.flush()
    async with AsyncSession(engine) as s:
        s.add(
            QuestionTag(
                question_id=qid, node_id=nid, source="llm", confidence=0.3, manual_review=True
            )
        )
        await s.flush()
        await s.commit()


# ── anki_note_tags: same canonical shape, node_id NULL-able ──────────────────


async def test_anki_note_tag_node_id_nullable_and_canonical(engine):
    _, nid, note_id = await _seed(engine)
    async with AsyncSession(engine) as s:
        # Unparsed tag → no node.
        s.add(
            AnkiNoteTag(
                note_id=note_id,
                tag_raw="#AK::unparsed",
                node_id=None,
                parsed_kind="unparsed",
                source="schema_map",
                confidence=None,
            )
        )
        # Resolved tag → node, llm source needs confidence.
        s.add(
            AnkiNoteTag(
                note_id=note_id,
                tag_raw="#AK::resolved",
                node_id=nid,
                parsed_kind="resolved",
                source="llm",
                confidence=0.8,
            )
        )
        await s.flush()
        await s.commit()
    async with AsyncSession(engine) as s:
        n = (
            (await s.execute(select(AnkiNoteTag).where(AnkiNoteTag.note_id == note_id)))
            .scalars()
            .all()
        )
    assert len(n) == 2


async def test_anki_note_tag_bad_source_rejected(engine):
    _, nid, note_id = await _seed(engine)
    async with AsyncSession(engine) as s:
        s.add(
            AnkiNoteTag(
                note_id=note_id,
                tag_raw="#AK::regex",
                node_id=nid,
                parsed_kind="resolved",
                source="regex",  # retired enum value
                confidence=None,
            )
        )
        with pytest.raises(IntegrityError):
            await s.flush()
