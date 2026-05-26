"""T30 — app/services/kb/persist_tags.py contract tests (V-T2, V-T3, V-E1, V-KB1).

V-T2: re-run pattern — DELETE source='llm' then INSERT; manual / schema_map
rows untouched.
V-T3: confidence required for source='llm'; calibrated <0.5 ⇒ manual_review,
persisted (surfaced) not silently dropped.
V-KB1: the new atomic_fact_tags substrate persists idempotently on re-run.

GroundedResult is constructed directly here — no LLM call — so these tests
exercise the persistence seam in isolation (the grounded/calibrator paths
are covered by test_grounded.py).
"""

from __future__ import annotations

import uuid as _uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question, QuestionTag
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.persist_tags import (
    EntityNotFoundError,
    persist_grounded_tags,
)
from app.services.llm.grounded import GroundedResult, GroundedTag

EXTRACTOR = "grounded-v1"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


async def _make_course_and_nodes(
    session: AsyncSession, *, n: int
) -> tuple[Course, list[OutlineNode]]:
    course = Course(slug=f"persist-{_uuid.uuid4().hex[:8]}", name="Persist")
    session.add(course)
    await session.flush()
    nodes: list[OutlineNode] = []
    for i in range(n):
        node = OutlineNode(
            course_id=course.id,
            parent_id=None,
            kind="concept",
            name=f"concept-{i}-{_uuid.uuid4().hex[:4]}",
            depth=0,
            position=i,
        )
        session.add(node)
        nodes.append(node)
    await session.flush()
    return course, nodes


async def _make_question(session: AsyncSession) -> Question:
    q = Question(
        source="uworld",
        qid=f"q-{_uuid.uuid4().hex[:10]}",
        stem_html="<p>stem</p>",
        stem_plain="stem",
        choices=[{"id": "A", "text": "x"}],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()
    return q


async def _make_atomic_fact(session: AsyncSession, course: Course) -> AtomicFact:
    pdf = PdfSource(
        course_id=course.id,
        filename="lecture.pdf",
        sha256=_uuid.uuid4().hex,
        status="ingested",
    )
    session.add(pdf)
    await session.flush()
    fact = AtomicFact(
        course_id=course.id,
        pdf_source_id=pdf.id,
        text="glycolysis nets 2 ATP",
        content_hash=_uuid.uuid4().hex,
    )
    session.add(fact)
    await session.flush()
    return fact


def _tag(node_id: int, *, conf: float, review: bool, rationale: str = "r") -> GroundedTag:
    return GroundedTag(
        node_id=node_id,
        path=None,
        candidate_index=1,
        via="embedding",
        rationale=rationale,
        calibrated_confidence=conf,
        manual_review=review,
    )


def _result(*tags: GroundedTag) -> GroundedResult:
    return GroundedResult(
        tags=list(tags),
        extractor_version=EXTRACTOR,
        model="gpt-4.1-mini",
        calibrator_model="gpt-4.1-mini",
    )


# --------------------------------------------------------------------------- #
# question path
# --------------------------------------------------------------------------- #


async def test_question_persist_creates_llm_rows(db_session: AsyncSession):
    _, nodes = await _make_course_and_nodes(db_session, n=2)
    q = await _make_question(db_session)
    res = _result(
        _tag(nodes[0].id, conf=0.91, review=False),
        _tag(nodes[1].id, conf=0.72, review=False),
    )

    out = await persist_grounded_tags(
        db_session, entity_kind="question", entity_id=q.id, result=res
    )

    assert out.persisted == 2
    assert out.replaced == 0
    assert out.manual_review_flagged == 0
    assert out.primary_node_id is None  # questions carry no denormalized node

    rows = (
        await db_session.execute(
            select(QuestionTag).where(QuestionTag.question_id == q.id)
        )
    ).scalars().all()
    assert len(rows) == 2
    for r in rows:
        assert r.source == "llm"
        assert r.extractor_version == EXTRACTOR
        assert r.confidence is not None
        assert r.manual_review is False


async def test_question_rerun_replaces_llm_preserves_manual(db_session: AsyncSession):
    """V-T2: a re-run deletes only source='llm' rows; manual survives."""
    _, nodes = await _make_course_and_nodes(db_session, n=2)
    q = await _make_question(db_session)

    # Seed a human manual tag (confidence NULL per V-T3) + an old llm tag.
    db_session.add(
        QuestionTag(question_id=q.id, node_id=nodes[0].id, source="manual", confidence=None)
    )
    db_session.add(
        QuestionTag(
            question_id=q.id,
            node_id=nodes[1].id,
            source="llm",
            confidence=0.60,
            extractor_version="old",
        )
    )
    await db_session.flush()

    res = _result(_tag(nodes[0].id, conf=0.95, review=False))
    out = await persist_grounded_tags(
        db_session, entity_kind="question", entity_id=q.id, result=res
    )

    assert out.replaced == 1   # the one prior llm row
    assert out.persisted == 1

    rows = (
        await db_session.execute(
            select(QuestionTag).where(QuestionTag.question_id == q.id)
        )
    ).scalars().all()
    by_source = sorted((r.source, r.node_id) for r in rows)
    # manual row preserved; single fresh llm row; old llm row gone.
    assert ("manual", nodes[0].id) in by_source
    llm_rows = [r for r in rows if r.source == "llm"]
    assert len(llm_rows) == 1
    assert llm_rows[0].node_id == nodes[0].id
    assert llm_rows[0].extractor_version == EXTRACTOR


async def test_low_confidence_persisted_with_manual_review(db_session: AsyncSession):
    """V-T3: calibrated <0.5 ⇒ manual_review, persisted not dropped."""
    _, nodes = await _make_course_and_nodes(db_session, n=1)
    q = await _make_question(db_session)
    res = _result(_tag(nodes[0].id, conf=0.31, review=True))

    out = await persist_grounded_tags(
        db_session, entity_kind="question", entity_id=q.id, result=res
    )

    assert out.persisted == 1
    assert out.manual_review_flagged == 1

    row = (
        await db_session.execute(
            select(QuestionTag).where(QuestionTag.question_id == q.id)
        )
    ).scalar_one()
    assert row.manual_review is True
    assert float(row.confidence) < 0.5  # surfaced for review, not discarded


# --------------------------------------------------------------------------- #
# atomic_fact path
# --------------------------------------------------------------------------- #


async def test_atomic_fact_persist_sets_primary_node(db_session: AsyncSession):
    course, nodes = await _make_course_and_nodes(db_session, n=2)
    fact = await _make_atomic_fact(db_session, course)
    res = _result(
        _tag(nodes[0].id, conf=0.70, review=False),
        _tag(nodes[1].id, conf=0.92, review=False),  # highest → primary
    )

    out = await persist_grounded_tags(
        db_session, entity_kind="atomic_fact", entity_id=fact.id, result=res
    )

    assert out.persisted == 2
    assert out.primary_node_id == nodes[1].id

    rows = (
        await db_session.execute(
            select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id == fact.id)
        )
    ).scalars().all()
    assert len(rows) == 2
    assert all(r.source == "llm" and r.extractor_version == EXTRACTOR for r in rows)

    await db_session.refresh(fact)
    assert fact.node_id == nodes[1].id  # denormalized primary


async def test_atomic_fact_low_conf_only_leaves_node_null(db_session: AsyncSession):
    """V-T3: low-conf fact tag persisted + flagged, but no auto primary node."""
    course, nodes = await _make_course_and_nodes(db_session, n=1)
    fact = await _make_atomic_fact(db_session, course)
    res = _result(_tag(nodes[0].id, conf=0.20, review=True))

    out = await persist_grounded_tags(
        db_session, entity_kind="atomic_fact", entity_id=fact.id, result=res
    )

    assert out.persisted == 1
    assert out.manual_review_flagged == 1
    assert out.primary_node_id is None

    row = (
        await db_session.execute(
            select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id == fact.id)
        )
    ).scalar_one()
    assert row.manual_review is True
    await db_session.refresh(fact)
    assert fact.node_id is None  # ⊥ auto-assign a low-confidence node


# --------------------------------------------------------------------------- #
# V-KB1 idempotency + guards
# --------------------------------------------------------------------------- #


async def test_rerun_idempotent(db_session: AsyncSession):
    """V-KB1: persisting the same result twice leaves a stable row set."""
    course, nodes = await _make_course_and_nodes(db_session, n=1)
    fact = await _make_atomic_fact(db_session, course)
    res = _result(_tag(nodes[0].id, conf=0.88, review=False))

    first = await persist_grounded_tags(
        db_session, entity_kind="atomic_fact", entity_id=fact.id, result=res
    )
    second = await persist_grounded_tags(
        db_session, entity_kind="atomic_fact", entity_id=fact.id, result=res
    )

    assert first.replaced == 0 and first.persisted == 1
    assert second.replaced == 1 and second.persisted == 1

    rows = (
        await db_session.execute(
            select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id == fact.id)
        )
    ).scalars().all()
    assert len(rows) == 1  # not duplicated


async def test_unknown_entity_kind_raises(db_session: AsyncSession):
    with pytest.raises(ValueError):
        await persist_grounded_tags(
            db_session, entity_kind="flashcard", entity_id=1, result=_result()
        )


async def test_missing_entity_raises(db_session: AsyncSession):
    with pytest.raises(EntityNotFoundError):
        await persist_grounded_tags(
            db_session, entity_kind="question", entity_id=999_999, result=_result()
        )
