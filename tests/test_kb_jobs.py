"""T50 — LLM4Tag runner cores: embed_pending + tag_pending (V-L3, V-E1, V41).

embed_pending is exercised against a mocked OpenAI embeddings client (V16).
tag_pending wires recall → grounded → persist; the grounded LLM pass
(generate_grounded_tags — itself unit-tested, incl. inline V69 calibration) is
patched so these tests assert the *orchestration*: pending-entity selection,
embedding-gating, course resolution, node_id denormalization, and V41
isolation.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question, QuestionTag
from app.models.content_embedding import ContentEmbedding
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.embeddings import current_version
from app.services.kb.jobs import embed_pending, tag_pending
from app.services.llm.grounded import GroundedResult, GroundedTag

_DIM = 1536


def _vec(i: int) -> list[float]:
    v = [0.0] * _DIM
    v[i % _DIM] = 1.0
    return v


def _embed_client(*, prompt_tokens: int = 3) -> MagicMock:
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=_vec(0))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens),
        )
    )
    return client


def _grounded(node_id: int, *, conf: float = 0.9, manual_review: bool = False) -> GroundedResult:
    return GroundedResult(
        tags=[
            GroundedTag(
                node_id=node_id,
                path=None,
                candidate_index=1,
                via="embedding",
                rationale="because",
                calibrated_confidence=conf,
                manual_review=manual_review,
            )
        ],
        extractor_version="grounded-v1",
        model="m",
        calibrator_model="c",
        input_tokens=10,
        output_tokens=2,
        cached_tokens=0,
    )


async def _course(session: AsyncSession) -> Course:
    c = Course(slug=f"c-{uuid.uuid4().hex[:8]}", name="C")
    session.add(c)
    await session.flush()
    return c


async def _node(session: AsyncSession, course_id: int, name: str) -> OutlineNode:
    n = OutlineNode(
        course_id=course_id, parent_id=None, kind="concept", name=name, depth=0, position=0
    )
    session.add(n)
    await session.flush()
    return n


async def _fact(session: AsyncSession, course_id: int, text: str) -> AtomicFact:
    pdf = PdfSource(
        course_id=course_id, filename="x.pdf", sha256=uuid.uuid4().hex, status="ingested"
    )
    session.add(pdf)
    await session.flush()
    f = AtomicFact(
        course_id=course_id, pdf_source_id=pdf.id, text=text, content_hash=uuid.uuid4().hex
    )
    session.add(f)
    await session.flush()
    return f


async def _question(
    session: AsyncSession, stem: str, course_id: int | None = None
) -> Question:
    q = Question(
        source="manual",
        qid=f"q-{uuid.uuid4().hex[:10]}",
        course_id=course_id,
        stem_html=f"<p>{stem}</p>",
        stem_plain=stem,
        choices=[{"key": "A", "html": "a", "plain": "a", "media_ids": []}],
        correct_choice="A",
        needs_categorization=True,
    )
    session.add(q)
    await session.flush()
    return q


async def _embed(session: AsyncSession, kind: str, entity_id: int, vec: list[float]) -> None:
    session.add(
        ContentEmbedding(
            entity_kind=kind,
            entity_id=entity_id,
            embedding=vec,
            embedding_version=current_version(),
        )
    )
    await session.flush()


# --------------------------------------------------------------------------- #
# embed_pending
# --------------------------------------------------------------------------- #


async def test_embed_pending_embeds_all_kinds(db_session: AsyncSession):
    course = await _course(db_session)
    await _node(db_session, course.id, "Glycolysis")
    await _fact(db_session, course.id, "Glycolysis yields pyruvate.")
    await _question(db_session, "What does glycolysis yield?")

    client = _embed_client()
    report = await embed_pending(db_session, openai_client=client)

    assert report.embedded == 3  # one node, one fact, one question
    assert report.reused == 0
    assert not report.partial_failure

    kinds = {
        row.entity_kind
        for row in (await db_session.execute(select(ContentEmbedding))).scalars().all()
    }
    assert kinds == {"outline_node", "atomic_fact", "question"}


async def test_embed_pending_embeds_outline_node_by_full_path(db_session: AsyncSession):
    """B: outline nodes embed their ancestor-qualified ``>>`` path, not the
    bare leaf name — the recall cosine matches a fact's prose against the
    tag's meaning, and the path carries far more of it."""

    course = await _course(db_session)
    parent = await _node(db_session, course.id, "Metabolism")
    child = OutlineNode(
        course_id=course.id,
        parent_id=parent.id,
        kind="concept",
        name="Glycolysis",
        depth=1,
        position=0,
    )
    db_session.add(child)
    await db_session.flush()

    client = _embed_client()
    await embed_pending(db_session, openai_client=client)

    inputs = [c.kwargs["input"] for c in client.embeddings.create.call_args_list]
    assert "Metabolism >> Glycolysis" in inputs  # child embedded by full path
    assert "Metabolism" in inputs                # root = its own name


async def test_embed_pending_idempotent(db_session: AsyncSession):
    course = await _course(db_session)
    await _node(db_session, course.id, "TCA cycle")
    client = _embed_client()

    first = await embed_pending(db_session, openai_client=client)
    second = await embed_pending(db_session, openai_client=client)

    # Idempotent: already-embedded entities are filtered out of the second
    # sweep, so it embeds nothing and the row count stays put.
    assert first.embedded == 1
    assert second.embedded == 0
    rows = (await db_session.execute(select(ContentEmbedding))).scalars().all()
    assert len(rows) == 1
    # The embeddings client was called exactly once across both runs.
    assert client.embeddings.create.await_count == 1


# --------------------------------------------------------------------------- #
# tag_pending
# --------------------------------------------------------------------------- #


async def test_tag_pending_tags_fact_and_sets_node_id(db_session: AsyncSession):
    course = await _course(db_session)
    node = await _node(db_session, course.id, "Glycolysis")
    await _embed(db_session, "outline_node", node.id, _vec(0))
    fact = await _fact(db_session, course.id, "Glycolysis yields pyruvate.")
    await _embed(db_session, "atomic_fact", fact.id, _vec(0))  # cosine 1.0 with node

    with patch(
        "app.services.kb.jobs.generate_grounded_tags",
        new=AsyncMock(return_value=_grounded(node.id)),
    ):
        report = await tag_pending(
            db_session, tagging_client=MagicMock(), calibrator_client=MagicMock()
        )

    assert report.facts_tagged == 1
    assert report.tags_persisted == 1
    assert not report.partial_failure

    refreshed = (
        await db_session.execute(select(AtomicFact).where(AtomicFact.id == fact.id))
    ).scalar_one()
    assert refreshed.node_id == node.id
    tags = (
        await db_session.execute(
            select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id == fact.id)
        )
    ).scalars().all()
    assert len(tags) == 1
    assert tags[0].source == "llm"


async def test_tag_pending_skips_fact_without_embedding(db_session: AsyncSession):
    course = await _course(db_session)
    await _node(db_session, course.id, "Glycolysis")
    await _fact(db_session, course.id, "Untagged, unembedded fact.")  # no embedding

    gen = AsyncMock(return_value=_grounded(1))
    with patch("app.services.kb.jobs.generate_grounded_tags", new=gen):
        report = await tag_pending(db_session, tagging_client=MagicMock())

    assert report.facts_tagged == 0
    assert report.facts_skipped_no_embedding == 1
    gen.assert_not_called()  # no embedding → no recall/LLM pass


async def test_tag_pending_tags_question_when_single_course(db_session: AsyncSession):
    course = await _course(db_session)
    node = await _node(db_session, course.id, "Glycolysis")
    await _embed(db_session, "outline_node", node.id, _vec(0))
    q = await _question(db_session, "What does glycolysis yield?")
    await _embed(db_session, "question", q.id, _vec(0))

    with patch(
        "app.services.kb.jobs.generate_grounded_tags",
        new=AsyncMock(return_value=_grounded(node.id)),
    ):
        report = await tag_pending(db_session, tagging_client=MagicMock())

    assert report.questions_tagged == 1
    refreshed = (
        await db_session.execute(select(Question).where(Question.id == q.id))
    ).scalar_one()
    assert refreshed.needs_categorization is False
    qtags = (
        await db_session.execute(select(QuestionTag).where(QuestionTag.question_id == q.id))
    ).scalars().all()
    assert len(qtags) == 1


async def test_tag_pending_skips_questions_when_course_ambiguous(db_session: AsyncSession):
    # Two courses → a pre-tag question has no resolvable course scope.
    await _course(db_session)
    await _course(db_session)
    q = await _question(db_session, "Ambiguous-course question?")

    gen = AsyncMock(return_value=_grounded(1))
    with patch("app.services.kb.jobs.generate_grounded_tags", new=gen):
        report = await tag_pending(db_session, tagging_client=MagicMock())

    assert report.questions_tagged == 0
    assert report.questions_skipped == 1
    gen.assert_not_called()
    refreshed = (
        await db_session.execute(select(Question).where(Question.id == q.id))
    ).scalar_one()
    assert refreshed.needs_categorization is True  # untouched


async def test_tag_pending_tags_course_stamped_question_with_multiple_courses(
    db_session: AsyncSession,
):
    # V-CAP2: two courses exist (globally ambiguous), but the question carries
    # its own course_id — recall scopes to it, so it tags rather than skips.
    course_a = await _course(db_session)
    await _course(db_session)  # second course → count > 1
    node = await _node(db_session, course_a.id, "Glycolysis")
    await _embed(db_session, "outline_node", node.id, _vec(0))
    q = await _question(db_session, "What does glycolysis yield?", course_id=course_a.id)
    await _embed(db_session, "question", q.id, _vec(0))

    with patch(
        "app.services.kb.jobs.generate_grounded_tags",
        new=AsyncMock(return_value=_grounded(node.id)),
    ):
        report = await tag_pending(db_session, tagging_client=MagicMock())

    assert report.questions_tagged == 1
    assert report.questions_skipped == 0
    refreshed = (
        await db_session.execute(select(Question).where(Question.id == q.id))
    ).scalar_one()
    assert refreshed.needs_categorization is False


async def test_tag_pending_isolates_per_entity_failure(db_session: AsyncSession):
    course = await _course(db_session)
    node = await _node(db_session, course.id, "Glycolysis")
    await _embed(db_session, "outline_node", node.id, _vec(0))
    fact = await _fact(db_session, course.id, "Boom fact.")
    await _embed(db_session, "atomic_fact", fact.id, _vec(0))

    with patch(
        "app.services.kb.jobs.generate_grounded_tags",
        new=AsyncMock(side_effect=ValueError("LLM blew up")),
    ):
        report = await tag_pending(db_session, tagging_client=MagicMock())

    assert report.facts_tagged == 0
    assert report.partial_failure
    assert len(report.failures) == 1
    assert "atomic_fact" in report.failures[0]
