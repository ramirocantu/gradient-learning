"""T28 — app/services/kb/recall.py contract tests (V-L3, V-E2).

V-L3: tagging prompts must be constrained by retrieved outline-node
candidates (embeddings + ``concept_edges.kind='similarity'`` + optional
exemplars from prior calibrated tags). ⊥ free-form judgment over the
full outline.

V-E2: similarity edges are derived; manual edges are human-verified
and ⊥ followed by the recall pass. Recall ⊥ weight
``Attempt.time_seconds``.
"""

from __future__ import annotations

import pathlib
import uuid as _uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question, QuestionTag
from app.models.concept_edge import ConceptEdge
from app.models.content_embedding import ContentEmbedding
from app.models.outline import Course, OutlineNode
from app.models.pdf_source import PdfSource
from app.services.kb.embeddings import current_version
from app.services.kb.recall import (
    Candidate,
    RecallResult,
    format_candidates_for_prompt,
    load_embedding,
    retrieve_candidates,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


async def _make_course_and_tree(
    session: AsyncSession, *, leaves: int
) -> tuple[Course, OutlineNode, list[OutlineNode]]:
    """Course → one root → ``leaves`` leaf concepts. Returns
    ``(course, root, leaf_nodes)``.
    """

    course = Course(slug=f"recall-{_uuid.uuid4().hex[:8]}", name="Recall")
    session.add(course)
    await session.flush()

    root = OutlineNode(
        course_id=course.id,
        parent_id=None,
        kind="section",
        name=f"root-{_uuid.uuid4().hex[:4]}",
        depth=0,
        position=0,
    )
    session.add(root)
    await session.flush()

    leaf_nodes: list[OutlineNode] = []
    for i in range(leaves):
        n = OutlineNode(
            course_id=course.id,
            parent_id=root.id,
            kind="concept",
            name=f"leaf-{i}-{_uuid.uuid4().hex[:4]}",
            depth=1,
            position=i,
        )
        session.add(n)
        leaf_nodes.append(n)
    await session.flush()
    return course, root, leaf_nodes


def _emb_row(
    node_id: int,
    vec: list[float],
    *,
    version: str | None = None,
    entity_kind: str = "outline_node",
) -> ContentEmbedding:
    return ContentEmbedding(
        entity_kind=entity_kind,
        entity_id=node_id,
        embedding=vec,
        embedding_version=version or current_version(),
    )


async def _make_question(session: AsyncSession, *, stem: str = "stem text") -> Question:
    q = Question(
        source="uworld",
        qid=f"q-{_uuid.uuid4().hex[:10]}",
        stem_html=f"<p>{stem}</p>",
        stem_plain=stem,
        choices=[{"id": "A", "text": "x"}, {"id": "B", "text": "y"}],
        correct_choice="A",
    )
    session.add(q)
    await session.flush()
    return q


def _tag(
    question_id: int,
    node_id: int,
    *,
    source: str,
    confidence: float | None,
    manual_review: bool = False,
) -> QuestionTag:
    return QuestionTag(
        question_id=question_id,
        node_id=node_id,
        source=source,
        confidence=confidence,
        manual_review=manual_review,
    )


async def _make_fact(
    session: AsyncSession, *, course_id: int, text: str = "fact text"
) -> AtomicFact:
    pdf = PdfSource(
        course_id=course_id,
        filename="x.pdf",
        sha256=_uuid.uuid4().hex,
        status="ingested",
    )
    session.add(pdf)
    await session.flush()
    f = AtomicFact(
        course_id=course_id,
        pdf_source_id=pdf.id,
        text=text,
        content_hash=_uuid.uuid4().hex,
    )
    session.add(f)
    await session.flush()
    return f


def _fact_tag(
    atomic_fact_id: int,
    node_id: int,
    *,
    source: str,
    confidence: float | None,
    manual_review: bool = False,
) -> AtomicFactTag:
    return AtomicFactTag(
        atomic_fact_id=atomic_fact_id,
        node_id=node_id,
        source=source,
        confidence=confidence,
        manual_review=manual_review,
    )


# --------------------------------------------------------------------------- #
# load_embedding
# --------------------------------------------------------------------------- #


async def test_load_embedding_returns_persisted_vector(db_session: AsyncSession):
    _, _, leaves = await _make_course_and_tree(db_session, leaves=1)
    db_session.add(_emb_row(leaves[0].id, [0.1, 0.2, 0.3]))
    await db_session.flush()

    vec = await load_embedding(db_session, entity_kind="outline_node", entity_id=leaves[0].id)
    assert vec == [0.1, 0.2, 0.3]


async def test_load_embedding_missing_returns_none(db_session: AsyncSession):
    vec = await load_embedding(db_session, entity_kind="outline_node", entity_id=99999999)
    assert vec is None


async def test_load_embedding_version_filter_excludes_other_version(
    db_session: AsyncSession,
):
    """V-E1: a different version must not surface — mixed-dim coexistence
    is the whole reason `embedding_version` is keyed into the UQ."""

    _, _, leaves = await _make_course_and_tree(db_session, leaves=1)
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0], version="other-v1"))
    await db_session.flush()

    vec = await load_embedding(
        db_session,
        entity_kind="outline_node",
        entity_id=leaves[0].id,
        embedding_version="missing-v1",
    )
    assert vec is None


# --------------------------------------------------------------------------- #
# retrieve_candidates — embedding rank
# --------------------------------------------------------------------------- #


async def test_retrieve_ranks_by_cosine_within_course(db_session: AsyncSession):
    course, _, leaves = await _make_course_and_tree(db_session, leaves=3)
    # node 0 = identical to query, node 1 = near, node 2 = orthogonal.
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0, 0.0]))
    db_session.add(_emb_row(leaves[1].id, [0.9, 0.1, 0.0]))
    db_session.add(_emb_row(leaves[2].id, [0.0, 1.0, 0.0]))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
        edge_expansion=False,
    )

    assert [c.node_id for c in result.candidates] == [leaves[0].id, leaves[1].id]
    assert all(c.via == "embedding" for c in result.candidates)
    assert result.candidates[0].score > result.candidates[1].score
    # V-O4 path rendering walks parent_id chain.
    assert result.candidates[0].path is not None
    assert " >> " in result.candidates[0].path


async def test_retrieve_excludes_other_course(db_session: AsyncSession):
    """Embedding candidates are course-scoped via the OutlineNode join."""

    course_a, _, leaves_a = await _make_course_and_tree(db_session, leaves=1)
    course_b, _, leaves_b = await _make_course_and_tree(db_session, leaves=1)
    db_session.add(_emb_row(leaves_a[0].id, [1.0, 0.0]))
    db_session.add(_emb_row(leaves_b[0].id, [1.0, 0.0]))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course_a.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=False,
    )
    assert [c.node_id for c in result.candidates] == [leaves_a[0].id]


async def test_retrieve_version_filter_v_e1(db_session: AsyncSession):
    """V-E1: only the requested version is ranked. Mixed-dim coexistence
    during a re-embed sweep must not leak old vectors into recall.
    """

    course, _, leaves = await _make_course_and_tree(db_session, leaves=2)
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0], version="vNEW"))
    db_session.add(_emb_row(leaves[1].id, [1.0, 0.0], version="vOLD"))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=False,
        embedding_version="vNEW",
    )
    assert [c.node_id for c in result.candidates] == [leaves[0].id]
    assert result.embedding_version == "vNEW"


# --------------------------------------------------------------------------- #
# retrieve_candidates — edge expansion (V-E2)
# --------------------------------------------------------------------------- #


async def test_edge_expansion_follows_similarity_not_manual(
    db_session: AsyncSession,
):
    """V-E2: similarity edges are derived; manual edges human-verified.
    Recall expands via similarity edges only — manual edges must not
    surface neighbors.
    """

    course, _, leaves = await _make_course_and_tree(db_session, leaves=4)
    # leaves[0] = the only one with an embedding → sole seed candidate.
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0]))
    await db_session.flush()

    def _ordered(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    sim_src, sim_dst = _ordered(leaves[0].id, leaves[1].id)
    db_session.add(
        ConceptEdge(
            src_node_id=sim_src,
            dst_node_id=sim_dst,
            kind="similarity",
            score=0.85,
        )
    )
    man_src, man_dst = _ordered(leaves[0].id, leaves[2].id)
    db_session.add(
        ConceptEdge(
            src_node_id=man_src,
            dst_node_id=man_dst,
            kind="manual",
            score=None,
        )
    )
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=True,
    )

    node_ids = [c.node_id for c in result.candidates]
    assert leaves[0].id in node_ids  # seed via embedding
    assert leaves[1].id in node_ids  # via similarity edge
    assert leaves[2].id not in node_ids  # manual edge ⊥ followed (V-E2)

    edge_cand = next(c for c in result.candidates if c.node_id == leaves[1].id)
    assert edge_cand.via == "edge"
    assert edge_cand.score == pytest.approx(0.85)


async def test_edge_expansion_disabled_yields_only_embedding_hits(
    db_session: AsyncSession,
):
    course, _, leaves = await _make_course_and_tree(db_session, leaves=2)
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0]))
    await db_session.flush()
    a, b = sorted([leaves[0].id, leaves[1].id])
    db_session.add(ConceptEdge(src_node_id=a, dst_node_id=b, kind="similarity", score=0.9))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=False,
    )
    assert [c.node_id for c in result.candidates] == [leaves[0].id]


# --------------------------------------------------------------------------- #
# retrieve_candidates — exemplars (V-L3, V-T2, V-T3)
# --------------------------------------------------------------------------- #


async def test_exemplars_filter_source_and_manual_review(db_session: AsyncSession):
    """Few-shot exemplars: only `source='llm'`, `manual_review=false`,
    confidence above the floor. Manual + schema_map + flagged-for-review
    rows ⊥ surface as exemplars (V-T2, V-T3).
    """

    course, _, leaves = await _make_course_and_tree(db_session, leaves=1)
    node_id = leaves[0].id
    db_session.add(_emb_row(node_id, [1.0, 0.0]))

    q_good = await _make_question(db_session, stem="good calibrated stem")
    q_low = await _make_question(db_session, stem="below-threshold stem")
    q_flagged = await _make_question(db_session, stem="flagged stem")
    q_manual = await _make_question(db_session, stem="human-written stem")
    q_schema = await _make_question(db_session, stem="schema-mapped stem")

    db_session.add(_tag(q_good.id, node_id, source="llm", confidence=0.9))
    # below threshold → excluded (under min_exemplar_confidence)
    # V-T3 requires the row to also carry manual_review when confidence<0.5,
    # so we keep this one above the CHECK floor (0.5) but below our
    # exemplar floor (0.7) to exercise the recall-level filter cleanly.
    db_session.add(_tag(q_low.id, node_id, source="llm", confidence=0.6))
    # manual_review=True → excluded even though confidence is otherwise fine
    db_session.add(_tag(q_flagged.id, node_id, source="llm", confidence=0.95, manual_review=True))
    db_session.add(_tag(q_manual.id, node_id, source="manual", confidence=None))
    db_session.add(_tag(q_schema.id, node_id, source="schema_map", confidence=None))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=False,
        exemplars_per_node=5,
        min_exemplar_confidence=0.7,
    )

    assert len(result.candidates) == 1
    cand = result.candidates[0]
    exemplar_qids = {ex.entity_id for ex in cand.exemplars}
    assert exemplar_qids == {q_good.id}
    assert all(ex.entity_kind == "question" for ex in cand.exemplars)
    assert cand.exemplars[0].text == "good calibrated stem"


async def test_exemplars_off_by_default(db_session: AsyncSession):
    course, _, leaves = await _make_course_and_tree(db_session, leaves=1)
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0]))
    q = await _make_question(db_session, stem="anything")
    db_session.add(_tag(q.id, leaves[0].id, source="llm", confidence=0.99))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=False,
        # exemplars_per_node defaults to 0
    )
    assert result.candidates[0].exemplars == []


# --------------------------------------------------------------------------- #
# format_candidates_for_prompt — V-L3 constrained surface
# --------------------------------------------------------------------------- #


def test_format_candidates_numbered_constrained_list():
    """V-L3: prompt surface is the candidate set, ⊥ the full outline."""

    res = RecallResult(
        embedding_version="v1",
        candidates=[
            Candidate(node_id=1, path="root >> a", score=0.92, via="embedding"),
            Candidate(node_id=2, path=None, score=0.71, via="edge"),
        ],
    )
    text = format_candidates_for_prompt(res, include_exemplars=False)
    lines = text.splitlines()
    assert lines[0].startswith("1. ")
    assert "embedding" in lines[0]
    assert "root >> a" in lines[0]
    assert lines[1].startswith("2. ")
    assert "edge" in lines[1]
    assert "node:2" in lines[1]  # unresolved path → node:<id> fallback


def test_format_candidates_empty_yields_placeholder():
    res = RecallResult(embedding_version="v1", candidates=[])
    assert format_candidates_for_prompt(res) == "(no candidates retrieved)"


def test_format_candidates_renders_exemplars():
    from app.services.kb.recall import Exemplar

    res = RecallResult(
        embedding_version="v1",
        candidates=[
            Candidate(
                node_id=1,
                path="root >> a",
                score=0.9,
                via="embedding",
                exemplars=[
                    Exemplar(
                        entity_kind="question",
                        entity_id=7,
                        text="ex stem",
                        score=0.88,
                    )
                ],
            )
        ],
    )
    text = format_candidates_for_prompt(res)
    assert "exemplar [question] (score=0.88)" in text
    assert "ex stem" in text


# --------------------------------------------------------------------------- #
# D — δ floor on embedding candidates
# --------------------------------------------------------------------------- #


async def test_embedding_floor_drops_below_min_score(db_session: AsyncSession):
    """D: a candidate whose cosine is below ``min_score`` is dropped rather
    than handed to the LLM as a 'least-bad' pick."""

    course, _, leaves = await _make_course_and_tree(db_session, leaves=2)
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0]))  # cosine 1.0 — kept
    db_session.add(_emb_row(leaves[1].id, [0.1, 1.0]))  # cosine ~0.10 — dropped
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=False,
        content_expansion=False,
        min_score=0.25,
    )
    assert [c.node_id for c in result.candidates] == [leaves[0].id]


# --------------------------------------------------------------------------- #
# E — edge-expansion cap
# --------------------------------------------------------------------------- #


async def test_edge_expansion_capped_at_top_n(db_session: AsyncSession):
    """E: the T2T fan-out is bounded so it can't flood the candidate list."""

    course, _, leaves = await _make_course_and_tree(db_session, leaves=7)
    db_session.add(_emb_row(leaves[0].id, [1.0, 0.0]))  # sole seed
    await db_session.flush()
    for i in range(1, 7):  # 6 similarity neighbours of the seed
        a, b = sorted([leaves[0].id, leaves[i].id])
        db_session.add(
            ConceptEdge(src_node_id=a, dst_node_id=b, kind="similarity", score=0.5 + i * 0.01)
        )
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        top_k=5,
        edge_expansion=True,
        content_expansion=False,
        edge_top_n=5,
    )
    edge_cands = [c for c in result.candidates if c.via == "edge"]
    assert len(edge_cands) == 5  # capped at edge_top_n, not all 6


# --------------------------------------------------------------------------- #
# A — C2C2T content recall (content → similar tagged content → its tag)
# --------------------------------------------------------------------------- #


async def test_content_recall_borrows_gold_tag_from_similar_fact(
    db_session: AsyncSession,
):
    """A: a target with NO matching node embedding still recalls the right
    node via a similar already-tagged fact (gold = manual/schema second hop).
    This is the path C2T cannot reach."""

    course, _, leaves = await _make_course_and_tree(db_session, leaves=2)
    target_node = leaves[0].id
    # Deliberately NO outline_node embeddings → C2T finds nothing.
    neighbour = await _make_fact(db_session, course_id=course.id, text="neighbour fact")
    db_session.add(_emb_row(neighbour.id, [1.0, 0.0], entity_kind="atomic_fact"))
    db_session.add(_fact_tag(neighbour.id, target_node, source="manual", confidence=None))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        edge_expansion=False,
    )

    cand = next((c for c in result.candidates if c.node_id == target_node), None)
    assert cand is not None
    assert cand.via == "content-gold"
    assert cand.exemplars and cand.exemplars[0].entity_kind == "atomic_fact"
    assert cand.exemplars[0].entity_id == neighbour.id


async def test_content_recall_silver_is_discounted_and_review_gated(
    db_session: AsyncSession,
):
    """A: a prior ``llm`` tag is borrowed as 'silver' but discounted by
    ``silver_factor``; a ``manual_review`` llm tag is NOT borrowed (echo
    guard)."""

    course, _, leaves = await _make_course_and_tree(db_session, leaves=2)
    silver_node, flagged_node = leaves[0].id, leaves[1].id

    n_silver = await _make_fact(db_session, course_id=course.id, text="silver neighbour")
    db_session.add(_emb_row(n_silver.id, [1.0, 0.0], entity_kind="atomic_fact"))
    db_session.add(_fact_tag(n_silver.id, silver_node, source="llm", confidence=0.8))

    n_flagged = await _make_fact(db_session, course_id=course.id, text="flagged neighbour")
    db_session.add(_emb_row(n_flagged.id, [1.0, 0.0], entity_kind="atomic_fact"))
    db_session.add(
        _fact_tag(n_flagged.id, flagged_node, source="llm", confidence=0.95, manual_review=True)
    )
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        edge_expansion=False,
        silver_factor=0.6,
    )

    by_node = {c.node_id: c for c in result.candidates}
    assert silver_node in by_node
    assert by_node[silver_node].via == "content-silver"
    # neighbour cosine 1.0 × silver_factor 0.6 → score ~0.6, the discount.
    assert by_node[silver_node].score == pytest.approx(0.6, abs=1e-6)
    assert flagged_node not in by_node  # manual_review llm tag ⊥ borrowed


async def test_content_recall_excludes_self(db_session: AsyncSession):
    """A: the target fact must not recall itself as its own neighbour."""

    course, _, leaves = await _make_course_and_tree(db_session, leaves=1)
    target = await _make_fact(db_session, course_id=course.id, text="the target")
    db_session.add(_emb_row(target.id, [1.0, 0.0], entity_kind="atomic_fact"))
    # Give the target a (stale) tag so, if not excluded, it would self-borrow.
    db_session.add(_fact_tag(target.id, leaves[0].id, source="manual", confidence=None))
    await db_session.flush()

    result = await retrieve_candidates(
        db_session,
        course_id=course.id,
        query_embedding=[1.0, 0.0],
        edge_expansion=False,
        exclude_entity=("atomic_fact", target.id),
    )
    assert result.candidates == []


# --------------------------------------------------------------------------- #
# V-E2 negative-constraint guard: recall ⊥ weight Attempt.time_seconds.
# --------------------------------------------------------------------------- #


def test_recall_source_does_not_touch_time_seconds():
    """V-E2: recall ⊥ weight ``Attempt.time_seconds``. Source-level guard —
    a future change that imports ``Attempt`` or references ``time_seconds``
    inside the recall module trips this test and forces an explicit
    invariant review.

    AST-based so docstring/comment mentions of the invariant don't
    self-trip the guard — only real code references count.
    """

    import ast

    src = pathlib.Path(__file__).resolve().parents[1] / "app" / "services" / "kb" / "recall.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "time_seconds", "recall.py ⊥ reference .time_seconds (V-E2)"
        if isinstance(node, ast.Name):
            assert node.id != "time_seconds", "recall.py ⊥ name time_seconds (V-E2)"
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "Attempt", "recall.py ⊥ import Attempt (V-E2)"
                assert (alias.asname or "") != "Attempt"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "Attempt" not in alias.name, "recall.py ⊥ import Attempt (V-E2)"
