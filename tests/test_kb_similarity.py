"""T26 — app/services/kb/similarity.py contract tests (V-E2).

V-E2: similarity edges are derived (cosine over embeddings); manual
edges are human-verified and ⊥ touched by the derivation pass.
Derivation is idempotent (a re-run is a no-op once the threshold
boundary stabilizes).
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.concept_edge import ConceptEdge
from app.models.content_embedding import ContentEmbedding
from app.models.outline import Course, OutlineNode
from app.services.kb.similarity import cosine, derive_similarity_edges


# --------------------------------------------------------------------------- #
# Pure helper tests
# --------------------------------------------------------------------------- #


def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_norm_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_empty_returns_zero():
    assert cosine([], [1.0, 1.0]) == 0.0


def test_cosine_dim_mismatch_raises():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0])


# --------------------------------------------------------------------------- #
# DB integration
# --------------------------------------------------------------------------- #


async def _make_course_with_nodes(session: AsyncSession, n: int) -> list[OutlineNode]:
    import uuid as _uuid

    course = Course(slug=f"sim-{_uuid.uuid4().hex[:8]}", name="Sim")
    session.add(course)
    await session.flush()
    nodes: list[OutlineNode] = []
    for i in range(n):
        node = OutlineNode(
            course_id=course.id,
            parent_id=None,
            kind="concept",
            name=f"node-{i}-{_uuid.uuid4().hex[:6]}",
            depth=0,
            position=i,
        )
        session.add(node)
        nodes.append(node)
    await session.flush()
    return nodes


def _emb(node_id: int, vec: list[float], version: str = "v1") -> ContentEmbedding:
    return ContentEmbedding(
        entity_kind="outline_node",
        entity_id=node_id,
        embedding=vec,
        embedding_version=version,
    )


async def test_derive_writes_above_threshold(db_session: AsyncSession):
    nodes = await _make_course_with_nodes(db_session, 3)
    db_session.add(_emb(nodes[0].id, [1.0, 0.0, 0.0]))
    db_session.add(_emb(nodes[1].id, [0.99, 0.1, 0.0]))   # ~0.995 cosine vs node 0
    db_session.add(_emb(nodes[2].id, [0.0, 1.0, 0.0]))     # orthogonal
    await db_session.flush()

    report = await derive_similarity_edges(db_session, threshold=0.7)

    assert report.inspected_pairs == 3
    assert report.new_edges == 1
    assert report.reused_edges == 0

    edges = (
        await db_session.execute(
            select(ConceptEdge).where(ConceptEdge.kind == "similarity")
        )
    ).scalars().all()
    assert len(edges) == 1
    e = edges[0]
    assert float(e.score) > 0.95
    assert {e.src_node_id, e.dst_node_id} == {nodes[0].id, nodes[1].id}


async def test_derive_idempotent(db_session: AsyncSession):
    nodes = await _make_course_with_nodes(db_session, 2)
    db_session.add(_emb(nodes[0].id, [1.0, 0.0]))
    db_session.add(_emb(nodes[1].id, [0.95, 0.05]))
    await db_session.flush()

    r1 = await derive_similarity_edges(db_session, threshold=0.5)
    r2 = await derive_similarity_edges(db_session, threshold=0.5)

    assert r1.new_edges == 1
    assert r2.new_edges == 0
    assert r2.reused_edges == 1

    edges = (
        await db_session.execute(
            select(ConceptEdge).where(ConceptEdge.kind == "similarity")
        )
    ).scalars().all()
    assert len(edges) == 1


async def test_derive_skips_manual_edges(db_session: AsyncSession):
    """V-E2: manual edges human-verified, ⊥ inspected by derivation.

    Setup writes a manual edge for the same pair we expect derivation to
    cover; after derivation, both should exist (different `kind` rows;
    UQ is on `(src, dst, kind)`).
    """

    nodes = await _make_course_with_nodes(db_session, 2)
    n0_id, n1_id = nodes[0].id, nodes[1].id
    src, dst = (n0_id, n1_id) if n0_id < n1_id else (n1_id, n0_id)
    db_session.add(
        ConceptEdge(src_node_id=src, dst_node_id=dst, kind="manual", score=None)
    )
    db_session.add(_emb(n0_id, [1.0, 0.0]))
    db_session.add(_emb(n1_id, [0.99, 0.0]))
    await db_session.flush()

    await derive_similarity_edges(db_session, threshold=0.5)

    edges = (
        await db_session.execute(
            select(ConceptEdge).where(
                ConceptEdge.src_node_id == src, ConceptEdge.dst_node_id == dst
            )
        )
    ).scalars().all()
    kinds = {e.kind for e in edges}
    assert kinds == {"manual", "similarity"}


async def test_derive_below_threshold_no_edge(db_session: AsyncSession):
    nodes = await _make_course_with_nodes(db_session, 2)
    db_session.add(_emb(nodes[0].id, [1.0, 0.0]))
    db_session.add(_emb(nodes[1].id, [0.0, 1.0]))     # orthogonal → cosine 0
    await db_session.flush()

    report = await derive_similarity_edges(db_session, threshold=0.5)
    assert report.new_edges == 0


async def test_derive_version_filter(db_session: AsyncSession):
    nodes = await _make_course_with_nodes(db_session, 3)
    # node 0 has only v1; nodes 1,2 only v2.
    db_session.add(_emb(nodes[0].id, [1.0, 0.0], version="v1"))
    db_session.add(_emb(nodes[1].id, [1.0, 0.0], version="v2"))
    db_session.add(_emb(nodes[2].id, [0.99, 0.0], version="v2"))
    await db_session.flush()

    report_v2 = await derive_similarity_edges(
        db_session, embedding_version="v2", threshold=0.5
    )
    # Only the v2 pair (1,2) inspected: 2C2 = 1 pair.
    assert report_v2.inspected_pairs == 1
    assert report_v2.new_edges == 1
