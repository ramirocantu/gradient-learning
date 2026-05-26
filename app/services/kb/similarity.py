"""Similarity-edge derivation (T26, V-E2).

Cosine over ``content_embeddings`` rows (entity_kind='outline_node')
→ ``concept_edges`` rows with ``kind='similarity'``. Manual edges
(``kind='manual'``) are never written or modified here (V-E2:
similarity = derived; manual = human-verified).

Computed Python-side over JSONB embeddings until T27 swaps the column
to pgvector ``vector(N)`` — at which point T28 may push this into the
DB via ``embedding <=> embedding`` cosine-distance operators.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.concept_edge import ConceptEdge
from app.models.content_embedding import ContentEmbedding

_logger = logging.getLogger("app.services.kb.similarity")


@dataclass
class DeriveReport:
    inspected_pairs: int
    new_edges: int
    reused_edges: int


def cosine(a: list[float], b: list[float]) -> float:
    """Pure cosine similarity. Returns 0.0 for empty or zero-norm vectors
    rather than raising; the caller filters by threshold and that's
    cheaper than try/except on the hot path.
    """

    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


async def derive_similarity_edges(
    session: AsyncSession,
    *,
    embedding_version: str | None = None,
    threshold: float = 0.7,
) -> DeriveReport:
    """Pairwise-cosine over outline_node embeddings; persist
    ``concept_edges.kind='similarity'`` rows above ``threshold``.

    Idempotent: pair already present in ``concept_edges`` with
    ``kind='similarity'`` is left untouched (no score update).
    Manual edges (``kind='manual'``) are not inspected — V-E2 keeps
    those human-verified rows out of the derivation path entirely.
    The UQ on ``(src, dst, kind)`` lets a manual edge and a similarity
    edge over the same pair coexist.
    """

    embeddings_q = select(ContentEmbedding).where(
        ContentEmbedding.entity_kind == "outline_node"
    )
    if embedding_version is not None:
        embeddings_q = embeddings_q.where(
            ContentEmbedding.embedding_version == embedding_version
        )
    embeddings = (await session.execute(embeddings_q)).scalars().all()

    existing_pairs = {
        (e.src_node_id, e.dst_node_id)
        for e in (
            await session.execute(
                select(ConceptEdge).where(ConceptEdge.kind == "similarity")
            )
        ).scalars().all()
    }

    inspected = 0
    new = 0
    reused = 0
    for i, a in enumerate(embeddings):
        for b in embeddings[i + 1 :]:
            inspected += 1
            if not a.embedding or not b.embedding:
                continue
            score = cosine(a.embedding, b.embedding)
            if score < threshold:
                continue
            src, dst = (
                (a.entity_id, b.entity_id)
                if a.entity_id < b.entity_id
                else (b.entity_id, a.entity_id)
            )
            if (src, dst) in existing_pairs:
                reused += 1
                continue
            session.add(
                ConceptEdge(
                    src_node_id=src,
                    dst_node_id=dst,
                    kind="similarity",
                    score=round(score, 5),
                )
            )
            existing_pairs.add((src, dst))
            new += 1

    if new:
        await session.flush()

    return DeriveReport(inspected_pairs=inspected, new_edges=new, reused_edges=reused)
