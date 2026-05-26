"""Recall layer — candidate retrieval feeding tagging prompts (T28, V-L3, V-E2).

V-L3 forbids raw free-form judgment over the full outline. Tagging
prompts for atomic facts / questions must be constrained by a retrieved
candidate set built from:

1. Cosine similarity over ``content_embeddings`` (entity_kind='outline_node')
   for the target course, version-filtered (V-E1).
2. Optional expansion via ``concept_edges.kind='similarity'`` neighbors of
   the top embedding candidates. Manual edges (``kind='manual'``) are
   human-verified and never followed by the recall pass (V-E2).
3. Optional few-shot exemplars drawn from prior calibrated
   ``question_tags`` (``source='llm'``, ``manual_review=false``,
   ``confidence >= min_exemplar_confidence``).

V-E2 negative constraint: recall must not weight ``Attempt.time_seconds``.
This module does not import ``Attempt`` and does not reference
``time_seconds``; a guard test asserts the source stays clean.

T29 consumes ``RecallResult`` + ``format_candidates_for_prompt`` as the
constrained surface for the grounded-generation tagging prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag
from app.models.concept_edge import ConceptEdge
from app.models.content_embedding import ContentEmbedding
from app.models.outline import OUTLINE_PATH_DELIMITER, OutlineNode
from app.services.kb.embeddings import current_version
from app.services.kb.similarity import cosine

_logger = logging.getLogger("app.services.kb.recall")


# --------------------------------------------------------------------------- #
# Output dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exemplar:
    """One prior calibrated tag, used as a few-shot example.

    `text` is the source entity's surface text (``Question.stem_plain``
    for question tags). `confidence` is the calibrator-stamped score
    (V69) — already >= ``min_exemplar_confidence`` and never
    ``manual_review`` (V-T3).
    """

    question_id: int
    text: str
    confidence: float


@dataclass
class Candidate:
    """One retrieved outline-node candidate for a tagging prompt.

    `score` is the cosine score when ``via='embedding'`` or the edge
    score when ``via='edge'``. `path` is the rendered ``>>``-delimited
    outline path (V-O4); ``None`` when the node's ancestors aren't in
    the loaded course map (e.g. cross-course edge endpoint).
    """

    node_id: int
    path: str | None
    score: float
    via: str  # 'embedding' | 'edge'
    exemplars: list[Exemplar] = field(default_factory=list)


@dataclass
class RecallResult:
    candidates: list[Candidate]
    embedding_version: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def load_embedding(
    session: AsyncSession,
    *,
    entity_kind: str,
    entity_id: int,
    embedding_version: str | None = None,
) -> list[float] | None:
    """Fetch the persisted embedding vector for ``(entity_kind, entity_id)``.

    Returns ``None`` when no row exists. The V-E1 version stamp filters
    to a single dim per call; defaults to ``current_version()``.
    """

    version = embedding_version or current_version()
    row = (
        await session.execute(
            select(ContentEmbedding).where(
                ContentEmbedding.entity_kind == entity_kind,
                ContentEmbedding.entity_id == entity_id,
                ContentEmbedding.embedding_version == version,
            )
        )
    ).scalar_one_or_none()
    if row is None or row.embedding is None:
        return None
    return list(row.embedding)


async def _load_course_node_map(
    session: AsyncSession, *, course_id: int
) -> dict[int, OutlineNode]:
    rows = (
        await session.execute(
            select(OutlineNode).where(OutlineNode.course_id == course_id)
        )
    ).scalars().all()
    return {n.id: n for n in rows}


def _render_path(node_id: int, node_map: dict[int, OutlineNode]) -> str | None:
    """Walk ``parent_id`` chain to build a ``>>``-delimited path (V-O4).

    Returns ``None`` if any ancestor is missing from ``node_map`` — keeps
    cross-course / partially-loaded candidates from yielding bogus
    truncated paths.
    """

    n = node_map.get(node_id)
    if n is None:
        return None
    segs: list[str] = [n.name]
    while n.parent_id is not None:
        n = node_map.get(n.parent_id)
        if n is None:
            return None
        segs.append(n.name)
    return OUTLINE_PATH_DELIMITER.join(reversed(segs))


# --------------------------------------------------------------------------- #
# Core retrieval
# --------------------------------------------------------------------------- #


async def _embedding_candidates(
    session: AsyncSession,
    *,
    course_id: int,
    query_embedding: list[float],
    top_k: int,
    embedding_version: str,
) -> list[tuple[int, float]]:
    """Cosine-rank outline_node embeddings within ``course_id`` against the
    query. Returns ``[(node_id, score), ...]`` sorted desc, capped to
    ``top_k``.

    Joining ``content_embeddings`` ↔ ``outline_nodes`` on
    ``entity_id == OutlineNode.id`` enforces course scope without
    crossing into other courses' embedded nodes. V-E1 version filter
    keeps the cosine dim-consistent.
    """

    rows = (
        await session.execute(
            select(
                ContentEmbedding.entity_id,
                ContentEmbedding.embedding,
            )
            .join(OutlineNode, OutlineNode.id == ContentEmbedding.entity_id)
            .where(
                ContentEmbedding.entity_kind == "outline_node",
                ContentEmbedding.embedding_version == embedding_version,
                OutlineNode.course_id == course_id,
            )
        )
    ).all()

    scored: list[tuple[int, float]] = []
    for node_id, vec in rows:
        if not vec:
            continue
        if len(vec) != len(query_embedding):
            # Dim disagreement inside a single version is V-E1's
            # "⊥ mixed-dim vectors" violation. Skip and warn rather
            # than crash mid-rank.
            _logger.warning(
                "recall: dim mismatch node_id=%s (%d vs %d), version=%s",
                node_id,
                len(vec),
                len(query_embedding),
                embedding_version,
            )
            continue
        scored.append((node_id, cosine(list(vec), query_embedding)))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


async def _expand_via_edges(
    session: AsyncSession,
    *,
    seed_node_ids: Iterable[int],
    exclude: set[int],
) -> list[tuple[int, float]]:
    """Follow ``concept_edges.kind='similarity'`` from ``seed_node_ids`` to
    their neighbors. Manual edges (``kind='manual'``) ⊥ followed (V-E2).
    Edges are undirected for recall purposes: a row with src=S or
    dst=S contributes the other endpoint as a candidate.

    Returns ``[(node_id, score), ...]`` for neighbors not already in
    ``exclude``. When the same neighbor appears via multiple edges its
    highest score wins.
    """

    seeds = set(seed_node_ids)
    if not seeds:
        return []

    rows = (
        await session.execute(
            select(ConceptEdge).where(
                ConceptEdge.kind == "similarity",
                (ConceptEdge.src_node_id.in_(seeds))
                | (ConceptEdge.dst_node_id.in_(seeds)),
            )
        )
    ).scalars().all()

    best: dict[int, float] = {}
    for edge in rows:
        other = edge.dst_node_id if edge.src_node_id in seeds else edge.src_node_id
        if other in exclude or other in seeds:
            continue
        score = float(edge.score) if edge.score is not None else 0.0
        if score > best.get(other, float("-inf")):
            best[other] = score

    return sorted(best.items(), key=lambda t: t[1], reverse=True)


async def _exemplars_for_node(
    session: AsyncSession,
    *,
    node_id: int,
    limit: int,
    min_confidence: float,
) -> list[Exemplar]:
    """Pull prior calibrated question-tag exemplars for ``node_id``.

    Filter: ``source='llm'`` (V-T2 — only calibrated rows),
    ``manual_review=false`` (V-T3 — low-confidence rows are flagged for
    human review, ⊥ used as exemplars), ``confidence >= min_confidence``.
    Ordered by confidence desc; capped at ``limit``.
    """

    if limit <= 0:
        return []

    rows = (
        await session.execute(
            select(QuestionTag, Question)
            .join(Question, Question.id == QuestionTag.question_id)
            .where(
                QuestionTag.node_id == node_id,
                QuestionTag.source == "llm",
                QuestionTag.manual_review.is_(False),
                QuestionTag.confidence >= min_confidence,
            )
            .order_by(QuestionTag.confidence.desc())
            .limit(limit)
        )
    ).all()

    out: list[Exemplar] = []
    for tag, question in rows:
        out.append(
            Exemplar(
                question_id=question.id,
                text=question.stem_plain or "",
                confidence=float(tag.confidence) if tag.confidence is not None else 0.0,
            )
        )
    return out


async def retrieve_candidates(
    session: AsyncSession,
    *,
    course_id: int,
    query_embedding: list[float],
    top_k: int = 10,
    edge_expansion: bool = True,
    exemplars_per_node: int = 0,
    embedding_version: str | None = None,
    min_exemplar_confidence: float = 0.7,
) -> RecallResult:
    """Build the V-L3 constrained candidate set for a tagging prompt.

    Args:
        course_id: scope embedding-similarity to this course's outline.
        query_embedding: the entity-being-tagged's vector (use
            :func:`load_embedding` to fetch a persisted one). Must
            match the dim of ``embedding_version``'s stamped vectors.
        top_k: cap on the embedding-rank step before edge expansion.
        edge_expansion: when ``True``, augment with similarity-edge
            neighbors of the top_k embedding candidates (V-E2).
        exemplars_per_node: per-candidate few-shot exemplar cap (0 = off).
        embedding_version: defaults to :func:`current_version`; the V-E1
            stamp filtering vectors before cosine.
        min_exemplar_confidence: floor for exemplar selection.

    Returns ``RecallResult`` with candidates in score-desc order
    (embedding candidates first, then edge-expanded, exemplars attached).
    """

    version = embedding_version or current_version()

    emb_hits = await _embedding_candidates(
        session,
        course_id=course_id,
        query_embedding=query_embedding,
        top_k=top_k,
        embedding_version=version,
    )

    embedding_node_ids = {nid for nid, _ in emb_hits}

    edge_hits: list[tuple[int, float]] = []
    if edge_expansion and embedding_node_ids:
        edge_hits = await _expand_via_edges(
            session,
            seed_node_ids=embedding_node_ids,
            exclude=set(),  # we dedupe against embedding_node_ids inside
        )

    node_map = await _load_course_node_map(session, course_id=course_id)

    candidates: list[Candidate] = []
    seen: set[int] = set()
    for node_id, score in emb_hits:
        if node_id in seen:
            continue
        seen.add(node_id)
        candidates.append(
            Candidate(
                node_id=node_id,
                path=_render_path(node_id, node_map),
                score=score,
                via="embedding",
            )
        )
    for node_id, score in edge_hits:
        if node_id in seen:
            continue
        seen.add(node_id)
        candidates.append(
            Candidate(
                node_id=node_id,
                path=_render_path(node_id, node_map),
                score=score,
                via="edge",
            )
        )

    if exemplars_per_node > 0:
        for cand in candidates:
            cand.exemplars = await _exemplars_for_node(
                session,
                node_id=cand.node_id,
                limit=exemplars_per_node,
                min_confidence=min_exemplar_confidence,
            )

    return RecallResult(candidates=candidates, embedding_version=version)


# --------------------------------------------------------------------------- #
# Prompt-surface rendering — the V-L3 constraint into the tagging prompt
# --------------------------------------------------------------------------- #


def format_candidates_for_prompt(
    result: RecallResult,
    *,
    include_exemplars: bool = True,
    max_exemplar_chars: int = 280,
) -> str:
    """Render ``RecallResult`` as the constrained candidate surface for a
    tagging prompt (V-L3).

    Output shape — numbered list of candidates, each rendered as path or
    ``node:<id>`` when the path is unresolved, with optional exemplars.
    The tagging prompt downstream (T29) must pick FROM this list, ⊥ free
    text. Empty result → an explicit placeholder line so a downstream
    prompt template can short-circuit deterministically.
    """

    if not result.candidates:
        return "(no candidates retrieved)"

    lines: list[str] = []
    for i, cand in enumerate(result.candidates, start=1):
        label = cand.path or f"node:{cand.node_id}"
        lines.append(
            f"{i}. [{cand.via}, score={cand.score:.3f}] {label}"
        )
        if include_exemplars and cand.exemplars:
            for ex in cand.exemplars:
                snippet = ex.text.strip().replace("\n", " ")
                if len(snippet) > max_exemplar_chars:
                    snippet = snippet[: max_exemplar_chars - 1] + "…"
                lines.append(
                    f"   - exemplar (conf={ex.confidence:.2f}): {snippet}"
                )
    return "\n".join(lines)
