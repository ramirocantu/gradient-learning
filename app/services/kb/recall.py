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

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question, QuestionTag
from app.models.concept_edge import ConceptEdge
from app.models.content_embedding import ContentEmbedding
from app.models.outline import OUTLINE_PATH_DELIMITER, OutlineNode
from app.services.kb.embeddings import current_version
from app.services.kb.similarity import cosine

# Recall tuning defaults. δ floor (D) drops clear non-matches from the
# embedding rank rather than handing the LLM a "least-bad" candidate it can
# still pick (the calibrator is the only other backstop). PROVISIONAL —
# text-embedding-3-small's cosine scale differs from BGE's; retune on the
# V-L2 harness against real vectors. Caps (E, A) mirror the paper's
# C2T ≤15 / C2C2T ≤5 fan-out so the candidate list can't balloon and
# dilute the pick.
DEFAULT_MIN_SCORE = 0.25
DEFAULT_EDGE_TOP_N = 5
DEFAULT_CONTENT_NEIGHBOR_K = 20
DEFAULT_CONTENT_NODE_CAP = 5
DEFAULT_SILVER_FACTOR = 0.6
DEFAULT_MIN_SILVER_CONFIDENCE = 0.5

_logger = logging.getLogger("app.services.kb.recall")


# --------------------------------------------------------------------------- #
# Output dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exemplar:
    """One few-shot example attached to a candidate node.

    Two provenances share this shape:

    - **prior-tag exemplar** (``entity_kind='question'``): a calibrated
      ``question_tags`` row for the node. ``score`` is the V69 calibrated
      confidence — already >= ``min_exemplar_confidence`` and never
      ``manual_review`` (V-T3).
    - **content-neighbor exemplar** (``entity_kind='atomic_fact'``): the
      similar already-tagged fact that *recalled* this node via the C2C2T
      path (A). ``score`` is the content↔content cosine that bridged them.

    ``text`` is the source entity's surface text.
    """

    entity_kind: str
    entity_id: int
    text: str
    score: float


@dataclass
class Candidate:
    """One retrieved outline-node candidate for a tagging prompt.

    `score` meaning by ``via``:
      - ``'embedding'`` — cosine of the target vs the node vector (C2T).
      - ``'edge'``      — similarity-edge score, node↔node (T2T).
      - ``'content-gold'`` / ``'content-silver'`` — the C2C2T path (A):
        cosine to a similar already-tagged fact, times a source weight
        (gold = human/schema tag at 1.0; silver = a prior ``llm`` tag,
        discounted). The borrowed node is that neighbour's tag.

    `path` is the rendered ``>>``-delimited outline path (V-O4); ``None``
    when the node's ancestors aren't in the loaded course map (e.g. a
    cross-course edge endpoint).
    """

    node_id: int
    path: str | None
    score: float
    via: str  # 'embedding' | 'edge' | 'content-gold' | 'content-silver'
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


async def _load_course_node_map(session: AsyncSession, *, course_id: int) -> dict[int, OutlineNode]:
    rows = (
        (await session.execute(select(OutlineNode).where(OutlineNode.course_id == course_id)))
        .scalars()
        .all()
    )
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
    min_score: float = 0.0,
) -> list[tuple[int, float]]:
    """Cosine-rank outline_node embeddings within ``course_id`` against the
    query. Returns ``[(node_id, score), ...]`` sorted desc, capped to
    ``top_k``, dropping anything below ``min_score`` (D — δ floor).

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
        score = cosine(list(vec), query_embedding)
        if score < min_score:
            continue
        scored.append((node_id, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


async def _expand_via_edges(
    session: AsyncSession,
    *,
    seed_node_ids: Iterable[int],
    exclude: set[int],
    top_n: int = DEFAULT_EDGE_TOP_N,
) -> list[tuple[int, float]]:
    """Follow ``concept_edges.kind='similarity'`` from ``seed_node_ids`` to
    their neighbors. Manual edges (``kind='manual'``) ⊥ followed (V-E2).
    Edges are undirected for recall purposes: a row with src=S or
    dst=S contributes the other endpoint as a candidate.

    Returns ``[(node_id, score), ...]`` for neighbors not already in
    ``exclude``, score-desc, capped to ``top_n`` (E — bound the T2T
    fan-out so it can't flood the candidate list). When the same neighbor
    appears via multiple edges its highest score wins.
    """

    seeds = set(seed_node_ids)
    if not seeds:
        return []

    rows = (
        (
            await session.execute(
                select(ConceptEdge).where(
                    ConceptEdge.kind == "similarity",
                    (ConceptEdge.src_node_id.in_(seeds)) | (ConceptEdge.dst_node_id.in_(seeds)),
                )
            )
        )
        .scalars()
        .all()
    )

    best: dict[int, float] = {}
    for edge in rows:
        other = edge.dst_node_id if edge.src_node_id in seeds else edge.src_node_id
        if other in exclude or other in seeds:
            continue
        score = float(edge.score) if edge.score is not None else 0.0
        if score > best.get(other, float("-inf")):
            best[other] = score

    ranked = sorted(best.items(), key=lambda t: t[1], reverse=True)
    return ranked[:top_n] if top_n > 0 else ranked


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
                entity_kind="question",
                entity_id=question.id,
                text=question.stem_plain or "",
                score=float(tag.confidence) if tag.confidence is not None else 0.0,
            )
        )
    return out


async def _content_candidates(
    session: AsyncSession,
    *,
    course_id: int,
    query_embedding: list[float],
    embedding_version: str,
    neighbor_k: int,
    node_cap: int,
    min_score: float,
    silver_factor: float,
    min_silver_confidence: float,
    exclude_entity: tuple[str, int] | None,
    neighbor_exemplars_per_node: int = 2,
) -> list[tuple[int, float, str, list[Exemplar]]]:
    """C2C2T (A): content → similar already-tagged content → its tags.

    The paper's hard-case rescuer. Cosine the target against other
    ``atomic_fact`` vectors in the course (the content↔content first hop),
    then borrow each similar fact's *tags* (the second hop). The borrowed
    node is scored ``neighbour_cosine × weight`` where weight encodes the
    second hop's trust:

    - **gold** (``source ∈ {manual, schema_map}``): a human / schema tag,
      weight ``1.0`` — the strongest signal.
    - **silver** (``source='llm'``, ``manual_review=false``, calibrated
      ``confidence ≥ min_silver_confidence``): a prior model guess,
      discounted by ``silver_factor`` to damp the echo / feedback loop.

    Borrowed nodes are aggregated (max score; gold via label wins over
    silver), sorted desc, capped to ``node_cap`` (paper's C2C2T ≤5). Each
    carries the bridging neighbour fact as an exemplar. Returns
    ``[(node_id, score, via, exemplars), ...]``.
    """

    rows = (
        await session.execute(
            select(AtomicFact.id, ContentEmbedding.embedding, AtomicFact.text)
            .join(ContentEmbedding, ContentEmbedding.entity_id == AtomicFact.id)
            .where(
                ContentEmbedding.entity_kind == "atomic_fact",
                ContentEmbedding.embedding_version == embedding_version,
                AtomicFact.course_id == course_id,
            )
        )
    ).all()

    exclude_fact_id = (
        exclude_entity[1]
        if exclude_entity is not None and exclude_entity[0] == "atomic_fact"
        else None
    )

    scored: list[tuple[int, float, str]] = []
    for fid, vec, text in rows:
        if fid == exclude_fact_id or not vec or len(vec) != len(query_embedding):
            continue
        s = cosine(list(vec), query_embedding)
        if s < min_score:
            continue
        scored.append((fid, s, text or ""))
    scored.sort(key=lambda t: t[1], reverse=True)
    scored = scored[:neighbor_k]
    if not scored:
        return []

    neighbor_ids = [fid for fid, _, _ in scored]
    tag_rows = (
        (
            await session.execute(
                select(AtomicFactTag).where(AtomicFactTag.atomic_fact_id.in_(neighbor_ids))
            )
        )
        .scalars()
        .all()
    )
    tags_by_fact: dict[int, list[AtomicFactTag]] = {}
    for t in tag_rows:
        tags_by_fact.setdefault(t.atomic_fact_id, []).append(t)

    # node_id -> {'score', 'via', 'exemplars'}
    agg: dict[int, dict] = {}
    for fid, cos, text in scored:
        for t in tags_by_fact.get(fid, []):
            if t.source in ("manual", "schema_map"):
                weight, via = 1.0, "content-gold"
            elif (
                t.source == "llm"
                and not t.manual_review
                and t.confidence is not None
                and float(t.confidence) >= min_silver_confidence
            ):
                weight, via = silver_factor, "content-silver"
            else:
                continue
            score = cos * weight
            ex = Exemplar(entity_kind="atomic_fact", entity_id=fid, text=text, score=cos)
            cur = agg.get(t.node_id)
            if cur is None:
                agg[t.node_id] = {"score": score, "via": via, "exemplars": [ex]}
            else:
                if score > cur["score"]:
                    cur["score"] = score
                if via == "content-gold":
                    cur["via"] = "content-gold"
                if len(cur["exemplars"]) < neighbor_exemplars_per_node:
                    cur["exemplars"].append(ex)

    items = sorted(agg.items(), key=lambda kv: kv[1]["score"], reverse=True)[:node_cap]
    return [
        (nid, d["score"], d["via"], d["exemplars"][:neighbor_exemplars_per_node])
        for nid, d in items
    ]


async def retrieve_candidates(
    session: AsyncSession,
    *,
    course_id: int,
    query_embedding: list[float],
    top_k: int = 10,
    edge_expansion: bool = True,
    content_expansion: bool = True,
    exemplars_per_node: int = 0,
    embedding_version: str | None = None,
    min_exemplar_confidence: float = 0.7,
    min_score: float = DEFAULT_MIN_SCORE,
    edge_top_n: int = DEFAULT_EDGE_TOP_N,
    content_neighbor_k: int = DEFAULT_CONTENT_NEIGHBOR_K,
    content_node_cap: int = DEFAULT_CONTENT_NODE_CAP,
    silver_factor: float = DEFAULT_SILVER_FACTOR,
    min_silver_confidence: float = DEFAULT_MIN_SILVER_CONFIDENCE,
    exclude_entity: tuple[str, int] | None = None,
) -> RecallResult:
    """Build the V-L3 constrained candidate set for a tagging prompt.

    Three recall paths merge, deduped by node (first wins):
      1. **C2T** — cosine vs outline-node vectors, floored at ``min_score``
         (D), capped at ``top_k``.
      2. **C2C2T** (A) — similar already-tagged facts → their tags, gold /
         silver weighted, capped at ``content_node_cap``.
      3. **T2T** — similarity-edge neighbours of the C2T hits, capped at
         ``edge_top_n`` (E).

    Args:
        course_id: scope all recall to this course's outline / content.
        query_embedding: the entity-being-tagged's vector (use
            :func:`load_embedding`). Must match ``embedding_version``'s dim.
        top_k: cap on the C2T embedding rank.
        edge_expansion / content_expansion: toggle T2T / C2C2T paths.
        exemplars_per_node: extra prior-tag (question) exemplars appended
            per candidate (0 = off); additive to C2C2T neighbour exemplars.
        embedding_version: defaults to :func:`current_version` (V-E1 stamp).
        min_score: δ floor (D) for embedding + content cosine.
        edge_top_n / content_node_cap: fan-out caps (E / A).
        silver_factor / min_silver_confidence: C2C2T silver-hop discount +
            gate (A).
        exclude_entity: ``(kind, id)`` of the target, skipped as its own
            content neighbour.

    Returns ``RecallResult`` — candidates score-desc within path, ordered
    C2T → C2C2T → T2T, exemplars attached.
    """

    version = embedding_version or current_version()

    emb_hits = await _embedding_candidates(
        session,
        course_id=course_id,
        query_embedding=query_embedding,
        top_k=top_k,
        embedding_version=version,
        min_score=min_score,
    )
    embedding_node_ids = {nid for nid, _ in emb_hits}

    content_hits: list[tuple[int, float, str, list[Exemplar]]] = []
    if content_expansion:
        content_hits = await _content_candidates(
            session,
            course_id=course_id,
            query_embedding=query_embedding,
            embedding_version=version,
            neighbor_k=content_neighbor_k,
            node_cap=content_node_cap,
            min_score=min_score,
            silver_factor=silver_factor,
            min_silver_confidence=min_silver_confidence,
            exclude_entity=exclude_entity,
        )

    edge_hits: list[tuple[int, float]] = []
    if edge_expansion and embedding_node_ids:
        edge_hits = await _expand_via_edges(
            session,
            seed_node_ids=embedding_node_ids,
            exclude=set(),  # we dedupe against seen below
            top_n=edge_top_n,
        )

    node_map = await _load_course_node_map(session, course_id=course_id)

    candidates: list[Candidate] = []
    seen: set[int] = set()

    def _add(node_id: int, score: float, via: str, exemplars: list[Exemplar]) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        candidates.append(
            Candidate(
                node_id=node_id,
                path=_render_path(node_id, node_map),
                score=score,
                via=via,
                exemplars=list(exemplars),
            )
        )

    for node_id, score in emb_hits:
        _add(node_id, score, "embedding", [])
    for node_id, score, via, exemplars in content_hits:
        _add(node_id, score, via, exemplars)
    for node_id, score in edge_hits:
        _add(node_id, score, "edge", [])

    if exemplars_per_node > 0:
        for cand in candidates:
            q_ex = await _exemplars_for_node(
                session,
                node_id=cand.node_id,
                limit=exemplars_per_node,
                min_confidence=min_exemplar_confidence,
            )
            if q_ex:
                cand.exemplars = cand.exemplars + q_ex

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
        lines.append(f"{i}. [{cand.via}, score={cand.score:.3f}] {label}")
        if include_exemplars and cand.exemplars:
            for ex in cand.exemplars:
                snippet = ex.text.strip().replace("\n", " ")
                if len(snippet) > max_exemplar_chars:
                    snippet = snippet[: max_exemplar_chars - 1] + "…"
                lines.append(f"   - exemplar [{ex.entity_kind}] (score={ex.score:.2f}): {snippet}")
    return "\n".join(lines)
