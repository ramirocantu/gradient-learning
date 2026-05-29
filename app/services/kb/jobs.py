"""LLM4Tag runner cores — embed + grounded-tag (T50, V-L3, V69, V-E1, V41).

Two session-accepting orchestrators the scheduler wraps:

- :func:`embed_pending` — embed every outline_node / atomic_fact / question
  that lacks a current-version ``content_embeddings`` row (V-E1). Outline-node
  vectors are the recall layer's candidate index, so this must run before
  tagging can find anything.
- :func:`tag_pending` — for each untagged atomic_fact (and, when the course is
  unambiguous, each ``needs_categorization`` question): recall candidates
  (V-L3) → grounded pick + inline V69 calibration → persist (V-T2/V-T3,
  ``persist_grounded_tags`` denormalizes ``atomic_facts.node_id``).

Calibration is inline (``generate_grounded_tags`` → ``calibrate_tag``); there
is no separate calibrate runner (§I — calibration runs inline in grounded).

V41: per-item failures are caught and recorded so the scheduler still reaches
``commit()`` and marks the run ``succeeded`` (partial). V16: the OpenAI clients
are injected so tests mock at the SDK boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.atomic_fact import AtomicFact
from app.models.atomic_fact_tag import AtomicFactTag
from app.models.captures import Question
from app.models.content_embedding import ContentEmbedding
from app.models.outline import OUTLINE_PATH_DELIMITER, Course, OutlineNode
from app.services.kb.embeddings import current_version, embed_and_persist
from app.services.kb.persist_tags import ATOMIC_FACT, QUESTION, persist_grounded_tags
from app.services.kb.recall import load_embedding, retrieve_candidates
from app.services.llm.grounded import generate_grounded_tags

_logger = logging.getLogger("app.services.kb.jobs")

# C: the live job runs few-shot exemplars ON (the design kept SRKI but the
# scheduler had been calling recall with the 0 default — SRKI inert). 3 prior
# calibrated tags per candidate is the paper's retrieved-ICL injection budget.
EXEMPLARS_PER_NODE = 3


@dataclass
class EmbedReport:
    embedded: int = 0
    reused: int = 0
    tokens: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def partial_failure(self) -> bool:
        return bool(self.failures)


@dataclass
class TagReport:
    facts_tagged: int = 0
    facts_skipped_no_embedding: int = 0
    questions_tagged: int = 0
    questions_skipped: int = 0
    tags_persisted: int = 0
    manual_review_flagged: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def partial_failure(self) -> bool:
        return bool(self.failures)


# --------------------------------------------------------------------------- #
# embed_pending (V-E1)
# --------------------------------------------------------------------------- #


def _render_node_path(node: OutlineNode, by_id: dict[int, OutlineNode]) -> str:
    """Walk ``parent_id`` to the root, rendering the ``>>``-joined path (V-O4).

    Mirrors ``recall._render_path`` but returns the leaf name alone if an
    ancestor is missing rather than ``None`` — we always want *some* text to
    embed.
    """

    segs = [node.name]
    cur: OutlineNode | None = node
    while cur is not None and cur.parent_id is not None:
        cur = by_id.get(cur.parent_id)
        if cur is None:
            break
        segs.append(cur.name)
    return OUTLINE_PATH_DELIMITER.join(reversed(segs))


async def _embed_outline_nodes(
    session: AsyncSession,
    *,
    openai_client: Any,
    version: str,
    report: EmbedReport,
) -> None:
    """Embed outline nodes by their FULL ``>>`` path, not the bare leaf name
    (B). The recall cosine matches a fact's prose against the tag's *meaning*;
    an ancestor-qualified path ("Metabolism >> Glycolysis >> Regulation")
    carries far more of that than the leaf token alone. Changing this text
    invalidates prior vectors ⇒ bump :func:`current_version` (V-E1)."""

    nodes = (await session.execute(select(OutlineNode))).scalars().all()
    by_id = {n.id: n for n in nodes}
    embedded_ids = set(
        (
            await session.execute(
                select(ContentEmbedding.entity_id).where(
                    ContentEmbedding.entity_kind == "outline_node",
                    ContentEmbedding.embedding_version == version,
                )
            )
        ).scalars().all()
    )

    for node in nodes:
        if node.id in embedded_ids:
            continue
        text = _render_node_path(node, by_id)
        if not text.strip():
            continue
        try:
            result = await embed_and_persist(
                session,
                openai_client=openai_client,
                entity_kind="outline_node",
                entity_id=node.id,
                text=text,
                version=version,
            )
        except Exception as exc:  # noqa: BLE001 — V41 per-item isolation
            report.failures.append(f"outline_node:{node.id}: {exc}")
            _logger.warning("embed_pending: failed on outline_node:%s: %s", node.id, exc)
            continue
        if result.reused:
            report.reused += 1
        else:
            report.embedded += 1
            report.tokens += result.tokens


async def embed_pending(
    session: AsyncSession,
    *,
    openai_client: Any,
    version: str | None = None,
) -> EmbedReport:
    """Embed every outline_node / atomic_fact / question missing a
    ``content_embeddings`` row for ``version`` (defaults to
    :func:`current_version`). Idempotent (``embed_and_persist`` skips an
    existing triple); V41 isolates per-item failures."""

    version = version or current_version()
    report = EmbedReport()

    # Outline nodes embed their >>-path (B); see _embed_outline_nodes.
    await _embed_outline_nodes(
        session, openai_client=openai_client, version=version, report=report
    )

    # (entity_kind, id column, text column) — surface text embedded directly.
    specs = [
        (ATOMIC_FACT, AtomicFact.id, AtomicFact.text),
        (QUESTION, Question.id, Question.stem_plain),
    ]
    for kind, id_col, text_col in specs:
        embedded_ids = select(ContentEmbedding.entity_id).where(
            ContentEmbedding.entity_kind == kind,
            ContentEmbedding.embedding_version == version,
        )
        rows = (
            await session.execute(
                select(id_col, text_col).where(id_col.notin_(embedded_ids))
            )
        ).all()
        for entity_id, text in rows:
            if not (text or "").strip():
                continue
            try:
                result = await embed_and_persist(
                    session,
                    openai_client=openai_client,
                    entity_kind=kind,
                    entity_id=entity_id,
                    text=text,
                    version=version,
                )
            except Exception as exc:  # noqa: BLE001 — V41 per-item isolation
                report.failures.append(f"{kind}:{entity_id}: {exc}")
                _logger.warning("embed_pending: failed on %s:%s: %s", kind, entity_id, exc)
                continue
            if result.reused:
                report.reused += 1
            else:
                report.embedded += 1
                report.tokens += result.tokens

    _logger.info(
        "embed_pending: embedded=%d reused=%d tokens=%d failures=%d",
        report.embedded,
        report.reused,
        report.tokens,
        len(report.failures),
    )
    return report


# --------------------------------------------------------------------------- #
# tag_pending (V-L3, V69, V-T2/V-T3)
# --------------------------------------------------------------------------- #


async def _tag_one(
    session: AsyncSession,
    *,
    entity_kind: str,
    entity_id: int,
    course_id: int,
    entity_text: str,
    version: str,
    tagging_client: Any,
    calibrator_client: Any,
    report: TagReport,
) -> bool:
    """Recall → grounded → persist one entity. Returns True if an LLM pass ran
    (embedding present), False if skipped for want of an embedding."""

    embedding = await load_embedding(
        session, entity_kind=entity_kind, entity_id=entity_id, embedding_version=version
    )
    if embedding is None:
        return False

    recall = await retrieve_candidates(
        session,
        course_id=course_id,
        query_embedding=embedding,
        embedding_version=version,
        exemplars_per_node=EXEMPLARS_PER_NODE,  # C: SRKI on in the live job
        exclude_entity=(entity_kind, entity_id),  # A: don't recall self
    )
    result = await generate_grounded_tags(
        entity_text=entity_text,
        recall_result=recall,
        tagging_client=tagging_client,
        calibrator_client=calibrator_client,
    )
    persisted = await persist_grounded_tags(
        session, entity_kind=entity_kind, entity_id=entity_id, result=result
    )
    report.tags_persisted += persisted.persisted
    report.manual_review_flagged += persisted.manual_review_flagged
    report.input_tokens += result.input_tokens
    report.output_tokens += result.output_tokens
    report.cached_tokens += result.cached_tokens
    return True


async def tag_pending(
    session: AsyncSession,
    *,
    tagging_client: Any,
    calibrator_client: Any | None = None,
    version: str | None = None,
) -> TagReport:
    """Tag untagged atomic_facts + needs_categorization questions (V-L3, V69).
    Atomic facts always carry their ``course_id``. A question is scoped to its
    OWN ``course_id`` (V-CAP2 — the capture adapter stamps it from the chosen
    course), so multi-course installs tag normally. A question with no
    ``course_id`` (legacy/unscoped) falls back to the sole course iff exactly
    one exists, else is skipped (course ambiguous) with a log. V41 isolates
    per-entity failures."""

    version = version or current_version()
    calibrator_client = calibrator_client or tagging_client
    report = TagReport()

    # --- atomic facts: untagged (no primary node, no prior llm tag) ---------
    tagged_facts = select(AtomicFactTag.atomic_fact_id).where(AtomicFactTag.source == "llm")
    facts = (
        await session.execute(
            select(AtomicFact).where(
                AtomicFact.node_id.is_(None),
                AtomicFact.id.notin_(tagged_facts),
            )
        )
    ).scalars().all()
    for fact in facts:
        try:
            ran = await _tag_one(
                session,
                entity_kind=ATOMIC_FACT,
                entity_id=fact.id,
                course_id=fact.course_id,
                entity_text=fact.text,
                version=version,
                tagging_client=tagging_client,
                calibrator_client=calibrator_client,
                report=report,
            )
        except Exception as exc:  # noqa: BLE001 — V41
            report.failures.append(f"atomic_fact:{fact.id}: {exc}")
            _logger.warning("tag_pending: fact %s failed: %s", fact.id, exc)
            continue
        if ran:
            report.facts_tagged += 1
        else:
            report.facts_skipped_no_embedding += 1

    # --- questions: scope to the question's OWN course_id (V-CAP2). A question
    # with no course_id (legacy/unscoped) falls back to the sole course iff
    # exactly one exists, else is skipped (course ambiguous) -----------------
    questions = (
        await session.execute(
            select(Question).where(Question.needs_categorization.is_(True))
        )
    ).scalars().all()
    if questions:
        course_count = (
            await session.execute(select(func.count()).select_from(Course))
        ).scalar_one()
        fallback_course_id: int | None = None
        if course_count == 1:
            fallback_course_id = (await session.execute(select(Course.id))).scalar_one()

        ambiguous_skipped = 0
        for q in questions:
            course_id = q.course_id if q.course_id is not None else fallback_course_id
            if course_id is None:
                report.questions_skipped += 1
                ambiguous_skipped += 1
                continue
            try:
                ran = await _tag_one(
                    session,
                    entity_kind=QUESTION,
                    entity_id=q.id,
                    course_id=course_id,
                    entity_text=q.stem_plain,
                    version=version,
                    tagging_client=tagging_client,
                    calibrator_client=calibrator_client,
                    report=report,
                )
            except Exception as exc:  # noqa: BLE001 — V41
                report.failures.append(f"question:{q.id}: {exc}")
                _logger.warning("tag_pending: question %s failed: %s", q.id, exc)
                continue
            if ran:
                q.needs_categorization = False
                report.questions_tagged += 1
            else:
                report.questions_skipped += 1
        await session.flush()
        if ambiguous_skipped:
            _logger.info(
                "tag_pending: %d unscoped question(s) skipped — course ambiguous "
                "(%d courses, no course_id stamped)",
                ambiguous_skipped,
                course_count,
            )

    _logger.info(
        "tag_pending: facts_tagged=%d facts_no_emb=%d q_tagged=%d q_skipped=%d "
        "tags=%d flagged=%d failures=%d",
        report.facts_tagged,
        report.facts_skipped_no_embedding,
        report.questions_tagged,
        report.questions_skipped,
        report.tags_persisted,
        report.manual_review_flagged,
        len(report.failures),
    )
    return report
