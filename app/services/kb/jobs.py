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
from app.models.outline import Course, OutlineNode
from app.services.kb.embeddings import current_version, embed_and_persist
from app.services.kb.persist_tags import ATOMIC_FACT, QUESTION, persist_grounded_tags
from app.services.kb.recall import load_embedding, retrieve_candidates
from app.services.llm.grounded import generate_grounded_tags

_logger = logging.getLogger("app.services.kb.jobs")


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

    # (entity_kind, id column, text column) — text is what we embed.
    specs = [
        ("outline_node", OutlineNode.id, OutlineNode.name),
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
    """Tag untagged atomic_facts + (single-course) needs_categorization
    questions (V-L3, V69). Atomic facts always carry their ``course_id``;
    questions are only tagged when exactly one course exists (recall needs a
    course scope and a pre-tag question has no binding) — otherwise skipped
    with a log. V41 isolates per-entity failures."""

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

    # --- questions: only when the course is unambiguous ---------------------
    questions = (
        await session.execute(
            select(Question).where(Question.needs_categorization.is_(True))
        )
    ).scalars().all()
    if questions:
        course_count = (
            await session.execute(select(func.count()).select_from(Course))
        ).scalar_one()
        if course_count == 1:
            course_id = (await session.execute(select(Course.id))).scalar_one()
            for q in questions:
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
        else:
            report.questions_skipped += len(questions)
            _logger.info(
                "tag_pending: %d question(s) skipped — course ambiguous (%d courses)",
                len(questions),
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
