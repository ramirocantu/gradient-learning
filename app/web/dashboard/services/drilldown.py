"""Data-fetching helpers for the CC drilldown page.

Pure async functions over an ``AsyncSession``. All ORM reads live here so
the route module stays declarative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import desc, func, select, union
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Passage, Question, QuestionTag
from app.models.media import Media
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic
from app.services.attempt_notes import list_notes


_HASH_ATTR_RE = re.compile(r'data-media-content-hash="([^"]+)"')


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CCInfo:
    id: int
    code: str
    name: str
    section_code: str
    section_name: str


@dataclass(frozen=True)
class TagSummary:
    tag_id: int
    label: str
    source: str
    kind: str  # 'topic' | 'content_category' | 'skill'
    confidence: float = 0.0
    rationale: str | None = None


@dataclass(frozen=True)
class QuestionAttemptSummary:
    is_correct: bool
    selected_choice: str
    correct_choice: str
    attempted_at: datetime


@dataclass(frozen=True)
class QuestionCard:
    question_id: int
    qid: str
    stem_preview: str
    stem_truncated: bool
    last_attempt: QuestionAttemptSummary | None
    tags: list[TagSummary]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def truncate_to_chars(text: str, n: int) -> tuple[str, bool]:
    """Return ``(snippet, was_truncated)``."""
    text = (text or "").strip()
    if len(text) <= n:
        return text, False
    return text[:n].rstrip() + "...", True


async def get_cc_info(session: AsyncSession, cc_code: str) -> CCInfo | None:
    """Look up a CC by code and return its descriptive info + section."""
    stmt = (
        select(
            ContentCategory.id,
            ContentCategory.code,
            ContentCategory.name,
            Section.code.label("section_code"),
            Section.name.label("section_name"),
        )
        .join(
            FoundationalConcept,
            FoundationalConcept.id == ContentCategory.foundational_concept_id,
        )
        .join(Section, Section.id == FoundationalConcept.section_id)
        .where(ContentCategory.code == cc_code)
    )
    row = (await session.execute(stmt)).one_or_none()
    if row is None:
        return None
    return CCInfo(
        id=row.id,
        code=row.code,
        name=row.name,
        section_code=row.section_code,
        section_name=row.section_name,
    )


async def get_questions_for_cc(session: AsyncSession, cc_id: int) -> list[int]:
    """All distinct question_ids tagged to this CC (direct OR via topic)."""
    direct = select(QuestionTag.question_id).where(QuestionTag.content_category_id == cc_id)
    via_topic = (
        select(QuestionTag.question_id)
        .join(Topic, Topic.id == QuestionTag.topic_id)
        .where(Topic.content_category_id == cc_id)
    )
    sub = union(direct, via_topic).subquery()
    stmt = select(sub.c.question_id).distinct()
    rows = (await session.execute(stmt)).all()
    return [r.question_id for r in rows]


async def get_questions_for_topic_subtree(session: AsyncSession, topic_id: int) -> list[int]:
    """All distinct question_ids with a tag whose topic_id ∈ subtree(topic_id).

    Subtree per §V31 — recursive CTE on `topics.parent_topic_id`.
    Direct-CC tags do NOT contribute (topic-level scope only, since
    these questions have no resolved topic_id at this granularity).
    """
    from sqlalchemy import text

    sql = text(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT id FROM topics WHERE id = :topic_id
            UNION ALL
            SELECT child.id
            FROM topics child
            JOIN subtree s ON child.parent_topic_id = s.id
        )
        SELECT DISTINCT qt.question_id
        FROM question_tags qt
        WHERE qt.topic_id IN (SELECT id FROM subtree)
        """
    )
    rows = (await session.execute(sql, {"topic_id": topic_id})).all()
    return [int(r[0]) for r in rows]


async def list_question_cards(
    session: AsyncSession,
    question_ids: list[int],
    *,
    page: int,
    per_page: int,
) -> tuple[list[QuestionCard], int]:
    """Return paginated cards (most recent attempt first; no-attempt last)."""
    if not question_ids:
        return [], 0

    # Pre-aggregate: most recent attempt per question.
    last_attempt_sub = (
        select(
            Attempt.question_id.label("question_id"),
            func.max(Attempt.attempted_at).label("last_attempted_at"),
        )
        .where(Attempt.question_id.in_(question_ids))
        .group_by(Attempt.question_id)
        .subquery()
    )

    order_col = func.coalesce(last_attempt_sub.c.last_attempted_at, None)

    stmt = (
        select(Question, last_attempt_sub.c.last_attempted_at)
        .outerjoin(last_attempt_sub, last_attempt_sub.c.question_id == Question.id)
        .where(Question.id.in_(question_ids))
        .order_by(order_col.desc().nulls_last(), Question.id.desc())
    )

    total = len(question_ids)
    offset = (page - 1) * per_page
    stmt = stmt.limit(per_page).offset(offset)

    rows = (await session.execute(stmt)).all()
    qs: list[Question] = [r[0] for r in rows]
    if not qs:
        return [], total

    qids = [q.id for q in qs]
    last_attempts = await _last_attempts_for(session, qids)
    tags_by_q = await _tags_summaries_for(session, qids)

    cards: list[QuestionCard] = []
    for q in qs:
        snippet, truncated = truncate_to_chars(q.stem_plain, 200)
        last = last_attempts.get(q.id)
        cards.append(
            QuestionCard(
                question_id=q.id,
                qid=q.qid,
                stem_preview=snippet,
                stem_truncated=truncated,
                last_attempt=last,
                tags=tags_by_q.get(q.id, []),
            )
        )
    return cards, total


async def get_question_card(session: AsyncSession, question_id: int) -> QuestionCard | None:
    """Single-card lookup for the post-retag refresh."""
    cards, _ = await list_question_cards(session, [question_id], page=1, per_page=1)
    return cards[0] if cards else None


async def _last_attempts_for(
    session: AsyncSession, question_ids: list[int]
) -> dict[int, QuestionAttemptSummary]:
    if not question_ids:
        return {}

    sub = (
        select(
            Attempt.question_id,
            func.max(Attempt.attempted_at).label("max_at"),
        )
        .where(Attempt.question_id.in_(question_ids))
        .group_by(Attempt.question_id)
        .subquery()
    )
    stmt = (
        select(Attempt, Question.correct_choice)
        .join(
            sub,
            (sub.c.question_id == Attempt.question_id) & (sub.c.max_at == Attempt.attempted_at),
        )
        .join(Question, Question.id == Attempt.question_id)
    )
    rows = (await session.execute(stmt)).all()
    out: dict[int, QuestionAttemptSummary] = {}
    for attempt, correct_choice in rows:
        # Guard against duplicate timestamps — first wins.
        out.setdefault(
            attempt.question_id,
            QuestionAttemptSummary(
                is_correct=attempt.is_correct,
                selected_choice=attempt.selected_choice,
                correct_choice=correct_choice,
                attempted_at=attempt.attempted_at,
            ),
        )
    return out


async def _tags_summaries_for(
    session: AsyncSession, question_ids: list[int]
) -> dict[int, list[TagSummary]]:
    if not question_ids:
        return {}

    rows = (
        (
            await session.execute(
                select(QuestionTag)
                .where(QuestionTag.question_id.in_(question_ids))
                .where(QuestionTag.is_overridden == False)  # noqa: E712
                .order_by(QuestionTag.question_id, QuestionTag.source, QuestionTag.id)
            )
        )
        .scalars()
        .all()
    )

    if not rows:
        return {qid: [] for qid in question_ids}

    topic_ids = {r.topic_id for r in rows if r.topic_id is not None}
    cc_ids = {r.content_category_id for r in rows if r.content_category_id is not None}

    topics: dict[int, Topic] = {}
    if topic_ids:
        topics = {
            t.id: t
            for t in (await session.execute(select(Topic).where(Topic.id.in_(topic_ids)))).scalars()
        }
        cc_ids |= {t.content_category_id for t in topics.values()}

    ccs: dict[int, ContentCategory] = {}
    if cc_ids:
        ccs = {
            cc.id: cc
            for cc in (
                await session.execute(select(ContentCategory).where(ContentCategory.id.in_(cc_ids)))
            ).scalars()
        }

    out: dict[int, list[TagSummary]] = {qid: [] for qid in question_ids}
    for r in rows:
        if r.topic_id is not None:
            topic = topics.get(r.topic_id)
            cc = ccs.get(topic.content_category_id) if topic else None
            cc_code = cc.code if cc else "?"
            label = f"{cc_code} / {topic.name}" if topic else f"topic#{r.topic_id}"
            kind = "topic"
        elif r.content_category_id is not None:
            cc = ccs.get(r.content_category_id)
            label = f"{cc.code} — {cc.name}" if cc else f"cc#{r.content_category_id}"
            kind = "content_category"
        elif r.skill is not None:
            label = f"Skill {r.skill}"
            kind = "skill"
        else:
            label = "(no target)"
            kind = "unknown"
        out[r.question_id].append(
            TagSummary(
                tag_id=r.id,
                label=label,
                source=r.source,
                kind=kind,
                confidence=float(r.confidence) if r.confidence is not None else 0.0,
                rationale=r.rationale,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Topic dropdown options
# --------------------------------------------------------------------------- #


async def list_all_ccs(session: AsyncSession) -> list[ContentCategory]:
    rows = (await session.execute(select(ContentCategory).order_by(ContentCategory.code))).scalars()
    return list(rows)


async def list_topics_for_cc(session: AsyncSession, cc_code: str) -> list[Topic]:
    stmt = (
        select(Topic)
        .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
        .where(ContentCategory.code == cc_code)
        .order_by(Topic.name)
    )
    return list((await session.execute(stmt)).scalars())


# --------------------------------------------------------------------------- #
# Full-question rendering data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FullChoice:
    key: str
    html: str
    is_correct: bool
    is_selected: bool


@dataclass(frozen=True)
class FullQuestion:
    question_id: int
    qid: str
    stem_html: str
    explanation_html: str
    passage_html: str
    choices: list[FullChoice]
    last_attempt: QuestionAttemptSummary | None


def _hashes_in_html(*htmls: str | None) -> set[str]:
    found: set[str] = set()
    for h in htmls:
        if h:
            found.update(_HASH_ATTR_RE.findall(h))
    return found


async def media_by_hash_for_question(session: AsyncSession, question_id: int) -> dict[str, str]:
    """{content_hash: local_path} for media a question references."""
    q = await session.get(Question, question_id)
    if q is None:
        return {}

    media_ids: set[int] = set()
    for choice in q.choices or []:
        for mid in choice.get("media_ids") or []:
            if isinstance(mid, int):
                media_ids.add(mid)

    hashes: set[str] = set()
    if media_ids:
        rows = await session.execute(select(Media.content_hash).where(Media.id.in_(media_ids)))
        hashes.update(rows.scalars().all())

    passage_html = None
    if q.passage_id is not None:
        passage = await session.get(Passage, q.passage_id)
        if passage is not None:
            passage_html = passage.html

    hashes |= _hashes_in_html(q.stem_html, q.explanation_html, passage_html)

    if not hashes:
        return {}

    rows = await session.execute(
        select(Media.content_hash, Media.local_path).where(Media.content_hash.in_(hashes))
    )
    return {row.content_hash: row.local_path for row in rows}


async def get_full_question(session: AsyncSession, question_id: int) -> FullQuestion | None:
    q = await session.get(Question, question_id)
    if q is None:
        return None

    media_map = await media_by_hash_for_question(session, question_id)

    from app.web.dashboard.services.html_rewriter import rewrite_media_refs

    stem_html = rewrite_media_refs(q.stem_html or "", media_map)
    explanation_html = rewrite_media_refs(q.explanation_html or "", media_map)

    passage_html = ""
    if q.passage_id is not None:
        passage = await session.get(Passage, q.passage_id)
        if passage is not None:
            passage_html = rewrite_media_refs(passage.html or "", media_map)

    last = (
        await session.execute(
            select(Attempt)
            .where(Attempt.question_id == question_id)
            .order_by(desc(Attempt.attempted_at))
            .limit(1)
        )
    ).scalar_one_or_none()

    last_summary = None
    selected = None
    if last is not None:
        last_summary = QuestionAttemptSummary(
            is_correct=last.is_correct,
            selected_choice=last.selected_choice,
            correct_choice=q.correct_choice,
            attempted_at=last.attempted_at,
        )
        selected = last.selected_choice

    choices: list[FullChoice] = []
    for c in q.choices or []:
        key = c.get("key") or c.get("label") or ""
        choices.append(
            FullChoice(
                key=key,
                html=rewrite_media_refs(c.get("html") or "", media_map),
                is_correct=key == q.correct_choice,
                is_selected=key == selected,
            )
        )

    return FullQuestion(
        question_id=q.id,
        qid=q.qid,
        stem_html=stem_html,
        explanation_html=explanation_html,
        passage_html=passage_html,
        choices=choices,
        last_attempt=last_summary,
    )


# --------------------------------------------------------------------------- #
# Standalone question detail page (Ticket 6.6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuestionDetail:
    question: Question
    passage: Passage | None
    latest_attempt: Attempt | None
    tags: list[TagSummary]
    notes: list[AttemptNote]


async def get_question_detail(session: AsyncSession, question_id: int) -> QuestionDetail | None:
    """Fetch question + passage + latest attempt + tag summaries + notes for the detail page."""
    q = await session.get(Question, question_id)
    if q is None:
        return None

    passage: Passage | None = None
    if q.passage_id is not None:
        passage = await session.get(Passage, q.passage_id)

    latest_attempt = (
        await session.execute(
            select(Attempt)
            .where(Attempt.question_id == question_id)
            .order_by(desc(Attempt.attempted_at))
            .limit(1)
        )
    ).scalar_one_or_none()

    tags_map = await _tags_summaries_for(session, [question_id])
    tags = tags_map.get(question_id, [])

    notes: list[AttemptNote] = []
    if latest_attempt is not None:
        notes = await list_notes(session, attempt_id=latest_attempt.id)

    return QuestionDetail(
        question=q,
        passage=passage,
        latest_attempt=latest_attempt,
        tags=tags,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Topic-level table for the drilldown
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopicRow:
    topic_id: int
    name: str
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float


def _filter_topics_for_cc(by_topic: list[Any], cc_code: str) -> list[TopicRow]:
    """Pick out the AccuracyStat rows whose code matches this CC."""
    out: list[TopicRow] = []
    for stat in by_topic:
        if stat.code != cc_code:
            continue
        # AccuracyStat label is "{cc_code} — {topic_name}". Strip prefix.
        prefix = f"{cc_code} — "
        topic_name = stat.label[len(prefix) :] if stat.label.startswith(prefix) else stat.label
        out.append(
            TopicRow(
                topic_id=stat.target_id,
                name=topic_name,
                attempts=stat.attempts,
                correct=stat.correct,
                accuracy=stat.accuracy,
                wilson_lower=stat.wilson_lower,
            )
        )
    out.sort(key=lambda t: (t.wilson_lower, -t.attempts, t.name))
    return out
