"""Data-fetching helpers for the CC drilldown page — partially FENCED
(T17, V-RB1, V-O5).

Outline-dependent helpers (`get_cc_info`, `get_questions_for_cc`,
`get_questions_for_topic_subtree`, `list_question_cards`, `list_all_ccs`,
`list_topics_for_cc`, `_filter_topics_for_cc`, `_tags_summaries_for`) are
FENCED — the dashboard mastery / topics routes that consume them are
unmounted in `app/web/dashboard/main.py`; restoration is tied to the T34
SPA reassessment.

Outline-free helpers (`get_question_card`, `get_full_question`,
`get_question_detail`, `media_by_hash_for_question`) remain real — they
read `questions` / `passages` / `attempts` / `media` / `attempt_notes`
directly and back the `questions` dashboard route which stays mounted.

The FENCED helpers stay importable so the route modules load even while
unmounted; they return empty / None / placeholder values and log a
FENCED warning. This file is partially FENCED, not a stub: behavior is
deliberate, not in-progress.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt, Passage, Question
from app.models.media import Media
from app.services.attempt_notes import list_notes

logger = logging.getLogger(__name__)


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
    kind: str  # FENCED — legacy 3-target shape; node_id port (T34) reshapes.
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


@dataclass(frozen=True)
class QuestionDetail:
    question: Question
    passage: Passage | None
    latest_attempt: Attempt | None
    tags: list[TagSummary]
    notes: list[AttemptNote]


@dataclass(frozen=True)
class TopicRow:
    topic_id: int
    name: str
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def truncate_to_chars(text: str, n: int) -> tuple[str, bool]:
    if len(text) <= n:
        return text, False
    return text[:n].rstrip() + "…", True


# --------------------------------------------------------------------------- #
# Outline-dependent — FENCED.
# --------------------------------------------------------------------------- #


_FENCED_MSG = (
    "dashboard.services.drilldown outline-dependent helpers are FENCED "
    "(T17, V-RB1) — consuming routes unmounted; restoration tied to T34"
)


async def get_cc_info(session: AsyncSession, cc_code: str) -> CCInfo | None:
    """FENCED — returns None."""
    logger.warning(_FENCED_MSG)
    return None


async def get_questions_for_cc(session: AsyncSession, cc_id: int) -> list[int]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def get_questions_for_topic_subtree(
    session: AsyncSession, topic_id: int
) -> list[int]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def list_question_cards(session: AsyncSession, *args, **kwargs) -> list[QuestionCard]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def list_all_ccs(session: AsyncSession) -> list[Any]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


async def list_topics_for_cc(session: AsyncSession, cc_code: str) -> list[Any]:
    """FENCED — returns empty list."""
    logger.warning(_FENCED_MSG)
    return []


def _filter_topics_for_cc(by_topic: list[Any], cc_code: str) -> list[TopicRow]:
    """FENCED — by_topic comes from compute_mastery (also FENCED); returns []."""
    return []


async def _tags_summaries_for(
    session: AsyncSession, question_ids: list[int]
) -> dict[int, list[TagSummary]]:
    """FENCED — returns empty mapping."""
    logger.warning(_FENCED_MSG)
    return {}


# --------------------------------------------------------------------------- #
# Outline-free question detail — preserved.
# --------------------------------------------------------------------------- #


async def get_question_card(
    session: AsyncSession, question_id: int
) -> QuestionCard | None:
    """Minimal preserved version — fetches the question + latest attempt; tags
    list is empty (depends on `_tags_summaries_for` which is FENCED)."""
    q = await session.get(Question, question_id)
    if q is None:
        return None
    last = await _last_attempts_for(session, [question_id])
    preview, truncated = truncate_to_chars(q.stem_plain or "", 200)
    return QuestionCard(
        question_id=q.id,
        qid=q.qid,
        stem_preview=preview,
        stem_truncated=truncated,
        last_attempt=last.get(question_id),
        tags=[],
    )


async def _last_attempts_for(
    session: AsyncSession, question_ids: list[int]
) -> dict[int, QuestionAttemptSummary]:
    if not question_ids:
        return {}
    rows = (
        await session.execute(
            select(Attempt)
            .where(Attempt.question_id.in_(question_ids))
            .order_by(Attempt.question_id, desc(Attempt.attempted_at))
        )
    ).scalars().all()
    out: dict[int, QuestionAttemptSummary] = {}
    seen: set[int] = set()
    for a in rows:
        if a.question_id in seen:
            continue
        seen.add(a.question_id)
        # Look up correct_choice on the question — one query per question is OK here.
        q = await session.get(Question, a.question_id)
        out[a.question_id] = QuestionAttemptSummary(
            is_correct=a.is_correct,
            selected_choice=a.selected_choice,
            correct_choice=q.correct_choice if q else "",
            attempted_at=a.attempted_at,
        )
    return out


def _hashes_in_html(*htmls: str | None) -> set[str]:
    found: set[str] = set()
    for h in htmls:
        if h:
            found.update(_HASH_ATTR_RE.findall(h))
    return found


async def media_by_hash_for_question(
    session: AsyncSession, question_id: int
) -> dict[str, str]:
    """{content_hash: local_path} for media a question references."""
    q = await session.get(Question, question_id)
    if q is None:
        return {}

    media_ids: set[int] = set()
    for choice in q.choices or []:
        for mid in choice.get("media_ids") or []:
            if isinstance(mid, int):
                media_ids.add(mid)

    out: dict[str, str] = {}
    if media_ids:
        rows = (
            await session.execute(
                select(Media.content_hash, Media.local_path).where(Media.id.in_(media_ids))
            )
        ).all()
        out.update({h: p for h, p in rows})
    return out


async def get_full_question(
    session: AsyncSession, question_id: int
) -> FullQuestion | None:
    q = await session.get(Question, question_id)
    if q is None:
        return None
    passage_html = ""
    if q.passage_id is not None:
        p = await session.get(Passage, q.passage_id)
        if p is not None:
            passage_html = p.html
    last = await _last_attempts_for(session, [question_id])
    last_attempt = last.get(question_id)
    selected = last_attempt.selected_choice if last_attempt else ""
    choices = [
        FullChoice(
            key=c.get("key", ""),
            html=c.get("html", ""),
            is_correct=c.get("key") == q.correct_choice,
            is_selected=c.get("key") == selected,
        )
        for c in (q.choices or [])
    ]
    return FullQuestion(
        question_id=q.id,
        qid=q.qid,
        stem_html=q.stem_html,
        explanation_html=q.explanation_html or "",
        passage_html=passage_html,
        choices=choices,
        last_attempt=last_attempt,
    )


async def get_question_detail(
    session: AsyncSession, question_id: int
) -> QuestionDetail | None:
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
    notes: list[AttemptNote] = []
    if latest_attempt is not None:
        notes = await list_notes(session, attempt_id=latest_attempt.id)
    return QuestionDetail(
        question=q,
        passage=passage,
        latest_attempt=latest_attempt,
        tags=[],
        notes=notes,
    )
