"""Topic-level analytics rollups (Ticket 5.1).

Pure async functions over an ``AsyncSession``. No HTTP concerns; the router in
``app.api.v1.analytics`` wraps these and serializes via Pydantic.

Multi-tag handling: a question may carry N topic tags + M direct CC tags. For
per-CC and per-section rollups, the CC set for a question is the union of
(direct CC tags) and (the CC reachable via each topic's ``content_category_id``).
A single attempt on a multi-CC question counts once per CC in that set, so
cross-CC totals can (correctly) exceed the total attempt count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Literal

from sqlalchemy import (
    Integer,
    and_,
    cast,
    func,
    literal,
    literal_column,
    select,
    text,
    union,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic


AccuracyKind = Literal["section", "content_category", "topic", "skill"]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AccuracyStat:
    """Per-dimension accuracy rollup.

    ``attempts`` = unique questions attempted in this dimension.
    ``correct`` = unique questions where the *latest* attempt was correct.
    ``accuracy`` = correct / attempts (0.0 when attempts == 0).
    """

    label: str
    code: str | None
    kind: AccuracyKind
    target_id: int | None
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float


@dataclass(frozen=True)
class TimingStat:
    median_seconds_discrete: float | None
    median_seconds_passage_based: float | None
    questions_over_target_discrete: int
    questions_over_target_passage: int


@dataclass(frozen=True)
class TrendPoint:
    period_start: date
    accuracy: float
    attempts: int


@dataclass(frozen=True)
class MasteryReport:
    by_section: list[AccuracyStat]
    by_content_category: list[AccuracyStat]
    by_topic: list[AccuracyStat]
    by_skill: list[AccuracyStat]
    timing: TimingStat
    trend_7d: list[TrendPoint]
    trend_30d: list[TrendPoint]
    uncategorized_question_count: int
    total_attempts: int
    total_questions: int


# --------------------------------------------------------------------------- #
# Wilson lower bound
# --------------------------------------------------------------------------- #


def wilson_lower(correct: int, attempts: int, z: float = 1.96) -> float:
    """95% Wilson score lower bound on the success-rate proportion."""
    if attempts == 0:
        return 0.0
    p = correct / attempts
    denominator = 1 + z**2 / attempts
    center = p + z**2 / (2 * attempts)
    margin = z * sqrt(p * (1 - p) / attempts + z**2 / (4 * attempts**2))
    return max(0.0, (center - margin) / denominator)


def _stat(
    *,
    label: str,
    code: str | None,
    kind: AccuracyKind,
    target_id: int | None,
    attempts: int,
    correct: int,
) -> AccuracyStat:
    accuracy = correct / attempts if attempts else 0.0
    return AccuracyStat(
        label=label,
        code=code,
        kind=kind,
        target_id=target_id,
        attempts=attempts,
        correct=correct,
        accuracy=accuracy,
        wilson_lower=wilson_lower(correct, attempts),
    )


# --------------------------------------------------------------------------- #
# Helpers — derive (question_id, cc_id) pairs from both direct CC tags and
# topic tags. This is the seed for both per-CC and per-section rollups.
# --------------------------------------------------------------------------- #


def _question_cc_pairs():
    """Returns a SELECT yielding distinct (question_id, content_category_id)."""
    direct = select(
        QuestionTag.question_id.label("question_id"),
        QuestionTag.content_category_id.label("content_category_id"),
    ).where(QuestionTag.content_category_id.is_not(None))

    via_topic = (
        select(
            QuestionTag.question_id.label("question_id"),
            Topic.content_category_id.label("content_category_id"),
        )
        .join(Topic, Topic.id == QuestionTag.topic_id)
        .where(QuestionTag.topic_id.is_not(None))
    )

    return union(direct, via_topic).subquery("question_ccs")


def _latest_attempts():
    """One row per question: the most recent attempt (by attempted_at)."""
    return (
        select(
            Attempt.question_id,
            Attempt.is_correct,
            Attempt.attempted_at,
            Attempt.time_seconds,
        )
        .distinct(Attempt.question_id)
        .order_by(Attempt.question_id, Attempt.attempted_at.desc())
        .subquery("latest_attempts")
    )


# --------------------------------------------------------------------------- #
# Per-content-category accuracy
# --------------------------------------------------------------------------- #


async def _by_content_category(session: AsyncSession) -> list[AccuracyStat]:
    qccs = _question_cc_pairs()
    latest = _latest_attempts()

    # Derive per-CC unique-question counts via the multi-tag union + latest attempt.
    cc_attempts = (
        select(
            qccs.c.content_category_id.label("cc_id"),
            func.count(latest.c.question_id).label("attempts"),
            func.coalesce(func.sum(cast(latest.c.is_correct, Integer)), 0).label("correct"),
        )
        .join(latest, latest.c.question_id == qccs.c.question_id)
        .group_by(qccs.c.content_category_id)
        .subquery("cc_attempts")
    )

    # Start FROM the outline table, LEFT JOIN to include zero-attempt CCs.
    stmt = (
        select(
            ContentCategory.id,
            ContentCategory.code,
            ContentCategory.name,
            func.coalesce(cc_attempts.c.attempts, 0).label("attempts"),
            func.coalesce(cc_attempts.c.correct, 0).label("correct"),
        )
        .outerjoin(cc_attempts, cc_attempts.c.cc_id == ContentCategory.id)
        .order_by(ContentCategory.code)
    )

    rows = (await session.execute(stmt)).all()
    return [
        _stat(
            label=f"{code} — {name}",
            code=code,
            kind="content_category",
            target_id=cc_id,
            attempts=int(attempts),
            correct=int(correct),
        )
        for cc_id, code, name, attempts, correct in rows
    ]


# --------------------------------------------------------------------------- #
# Per-section rollup (via CC -> FC -> Section)
# --------------------------------------------------------------------------- #


async def _by_section(session: AsyncSession) -> list[AccuracyStat]:
    qccs = _question_cc_pairs()
    latest = _latest_attempts()

    # Each (question_id, section_id) pair must be distinct, even when a
    # question's CC set spans multiple FCs that share a section. Without
    # DISTINCT, a question tagged to two CCs in the same section would
    # double-count at the section level.
    question_sections = (
        select(
            qccs.c.question_id.label("question_id"),
            Section.id.label("section_id"),
        )
        .join(ContentCategory, ContentCategory.id == qccs.c.content_category_id)
        .join(
            FoundationalConcept,
            FoundationalConcept.id == ContentCategory.foundational_concept_id,
        )
        .join(Section, Section.id == FoundationalConcept.section_id)
        .distinct()
        .subquery("question_sections")
    )

    # Pre-aggregate unique-question counts per section.
    section_attempts = (
        select(
            question_sections.c.section_id.label("section_id"),
            func.count(latest.c.question_id).label("attempts"),
            func.coalesce(func.sum(cast(latest.c.is_correct, Integer)), 0).label("correct"),
        )
        .join(latest, latest.c.question_id == question_sections.c.question_id)
        .group_by(question_sections.c.section_id)
        .subquery("section_attempts")
    )

    # Start FROM sections, LEFT JOIN to include zero-attempt sections.
    stmt = (
        select(
            Section.id,
            Section.code,
            Section.name,
            Section.position,
            func.coalesce(section_attempts.c.attempts, 0).label("attempts"),
            func.coalesce(section_attempts.c.correct, 0).label("correct"),
        )
        .outerjoin(section_attempts, section_attempts.c.section_id == Section.id)
        .order_by(Section.position)
    )

    rows = (await session.execute(stmt)).all()
    return [
        _stat(
            label=f"{code} — {name}",
            code=code,
            kind="section",
            target_id=section_id,
            attempts=int(attempts),
            correct=int(correct),
        )
        for section_id, code, name, _position, attempts, correct in rows
    ]


# --------------------------------------------------------------------------- #
# Per-topic rollup (leaf topics as emitted by the LLM categorizer)
# --------------------------------------------------------------------------- #


async def _by_topic(session: AsyncSession) -> list[AccuracyStat]:
    latest = _latest_attempts()

    # Pre-aggregate unique-question counts per topic.
    topic_attempts = (
        select(
            QuestionTag.topic_id.label("topic_id"),
            func.count(latest.c.question_id).label("attempts"),
            func.coalesce(func.sum(cast(latest.c.is_correct, Integer)), 0).label("correct"),
        )
        .join(latest, latest.c.question_id == QuestionTag.question_id)
        .where(QuestionTag.topic_id.is_not(None))
        .group_by(QuestionTag.topic_id)
        .subquery("topic_attempts")
    )

    # Start FROM topics, LEFT JOIN to include zero-attempt topics.
    stmt = (
        select(
            Topic.id,
            Topic.name,
            ContentCategory.code.label("cc_code"),
            func.coalesce(topic_attempts.c.attempts, 0).label("attempts"),
            func.coalesce(topic_attempts.c.correct, 0).label("correct"),
        )
        .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
        .outerjoin(topic_attempts, topic_attempts.c.topic_id == Topic.id)
        .order_by(ContentCategory.code, Topic.name)
    )

    rows = (await session.execute(stmt)).all()
    return [
        _stat(
            label=f"{cc_code} — {name}",
            code=cc_code,
            kind="topic",
            target_id=topic_id,
            attempts=int(attempts),
            correct=int(correct),
        )
        for topic_id, name, cc_code, attempts, correct in rows
    ]


# --------------------------------------------------------------------------- #
# Per-skill rollup
# --------------------------------------------------------------------------- #


async def _by_skill(session: AsyncSession) -> list[AccuracyStat]:
    latest = _latest_attempts()

    # Pre-aggregate unique-question counts per skill.
    skill_attempts = (
        select(
            QuestionTag.skill.label("skill"),
            func.count(latest.c.question_id).label("attempts"),
            func.coalesce(func.sum(cast(latest.c.is_correct, Integer)), 0).label("correct"),
        )
        .join(latest, latest.c.question_id == QuestionTag.question_id)
        .where(QuestionTag.skill.is_not(None))
        .group_by(QuestionTag.skill)
        .subquery("skill_attempts")
    )

    # Generate literal rows for all 4 MCAT skills.
    all_skills = (
        select(literal_column("s.skill").label("skill"))
        .select_from(text("(VALUES (1), (2), (3), (4)) AS s(skill)"))
        .subquery("all_skills")
    )

    stmt = (
        select(
            all_skills.c.skill,
            func.coalesce(skill_attempts.c.attempts, 0).label("attempts"),
            func.coalesce(skill_attempts.c.correct, 0).label("correct"),
        )
        .outerjoin(skill_attempts, skill_attempts.c.skill == all_skills.c.skill)
        .order_by(all_skills.c.skill)
    )

    rows = (await session.execute(stmt)).all()
    return [
        _stat(
            label=f"Skill {skill}",
            code=str(skill),
            kind="skill",
            target_id=None,
            attempts=int(attempts),
            correct=int(correct),
        )
        for skill, attempts, correct in rows
    ]


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


async def _timing(session: AsyncSession) -> TimingStat:
    # Timing uses raw attempts intentionally — each attempt has its own duration.
    # Discrete: Question.passage_id IS NULL; passage-based: NOT NULL.
    discrete_flag = Question.passage_id.is_(None)

    median_discrete = func.percentile_cont(0.5).within_group(Attempt.time_seconds)
    median_passage = func.percentile_cont(0.5).within_group(Attempt.time_seconds)

    stmt = select(
        func.coalesce(
            func.sum(cast(and_(discrete_flag, Attempt.time_seconds.is_not(None)), Integer)),
            0,
        ).label("discrete_n"),
        func.coalesce(
            func.sum(cast(and_(~discrete_flag, Attempt.time_seconds.is_not(None)), Integer)),
            0,
        ).label("passage_n"),
        median_discrete.filter(and_(discrete_flag, Attempt.time_seconds.is_not(None))).label(
            "median_discrete"
        ),
        median_passage.filter(and_(~discrete_flag, Attempt.time_seconds.is_not(None))).label(
            "median_passage"
        ),
        func.coalesce(
            func.sum(
                cast(
                    and_(discrete_flag, Attempt.time_seconds > literal(60)),
                    Integer,
                )
            ),
            0,
        ).label("over_discrete"),
        func.coalesce(
            func.sum(
                cast(
                    and_(~discrete_flag, Attempt.time_seconds > literal(95)),
                    Integer,
                )
            ),
            0,
        ).label("over_passage"),
    ).select_from(Attempt.__table__.join(Question.__table__, Question.id == Attempt.question_id))

    row = (await session.execute(stmt)).one()

    return TimingStat(
        median_seconds_discrete=(
            float(row.median_discrete) if row.median_discrete is not None else None
        ),
        median_seconds_passage_based=(
            float(row.median_passage) if row.median_passage is not None else None
        ),
        questions_over_target_discrete=int(row.over_discrete or 0),
        questions_over_target_passage=int(row.over_passage or 0),
    )


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #


async def _trend(
    session: AsyncSession, *, unit: Literal["week", "month"], limit: int
) -> list[TrendPoint]:
    # Trends use raw attempts intentionally — tracks practice velocity over time.
    period = func.date_trunc(unit, Attempt.attempted_at)

    stmt = (
        select(
            period.label("period"),
            func.count(Attempt.id).label("attempts"),
            func.sum(cast(Attempt.is_correct, Integer)).label("correct"),
        )
        .group_by(period)
        .order_by(period.desc())
        .limit(limit)
    )

    rows = list((await session.execute(stmt)).all())
    rows.reverse()  # ascending by period_start for downstream rendering

    points: list[TrendPoint] = []
    for period_value, attempts_n, correct_n in rows:
        attempts_int = int(attempts_n or 0)
        if attempts_int == 0:  # safety: skip zero-attempt windows
            continue
        correct_int = int(correct_n or 0)
        # date_trunc returns a timestamp; coerce to a python date.
        period_start = period_value.date() if hasattr(period_value, "date") else period_value
        points.append(
            TrendPoint(
                period_start=period_start,
                accuracy=correct_int / attempts_int,
                attempts=attempts_int,
            )
        )
    return points


# --------------------------------------------------------------------------- #
# Uncategorized count + totals
# --------------------------------------------------------------------------- #


async def _uncategorized_question_count(session: AsyncSession) -> int:
    categorized = (
        select(QuestionTag.question_id)
        .where((QuestionTag.topic_id.is_not(None)) | (QuestionTag.content_category_id.is_not(None)))
        .distinct()
        .subquery("categorized_qids")
    )
    stmt = select(func.count(Question.id)).where(
        Question.id.notin_(select(categorized.c.question_id))
    )
    return int((await session.execute(stmt)).scalar_one())


async def _total_attempts(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count(Attempt.id)))).scalar_one())


async def _total_questions(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count(Question.id)))).scalar_one())


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


async def compute_mastery(session: AsyncSession) -> MasteryReport:
    by_section = await _by_section(session)
    by_cc = await _by_content_category(session)
    by_topic = await _by_topic(session)
    by_skill = await _by_skill(session)
    timing = await _timing(session)
    trend_7d = await _trend(session, unit="week", limit=12)
    trend_30d = await _trend(session, unit="month", limit=6)
    uncategorized = await _uncategorized_question_count(session)
    total_attempts = await _total_attempts(session)
    total_questions = await _total_questions(session)

    return MasteryReport(
        by_section=by_section,
        by_content_category=by_cc,
        by_topic=by_topic,
        by_skill=by_skill,
        timing=timing,
        trend_7d=trend_7d,
        trend_30d=trend_30d,
        uncategorized_question_count=uncategorized,
        total_attempts=total_attempts,
        total_questions=total_questions,
    )
