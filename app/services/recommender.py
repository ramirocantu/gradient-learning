"""Study-next recommender — Ticket 5.2.

Pure Python scoring over existing mastery and feature-pattern data.
No LLM, no external I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Literal

from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic
from app.services.analytics import AccuracyStat, _question_cc_pairs, compute_mastery
from app.services.analyzer.patterns import AnalysisFilter, FeatureFinding, analyze


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MIN_ATTEMPTS = 3

SECTION_EXAM_WEIGHT: dict[str, float] = {
    "CP": 1.00,
    "CARS": 0.90,
    "BB": 1.00,
    "PS": 1.00,
}

RECENCY_DAYS = 30

# Only findings more negative than this generate feature_pattern recommendations
# or topic-level feature bonuses.
FEATURE_DELTA_THRESHOLD = -0.15

FEATURE_BONUS_WEIGHT = 0.3


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

RecommendationKind = Literal["topic_weakness", "feature_pattern"]


@dataclass(frozen=True)
class StudyRecommendation:
    kind: RecommendationKind
    label: str | None
    code: str | None
    target_id: int | None
    accuracy: float | None
    wilson_lower: float | None
    attempts: int | None
    feature_name: str | None
    feature_value: str | None
    accuracy_with: float | None
    accuracy_without: float | None
    priority_score: float
    reason: str
    representative_qids: list[str]


@dataclass
class RecommendationResult:
    recommendations: list[StudyRecommendation]
    total_candidates_scored: int


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _recency_mod(overall_acc: float, recent_acc: float | None) -> float:
    if recent_acc is None:
        return 1.0
    decline = max(0.0, overall_acc - recent_acc)
    return 1.0 + min(decline, 0.5)


def _compute_feature_bonus(
    stat: AccuracyStat,
    missed_qid_to_targets: dict[str, set[int]],
    feature_findings: list[FeatureFinding],
) -> tuple[float, list[FeatureFinding]]:
    """Returns (bonus, matched_findings) for a topic/CC stat."""
    bonus = 0.0
    matched: list[FeatureFinding] = []
    for finding in feature_findings:
        if finding.confident_delta >= FEATURE_DELTA_THRESHOLD:
            continue
        for qid in finding.representative_missed_qids:
            target_ids = missed_qid_to_targets.get(qid, set())
            if stat.target_id in target_ids:
                bonus += abs(finding.confident_delta)
                matched.append(finding)
                break
    return bonus, matched


# --------------------------------------------------------------------------- #
# Async query helpers
# --------------------------------------------------------------------------- #


async def _batch_recent_accuracy(
    session: AsyncSession,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return (topic_recent, cc_recent) — target_id → recent accuracy."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=RECENCY_DAYS)

    # Most recent attempt per question, only those within the recency window.
    recent_sq = (
        select(
            Attempt.question_id,
            Attempt.is_correct,
        )
        .distinct(Attempt.question_id)
        .where(Attempt.attempted_at >= cutoff)
        .order_by(Attempt.question_id, Attempt.attempted_at.desc())
        .subquery("recent_attempts")
    )

    topic_rows = (
        await session.execute(
            select(
                QuestionTag.topic_id,
                func.count(recent_sq.c.question_id).label("attempts"),
                func.coalesce(func.sum(cast(recent_sq.c.is_correct, Integer)), 0).label("correct"),
            )
            .join(QuestionTag, QuestionTag.question_id == recent_sq.c.question_id)
            .where(QuestionTag.topic_id.is_not(None))
            .group_by(QuestionTag.topic_id)
        )
    ).all()

    topic_recent: dict[int, float] = {
        int(row.topic_id): int(row.correct) / int(row.attempts)
        for row in topic_rows
        if int(row.attempts) > 0
    }

    qccs = _question_cc_pairs()
    cc_rows = (
        await session.execute(
            select(
                qccs.c.content_category_id,
                func.count(recent_sq.c.question_id).label("attempts"),
                func.coalesce(func.sum(cast(recent_sq.c.is_correct, Integer)), 0).label("correct"),
            )
            .join(recent_sq, recent_sq.c.question_id == qccs.c.question_id)
            .group_by(qccs.c.content_category_id)
        )
    ).all()

    cc_recent: dict[int, float] = {
        int(row.content_category_id): int(row.correct) / int(row.attempts)
        for row in cc_rows
        if int(row.attempts) > 0
    }

    return topic_recent, cc_recent


async def _build_qid_target_map(
    session: AsyncSession,
    missed_qids: set[str],
) -> dict[str, set[int]]:
    """Map qid strings → set of topic_ids and cc_ids for those questions.

    Includes both the topic_id from a topic tag and the CC that topic belongs to,
    so both topic-kind and content_category-kind stats can match.
    """
    if not missed_qids:
        return {}

    rows = (
        await session.execute(
            select(
                Question.qid,
                QuestionTag.topic_id,
                QuestionTag.content_category_id,
                Topic.content_category_id.label("topic_cc_id"),
            )
            .join(QuestionTag, QuestionTag.question_id == Question.id)
            .outerjoin(Topic, Topic.id == QuestionTag.topic_id)
            .where(Question.qid.in_(list(missed_qids)))
        )
    ).all()

    result: dict[str, set[int]] = {}
    for row in rows:
        targets = result.setdefault(row.qid, set())
        if row.topic_id is not None:
            targets.add(int(row.topic_id))
        if row.content_category_id is not None:
            targets.add(int(row.content_category_id))
        if row.topic_cc_id is not None:
            targets.add(int(row.topic_cc_id))

    return result


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


async def recommend(
    session: AsyncSession,
    *,
    n: int = 5,
) -> RecommendationResult:
    """Return top-N study recommendations ranked by priority_score descending."""

    # CC → section lookup (built once)
    cc_section_rows = (
        await session.execute(
            select(ContentCategory.code, Section.code.label("section_code"))
            .join(
                FoundationalConcept,
                FoundationalConcept.id == ContentCategory.foundational_concept_id,
            )
            .join(Section, Section.id == FoundationalConcept.section_id)
        )
    ).all()
    cc_to_section: dict[str, str] = {row.code: row.section_code for row in cc_section_rows}

    # Corpus-wide feature analysis — use MIN_ATTEMPTS as sample floor so the
    # recommender stays useful on the current ~75-question corpus.
    report = await analyze(AnalysisFilter(min_sample_size=MIN_ATTEMPTS), session)
    feature_findings = report.findings
    total_qs = report.total_questions_in_scope

    # Gather all representative missed qids across findings once.
    all_missed_qids: set[str] = set()
    for finding in feature_findings:
        all_missed_qids.update(finding.representative_missed_qids)
    qid_targets = await _build_qid_target_map(session, all_missed_qids)

    topic_recent, cc_recent = await _batch_recent_accuracy(session)

    mastery = await compute_mastery(session)
    candidate_stats = [*mastery.by_topic, *mastery.by_content_category]

    # ------------------------------------------------------------------ #
    # Pool 1 — topic / CC weakness candidates
    # ------------------------------------------------------------------ #
    pool1: list[StudyRecommendation] = []

    for stat in candidate_stats:
        if stat.attempts < MIN_ATTEMPTS:
            continue
        if stat.code == "CARS":
            continue

        section_code = cc_to_section.get(stat.code or "", "CP")
        aamc_weight = SECTION_EXAM_WEIGHT.get(section_code, 1.0)

        if stat.kind == "topic":
            recent_acc = topic_recent.get(stat.target_id) if stat.target_id else None
        else:
            recent_acc = cc_recent.get(stat.target_id) if stat.target_id else None

        recency_mod = _recency_mod(stat.accuracy, recent_acc)
        base_priority = 1.0 - stat.wilson_lower
        feat_bonus, matched_findings = _compute_feature_bonus(stat, qid_targets, feature_findings)
        priority_score = (
            base_priority * aamc_weight * recency_mod + feat_bonus * FEATURE_BONUS_WEIGHT
        )

        # Build reason field
        if matched_findings:
            feat_name = matched_findings[0].feature_name
            reason = (
                f"Low accuracy in {stat.label} ({stat.accuracy:.0%}, n={stat.attempts}); "
                f"also appears in {len(matched_findings)} feature-pattern finding(s) "
                f"(e.g. {feat_name})."
            )
        else:
            reason = (
                f"Low accuracy in {stat.label} ({stat.accuracy:.0%} correct, "
                f"{stat.attempts} questions, Wilson lower {stat.wilson_lower:.2f})."
            )

        if recent_acc is not None and stat.accuracy > recent_acc:
            reason += (
                f" Performance has declined in the last 30 days "
                f"({recent_acc:.0%} vs {stat.accuracy:.0%} overall)."
            )

        # Representative qids: pull from matched feature findings
        rep_qids: list[str] = []
        for finding in matched_findings[:2]:
            rep_qids.extend(finding.representative_missed_qids[:2])
        rep_qids = list(dict.fromkeys(rep_qids))[:3]

        pool1.append(
            StudyRecommendation(
                kind="topic_weakness",
                label=stat.label,
                code=stat.code,
                target_id=stat.target_id,
                accuracy=stat.accuracy,
                wilson_lower=stat.wilson_lower,
                attempts=stat.attempts,
                feature_name=None,
                feature_value=None,
                accuracy_with=None,
                accuracy_without=None,
                priority_score=priority_score,
                reason=reason,
                representative_qids=rep_qids,
            )
        )

    # ------------------------------------------------------------------ #
    # Pool 2 — feature-pattern candidates
    # ------------------------------------------------------------------ #
    pool2: list[StudyRecommendation] = []

    for finding in feature_findings:
        if finding.confident_delta >= FEATURE_DELTA_THRESHOLD:
            continue
        if finding.attempts_with < MIN_ATTEMPTS:
            continue

        effect_fraction = finding.attempts_with / max(total_qs, 1)
        priority_score = abs(finding.confident_delta) * sqrt(max(effect_fraction, 0.01))

        qids_str = ", ".join(finding.representative_missed_qids[:3])
        reason = (
            f"When {finding.feature_name}={finding.feature_value}, accuracy drops from "
            f"{finding.accuracy_without:.0%} to {finding.accuracy_with:.0%} "
            f"({finding.attempts_with} questions affected). Representative misses: {qids_str}."
        )

        pool2.append(
            StudyRecommendation(
                kind="feature_pattern",
                label=None,
                code=None,
                target_id=None,
                accuracy=None,
                wilson_lower=None,
                attempts=None,
                feature_name=finding.feature_name,
                feature_value=finding.feature_value,
                accuracy_with=finding.accuracy_with,
                accuracy_without=finding.accuracy_without,
                priority_score=priority_score,
                reason=reason,
                representative_qids=list(finding.representative_missed_qids[:3]),
            )
        )

    # ------------------------------------------------------------------ #
    # Merge, sort, cap
    # ------------------------------------------------------------------ #
    all_candidates = [*pool1, *pool2]
    all_candidates.sort(key=lambda r: r.priority_score, reverse=True)

    return RecommendationResult(
        recommendations=all_candidates[:n],
        total_candidates_scored=len(pool1) + len(pool2),
    )
