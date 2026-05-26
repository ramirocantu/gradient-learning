"""Pattern aggregator — Phase 4.4.

Pure SQL + Python stats. No LLM, no external I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import sqrt

from sqlalchemy import and_, select, union
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic
from app.services.analytics import wilson_lower
from app.services.analyzer.feature_extractor import EXTRACTOR_VERSION


# --------------------------------------------------------------------------- #
# Wilson helpers
# --------------------------------------------------------------------------- #


def wilson_upper(correct: int, attempts: int, z: float = 1.96) -> float:
    """95% Wilson score upper bound — same formula as wilson_lower but + margin."""
    if attempts == 0:
        return 1.0
    p = correct / attempts
    denominator = 1 + z**2 / attempts
    center = p + z**2 / (2 * attempts)
    margin = z * sqrt(p * (1 - p) / attempts + z**2 / (4 * attempts**2))
    return min(1.0, (center + margin) / denominator)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnalysisFilter:
    section_code: str | None = None
    content_category_code: str | None = None
    topic_id: int | None = None
    skill: int | None = None
    since: date | None = None
    until: date | None = None
    min_sample_size: int = 10


@dataclass(frozen=True)
class FeatureFinding:
    feature_name: str
    feature_value: str
    accuracy_with: float
    accuracy_without: float
    attempts_with: int
    attempts_without: int
    correct_with: int
    correct_without: int
    accuracy_delta: float
    wilson_lower_with: float
    wilson_lower_without: float
    confident_delta: float
    representative_missed_qids: list[str]


@dataclass(frozen=True)
class CoverageStats:
    questions_with_features: int
    questions_without_features: int
    feature_extractor_version: str


@dataclass(frozen=True)
class InsightReport:
    filter_applied: AnalysisFilter
    total_attempts_in_scope: int
    total_questions_in_scope: int
    baseline_accuracy: float
    baseline_wilson_lower: float
    findings: list[FeatureFinding]
    coverage: CoverageStats


# --------------------------------------------------------------------------- #
# Feature column specs
# --------------------------------------------------------------------------- #

_BOOL_FEATURES = [
    "requires_calculation",
    "involves_graph_or_figure",
    "involves_data_table",
    "has_negative_phrasing",
    "trap_distractor_present",
]

_ENUM_FEATURES = [
    "question_format",
    "reasoning_type",
    "distractor_difficulty",
    "passage_length_bucket",
    "passage_type",
    "jargon_density",
]

_CALC_BUCKETS = ("0", "1-2", "3+")


def _calc_bucket(steps: int) -> str:
    if steps == 0:
        return "0"
    if steps <= 2:
        return "1-2"
    return "3+"


# --------------------------------------------------------------------------- #
# Scope subquery helpers
# --------------------------------------------------------------------------- #

# Each helper returns a subquery with a single column named "question_id".


def _section_qids(section_code: str):
    direct = (
        select(QuestionTag.question_id.label("question_id"))
        .join(ContentCategory, ContentCategory.id == QuestionTag.content_category_id)
        .join(
            FoundationalConcept,
            FoundationalConcept.id == ContentCategory.foundational_concept_id,
        )
        .join(Section, Section.id == FoundationalConcept.section_id)
        .where(
            and_(
                QuestionTag.content_category_id.is_not(None),
                Section.code == section_code,
            )
        )
    )
    via_topic = (
        select(QuestionTag.question_id.label("question_id"))
        .join(Topic, Topic.id == QuestionTag.topic_id)
        .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
        .join(
            FoundationalConcept,
            FoundationalConcept.id == ContentCategory.foundational_concept_id,
        )
        .join(Section, Section.id == FoundationalConcept.section_id)
        .where(
            and_(
                QuestionTag.topic_id.is_not(None),
                Section.code == section_code,
            )
        )
    )
    return union(direct, via_topic).subquery()


def _cc_qids(cc_code: str):
    direct = (
        select(QuestionTag.question_id.label("question_id"))
        .join(ContentCategory, ContentCategory.id == QuestionTag.content_category_id)
        .where(
            and_(
                QuestionTag.content_category_id.is_not(None),
                ContentCategory.code == cc_code,
            )
        )
    )
    via_topic = (
        select(QuestionTag.question_id.label("question_id"))
        .join(Topic, Topic.id == QuestionTag.topic_id)
        .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
        .where(
            and_(
                QuestionTag.topic_id.is_not(None),
                ContentCategory.code == cc_code,
            )
        )
    )
    return union(direct, via_topic).subquery()


# --------------------------------------------------------------------------- #
# Finding builder
# --------------------------------------------------------------------------- #

# Each entry in a group is (is_correct, qid_str, attempted_at).
_Group = list[tuple[bool, str, datetime]]


def _build_finding(
    feature_name: str,
    feature_value: str,
    with_grp: _Group,
    without_grp: _Group,
    min_n: int,
) -> FeatureFinding | None:
    aw = len(with_grp)
    awout = len(without_grp)
    if aw < min_n or awout < min_n:
        return None

    cw = sum(1 for ic, _, _ in with_grp if ic)
    cwout = sum(1 for ic, _, _ in without_grp if ic)
    acc_with = cw / aw
    acc_without = cwout / awout

    wl_with = wilson_lower(cw, aw)
    wl_without = wilson_lower(cwout, awout)
    wu_without = wilson_upper(cwout, awout)
    confident_delta = wl_with - wu_without

    missed = sorted(
        [(qid_str, at) for ic, qid_str, at in with_grp if not ic],
        key=lambda x: x[1],
        reverse=True,
    )
    rep_qids = [q for q, _ in missed[:3]]

    return FeatureFinding(
        feature_name=feature_name,
        feature_value=feature_value,
        accuracy_with=acc_with,
        accuracy_without=acc_without,
        attempts_with=aw,
        attempts_without=awout,
        correct_with=cw,
        correct_without=cwout,
        accuracy_delta=acc_with - acc_without,
        wilson_lower_with=wl_with,
        wilson_lower_without=wl_without,
        confident_delta=confident_delta,
        representative_missed_qids=rep_qids,
    )


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


async def analyze(filter: AnalysisFilter, session: AsyncSession) -> InsightReport:
    """Compute accuracy deltas per feature value for the given scope."""
    stmt = select(
        Attempt.id.label("attempt_id"),
        Attempt.question_id,
        Question.qid,
        Attempt.is_correct,
        Attempt.attempted_at,
    ).join(Question, Question.id == Attempt.question_id)

    if filter.section_code is not None:
        sq = _section_qids(filter.section_code)
        stmt = stmt.where(Question.id.in_(select(sq.c.question_id)))

    if filter.content_category_code is not None:
        sq = _cc_qids(filter.content_category_code)
        stmt = stmt.where(Question.id.in_(select(sq.c.question_id)))

    if filter.topic_id is not None:
        stmt = stmt.where(
            Question.id.in_(
                select(QuestionTag.question_id).where(QuestionTag.topic_id == filter.topic_id)
            )
        )

    if filter.skill is not None:
        stmt = stmt.where(
            Question.id.in_(
                select(QuestionTag.question_id).where(QuestionTag.skill == filter.skill)
            )
        )

    if filter.since is not None:
        since_dt = datetime(
            filter.since.year, filter.since.month, filter.since.day, tzinfo=timezone.utc
        )
        stmt = stmt.where(Attempt.attempted_at >= since_dt)

    if filter.until is not None:
        until_dt = datetime(
            filter.until.year,
            filter.until.month,
            filter.until.day,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        )
        stmt = stmt.where(Attempt.attempted_at <= until_dt)

    rows = (await session.execute(stmt)).all()

    if not rows:
        return InsightReport(
            filter_applied=filter,
            total_attempts_in_scope=0,
            total_questions_in_scope=0,
            baseline_accuracy=0.0,
            baseline_wilson_lower=0.0,
            findings=[],
            coverage=CoverageStats(
                questions_with_features=0,
                questions_without_features=0,
                feature_extractor_version=EXTRACTOR_VERSION,
            ),
        )

    total_attempts = len(rows)
    total_correct = sum(1 for r in rows if r.is_correct)
    in_scope_qids: set[int] = {r.question_id for r in rows}
    total_questions = len(in_scope_qids)

    # Fetch feature rows for in-scope questions
    feat_rows = (
        (
            await session.execute(
                select(QuestionFeatures).where(
                    QuestionFeatures.question_id.in_(list(in_scope_qids))
                )
            )
        )
        .scalars()
        .all()
    )
    features_map: dict[int, QuestionFeatures] = {f.question_id: f for f in feat_rows}

    # Coverage — only rows at current extractor version count as "with features"
    questions_with_features = sum(
        1
        for qid in in_scope_qids
        if qid in features_map and features_map[qid].extractor_version == EXTRACTOR_VERSION
    )

    # Attach features to each attempt (None if missing or stale version)
    attempt_data: list[tuple[int, str, bool, datetime, QuestionFeatures | None]] = []
    for r in rows:
        feat = features_map.get(r.question_id)
        if feat is not None and feat.extractor_version != EXTRACTOR_VERSION:
            feat = None
        attempt_data.append((r.question_id, r.qid, r.is_correct, r.attempted_at, feat))

    # Only attempts from questions with current-version features enter feature analysis
    featured = [
        (qid_int, qid_str, ic, at, feat)
        for qid_int, qid_str, ic, at, feat in attempt_data
        if feat is not None
    ]

    findings: list[FeatureFinding] = []
    min_n = filter.min_sample_size

    # Boolean features — one finding for True (False is the complement)
    for col in _BOOL_FEATURES:
        with_grp: _Group = []
        without_grp: _Group = []
        for _, qid_str, ic, at, feat in featured:
            entry = (ic, qid_str, at)
            if getattr(feat, col):
                with_grp.append(entry)
            else:
                without_grp.append(entry)
        finding = _build_finding(col, "True", with_grp, without_grp, min_n)
        if finding:
            findings.append(finding)

    # Enum features — one finding per distinct non-null value in scope
    for col in _ENUM_FEATURES:
        value_groups: dict[str, _Group] = {}
        tagged: list[tuple[str, bool, str, datetime]] = []  # (val, ic, qid_str, at)
        for _, qid_str, ic, at, feat in featured:
            val = getattr(feat, col)
            if val is None:
                continue
            val_str = str(val)
            value_groups.setdefault(val_str, []).append((ic, qid_str, at))
            tagged.append((val_str, ic, qid_str, at))

        for val_str, with_grp in value_groups.items():
            without_grp = [(ic, qid_str, at) for v, ic, qid_str, at in tagged if v != val_str]
            finding = _build_finding(col, val_str, with_grp, without_grp, min_n)
            if finding:
                findings.append(finding)

    # calculation_steps bucketed into 0 / 1-2 / 3+
    bucket_groups: dict[str, _Group] = {b: [] for b in _CALC_BUCKETS}
    for _, qid_str, ic, at, feat in featured:
        bucket_groups[_calc_bucket(feat.calculation_steps)].append((ic, qid_str, at))

    for bucket in _CALC_BUCKETS:
        with_grp = bucket_groups[bucket]
        without_grp = [entry for b, grp in bucket_groups.items() if b != bucket for entry in grp]
        finding = _build_finding("calculation_steps", bucket, with_grp, without_grp, min_n)
        if finding:
            findings.append(finding)

    findings.sort(key=lambda ff: (ff.confident_delta, ff.accuracy_delta))

    return InsightReport(
        filter_applied=filter,
        total_attempts_in_scope=total_attempts,
        total_questions_in_scope=total_questions,
        baseline_accuracy=total_correct / total_attempts,
        baseline_wilson_lower=wilson_lower(total_correct, total_attempts),
        findings=findings,
        coverage=CoverageStats(
            questions_with_features=questions_with_features,
            questions_without_features=total_questions - questions_with_features,
            feature_extractor_version=EXTRACTOR_VERSION,
        ),
    )
