"""Mastery heatmap data builder (SPEC §T42 / §V29).

Joins UWorld accuracy (already on `AccuracyStat.wilson_lower`), Anki
state (`state_for_cc` → unlock%) + Anki retention (`retention_for_cc`
→ 30-day true retention) + UWorld trajectory (`trajectory_for_cc` →
arrow). One `HeatmapCell` per content category, plus an `is_cars`
discriminator so the template renders CARS as a single-cell section
per §V29 (CARS has no AAMC topics and the AnKing deck carries no
CARS AAMC tags, so Anki / retention surfaces are intentionally
empty).

All rollups read the same `(content_category_id = cc.id OR
topic.content_category_id = cc.id)` scope the per-domain helpers do
(retention_for_cc / state_for_cc / trajectory_for_cc) — they all
mirror each other so the four tile encodings stay axis-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic
from app.services.analytics import compute_mastery, wilson_lower
from app.services.analyzer.trajectory import trajectory_for_cc, trajectory_for_topic
from app.services.anki.queries import due_count_for_subtree
from app.services.anki.retention import retention_for_cc, retention_for_topic
from app.services.anki.state import state_for_cc, state_for_topic

# §V29 thresholds. Picked deliberately for Wilson lower bound — which
# tends to sit a few points below raw accuracy at small N — so the
# yellow band stays meaningful instead of collapsing.
WILSON_RED_MAX = 0.50
WILSON_YELLOW_MAX = 0.70

# §V29: opacity fade when UWorld N < 3 — "untested ghosted".
LOW_SIGNAL_N = 3

# §V29: retention 30-day window is the one the badge shows.
RETENTION_WINDOW_DAYS = 30


@dataclass(frozen=True)
class HeatmapCell:
    """Encoded view-model for one CC tile (or the CARS single cell).

    Anki fields (`unlock_pct`, `retention_30d`) are None when no Anki
    cards are linked to the CC scope. Trajectory `arrow` is None when
    either trajectory window has fewer than V36's minimum of 5
    attempts. `is_low_signal` toggles V29's ghost fade.
    """

    cc_id: int
    code: str
    name: str
    label: str
    section_code: str
    section_name: str
    is_cars: bool
    # UWorld
    attempts: int
    accuracy: float
    wilson_lower: float
    color_bucket: str  # "red" | "yellow" | "green" | "empty"
    is_low_signal: bool
    # Trajectory (§V36)
    arrow: str | None
    # Anki (§V27 / §V28) — None for CARS or when no in-scope cards
    unlock_pct: float | None
    retention_30d: float | None


def _color_bucket(wilson: float, attempts: int) -> str:
    if attempts == 0:
        return "empty"
    if wilson < WILSON_RED_MAX:
        return "red"
    if wilson < WILSON_YELLOW_MAX:
        return "yellow"
    return "green"


@dataclass(frozen=True)
class CCHeader:
    """View-model for the `/mastery/{cc}` page header (§V30).

    Two numbers side-by-side — never blended. The UWorld block surfaces
    Wilson lower bound + raw N; the Anki block surfaces an effective
    mastery percentage computed as `retention_30d × unlock_pct`. Either
    side can be missing (None) — zero UWorld attempts, or no AnKing
    coverage for the CC, or CARS (which has no AAMC topics and no
    AnKing AAMC tags by construction).
    """

    cc_code: str
    is_cars: bool
    # UWorld side
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float
    # Anki side — None when no in-scope Anki coverage (or CARS).
    unlock_pct: float | None
    retention_30d: float | None
    effective_mastery: float | None  # retention_30d * unlock_pct


async def cc_header(session: AsyncSession, *, cc_code: str) -> CCHeader:
    """Compose the §V30 header for one CC.

    Reads the same per-CC rollups the heatmap consumes
    (`compute_mastery` for UWorld, `state_for_cc` + `retention_for_cc`
    for Anki), so the drilldown header stays axis-aligned with the
    heatmap tile that links into it.
    """
    # Section code → is_cars discriminator (mirrors build_heatmap).
    section_stmt = (
        select(Section.code)
        .join(
            FoundationalConcept,
            FoundationalConcept.section_id == Section.id,
        )
        .join(
            ContentCategory,
            ContentCategory.foundational_concept_id == FoundationalConcept.id,
        )
        .where(ContentCategory.code == cc_code)
    )
    section_code = (await session.execute(section_stmt)).scalar_one_or_none() or ""
    is_cars = section_code == "CARS"

    mastery = await compute_mastery(session)
    stat = next(
        (s for s in mastery.by_content_category if s.code == cc_code),
        None,
    )
    attempts = stat.attempts if stat else 0
    correct = stat.correct if stat else 0
    accuracy = stat.accuracy if stat else 0.0
    wilson = stat.wilson_lower if stat else 0.0

    unlock_pct: float | None = None
    retention_30d: float | None = None
    effective: float | None = None
    if not is_cars:
        state = await state_for_cc(session, cc_code=cc_code)
        unlock_pct = state.unlock_pct
        ret = await retention_for_cc(session, cc_code=cc_code, windows=(RETENTION_WINDOW_DAYS,))
        retention_30d = ret.windows[RETENTION_WINDOW_DAYS].retention
        if unlock_pct is not None and retention_30d is not None:
            effective = retention_30d * unlock_pct

    return CCHeader(
        cc_code=cc_code,
        is_cars=is_cars,
        attempts=attempts,
        correct=correct,
        accuracy=accuracy,
        wilson_lower=wilson,
        unlock_pct=unlock_pct,
        retention_30d=retention_30d,
        effective_mastery=effective,
    )


async def build_heatmap(session: AsyncSession) -> dict[str, list[HeatmapCell]]:
    """Returns {section_name: [HeatmapCell, ...]} ordered by Section.position.

    Issues one bulk pass for UWorld accuracy (via `compute_mastery`),
    then per-CC queries for trajectory + Anki state + Anki retention.
    CARS CCs skip the Anki queries — CARS has no AnKing AAMC tags so
    those rollups would always be empty, and §V29 says the CARS cell
    surfaces accuracy + trajectory only.
    """
    mastery = await compute_mastery(session)

    # CC.id → (section_code, section_name, cc_name). One query.
    stmt = (
        select(
            ContentCategory.id,
            ContentCategory.code,
            ContentCategory.name,
            Section.code.label("section_code"),
            Section.name.label("section_name"),
            Section.position.label("section_position"),
        )
        .join(
            FoundationalConcept,
            FoundationalConcept.id == ContentCategory.foundational_concept_id,
        )
        .join(Section, Section.id == FoundationalConcept.section_id)
        .order_by(Section.position, ContentCategory.code)
    )
    rows = (await session.execute(stmt)).all()
    cc_meta = {
        row.id: (
            row.code,
            row.name,
            row.section_code,
            row.section_name,
            row.section_position,
        )
        for row in rows
    }

    # AccuracyStat by CC target_id (already includes zero-attempt CCs).
    by_cc = {stat.target_id: stat for stat in mastery.by_content_category}

    cells_by_section: dict[str, list[HeatmapCell]] = {}
    # Track section position to preserve outline order in the output.
    section_order: dict[str, int] = {}

    for cc_id, meta in cc_meta.items():
        code, name, section_code, section_name, section_position = meta
        section_order.setdefault(section_name, section_position)
        is_cars = section_code == "CARS"

        stat = by_cc.get(cc_id)
        attempts = stat.attempts if stat else 0
        accuracy = stat.accuracy if stat else 0.0
        wilson = stat.wilson_lower if stat else 0.0
        label = stat.label if stat else f"{code} — {name}"

        # Trajectory — same scope as Wilson (V36 arrow).
        traj = await trajectory_for_cc(session, cc_code=code)

        # Anki surfaces — skipped for CARS per §V29.
        unlock_pct: float | None = None
        retention_30d: float | None = None
        if not is_cars:
            state = await state_for_cc(session, cc_code=code)
            unlock_pct = state.unlock_pct
            ret = await retention_for_cc(session, cc_code=code, windows=(RETENTION_WINDOW_DAYS,))
            retention_30d = ret.windows[RETENTION_WINDOW_DAYS].retention

        cell = HeatmapCell(
            cc_id=cc_id,
            code=code,
            name=name,
            label=label,
            section_code=section_code,
            section_name=section_name,
            is_cars=is_cars,
            attempts=attempts,
            accuracy=accuracy,
            wilson_lower=wilson,
            color_bucket=_color_bucket(wilson, attempts),
            is_low_signal=attempts < LOW_SIGNAL_N,
            arrow=traj.arrow,
            unlock_pct=unlock_pct,
            retention_30d=retention_30d,
        )
        cells_by_section.setdefault(section_name, []).append(cell)

    # Return in section-position order.
    ordered: dict[str, list[HeatmapCell]] = {}
    for name in sorted(cells_by_section.keys(), key=lambda n: section_order[n]):
        ordered[name] = cells_by_section[name]
    return ordered


# --------------------------------------------------------------------------- #
# /mastery/{cc} sections — Anki state breakdown, review queue, topics tree
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateBreakdown:
    """Anki state + windowed retention for the §V34 "state breakdown" section."""

    total_cards: int
    assigned: int
    suspended: int
    new: int
    learning: int
    young: int
    mature: int
    unlock_pct: float | None
    retention_7d: float | None
    retention_30d: float | None
    retention_all: float | None


@dataclass(frozen=True)
class TopicTreeRow:
    """One row in the §V35 hierarchical topics table.

    Renderer indents per `depth`. Parent rows (`has_children=True`)
    print bold; leaves print non-bold. `drilldown_url` is the V32 id-path
    URL for the topic; the corresponding route ships in T45.
    """

    topic_id: int
    name: str
    depth: int  # 0 = CC root topic; +1 per descent
    has_children: bool
    drilldown_url: str
    # UWorld
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float
    # Per V36
    arrow: str | None
    # Anki — None when no in-scope cards
    unlock_pct: float | None
    retention_30d: float | None
    due_count: int


async def cc_anki_overview(session: AsyncSession, *, cc_code: str) -> StateBreakdown:
    """Compose state buckets + windowed retention for the CC's Anki
    state-breakdown section (§V34)."""
    state = await state_for_cc(session, cc_code=cc_code)
    ret = await retention_for_cc(session, cc_code=cc_code, windows=(7, 30, 0))
    return StateBreakdown(
        total_cards=state.total_cards,
        assigned=state.assigned,
        suspended=state.suspended,
        new=state.new,
        learning=state.learning,
        young=state.young,
        mature=state.mature,
        unlock_pct=state.unlock_pct,
        retention_7d=ret.windows[7].retention,
        retention_30d=ret.windows[30].retention,
        retention_all=ret.windows[0].retention,
    )


async def _topic_attempts_for_cc(
    session: AsyncSession, *, cc_id: int
) -> dict[int, tuple[int, int]]:
    """Returns {topic_id: (attempts, correct)} for every topic under the CC.

    Driven by `question_tags.topic_id` joined against topics in the CC.
    Uses the latest-attempt-per-question convention (mirrors
    analytics._by_topic) so a topic-tagged question counts once with
    its most-recent attempt's outcome.
    """
    from sqlalchemy import Integer, cast, func

    from app.models.captures import Attempt, QuestionTag

    latest = (
        select(
            Attempt.question_id,
            Attempt.is_correct,
        )
        .distinct(Attempt.question_id)
        .order_by(Attempt.question_id, Attempt.attempted_at.desc())
        .subquery("latest_attempts")
    )

    stmt = (
        select(
            QuestionTag.topic_id.label("topic_id"),
            func.count(latest.c.question_id).label("attempts"),
            func.coalesce(func.sum(cast(latest.c.is_correct, Integer)), 0).label("correct"),
        )
        .join(Topic, Topic.id == QuestionTag.topic_id)
        .join(latest, latest.c.question_id == QuestionTag.question_id)
        .where(Topic.content_category_id == cc_id)
        .group_by(QuestionTag.topic_id)
    )
    rows = (await session.execute(stmt)).all()
    return {int(r.topic_id): (int(r.attempts), int(r.correct)) for r in rows}


async def cc_topics_tree(session: AsyncSession, *, cc_id: int, cc_code: str) -> list[TopicTreeRow]:
    """Flat-tree DFS of every Topic under the CC with full row metrics.

    Output order = section-style DFS: each parent immediately followed
    by its descendants, depth-marked for indent rendering (§V35).
    """
    topics = (
        (
            await session.execute(
                select(Topic)
                .where(Topic.content_category_id == cc_id)
                .order_by(Topic.position, Topic.id)
            )
        )
        .scalars()
        .all()
    )

    # Build parent → children index.
    by_parent: dict[int | None, list[Topic]] = {}
    by_id: dict[int, Topic] = {}
    for t in topics:
        by_id[t.id] = t
        by_parent.setdefault(t.parent_topic_id, []).append(t)

    # Pre-pull per-topic attempts/correct in one query.
    attempts_map = await _topic_attempts_for_cc(session, cc_id=cc_id)

    due_before = datetime.now(tz=timezone.utc) + timedelta(days=1)

    # Walk DFS from root topics (parent_topic_id IS NULL within CC).
    def _path_to(topic_id: int) -> list[int]:
        chain: list[int] = []
        node = by_id[topic_id]
        chain.append(node.id)
        while node.parent_topic_id is not None and node.parent_topic_id in by_id:
            node = by_id[node.parent_topic_id]
            chain.append(node.id)
        chain.reverse()
        return chain

    rows: list[TopicTreeRow] = []

    async def _visit(topic: Topic, depth: int) -> None:
        children = by_parent.get(topic.id, [])
        has_children = bool(children)
        attempts, correct = attempts_map.get(topic.id, (0, 0))
        accuracy = (correct / attempts) if attempts else 0.0
        wilson = wilson_lower(correct, attempts) if attempts else 0.0

        traj = await trajectory_for_topic(session, topic_id=topic.id)
        state = await state_for_topic(session, topic_id=topic.id)
        ret = await retention_for_topic(session, topic_id=topic.id, windows=(30,))
        due = await due_count_for_subtree(session, topic_id=topic.id, due_before=due_before)

        path = _path_to(topic.id)
        drilldown_url = "/mastery/" + cc_code + "/topics/" + "/".join(str(x) for x in path)

        rows.append(
            TopicTreeRow(
                topic_id=topic.id,
                name=topic.name,
                depth=depth,
                has_children=has_children,
                drilldown_url=drilldown_url,
                attempts=attempts,
                correct=correct,
                accuracy=accuracy,
                wilson_lower=wilson,
                arrow=traj.arrow,
                unlock_pct=state.unlock_pct,
                retention_30d=ret.windows[30].retention,
                due_count=due,
            )
        )
        for child in children:
            await _visit(child, depth + 1)

    for root in by_parent.get(None, []):
        await _visit(root, 0)

    return rows


# --------------------------------------------------------------------------- #
# Topic page (§T45) — subtree-scoped header, state, children tree
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopicHeader:
    """View-model for the topic drilldown header (§V30 shape, subtree-scoped)."""

    cc_code: str
    topic_id: int
    topic_name: str
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float
    unlock_pct: float | None
    retention_30d: float | None
    effective_mastery: float | None


@dataclass(frozen=True)
class BreadcrumbItem:
    """One node in the topic-page breadcrumb (CC → root topic → … → leaf)."""

    label: str
    href: str


async def _topic_subtree_accuracy(session: AsyncSession, *, topic_id: int) -> tuple[int, int]:
    """`(attempts, correct)` over latest-attempt-per-question for questions
    tagged with any topic in `subtree(topic_id)` per §V31."""
    sql = text(
        """
        WITH RECURSIVE subtree(id) AS (
            SELECT id FROM topics WHERE id = :topic_id
            UNION ALL
            SELECT child.id
            FROM topics child
            JOIN subtree s ON child.parent_topic_id = s.id
        ),
        qids AS (
            SELECT DISTINCT qt.question_id
            FROM question_tags qt
            WHERE qt.topic_id IN (SELECT id FROM subtree)
        ),
        latest AS (
            SELECT DISTINCT ON (a.question_id) a.question_id, a.is_correct
            FROM attempts a
            JOIN qids ON qids.question_id = a.question_id
            ORDER BY a.question_id, a.attempted_at DESC
        )
        SELECT count(*) AS attempts,
               count(*) FILTER (WHERE is_correct) AS correct
        FROM latest
        """
    )
    row = (await session.execute(sql, {"topic_id": topic_id})).one()
    return int(row.attempts), int(row.correct)


async def topic_header(session: AsyncSession, *, cc_code: str, topic: Topic) -> TopicHeader:
    """Compose the §V30-shape header subtree-scoped to `topic`."""
    attempts, correct = await _topic_subtree_accuracy(session, topic_id=topic.id)
    accuracy = (correct / attempts) if attempts else 0.0
    wilson = wilson_lower(correct, attempts) if attempts else 0.0

    state = await state_for_topic(session, topic_id=topic.id)
    ret = await retention_for_topic(session, topic_id=topic.id, windows=(RETENTION_WINDOW_DAYS,))
    unlock_pct = state.unlock_pct
    retention_30d = ret.windows[RETENTION_WINDOW_DAYS].retention
    effective = (
        retention_30d * unlock_pct
        if (unlock_pct is not None and retention_30d is not None)
        else None
    )
    return TopicHeader(
        cc_code=cc_code,
        topic_id=topic.id,
        topic_name=topic.name,
        attempts=attempts,
        correct=correct,
        accuracy=accuracy,
        wilson_lower=wilson,
        unlock_pct=unlock_pct,
        retention_30d=retention_30d,
        effective_mastery=effective,
    )


async def topic_anki_overview(session: AsyncSession, *, topic_id: int) -> StateBreakdown:
    """Subtree-scoped Anki state buckets + windowed retention (§V33 sec 3)."""
    state = await state_for_topic(session, topic_id=topic_id)
    ret = await retention_for_topic(session, topic_id=topic_id, windows=(7, 30, 0))
    return StateBreakdown(
        total_cards=state.total_cards,
        assigned=state.assigned,
        suspended=state.suspended,
        new=state.new,
        learning=state.learning,
        young=state.young,
        mature=state.mature,
        unlock_pct=state.unlock_pct,
        retention_7d=ret.windows[7].retention,
        retention_30d=ret.windows[30].retention,
        retention_all=ret.windows[0].retention,
    )


def _path_chain_to(by_id: dict[int, Topic], topic_id: int) -> list[int]:
    chain: list[int] = []
    node = by_id[topic_id]
    chain.append(node.id)
    while node.parent_topic_id is not None and node.parent_topic_id in by_id:
        node = by_id[node.parent_topic_id]
        chain.append(node.id)
    chain.reverse()
    return chain


async def topic_children_tree(
    session: AsyncSession, *, cc_code: str, root_topic_id: int
) -> list[TopicTreeRow]:
    """Flat-tree DFS of all DESCENDANTS of `root_topic_id` (§V35).

    Excludes the root itself — the root's metrics live in the header.
    `depth` is relative to the root: immediate children = 0, grandchildren = 1.
    """
    # Pull the whole CC subset so we can resolve parent chains for URL building.
    root_topic = (
        await session.execute(select(Topic).where(Topic.id == root_topic_id))
    ).scalar_one()
    cc_id = root_topic.content_category_id
    topics = (
        (
            await session.execute(
                select(Topic)
                .where(Topic.content_category_id == cc_id)
                .order_by(Topic.position, Topic.id)
            )
        )
        .scalars()
        .all()
    )
    by_id = {t.id: t for t in topics}
    by_parent: dict[int | None, list[Topic]] = {}
    for t in topics:
        by_parent.setdefault(t.parent_topic_id, []).append(t)

    attempts_map = await _topic_attempts_for_cc(session, cc_id=cc_id)
    due_before = datetime.now(tz=timezone.utc) + timedelta(days=1)

    rows: list[TopicTreeRow] = []

    async def _visit(topic: Topic, depth: int) -> None:
        children = by_parent.get(topic.id, [])
        has_children = bool(children)
        attempts, correct = attempts_map.get(topic.id, (0, 0))
        accuracy = (correct / attempts) if attempts else 0.0
        wilson = wilson_lower(correct, attempts) if attempts else 0.0
        traj = await trajectory_for_topic(session, topic_id=topic.id)
        state = await state_for_topic(session, topic_id=topic.id)
        ret = await retention_for_topic(session, topic_id=topic.id, windows=(30,))
        due = await due_count_for_subtree(session, topic_id=topic.id, due_before=due_before)
        path = _path_chain_to(by_id, topic.id)
        drilldown_url = "/mastery/" + cc_code + "/topics/" + "/".join(str(x) for x in path)
        rows.append(
            TopicTreeRow(
                topic_id=topic.id,
                name=topic.name,
                depth=depth,
                has_children=has_children,
                drilldown_url=drilldown_url,
                attempts=attempts,
                correct=correct,
                accuracy=accuracy,
                wilson_lower=wilson,
                arrow=traj.arrow,
                unlock_pct=state.unlock_pct,
                retention_30d=ret.windows[30].retention,
                due_count=due,
            )
        )
        for child in children:
            await _visit(child, depth + 1)

    for child in by_parent.get(root_topic_id, []):
        await _visit(child, 0)
    return rows


async def validate_topic_chain(
    session: AsyncSession, *, cc_code: str, ids: list[int]
) -> tuple[list[Topic], list[BreadcrumbItem]] | None:
    """Validate the §V32 id-path. Returns `(topics_in_order, breadcrumb)` or
    None if any constraint fails.

    Constraints:
    - All ids exist as Topic rows.
    - `ids[0].parent_topic_id IS NULL` AND `ids[0].content_category_id`
      matches the CC referenced by `cc_code`.
    - `ids[k].parent_topic_id == ids[k-1]` for k > 0.
    """
    if not ids:
        return None
    # Fetch all topics in one query, preserve input order.
    topics = (await session.execute(select(Topic).where(Topic.id.in_(ids)))).scalars().all()
    by_id = {t.id: t for t in topics}
    if len(by_id) != len(ids):
        return None

    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == cc_code))
    ).scalar_one_or_none()
    if cc is None:
        return None

    first = by_id[ids[0]]
    if first.parent_topic_id is not None or first.content_category_id != cc.id:
        return None

    for prev_id, curr_id in zip(ids[:-1], ids[1:], strict=True):
        if by_id[curr_id].parent_topic_id != prev_id:
            return None

    topic_chain = [by_id[i] for i in ids]
    # Breadcrumb: CC → t1 → … → tn. Each href is the prefix-id-path.
    crumbs: list[BreadcrumbItem] = [
        BreadcrumbItem(label=f"{cc.code} — {cc.name}", href=f"/mastery/{cc.code}")
    ]
    for k in range(len(ids)):
        path = "/".join(str(x) for x in ids[: k + 1])
        crumbs.append(
            BreadcrumbItem(
                label=topic_chain[k].name,
                href=f"/mastery/{cc.code}/topics/{path}",
            )
        )
    return topic_chain, crumbs
