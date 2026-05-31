"""Assignment service — V51 lifecycle + V52 card-id resolution (SPEC T62).

`create_assignment` snapshots the resolved card-id set into the new
`anki_assignments` row at create-time and never re-resolves at unlock
(V52). The snapshot stores AnKing-native card_ids (BIGINT, per V52
amendment / §B11) so downstream callers (T63 scheduler) can pass them
straight to `AnkiConnectClient.unsuspend_cards`.

`mark_skipped` and `mark_completed_manual` implement the V51
non-AnkiConnect transitions:
  * skipped = study-plan accounting only; cards stay in whatever state
    Anki has them in (per V51 + V57 — review-push is the only thing
    that lives separately).
  * completed_manual lets the dashboard or MCP host close out an
    assignment ahead of the daily auto-completion job (T64). Both
    refuse to mutate terminal rows.

The AnkiConnect side-effect (pending → unlocked) and audit-row write
live in the T63 scheduler job, not here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment, AnkiWrite
from app.services.anki.client import (
    AnkiConnectError,
    AnkiConnectClient,
    AnkiUnreachableError,
    AnkiWriteFailed,
)
from app.services.outline_subtree import subtree_node_ids

logger = logging.getLogger(__name__)


PriorityKind = Literal["most_specific_first", "random", "mature_first", "young_first"]
_TERMINAL_STATES: frozenset[str] = frozenset({"completed", "skipped", "failed"})
_ACTIVE_STATES: frozenset[str] = frozenset({"pending", "unlocked"})


class AssignmentError(RuntimeError):
    """Base class for assignment-service errors."""


class AssignmentNotFoundError(AssignmentError):
    """No `anki_assignments` row matched the supplied id."""


class AssignmentTerminalError(AssignmentError):
    """Attempt to transition out of a terminal V51 state
    (completed | skipped | failed)."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One AnKing-native card surfaced for assignment, plus the columns
    needed to sort it under any V52 priority mode."""

    anki_card_id: int
    note_id: int
    queue: int
    interval_days: Optional[int]
    confidence: Optional[float]
    node_depth: int


# Candidate selection over the node_id subtree rollup (V-RB6, V-O5, V-O1).
# Membership comes from `outline_subtree.subtree_node_ids` (self + descendants,
# set rollup); this query then expands the in-scope node ids to suspended cards
# via `anki_note_tags.node_id` (sole tag target, V-T1) and attaches each node's
# tree depth (deeper = more specific) for the `most_specific_first` ordering.
_CANDIDATE_SQL = text(
    """
    WITH RECURSIVE node_depth(id, depth) AS (
        SELECT id, 0 FROM outline_nodes WHERE parent_id IS NULL
        UNION ALL
        SELECT n.id, nd.depth + 1
        FROM outline_nodes n
        JOIN node_depth nd ON n.parent_id = nd.id
    )
    SELECT DISTINCT
        c.anki_card_id,
        c.note_id,
        c.queue,
        c.interval_days,
        t.confidence,
        COALESCE(nd.depth, 0) AS depth
    FROM anki_cards c
    JOIN anki_note_tags t ON t.note_id = c.note_id
    LEFT JOIN node_depth nd ON nd.id = t.node_id
    WHERE c.queue = -1
      AND t.node_id = ANY(:node_ids)
      AND (t.confidence IS NULL OR t.confidence >= 0.5)
    ORDER BY c.anki_card_id, t.confidence NULLS LAST, depth
    """
)


async def _fetch_candidates(session: AsyncSession, *, node_id: int) -> list[_Candidate]:
    """Suspended (`queue=-1`) cards whose NOTE carries an in-scope `node_id`
    tag (V-T1), where in-scope = the subtree rooted at `node_id` (self +
    descendants, V-O1 set rollup via `outline_subtree`)."""
    node_ids = await subtree_node_ids(session, node_id)
    if not node_ids:
        return []
    rows = (await session.execute(_CANDIDATE_SQL, {"node_ids": list(node_ids)})).all()

    return [
        _Candidate(
            anki_card_id=int(r[0]),
            note_id=int(r[1]),
            queue=int(r[2]),
            interval_days=(int(r[3]) if r[3] is not None else None),
            confidence=(float(r[4]) if r[4] is not None else None),
            node_depth=int(r[5]),
        )
        for r in rows
    ]


def _seeded_shuffle(items: list[_Candidate], seed: int) -> list[_Candidate]:
    """Deterministic shuffle: same `(seed, anki_card_id)` pairs always
    produce the same order. Uses the python stdlib `random` seeded on a
    sha256 digest of the seed for stable cross-version reproducibility."""
    digest = hashlib.sha256(str(seed).encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    ordered = list(items)
    rng.shuffle(ordered)
    return ordered


def _apply_priority(
    candidates: list[_Candidate],
    priority: PriorityKind,
    random_seed: Optional[int],
) -> list[_Candidate]:
    if priority == "most_specific_first":
        # Confidence DESC NULLS LAST; topic depth DESC (deeper = more
        # specific); anki_card_id ASC (stable tiebreak per V52).
        return sorted(
            candidates,
            key=lambda c: (
                c.confidence is None,  # False (0) ahead of True (1) → non-NULL first
                -(c.confidence or 0.0),
                -c.node_depth,
                c.anki_card_id,
            ),
        )
    if priority == "random":
        if random_seed is None:
            raise AssignmentError(
                "priority='random' requires random_seed (use assignment.id post-flush)"
            )
        return _seeded_shuffle(candidates, random_seed)
    # mature_first / young_first key on interval_days; queue=-1 cards
    # may have a stale `interval_days` from before suspension. NULL
    # treated as 0 (no SRS history → least mature for mature_first,
    # most young for young_first).
    if priority == "mature_first":
        return sorted(
            candidates,
            key=lambda c: (-(c.interval_days or 0), c.anki_card_id),
        )
    if priority == "young_first":
        return sorted(
            candidates,
            key=lambda c: ((c.interval_days or 0), c.anki_card_id),
        )
    raise AssignmentError(f"unknown priority {priority!r}")


async def _resolve_targets(
    session: AsyncSession,
    *,
    scope_kind: str,
    scope_value: str,
    priority: PriorityKind,
    max_cards: Optional[int],
    random_seed: Optional[int],
) -> tuple[list[int], list[int]]:
    """Resolve `(card_ids, note_ids)` for the V52 scope (§V75).

    Candidates are note-scoped — a suspended card matches iff its NOTE
    carries an in-scope tag (`anki_note_tags.note_id = anki_cards.note_id`) —
    then expanded to the notes' suspended cards. Priority ordering +
    `max_cards` slice apply to cards (V52, unchanged). `note_ids` = the
    distinct notes among the selected cards, order-preserving — the canonical
    addTags target. Both lists are deduped (V64 belt: a cid belongs to exactly
    one note, so the card dedup already collapses cross-tag dups).

    Scope is a `node_id` subtree (V-RB6 / V-O5): `scope_value` carries the
    target outline node id (int-coercible); `scope_kind` ('cc'|'topic') is
    retained for storage/audit only and no longer steers resolution — a node
    is a node regardless of the AAMC `kind` label it wears (V-O1).
    """
    try:
        node_id = int(scope_value)
    except (TypeError, ValueError) as exc:
        raise AssignmentError(
            f"scope_value must be an int-coercible node_id; got {scope_value!r}"
        ) from exc
    candidates = await _fetch_candidates(session, node_id=node_id)
    ordered = _apply_priority(candidates, priority, random_seed)
    # Dedup by native card_id, order-preserving (V64): a card whose note
    # matched via multiple tags yields multiple candidate rows (the SQL
    # `SELECT DISTINCT` spans the full row incl confidence/depth, so it does
    # NOT collapse them). Keep the first occurrence — after `_apply_priority`
    # that is the most-specific / highest-confidence instance. Dedup BEFORE
    # the max_cards slice so the slice counts distinct cards, not dups.
    seen: set[int] = set()
    deduped: list[_Candidate] = []
    for c in ordered:
        if c.anki_card_id not in seen:
            seen.add(c.anki_card_id)
            deduped.append(c)
    if max_cards is not None:
        deduped = deduped[: int(max_cards)]
    card_ids = [c.anki_card_id for c in deduped]
    note_ids: list[int] = []
    seen_notes: set[int] = set()
    for c in deduped:
        if c.note_id not in seen_notes:
            seen_notes.add(c.note_id)
            note_ids.append(c.note_id)
    return card_ids, note_ids


async def resolve_card_ids(
    session: AsyncSession,
    *,
    scope_kind: str,
    scope_value: str,
    priority: PriorityKind = "most_specific_first",
    max_cards: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> list[int]:
    """Resolve the AnKing-native card_ids that satisfy the V52 scope.

    Filters: `anki_cards.queue=-1` (suspended only), confidence ≥ 0.5
    (or NULL — regex-derived rows have no confidence). Returns native
    card_ids ordered per `priority`; if `max_cards` set, sliced to that
    length. `random_seed` is required only when `priority='random'`. The
    note-id expansion is exposed via `_resolve_targets` (used by
    `create_assignment` for the addTags snapshot)."""
    card_ids, _note_ids = await _resolve_targets(
        session,
        scope_kind=scope_kind,
        scope_value=scope_value,
        priority=priority,
        max_cards=max_cards,
        random_seed=random_seed,
    )
    return card_ids


async def create_assignment(
    session: AsyncSession,
    *,
    scope_kind: str,
    scope_value: str,
    scheduled_unlock_at: datetime,
    max_cards: Optional[int] = None,
    priority: PriorityKind = "most_specific_first",
) -> AnkiAssignment:
    """Create a pending assignment with its card-id snapshot.

    Two-pass within the supplied session:
      1. Insert row with empty `card_ids` + status='pending'. Flush to
         get the auto-generated id (needed as the deterministic seed
         for priority='random' per V52).
      2. Resolve the scope to a native-id list, optionally sliced.
      3. UPDATE the row in place with the resolved list.

    Caller owns the outer transaction. The intermediate empty-array
    state is never visible outside this function on a clean commit.
    """
    assignment = AnkiAssignment(
        scope_kind=scope_kind,
        scope_value=scope_value,
        scheduled_unlock_at=scheduled_unlock_at,
        card_ids=[],
        note_ids=[],
        max_cards=max_cards,
        priority=priority,
        status="pending",
    )
    session.add(assignment)
    await session.flush()

    card_ids, note_ids = await _resolve_targets(
        session,
        scope_kind=scope_kind,
        scope_value=scope_value,
        priority=priority,
        max_cards=max_cards,
        random_seed=(assignment.id if priority == "random" else None),
    )
    assignment.card_ids = card_ids
    assignment.note_ids = note_ids
    await session.flush()
    return assignment


async def _load_or_raise(session: AsyncSession, assignment_id: int) -> AnkiAssignment:
    row = (
        await session.execute(select(AnkiAssignment).where(AnkiAssignment.id == assignment_id))
    ).scalar_one_or_none()
    if row is None:
        raise AssignmentNotFoundError(f"anki_assignments id={assignment_id} not found")
    return row


def _refuse_terminal(assignment: AnkiAssignment, target: str) -> None:
    if assignment.status in _TERMINAL_STATES:
        raise AssignmentTerminalError(
            f"cannot transition assignment id={assignment.id} from terminal "
            f"status={assignment.status!r} to {target!r}"
        )


async def mark_skipped(session: AsyncSession, assignment_id: int) -> AnkiAssignment:
    """V51 pending|unlocked → skipped. No AnkiConnect side-effect —
    already-unsuspended cards stay unsuspended (study-plan accounting only)."""
    assignment = await _load_or_raise(session, assignment_id)
    _refuse_terminal(assignment, "skipped")
    assignment.status = "skipped"
    await session.flush()
    return assignment


async def mark_completed_manual(session: AsyncSession, assignment_id: int) -> AnkiAssignment:
    """V51 pending|unlocked → completed (manual close-out). The daily
    auto-completion job (T64) handles the review-driven path; this is
    the human-override entry point."""
    assignment = await _load_or_raise(session, assignment_id)
    _refuse_terminal(assignment, "completed")
    assignment.status = "completed"
    await session.flush()
    return assignment


# ----------------------- T63 unlock scheduler --------------------------- #


_UNLOCK_FAILURE_CAP = 3
_UNLOCK_RETRYABLE_ERRORS = (AnkiUnreachableError, AnkiWriteFailed, AnkiConnectError)


@dataclass
class AssignmentUnlockSummary:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    terminal_failed: int = 0
    errors: list[str] = field(default_factory=list)


def _payload_hash_for_unsuspend(card_ids: list[int]) -> str:
    """Stable hash over the snapshot card_id list. Sort first so the same
    set of ids in any order hashes the same (Anki treats the cards list as
    a set anyway)."""
    encoded = json.dumps(sorted(int(c) for c in card_ids), separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _payload_hash_for_add_tags(note_ids: list[int], tags: list[str]) -> str:
    """Stable hash over the snapshot note_id list + tag list, scoped to the
    addTags action so it ⊥ collide with an unsuspend hash. Post-§V75 addTags
    targets notes (`notes=...`), so the hash keys on note ids."""
    encoded = json.dumps(
        {"action": "addTags", "notes": sorted(int(n) for n in note_ids), "tags": sorted(tags)},
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


async def run_unlock_due(
    session: AsyncSession,
    anki_client: AnkiConnectClient,
    *,
    now: Optional[datetime] = None,
) -> AssignmentUnlockSummary:
    """T63 unlock job per V51 + V55, with V50 `addTags` chain (T75).

    For each `pending` assignment whose `scheduled_unlock_at ≤ now`:
      * Call `anki_client.unsuspend_cards(card_ids)`. On success: insert
        an `anki_writes(unsuspend, succeeded)` audit row, flip
        `status='unlocked'`, stamp `actual_unlock_at=now`.
      * Then chain `anki_client.add_tags(card_ids, ["coach::assignment:{id}"])`
        per V50 (T75). On success: insert `anki_writes(addTags, succeeded)`.
        On failure: insert `anki_writes(addTags, failed)` — the tag is an
        audit-only write-only marker, ⊥ load-bearing on the assignment
        lifecycle. ⊥ revert `status='unlocked'` or bump `failure_count`
        on addTags failure (unsuspend already succeeded). Commit.
      * On `unsuspend_cards` failure
        (`AnkiUnreachableError` / `AnkiWriteFailed` / `AnkiConnectError`):
        insert an `anki_writes(unsuspend, failed)` audit row, bump
        `failure_count`; if `failure_count ≥ 3` flip `status='failed'` +
        set `error_text` (V55 retry cap). ⊥ chain addTags on unsuspend
        failure. The assignment row is otherwise left in `pending` so the
        next tick retries it (V55 idempotence: unsuspend on already-
        unsuspended cards is a no-op in Anki).

    Each assignment is committed individually so a transient mid-batch
    failure does not roll back successful unlocks earlier in the loop.
    """
    now = now or datetime.now(timezone.utc)
    pending = (
        (
            await session.execute(
                select(AnkiAssignment)
                .where(AnkiAssignment.status == "pending")
                .where(AnkiAssignment.scheduled_unlock_at <= now)
                .order_by(AnkiAssignment.scheduled_unlock_at.asc(), AnkiAssignment.id.asc())
            )
        )
        .scalars()
        .all()
    )

    summary = AssignmentUnlockSummary(processed=len(pending))
    for assignment in pending:
        payload_hash = _payload_hash_for_unsuspend(list(assignment.card_ids or []))
        try:
            result = await anki_client.unsuspend_cards(list(assignment.card_ids or []))
        except _UNLOCK_RETRYABLE_ERRORS as exc:
            err_text = str(exc)[:2000]
            session.add(
                AnkiWrite(
                    action="unsuspend",
                    payload_hash=payload_hash,
                    response_json=None,
                    status="failed",
                    error_text=err_text,
                    source="scheduler",
                    assignment_id=assignment.id,
                )
            )
            assignment.failure_count = (assignment.failure_count or 0) + 1
            terminal = assignment.failure_count >= _UNLOCK_FAILURE_CAP
            if terminal:
                assignment.status = "failed"
                assignment.error_text = err_text
                summary.terminal_failed += 1
            summary.failed += 1
            summary.errors.append(err_text)
            await session.commit()
            logger.warning(
                "anki_assignment_unlock id=%s failed (count=%d terminal=%s): %s",
                assignment.id,
                assignment.failure_count,
                terminal,
                err_text,
            )
            continue

        session.add(
            AnkiWrite(
                action="unsuspend",
                payload_hash=payload_hash,
                response_json={"result": bool(result)},
                status="succeeded",
                error_text=None,
                source="scheduler",
                assignment_id=assignment.id,
            )
        )
        assignment.status = "unlocked"
        assignment.actual_unlock_at = now

        # T75: chain addTags audit-trail write per V50 (§V75: tag the NOTES,
        # ⊥ cards — note_ids snapshot is the canonical addTags target).
        # Failure here ⊥ revert status or bump failure_count — the tag is a
        # write-only audit marker, not load-bearing on the lifecycle. Empty
        # note_ids → skip both the call (AnkiConnect add_tags short-circuits
        # anyway, client.py §V75) and the audit row, since an
        # AnkiWrite(succeeded) for a no-op would falsely claim a tag write
        # that never reached Anki. AnkiWrite.status CHECK only allows
        # succeeded|failed, so no "skipped" sentinel row.
        note_ids_list = list(assignment.note_ids or [])
        if note_ids_list:
            tag_value = f"coach::assignment:{assignment.id}"
            tag_payload_hash = _payload_hash_for_add_tags(note_ids_list, [tag_value])
            try:
                await anki_client.add_tags(note_ids_list, [tag_value])
            except _UNLOCK_RETRYABLE_ERRORS as exc:
                tag_err_text = str(exc)[:2000]
                session.add(
                    AnkiWrite(
                        action="addTags",
                        payload_hash=tag_payload_hash,
                        response_json=None,
                        status="failed",
                        error_text=tag_err_text,
                        source="scheduler",
                        assignment_id=assignment.id,
                    )
                )
                logger.warning(
                    "anki_assignment_unlock id=%s addTags failed (status=unlocked retained, ⊥ load-bearing): %s",
                    assignment.id,
                    tag_err_text,
                )
            else:
                session.add(
                    AnkiWrite(
                        action="addTags",
                        payload_hash=tag_payload_hash,
                        response_json={"tags": [tag_value]},
                        status="succeeded",
                        error_text=None,
                        source="scheduler",
                        assignment_id=assignment.id,
                    )
                )
        else:
            logger.info(
                "anki_assignment_unlock id=%s addTags skipped: empty note_ids (no audit row written)",
                assignment.id,
            )

        await session.commit()
        summary.succeeded += 1
        logger.info(
            "anki_assignment_unlock id=%s unlocked cards=%d",
            assignment.id,
            len(assignment.card_ids or []),
        )

    return summary


# ---------------------- T64 auto-complete scheduler --------------------- #


@dataclass
class AssignmentCompleteSummary:
    processed: int = 0
    completed: int = 0
    still_unlocked: int = 0


_COMPLETE_CHECK_SQL = text(
    """
    SELECT COUNT(DISTINCT ac.anki_card_id)
    FROM anki_cards ac
    JOIN anki_card_reviews r ON r.card_id = ac.id
    WHERE ac.anki_card_id = ANY(:native_ids)
      AND r.reviewed_at > :unlock_at
    """
)


async def run_complete_unlocked(
    session: AsyncSession,
) -> AssignmentCompleteSummary:
    """T64 auto-completion per V51.

    For each `unlocked` assignment, flip `status='completed'` iff
    every AnKing-native card_id in the snapshot has at least one
    `anki_card_reviews` row with `reviewed_at > actual_unlock_at`.
    Idempotent on re-run — already-completed assignments are skipped
    by the WHERE clause.

    The join goes `anki_card_reviews.card_id (local SERIAL) →
    anki_cards.id → anki_cards.anki_card_id (native BIGINT)` to bridge
    the local-vs-native id boundary (V52 / B11).

    Empty `card_ids` → vacuous universal truth → completes
    immediately. The unlock job (T63) does not generate such rows in
    practice (resolve_card_ids returns []) so this branch mostly
    matters for migration / hand-inserted rows.
    """
    unlocked = (
        (
            await session.execute(
                select(AnkiAssignment)
                .where(AnkiAssignment.status == "unlocked")
                .order_by(AnkiAssignment.actual_unlock_at.asc(), AnkiAssignment.id.asc())
            )
        )
        .scalars()
        .all()
    )

    summary = AssignmentCompleteSummary(processed=len(unlocked))
    for assignment in unlocked:
        card_ids = list(assignment.card_ids or [])
        if not card_ids:
            assignment.status = "completed"
            summary.completed += 1
            await session.flush()
            continue
        unique_native_ids = {int(cid) for cid in card_ids}
        reviewed_count = int(
            (
                await session.execute(
                    _COMPLETE_CHECK_SQL,
                    {
                        "native_ids": list(unique_native_ids),
                        "unlock_at": assignment.actual_unlock_at,
                    },
                )
            ).scalar_one()
        )
        if reviewed_count == len(unique_native_ids):
            assignment.status = "completed"
            summary.completed += 1
            await session.flush()
            logger.info(
                "anki_assignment_complete id=%s completed cards=%d",
                assignment.id,
                len(unique_native_ids),
            )
        else:
            summary.still_unlocked += 1

    return summary


__all__ = [
    "AssignmentCompleteSummary",
    "AssignmentError",
    "AssignmentNotFoundError",
    "AssignmentTerminalError",
    "AssignmentUnlockSummary",
    "PriorityKind",
    "create_assignment",
    "mark_completed_manual",
    "mark_skipped",
    "resolve_card_ids",
    "run_complete_unlocked",
    "run_unlock_due",
]
