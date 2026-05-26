"""Anki assignment API (SPEC T67, V51 + V52).

POST   /api/v1/anki/assignments        → create + resolve card_ids snapshot
GET    /api/v1/anki/assignments        → list (filter by status, window_days)
PATCH  /api/v1/anki/assignments/{id}   → mark_skipped | mark_completed_manual

All routes require `X-Coach-Token`. Card-id resolution and the
two human-driven lifecycle transitions live in
`app/services/anki/assignment.py` — handlers here are thin
adapters that translate service exceptions to HTTP status codes
(V51 terminal → 409, V52 invalid scope → 422, not-found → 404).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, verify_coach_token
from app.models.anki import AnkiAssignment
from app.schemas.anki_assign import (
    AnkiAssignmentCreateIn,
    AnkiAssignmentOut,
    AnkiAssignmentPatchIn,
)
from app.services.anki.assignment import (
    AssignmentError,
    AssignmentNotFoundError,
    AssignmentTerminalError,
    create_assignment,
    mark_completed_manual,
    mark_skipped,
)


router = APIRouter(prefix="/anki", tags=["anki"])


@router.post(
    "/assignments",
    response_model=AnkiAssignmentOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_coach_token)],
)
async def create_assignment_route(
    payload: AnkiAssignmentCreateIn,
    session: AsyncSession = Depends(get_session),
) -> AnkiAssignmentOut:
    try:
        row = await create_assignment(
            session,
            scope_kind=payload.scope_kind,
            scope_value=payload.scope_value,
            scheduled_unlock_at=payload.scheduled_unlock_at,
            max_cards=payload.max_cards,
            priority=payload.priority,
        )
    except AssignmentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AnkiAssignmentOut.model_validate(row)


@router.get(
    "/assignments",
    response_model=list[AnkiAssignmentOut],
    dependencies=[Depends(verify_coach_token)],
)
async def list_assignments_route(
    status_filter: str | None = Query(None, alias="status"),
    window_days: int | None = Query(None, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> list[AnkiAssignmentOut]:
    stmt = select(AnkiAssignment).order_by(
        AnkiAssignment.scheduled_unlock_at.asc(), AnkiAssignment.id.asc()
    )
    if status_filter is not None:
        stmt = stmt.where(AnkiAssignment.status == status_filter)
    if window_days is not None:
        # Window = scheduled_unlock_at ∈ [now, now+window_days] — the
        # near-future "what's about to unlock" view the dashboard wants.
        now = datetime.now(timezone.utc)
        stmt = stmt.where(AnkiAssignment.scheduled_unlock_at >= now).where(
            AnkiAssignment.scheduled_unlock_at <= now + timedelta(days=window_days)
        )
    rows = (await session.execute(stmt)).scalars().all()
    return [AnkiAssignmentOut.model_validate(r) for r in rows]


@router.patch(
    "/assignments/{assignment_id}",
    response_model=AnkiAssignmentOut,
    dependencies=[Depends(verify_coach_token)],
)
async def patch_assignment_route(
    assignment_id: int,
    payload: AnkiAssignmentPatchIn,
    session: AsyncSession = Depends(get_session),
) -> AnkiAssignmentOut:
    try:
        if payload.status == "skipped":
            row = await mark_skipped(session, assignment_id)
        else:
            row = await mark_completed_manual(session, assignment_id)
    except AssignmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AssignmentTerminalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AnkiAssignmentOut.model_validate(row)
