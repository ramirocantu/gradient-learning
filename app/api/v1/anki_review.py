"""Anki review API (SPEC T67 + T76 + T77, V53 amended).

POST  /api/v1/anki/reviews  → create pending review
GET   /api/v1/anki/reviews  → list (filter by status, window_days)
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.models.anki import AnkiReview
from app.schemas.anki_review import AnkiReviewCreateIn, AnkiReviewOut
from app.services.anki.review import create_review


router = APIRouter(prefix="/anki", tags=["anki"])


@router.post(
    "/reviews",
    response_model=AnkiReviewOut,
    status_code=status.HTTP_201_CREATED,)
async def create_review_route(
    payload: AnkiReviewCreateIn,
    session: AsyncSession = Depends(get_session),
) -> AnkiReviewOut:
    row = await create_review(
        session,
        card_ids=payload.card_ids,
        review_date=payload.review_date,
    )
    return AnkiReviewOut.model_validate(row)


@router.get(
    "/reviews",
    response_model=list[AnkiReviewOut],)
async def list_reviews_route(
    status_filter: str | None = Query(None, alias="status"),
    window_days: int | None = Query(None, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> list[AnkiReviewOut]:
    stmt = select(AnkiReview).order_by(AnkiReview.review_date.asc(), AnkiReview.id.asc())
    if status_filter is not None:
        stmt = stmt.where(AnkiReview.status == status_filter)
    if window_days is not None:
        today = date.today()
        stmt = stmt.where(AnkiReview.review_date >= today).where(
            AnkiReview.review_date <= today + timedelta(days=window_days)
        )
    rows = (await session.execute(stmt)).scalars().all()
    return [AnkiReviewOut.model_validate(r) for r in rows]
