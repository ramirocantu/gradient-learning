"""Anki load-config + plan-adherence API (SPEC T67, V54, V59, V60).

GET   /api/v1/anki/load-config       → read singleton (auto-seeds defaults on miss per V59)
POST  /api/v1/anki/load-config       → upsert singleton
GET   /api/v1/anki/load-adherence    → deterministic V54 payload (V60: data-only)

`load-config` follows V59's read-or-create on first access — the GET
handler will INSERT the seed row (200, 60) on miss so the dashboard
chip renders even when the T61 migration has not run (e.g. test DB
using `Base.metadata.create_all`). POST upserts in place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, verify_coach_token
from app.models.anki import AnkiLoadConfig
from app.schemas.anki_load import (
    AnkiLoadAdherenceOut,
    AnkiLoadConfigIn,
    AnkiLoadConfigOut,
)
from app.services.anki.load_adherence import compute_load_adherence


router = APIRouter(prefix="/anki", tags=["anki"])


_DEFAULT_CARD_BUDGET = 200
_DEFAULT_MINUTES_BUDGET = Decimal("60")


async def _get_or_seed(session: AsyncSession) -> AnkiLoadConfig:
    row = await session.get(AnkiLoadConfig, 1)
    if row is None:
        row = AnkiLoadConfig(
            id=1,
            daily_card_review_budget=_DEFAULT_CARD_BUDGET,
            daily_minutes_budget=_DEFAULT_MINUTES_BUDGET,
        )
        session.add(row)
        await session.flush()
    return row


@router.get(
    "/load-config",
    response_model=AnkiLoadConfigOut,
    dependencies=[Depends(verify_coach_token)],
)
async def get_load_config_route(
    session: AsyncSession = Depends(get_session),
) -> AnkiLoadConfigOut:
    row = await _get_or_seed(session)
    return AnkiLoadConfigOut.model_validate(row)


@router.post(
    "/load-config",
    response_model=AnkiLoadConfigOut,
    dependencies=[Depends(verify_coach_token)],
)
async def upsert_load_config_route(
    payload: AnkiLoadConfigIn,
    session: AsyncSession = Depends(get_session),
) -> AnkiLoadConfigOut:
    row = await session.get(AnkiLoadConfig, 1)
    if row is None:
        row = AnkiLoadConfig(
            id=1,
            daily_card_review_budget=payload.daily_card_review_budget,
            daily_minutes_budget=payload.daily_minutes_budget,
        )
        session.add(row)
    else:
        row.daily_card_review_budget = payload.daily_card_review_budget
        row.daily_minutes_budget = payload.daily_minutes_budget
        row.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return AnkiLoadConfigOut.model_validate(row)


@router.get(
    "/load-adherence",
    response_model=AnkiLoadAdherenceOut,
    dependencies=[Depends(verify_coach_token)],
)
async def get_load_adherence_route(
    window_days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> AnkiLoadAdherenceOut:
    result = await compute_load_adherence(session, window_days=window_days)
    return AnkiLoadAdherenceOut.model_validate(result)
