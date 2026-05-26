"""Anki API endpoints (SPEC §T4 + §T5 + §T39, P11 + mastery rebuild).

POST   /api/v1/anki/sync                       → runs a one-off sync of the configured deck.
GET    /api/v1/anki/cards?topic_id=N           → cards tagged for that AAMC topic.
GET    /api/v1/anki/review-queue?limit=N       → due-and-overdue cards, soonest first.
GET    /api/v1/anki/cards/by-qid/{qid}         → cards carrying `uworld::qid::{qid}`.
GET    /api/v1/anki/performance?cc_code=…      → raw state + retention windows (§T39).
GET    /api/v1/anki/performance?topic_id=…     → raw state + retention windows (§T39).

All routes require `X-Coach-Token` via `verify_coach_token`. The sync
route stays soft on AnkiConnect being down (returns the §V4 error
envelope, not a 500).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, verify_coach_token
from app.config import settings
from app.schemas.anki import (
    AnkiCardOut,
    AnkiPerformanceOut,
    AnkiRetentionOut,
    AnkiRetentionWindowOut,
    AnkiStateOut,
)
from app.services.anki.client import AnkiConnectClient
from app.services.anki.queries import (
    list_cards_for_qid,
    list_cards_for_topic,
    list_review_queue,
)
from app.services.anki.retention import (
    DEFAULT_WINDOWS,
    RetentionSummary,
    retention_for_cc,
    retention_for_topic,
)
from app.services.anki.state import (
    StateCounts,
    state_for_cc,
    state_for_topic,
)
from app.services.anki.sync import SyncSummary, sync_deck


router = APIRouter(prefix="/anki", tags=["anki"])


def _anki_client() -> AnkiConnectClient:
    """FastAPI dependency: per-request AnkiConnectClient.

    Tests override this dep (via app.dependency_overrides) so they can
    inject a MockTransport-backed client.
    """
    return AnkiConnectClient(settings.ANKICONNECT_URL)


def _summary_payload(summary: SyncSummary) -> dict:
    return {
        "synced_cards": summary.synced_cards,
        "linked_qids": summary.linked_qids,
        "reviews_synced": summary.reviews_synced,
        "error": summary.error,
    }


@router.post("/sync", dependencies=[Depends(verify_coach_token)])
async def sync_anki(
    session: AsyncSession = Depends(get_session),
    client: AnkiConnectClient = Depends(_anki_client),
) -> dict:
    """Trigger a one-off Anki deck sync against the configured deck."""
    try:
        summary = await sync_deck(session, client, deck_name=settings.ANKI_DECK_NAME)
    finally:
        await client.aclose()
    await session.commit()
    return _summary_payload(summary)


@router.get(
    "/cards",
    dependencies=[Depends(verify_coach_token)],
    response_model=list[AnkiCardOut],
)
async def get_cards_for_topic(
    topic_id: int = Query(..., ge=1),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[AnkiCardOut]:
    rows = await list_cards_for_topic(session, topic_id=topic_id, limit=limit)
    return [AnkiCardOut.model_validate(r) for r in rows]


@router.get(
    "/review-queue",
    dependencies=[Depends(verify_coach_token)],
    response_model=list[AnkiCardOut],
)
async def get_review_queue(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[AnkiCardOut]:
    rows = await list_review_queue(session, limit=limit)
    return [AnkiCardOut.model_validate(r) for r in rows]


@router.get(
    "/cards/by-qid/{qid}",
    dependencies=[Depends(verify_coach_token)],
    response_model=list[AnkiCardOut],
)
async def get_cards_for_qid(
    qid: str,
    session: AsyncSession = Depends(get_session),
) -> list[AnkiCardOut]:
    rows = await list_cards_for_qid(session, qid=qid)
    return [AnkiCardOut.model_validate(r) for r in rows]


def _state_to_out(counts: StateCounts) -> AnkiStateOut:
    return AnkiStateOut(
        scope=counts.scope,
        total_cards=counts.total_cards,
        assigned=counts.assigned,
        suspended=counts.suspended,
        new=counts.new,
        learning=counts.learning,
        young=counts.young,
        mature=counts.mature,
        unlock_pct=counts.unlock_pct,
    )


def _retention_to_out(summary: RetentionSummary) -> AnkiRetentionOut:
    windows = [
        AnkiRetentionWindowOut(
            window_days=w.window_days,
            pass_count=w.pass_count,
            fail_count=w.fail_count,
            total=w.total,
            retention=w.retention,
        )
        for w in summary.windows.values()
    ]
    return AnkiRetentionOut(scope=summary.scope, windows=windows)


@router.get(
    "/performance",
    dependencies=[Depends(verify_coach_token)],
    response_model=AnkiPerformanceOut,
)
async def get_anki_performance(
    cc_code: str | None = Query(None),
    topic_id: int | None = Query(None, ge=1),
    window_days: int | None = Query(None, ge=0),
    session: AsyncSession = Depends(get_session),
) -> AnkiPerformanceOut:
    """SPEC §T39 / §V37 — raw state + retention windows for one scope.

    Exactly one of `cc_code` or `topic_id` must be provided. `window_days`
    is optional: omit for the default `(7, 30, 0)` window set, or pass a
    single integer (≥ 0; 0 = all-time) for one window only.

    Per §V37 this surface is data exposure only — no "is this good?"
    verdict, no recommendations, no heuristics. The MCP-driven LLM
    decides what to do with the numbers.
    """
    if (cc_code is None) == (topic_id is None):
        raise HTTPException(
            status_code=422,
            detail="exactly one of cc_code or topic_id is required",
        )

    windows: tuple[int, ...] = DEFAULT_WINDOWS if window_days is None else (window_days,)

    if cc_code is not None:
        state = await state_for_cc(session, cc_code=cc_code)
        retention = await retention_for_cc(session, cc_code=cc_code, windows=windows)
    else:
        assert topic_id is not None
        state = await state_for_topic(session, topic_id=topic_id)
        retention = await retention_for_topic(session, topic_id=topic_id, windows=windows)

    return AnkiPerformanceOut(
        scope=state.scope,
        state=_state_to_out(state),
        retention=_retention_to_out(retention),
    )
