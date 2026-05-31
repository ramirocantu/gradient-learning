"""Admin-only endpoints.

POST   /api/v1/admin/questions/{question_id}/tags
DELETE /api/v1/admin/tags/{tag_id}

No auth — localhost-only service per CLAUDE.md.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, verify_coach_token
from app.config import settings
from app.services.anki.client import AnkiConnectClient
from app.services.llm.client import build_openai_client
from app.services.system_status import collect_system_status
from app.services.admin_tags import (
    ManualTagConflictError,
    ManualTagValidationError,
    create_manual_tag as _create_manual_tag_service,
)
from app.services.admin_tags import (
    QuestionNotFoundError as ManualTagQuestionNotFoundError,
)
from app.models.captures import QuestionTag

router = APIRouter(prefix="/admin", tags=["admin"])


# --------------------------------------------------------------------------- #
# Status-probe clients (T39). Separate deps so tests override each at the SDK
# boundary (V16). All three are mocked in tests/test_admin_status.py.
# --------------------------------------------------------------------------- #


async def _anki_status_client() -> AsyncIterator[AnkiConnectClient]:
    """Per-request AnkiConnect client for the health probe; closed on teardown."""
    client = AnkiConnectClient(settings.ANKICONNECT_URL)
    try:
        yield client
    finally:
        await client.aclose()


def _openai_status_client() -> AsyncOpenAI:
    """OpenAI client for the health probe. ``max_retries=0`` (not V41's ≥5,
    which is extractor-scoped): a status read must fail fast, not block for
    seconds retrying a down endpoint."""
    return build_openai_client(max_retries=0)


async def _notion_status_client() -> AsyncIterator[Any]:
    """Notion ``AsyncClient`` for the token-validity probe, or None when
    ``NOTION_API_TOKEN`` is unset (probe reports unconfigured, no call).
    Import is lazy so a missing token never imports notion-client."""
    if not settings.NOTION_API_TOKEN:
        yield None
        return
    from notion_client import AsyncClient

    client = AsyncClient(auth=settings.NOTION_API_TOKEN)
    try:
        yield client
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- #
# POST /questions/{question_id}/tags  (manual override)
# DELETE /tags/{tag_id}
# --------------------------------------------------------------------------- #


class ManualTagBody(BaseModel):
    node_id: int


def _tag_row_payload(row: QuestionTag) -> dict[str, Any]:
    return {
        "tag_id": row.id,
        "question_id": row.question_id,
        "node_id": row.node_id,
        "confidence": float(row.confidence) if row.confidence is not None else None,
        "source": row.source,
        "rationale": row.rationale,
        "extractor_version": row.extractor_version,
    }


@router.post(
    "/questions/{question_id}/tags",
    status_code=status.HTTP_201_CREATED,
)
async def create_manual_tag(
    question_id: int,
    body: ManualTagBody,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    try:
        row = await _create_manual_tag_service(
            session,
            question_id,
            node_id=body.node_id,
        )
    except ManualTagQuestionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ManualTagValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ManualTagConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _tag_row_payload(row)


# --------------------------------------------------------------------------- #
# GET /jobs  and  POST /jobs/{job_name}/trigger
#
# The dashboard's /admin page calls list_jobs_payload() and
# trigger_job_logic() directly (in-process). The HTTP routes below wrap
# the same helpers plus the X-Coach-Token guard, preserving the public
# JSON contract.
# --------------------------------------------------------------------------- #


_VALID_JOBS = {
    "run_anki_sync",
    "run_anki_assignment_unlock",
    "run_anki_assignment_complete",
    "run_anki_review",
}


async def list_jobs_payload() -> list[dict]:
    """Snapshot of scheduler job state. Caller owns auth/transport concerns."""
    from app.scheduler import scheduler

    return [
        {
            "job_id": job.id,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        }
        for job in scheduler.get_jobs()
    ]


async def trigger_job_logic(job_name: str) -> dict:
    """Validate + nudge scheduler to run `job_name` immediately.

    Raises HTTPException with the same status codes the JSON route used to
    return (404 unknown, 409 already in-flight, 503 scheduler not running)
    so callers can map them uniformly.
    """
    from app.scheduler import _inflight, _lock, scheduler

    if job_name not in _VALID_JOBS:
        raise HTTPException(404, detail=f"unknown job: {job_name}")
    async with _lock:
        if job_name in _inflight:
            raise HTTPException(409, detail=f"{job_name} already running")
    job = scheduler.get_job(job_name)
    if job is None:
        raise HTTPException(503, detail="scheduler not running")
    scheduler.modify_job(job_name, next_run_time=datetime.now(timezone.utc))
    return {"status": "triggered", "job": job_name}


@router.get("/jobs", dependencies=[Depends(verify_coach_token)])
async def list_jobs() -> list[dict]:
    return await list_jobs_payload()


@router.post(
    "/jobs/{job_name}/trigger",
    status_code=202,
    dependencies=[Depends(verify_coach_token)],
)
async def trigger_job(job_name: str) -> dict:
    return await trigger_job_logic(job_name)


# --------------------------------------------------------------------------- #
# GET /status  (T39)
#
# Real connection health for a client settings panel: probe AnkiConnect,
# OpenAI, and Notion reachability + fold each scheduler job's last TaskRun
# outcome into the next-run snapshot. Read-only public-seam JSON (V-D1).
# --------------------------------------------------------------------------- #


@router.get("/status", dependencies=[Depends(verify_coach_token)])
async def system_status(
    session: AsyncSession = Depends(get_session),
    anki_client: AnkiConnectClient = Depends(_anki_status_client),
    openai_client: AsyncOpenAI = Depends(_openai_status_client),
    notion_client: Any = Depends(_notion_status_client),
) -> dict[str, Any]:
    return await collect_system_status(
        session,
        anki_client=anki_client,
        openai_client=openai_client,
        notion_client=notion_client,
        job_snapshots=await list_jobs_payload(),
        openai_model=settings.OPENAI_MODEL,
        openai_configured=bool(settings.OPENAI_API_KEY),
        notion_configured=bool(settings.NOTION_API_TOKEN),
    )


@router.delete("/tags/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tag(
    tag_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    row = (
        await session.execute(select(QuestionTag).where(QuestionTag.id == tag_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"tag_id={tag_id} not found")
    if row.source == "manual":
        await session.delete(row)
    elif row.source == "llm":
        row.is_overridden = True
        row.overridden_at = datetime.now(timezone.utc)
    else:
        raise HTTPException(
            status_code=403,
            detail=(
                f"refusing to delete tag with source={row.source!r}; "
                "only manual and llm tags can be removed"
            ),
        )
    await session.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
