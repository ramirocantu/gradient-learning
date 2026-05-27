"""Admin-only endpoints.

POST   /api/v1/admin/questions/{question_id}/recategorize
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
from pydantic import BaseModel, model_validator
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
from app.services.categorizer.cache import CategorizerCache
from app.models.captures import QuestionTag
from app.services.categorizer import (
    QuestionNotFoundError,
    TagQuestionResult,
    serializable_suggestions,
    tag_question,
)
from app.services.categorizer.outline_lookup import OutlineLookup

router = APIRouter(prefix="/admin", tags=["admin"])


def _openai_client() -> AsyncOpenAI:
    """FastAPI dependency: per-request OpenAI client. V41 retries baked in."""
    return build_openai_client(max_retries=5)


def _categorizer_cache() -> CategorizerCache:
    """FastAPI dependency: per-request CategorizerCache.

    Each request opens its own connection to the shared SQLite file. Cheap;
    WAL mode (set by CategorizerCache) keeps reads/writes non-blocking.
    """
    return CategorizerCache(settings.CATEGORIZER_CACHE_PATH)


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
# POST /questions/{question_id}/recategorize
# --------------------------------------------------------------------------- #


def _tag_result_payload(result: TagQuestionResult) -> dict[str, Any]:
    return {
        "question_id": result.question_id,
        "qid": result.qid,
        "targets_persisted": result.targets_persisted,
        "targets_replaced": result.targets_replaced,
        "suggestions_unresolved": result.suggestions_unresolved,
        "manual_tags_preserved": result.manual_tags_preserved,
        "cache_hit": result.cache_hit,
        "cost_estimate_usd": result.cost_estimate_usd,
        "cost_saved_usd": result.cost_saved_usd,
        "extractor_version": result.extractor_version,
        "categorization": {
            "primary_aamc_section": result.categorize_result.primary_aamc_section,
            "suggestions": serializable_suggestions(result.categorize_result),
            "input_tokens": result.categorize_result.input_tokens,
            "output_tokens": result.categorize_result.output_tokens,
            "parse_warnings": result.categorize_result.parse_warnings,
            "model": result.categorize_result.model,
        },
    }


@router.post("/questions/{question_id}/recategorize")
async def recategorize_question(
    question_id: int,
    session: AsyncSession = Depends(get_session),
    client: AsyncOpenAI = Depends(_openai_client),
    cache: CategorizerCache = Depends(_categorizer_cache),
) -> dict[str, Any]:
    # NOTE: don't close `cache` here. Tests override the dep with a shared
    # instance; closing would prevent subsequent requests in the same test
    # from reading. In production the per-request CategorizerCache is GC'd
    # when its reference drops, which releases the SQLite connection.
    lookup = await OutlineLookup.load(session)
    try:
        result = await tag_question(
            question_id,
            session,
            lookup=lookup,
            openai_client=client,
            cache=cache,
        )
    except QuestionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _tag_result_payload(result)


# --------------------------------------------------------------------------- #
# POST /questions/{question_id}/tags  (manual override)
# DELETE /tags/{tag_id}
# --------------------------------------------------------------------------- #


class ManualTagBody(BaseModel):
    topic_id: int | None = None
    content_category_id: int | None = None
    skill: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ManualTagBody":
        provided = [
            v for v in (self.topic_id, self.content_category_id, self.skill) if v is not None
        ]
        if len(provided) != 1:
            raise ValueError(
                "exactly one of topic_id, content_category_id, skill must be provided"
                f" (got {len(provided)})"
            )
        return self


def _tag_row_payload(row: QuestionTag) -> dict[str, Any]:
    return {
        "tag_id": row.id,
        "question_id": row.question_id,
        "topic_id": row.topic_id,
        "content_category_id": row.content_category_id,
        "skill": row.skill,
        "confidence": float(row.confidence),
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
            topic_id=body.topic_id,
            content_category_id=body.content_category_id,
            skill=body.skill,
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
    "run_categorizer",
    "run_feature_extraction",
    "run_anki_sync",
    "run_anki_topic_resolver",
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
