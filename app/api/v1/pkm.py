"""PKM write seam — discriminator factors (T31, V-M1, V-M3).

`POST /api/v1/pkm/discriminators` persists a discriminator factor from the
tutor/MCP seam. Persist-only (V-M1): the body carries data, no verdict.
Append-only + deduped by `(question_id, factor_text)` (V-M3). Auth is
X-Coach-Token, shared with the ingest + tutor surfaces. Consumed by the MCP
server's `write_discriminator_factor` tool (separate repo, httpx proxy).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.pkm import DiscriminatorIn, DiscriminatorOut
from app.services.tutor import discriminators as disc_svc

router = APIRouter(prefix="/pkm", tags=["pkm"])


@router.post("/discriminators", response_model=DiscriminatorOut)
async def post_discriminator(
    payload: DiscriminatorIn,
    session: AsyncSession = Depends(get_session),) -> DiscriminatorOut:
    try:
        row = await disc_svc.write_discriminator_factor(
            session,
            question_id=payload.question_id,
            factor_text=payload.factor_text,
            node_id=payload.node_id,
        )
    except disc_svc.QuestionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"reason": "question_not_found", "question_id": payload.question_id},
        ) from exc
    except disc_svc.NodeNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"reason": "node_not_found", "node_id": payload.node_id},
        ) from exc
    return DiscriminatorOut.model_validate(row)
