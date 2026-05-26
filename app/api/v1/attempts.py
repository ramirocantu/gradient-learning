from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session, verify_coach_token
from app.models.attempt_note import AttemptNote
from app.schemas.attempt_notes import NoteOut
from app.services.attempt_notes import (
    AttemptNotFoundError,
    NoteNotFoundError,
    create_note,
    delete_note,
    list_notes,
)

router = APIRouter()


class NoteIn(BaseModel):
    note_text: str
    flag_for_review: bool = False
    source: Literal["user", "mcp"] = "user"

    @field_validator("note_text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("note_text must not be blank")
        return v


@router.get("/attempts/{attempt_id}/notes", response_model=list[NoteOut])
async def get_notes(
    attempt_id: int,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> list[AttemptNote]:
    return await list_notes(session, attempt_id=attempt_id)


@router.post("/attempts/{attempt_id}/notes", response_model=NoteOut, status_code=201)
async def add_note(
    attempt_id: int,
    body: NoteIn,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> AttemptNote:
    try:
        return await create_note(
            session,
            attempt_id=attempt_id,
            note_text=body.note_text,
            flag_for_review=body.flag_for_review,
            source=body.source,
        )
    except AttemptNotFoundError:
        raise HTTPException(404, detail="attempt not found")


@router.delete("/attempts/notes/{note_id}", status_code=204)
async def remove_note(
    note_id: int,
    session: AsyncSession = Depends(get_session),
    _: None = Depends(verify_coach_token),
) -> None:
    try:
        await delete_note(session, note_id=note_id)
    except NoteNotFoundError:
        raise HTTPException(404, detail="note not found")
