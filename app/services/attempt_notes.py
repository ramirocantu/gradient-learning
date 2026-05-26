from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attempt_note import AttemptNote
from app.models.captures import Attempt


class AttemptNotFoundError(Exception):
    pass


class NoteNotFoundError(Exception):
    pass


async def create_note(
    session: AsyncSession,
    *,
    attempt_id: int,
    note_text: str,
    flag_for_review: bool = False,
    source: str = "user",
) -> AttemptNote:
    attempt = await session.get(Attempt, attempt_id)
    if attempt is None:
        raise AttemptNotFoundError(attempt_id)
    note = AttemptNote(
        attempt_id=attempt_id,
        note_text=note_text.strip(),
        flag_for_review=flag_for_review,
        source=source,
    )
    session.add(note)
    await session.flush()
    return note


async def list_notes(session: AsyncSession, *, attempt_id: int) -> list[AttemptNote]:
    rows = (
        (
            await session.execute(
                select(AttemptNote)
                .where(AttemptNote.attempt_id == attempt_id)
                .order_by(AttemptNote.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def delete_note(session: AsyncSession, *, note_id: int) -> None:
    note = await session.get(AttemptNote, note_id)
    if note is None:
        raise NoteNotFoundError(note_id)
    await session.delete(note)
