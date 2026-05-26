from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class NoteOut(BaseModel):
    id: int
    attempt_id: int
    note_text: str
    flag_for_review: bool
    source: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
