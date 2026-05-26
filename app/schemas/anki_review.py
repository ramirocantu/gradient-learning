"""Pydantic schemas for SPEC T67 + T76 + T77 anki review routes (V53 amended)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AnkiReviewCreateIn(BaseModel):
    """V53 amended create-review payload. Reviews are standalone — no
    FK to assignments; deck name derived from the new row's own PK.
    `card_ids` carry AnKing-native ids (BIGINT) so AnkiConnect's
    filtered-deck `cid:<csv>` query receives them verbatim."""

    card_ids: list[int] = Field(min_length=1)
    review_date: date


class AnkiReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    review_date: date
    card_ids: list[int]
    deck_name: str
    status: str
    error_text: Optional[str]
    failure_count: int
    created_at: datetime
    pushed_at: Optional[datetime]
