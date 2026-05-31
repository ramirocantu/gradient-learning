"""Pydantic schemas for SPEC T67 anki assignment routes (V51, V52)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


_PriorityKind = Literal["most_specific_first", "random", "mature_first", "young_first"]


class AnkiAssignmentCreateIn(BaseModel):
    """V52 create-assignment payload. `scope_value` is an outline `node_id`
    (as str); candidates resolve over its subtree (T57, V-O1). `scope_kind`
    ('cc'|'topic') is retained for storage/audit and no longer steers
    resolution — a node is a node regardless of its AAMC `kind` label."""

    scope_kind: Literal["cc", "topic"]
    scope_value: str = Field(min_length=1, max_length=64)
    scheduled_unlock_at: datetime
    max_cards: Optional[int] = Field(default=None, gt=0)
    priority: _PriorityKind = "most_specific_first"


class AnkiAssignmentPatchIn(BaseModel):
    """V51 transition payload — only the two human-driven transitions
    (skipped, completed) live on this endpoint. unlock + auto-complete
    live on their respective scheduler jobs (T63, T64)."""

    status: Literal["skipped", "completed"]


class AnkiAssignmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scope_kind: str
    scope_value: str
    scheduled_unlock_at: datetime
    actual_unlock_at: Optional[datetime]
    card_ids: list[int]
    max_cards: Optional[int]
    priority: Optional[str]
    status: str
    error_text: Optional[str]
    failure_count: int
    created_at: datetime
    updated_at: datetime
