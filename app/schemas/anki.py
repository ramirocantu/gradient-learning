"""Pydantic response schemas for SPEC §T5 Anki read endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class AnkiCardTagOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    tag_raw: str
    parsed_kind: str
    topic_id: Optional[int] = None
    question_qid: Optional[str] = None


class AnkiCardOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    anki_card_id: int
    deck_name: str
    note_id: Optional[int] = None
    model_name: Optional[str] = None
    fields_json: Optional[dict[str, Any]] = None
    due_date: Optional[date] = None
    interval_days: Optional[int] = None
    ease: Optional[int] = None
    lapses: Optional[int] = None
    queue: Optional[int] = None
    sync_at: datetime
    tags: list[AnkiCardTagOut] = []


class AnkiStateOut(BaseModel):
    """SPEC §T39 / §V28 / §V37 — raw Anki state buckets for one scope."""

    scope: str
    total_cards: int
    assigned: int
    suspended: int
    new: int
    learning: int
    young: int
    mature: int
    unlock_pct: Optional[float] = None


class AnkiRetentionWindowOut(BaseModel):
    """SPEC §T39 / §V27 — raw pass/fail counts for one window."""

    window_days: int
    pass_count: int
    fail_count: int
    total: int
    retention: Optional[float] = None


class AnkiRetentionOut(BaseModel):
    scope: str
    windows: list[AnkiRetentionWindowOut]


class AnkiPerformanceOut(BaseModel):
    """SPEC §T39 / §V37 — `{state, retention windows}` for one CC or topic.

    Data-only per §V37 — no "is this good?" verdict, no heuristics. The
    consumer (LLM via MCP) interprets the numbers.
    """

    scope: str
    state: AnkiStateOut
    retention: AnkiRetentionOut
