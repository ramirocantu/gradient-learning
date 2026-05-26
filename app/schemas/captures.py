"""Pydantic schemas for the ingest endpoint (Ticket 2.2).

Defines the wire shape of the capture payload posted by the Chrome extension
and the response the ingest service returns. Schema-only — no endpoint here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


_STRICT = ConfigDict(extra="forbid")


class ParseWarning(BaseModel):
    model_config = _STRICT

    code: str
    message: str
    selector: Optional[str] = None


class MediaCapture(BaseModel):
    model_config = _STRICT

    content_hash: str
    mime_type: str
    bytes_b64: str
    original_url: Optional[str] = None
    width_px: Optional[int] = None
    height_px: Optional[int] = None


class ChoiceItem(BaseModel):
    model_config = _STRICT

    key: str
    html: str
    plain: str
    media_content_hashes: list[str] = Field(default_factory=list)


class PassageCapture(BaseModel):
    model_config = _STRICT

    uworld_passage_id: Optional[str] = None
    html: str
    plain: str


class ParsedCapture(BaseModel):
    model_config = _STRICT

    passage: Optional[PassageCapture] = None
    stem_html: str
    stem_plain: str
    choices: list[ChoiceItem]
    correct_choice: str
    explanation_html: Optional[str] = None
    explanation_plain: Optional[str] = None
    uworld_aamc_tags: list[str] = Field(default_factory=list)
    selected_choice: str
    is_correct: bool
    time_seconds: Optional[int] = None
    flagged: bool = False


class CapturePayload(BaseModel):
    model_config = _STRICT

    # Source discriminator (§A) — routes to the matching source adapter.
    # Defaults to uworld for back-compat with the current extension.
    source: str = "uworld"
    qid: str
    uworld_test_id: Optional[str] = None
    captured_at: datetime
    html: str
    parsed: ParsedCapture
    media: list[MediaCapture] = Field(default_factory=list)
    parse_warnings: list[ParseWarning] = Field(default_factory=list)
    extension_version: str


class IngestResponse(BaseModel):
    model_config = _STRICT

    capture_id: int
    question_id: int
    attempt_id: int
    passage_id: Optional[int] = None
    media_ids: list[int] = Field(default_factory=list)
