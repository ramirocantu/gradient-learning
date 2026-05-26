"""Pydantic response models for the analytics endpoints (Ticket 5.1)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict


_FROM_ATTRS = ConfigDict(from_attributes=True)


class AccuracyStatOut(BaseModel):
    model_config = _FROM_ATTRS

    label: str
    code: str | None
    kind: Literal["section", "content_category", "topic", "skill"]
    target_id: int | None
    attempts: int
    correct: int
    accuracy: float
    wilson_lower: float


class TimingStatOut(BaseModel):
    model_config = _FROM_ATTRS

    median_seconds_discrete: float | None
    median_seconds_passage_based: float | None
    questions_over_target_discrete: int
    questions_over_target_passage: int


class TrendPointOut(BaseModel):
    model_config = _FROM_ATTRS

    period_start: date
    accuracy: float
    attempts: int


class MasteryReportOut(BaseModel):
    model_config = _FROM_ATTRS

    by_section: list[AccuracyStatOut]
    by_content_category: list[AccuracyStatOut]
    by_topic: list[AccuracyStatOut]
    by_skill: list[AccuracyStatOut]
    timing: TimingStatOut
    trend_7d: list[TrendPointOut]
    trend_30d: list[TrendPointOut]
    uncategorized_question_count: int
    total_attempts: int
    total_questions: int
