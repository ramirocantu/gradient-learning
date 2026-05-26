"""Pydantic response models for the recommendations endpoint (Ticket 5.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StudyRecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    kind: Literal["topic_weakness", "feature_pattern"]
    label: str | None
    code: str | None
    target_id: int | None
    # 9.0: resolved section code for topic_weakness rows whose `code` is a CC code,
    # so the MCP layer can construct a plan-item write in one turn without a
    # separate outline lookup. None for feature_pattern rows (no target).
    resolved_section_code: str | None = None
    accuracy: float | None
    wilson_lower: float | None
    attempts: int | None
    feature_name: str | None
    feature_value: str | None
    accuracy_with: float | None
    accuracy_without: float | None
    priority_score: float
    reason: str
    representative_qids: list[str]


class StudyNextResponse(BaseModel):
    recommendations: list[StudyRecommendationOut]
    total_candidates_scored: int
    min_attempts_threshold: int
