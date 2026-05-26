"""Pydantic response models for the analyzer endpoints (Tickets 4.3, 4.4, 4.5)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

_FROM_ATTRS = ConfigDict(from_attributes=True)


class AnalysisFilterOut(BaseModel):
    model_config = _FROM_ATTRS

    section_code: str | None
    content_category_code: str | None
    topic_id: int | None
    skill: int | None
    since: date | None
    until: date | None
    min_sample_size: int


class FeatureFindingOut(BaseModel):
    model_config = _FROM_ATTRS

    feature_name: str
    feature_value: str
    accuracy_with: float
    accuracy_without: float
    attempts_with: int
    attempts_without: int
    correct_with: int
    correct_without: int
    accuracy_delta: float
    wilson_lower_with: float
    wilson_lower_without: float
    confident_delta: float
    representative_missed_qids: list[str]


class CoverageStatsOut(BaseModel):
    model_config = _FROM_ATTRS

    questions_with_features: int
    questions_without_features: int
    feature_extractor_version: str


class InsightReportOut(BaseModel):
    model_config = _FROM_ATTRS

    filter_applied: AnalysisFilterOut
    total_attempts_in_scope: int
    total_questions_in_scope: int
    baseline_accuracy: float
    baseline_wilson_lower: float
    findings: list[FeatureFindingOut]
    coverage: CoverageStatsOut


class InsightSynthesisResponse(BaseModel):
    model_config = _FROM_ATTRS

    markdown: str
    report: InsightReportOut
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    cost_saved_usd: float
    extractor_version: str
    model: str
