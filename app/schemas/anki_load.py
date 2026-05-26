"""Pydantic schemas for SPEC T67 anki load-config + load-adherence routes
(V54, V59, V60)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class AnkiLoadConfigIn(BaseModel):
    """V59 singleton upsert. Both budgets must be positive (mirrors the
    DB CHECK constraints + lets Pydantic short-circuit at the API layer
    instead of bouncing through an IntegrityError)."""

    daily_card_review_budget: int = Field(gt=0)
    daily_minutes_budget: Decimal = Field(gt=Decimal("0"))


class AnkiLoadConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    daily_card_review_budget: int
    daily_minutes_budget: Decimal
    updated_at: datetime


class AnkiLoadAdherenceOut(BaseModel):
    """V54 deterministic adherence shape. ⊥ a `recommended_changes`
    field per V60 — advisory lives in the MCP host chat."""

    model_config = ConfigDict(from_attributes=True)

    window_days: int
    projected_daily_load: int
    projected_daily_minutes: float
    daily_card_review_budget: int
    daily_minutes_budget: float
    headroom_card_review_pct: float
    headroom_minutes_pct: float
    status_label: str
