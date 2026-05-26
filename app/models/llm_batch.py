"""SQLAlchemy model for `llm_batch_runs` (SPEC §T51).

Tracks each Anthropic Message Batches API submission across extractors
(anki topic resolver, categorizer, feature extractor). The row's
`anthropic_batch_id` is the handle to retrieve status + stream results;
counts are mirrored locally so /admin doesn't have to round-trip the
Anthropic API on every page load.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import NUMERIC, DateTime, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class LlmBatchRun(Base):
    __tablename__ = "llm_batch_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    anthropic_batch_id: Mapped[str] = mapped_column(Text, nullable=False)
    extractor: Mapped[str] = mapped_column(Text, nullable=False)
    extractor_version: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    processing_status: Mapped[str] = mapped_column(Text, nullable=False)
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    succeeded_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errored_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    canceled_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expired_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_cost_usd: Mapped[Decimal | None] = mapped_column(NUMERIC(10, 4), nullable=True)
    result_persisted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("anthropic_batch_id", name="uq_llm_batch_runs_anthropic_id"),
        Index(
            "ix_llm_batch_runs_extractor_submitted",
            "extractor",
            "submitted_at",
        ),
    )
