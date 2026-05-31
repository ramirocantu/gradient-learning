from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.atomic_fact import AtomicFact


class AtomicFactTag(Base):
    """Canonical tag (V-T1): the only target is `node_id` → outline_nodes.

    The atomic-fact analogue of `QuestionTag` / `AnkiNoteTag` — same shape,
    only the FK target differs (`atomic_fact_id`). `source` records HOW the
    tag was derived (V-T2): grounded LLM4Tag (T29/T30) writes `source='llm'`;
    `schema_map` / `manual` reserved for deterministic / human tags. An LLM
    re-run replaces only its own `source='llm'` rows; `manual` + `schema_map`
    survive untouched (V-T2).

    `confidence` is the V69-calibrated logprob grade, required iff
    `source='llm'` and NULL otherwise; `<0.5` ⇒ `manual_review=true`,
    surfaced for review rather than silently dropped (V-T3).

    `AtomicFact.node_id` carries the denormalized "primary" node (the single
    highest-confidence, non-review pick) for cheap reads + Notion's one-page-
    per-node grouping (V-N2); these rows carry the full per-node provenance.
    """

    __tablename__ = "atomic_fact_tags"
    __table_args__ = (
        # V-T3: confidence required iff source='llm'.
        CheckConstraint(
            "(source = 'llm' AND confidence IS NOT NULL) "
            "OR (source <> 'llm' AND confidence IS NULL)",
            name="ck_atomic_fact_tags_confidence_when_llm",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)",
            name="ck_atomic_fact_tags_confidence_range",
        ),
        # V-T3: low-confidence surfaces for review, ⊥ silently dropped.
        CheckConstraint(
            "confidence IS NULL OR confidence >= 0.5 OR manual_review",
            name="ck_atomic_fact_tags_low_conf_flagged",
        ),
        CheckConstraint(
            "source IN ('schema_map', 'llm', 'manual')",
            name="ck_atomic_fact_tags_source",
        ),
        UniqueConstraint(
            "atomic_fact_id", "node_id", "source", name="uq_atomic_fact_tags_node_source"
        ),
        Index("ix_atomic_fact_tags_atomic_fact_id", "atomic_fact_id"),
        Index("ix_atomic_fact_tags_node_id", "node_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    atomic_fact_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("atomic_facts.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("outline_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extractor_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    manual_review: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    overridden_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    atomic_fact: Mapped["AtomicFact"] = relationship(back_populates="tags")
