from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class DiscriminatorFactor(Base):
    """Persistable discriminator factor from MCP/tutor seam (V-M1, V-M3, §I).

    Append-only by design (V-M3): a `(question_id, factor_text)` UQ
    deduplicates re-writes of the same factor while preserving link
    history. `node_id` ties the factor to an outline node (nullable
    when the host hasn't chosen one yet). `notion_block_id` records the
    Notion mirror anchor when V-N1/V-N2 sync has happened — append-only
    on the Notion side too.
    """

    __tablename__ = "discriminator_factors"
    __table_args__ = (
        UniqueConstraint(
            "question_id", "factor_text", name="uq_discriminator_factors_question_text"
        ),
        Index("ix_discriminator_factors_question_id", "question_id"),
        Index("ix_discriminator_factors_node_id", "node_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    factor_text: Mapped[str] = mapped_column(Text, nullable=False)
    node_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("outline_nodes.id", ondelete="SET NULL"), nullable=True
    )
    notion_block_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
