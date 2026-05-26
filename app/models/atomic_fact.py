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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AtomicFact(Base):
    """Grounded fact extracted from a PDF source (V-KB1, §I.schema).

    One row per atomic claim; `node_id` is the outline node it tags
    (nullable until LLM4Tag grounded tagging runs in T29/T30). `content_hash`
    dedupes within a course — same fact text from a re-ingested PDF maps
    to the existing row.
    """

    __tablename__ = "atomic_facts"
    __table_args__ = (
        UniqueConstraint("course_id", "content_hash", name="uq_atomic_facts_course_hash"),
        Index("ix_atomic_facts_course_id", "course_id"),
        Index("ix_atomic_facts_pdf_source_id", "pdf_source_id"),
        Index("ix_atomic_facts_node_id", "node_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    course_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    pdf_source_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pdf_sources.id", ondelete="CASCADE"), nullable=False
    )
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    node_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("outline_nodes.id", ondelete="SET NULL"), nullable=True
    )
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    pdf_source: Mapped["PdfSource"] = relationship(  # noqa: F821
        back_populates="atomic_facts"
    )
    tags: Mapped[list["AtomicFactTag"]] = relationship(  # noqa: F821
        back_populates="atomic_fact", cascade="all, delete-orphan"
    )
