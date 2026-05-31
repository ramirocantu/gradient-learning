from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
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


class PdfSource(Base):
    """An ingested classroom PDF, keyed by SHA-256 (V-KB1, §I.schema).

    One row per uploaded PDF per course; `sha256` deduplicates re-uploads
    of the same file. Atomic facts (V-L3 ground truth) hang off
    `pdf_source_id`. Status tracks the ingest pipeline lifecycle.
    """

    __tablename__ = "pdf_sources"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_pdf_sources_sha256"),
        CheckConstraint(
            "status IN ('pending','parsing','ingested','failed')",
            name="ck_pdf_sources_status",
        ),
        Index("ix_pdf_sources_course_id", "course_id"),
        Index("ix_pdf_sources_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    course_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    ingested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    atomic_facts: Mapped[list["AtomicFact"]] = relationship(  # noqa: F821
        back_populates="pdf_source", cascade="all, delete-orphan"
    )
