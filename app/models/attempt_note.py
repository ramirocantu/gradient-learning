from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AttemptNote(Base):
    __tablename__ = "attempt_notes"
    __table_args__ = (
        CheckConstraint("source IN ('user', 'mcp')", name="ck_attempt_notes_source"),
        Index("ix_attempt_notes_attempt_id", "attempt_id"),
        Index(
            "ix_attempt_notes_flagged",
            "flag_for_review",
            postgresql_where="flag_for_review = true",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    attempt_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("attempts.id", ondelete="CASCADE"), nullable=False
    )
    note_text: Mapped[str] = mapped_column(Text, nullable=False)
    flag_for_review: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    source: Mapped[str] = mapped_column(Text, nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )
