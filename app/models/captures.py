from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class RawCapture(Base):
    __tablename__ = "raw_captures"
    __table_args__ = (
        # Open source discriminator (§A): no closed enum — any registered
        # source adapter may write captures (uworld = reference adapter).
        Index("ix_raw_captures_qid", "qid"),
        Index("ix_raw_captures_captured_at", "captured_at"),
        Index(
            "ix_raw_captures_qid_with_warnings",
            "qid",
            postgresql_where=text("parse_warnings IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="uworld")
    qid: Mapped[str] = mapped_column(Text, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_html: Mapped[str] = mapped_column(Text, nullable=False)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    parse_warnings: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONB, nullable=True)
    extension_version: Mapped[str] = mapped_column(Text, nullable=False)
    uworld_test_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_passages_content_hash"),
        Index(
            "ix_passages_uworld_id",
            "uworld_passage_id",
            unique=True,
            postgresql_where=text("uworld_passage_id IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uworld_passage_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    html: Mapped[str] = mapped_column(Text, nullable=False)
    plain_text: Mapped[str] = mapped_column(Text, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    questions: Mapped[list[Question]] = relationship(back_populates="passage")


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        Index("ix_questions_passage_id", "passage_id"),
        Index("ix_questions_source", "source"),
        Index(
            "ix_questions_needs_categorization",
            "id",
            postgresql_where=text("needs_categorization = true"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Open source discriminator (§A plugin seam). `qid` stays the external key
    # for now; §I's rename to external_id + UQ(source, external_id) folds into
    # the T12–T14 reader ports.
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="uworld")
    qid: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    passage_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("passages.id", ondelete="SET NULL"),
        nullable=True,
    )
    stem_html: Mapped[str] = mapped_column(Text, nullable=False)
    stem_plain: Mapped[str] = mapped_column(Text, nullable=False)
    choices: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    correct_choice: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_html: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation_plain: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    uworld_aamc_tags: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    needs_categorization: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    passage: Mapped[Optional[Passage]] = relationship(back_populates="questions")
    attempts: Mapped[list[Attempt]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )
    tags: Mapped[list[QuestionTag]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class Attempt(Base):
    __tablename__ = "attempts"
    __table_args__ = (
        Index("ix_attempts_question_id", "question_id"),
        Index("ix_attempts_attempted_at", "attempted_at"),
        Index("ix_attempts_question_attempted", "question_id", "attempted_at"),
        Index("ix_attempts_uworld_test_id", "uworld_test_id"),
        Index("ix_attempts_source", "source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Open source discriminator (§A). §I's uworld_test_id→session_ref rename
    # folds into the T12–T14 ports.
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="uworld")
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    selected_choice: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    time_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    uworld_test_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    question: Mapped[Question] = relationship(back_populates="attempts")


class QuestionTag(Base):
    """Canonical tag (V-T1): the only target is `node_id` → outline_nodes.

    The PoC's 3-target (topic_id/content_category_id/skill) is retired. `source`
    records HOW the tag was derived (V-T2); `confidence` is required for
    `source='llm'`, NULL otherwise, and `<0.5` ⇒ `manual_review` (V-T3).
    """

    __tablename__ = "question_tags"
    __table_args__ = (
        # V-T3: confidence required iff source='llm'.
        CheckConstraint(
            "(source = 'llm' AND confidence IS NOT NULL) "
            "OR (source <> 'llm' AND confidence IS NULL)",
            name="ck_question_tags_confidence_when_llm",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)",
            name="ck_question_tags_confidence_range",
        ),
        # V-T3: low-confidence surfaces for review, ⊥ silently dropped.
        CheckConstraint(
            "confidence IS NULL OR confidence >= 0.5 OR manual_review",
            name="ck_question_tags_low_conf_flagged",
        ),
        CheckConstraint(
            "source IN ('schema_map', 'llm', 'manual')",
            name="ck_question_tags_source",
        ),
        UniqueConstraint(
            "question_id", "node_id", "source", name="uq_question_tags_node_source"
        ),
        Index("ix_question_tags_question_id", "question_id"),
        Index("ix_question_tags_node_id", "node_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("questions.id", ondelete="CASCADE"),
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

    question: Mapped[Question] = relationship(back_populates="tags")
