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
        CheckConstraint("source IN ('uworld')", name="ck_raw_captures_source"),
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
        Index(
            "ix_questions_needs_categorization",
            "id",
            postgresql_where=text("needs_categorization = true"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
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
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
    )
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
    __tablename__ = "question_tags"
    __table_args__ = (
        CheckConstraint(
            "((topic_id IS NOT NULL)::int + (content_category_id IS NOT NULL)::int "
            "+ (skill IS NOT NULL)::int) = 1",
            name="ck_question_tags_exactly_one_target",
        ),
        CheckConstraint(
            "skill IS NULL OR (skill BETWEEN 1 AND 4)",
            name="ck_question_tags_skill_range",
        ),
        CheckConstraint(
            "confidence BETWEEN 0.0 AND 1.0",
            name="ck_question_tags_confidence_range",
        ),
        CheckConstraint(
            "source IN ('uworld_map', 'llm', 'manual')",
            name="ck_question_tags_source",
        ),
        UniqueConstraint(
            "question_id",
            "topic_id",
            "content_category_id",
            "skill",
            "source",
            name="uq_question_tags_target_source",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_question_tags_question_id", "question_id"),
        Index("ix_question_tags_topic_id", "topic_id"),
        Index("ix_question_tags_content_category_id", "content_category_id"),
        Index("ix_question_tags_skill", "skill"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    topic_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("topics.id", ondelete="CASCADE"),
        nullable=True,
    )
    content_category_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("content_categories.id", ondelete="CASCADE"),
        nullable=True,
    )
    skill: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extractor_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    overridden_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    question: Mapped[Question] = relationship(back_populates="tags")
