from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class QuestionFeatures(Base):
    __tablename__ = "question_features"
    __table_args__ = (
        CheckConstraint(
            "question_format IN ('discrete', 'passage_based')",
            name="ck_question_features_question_format",
        ),
        CheckConstraint(
            "reasoning_type IN ('recall', 'comprehension', 'application', 'analysis', 'inference')",
            name="ck_question_features_reasoning_type",
        ),
        CheckConstraint(
            "calculation_steps >= 0",
            name="ck_question_features_calculation_steps_nonneg",
        ),
        CheckConstraint(
            "passage_length_bucket IN ('short', 'medium', 'long') OR passage_length_bucket IS NULL",
            name="ck_question_features_passage_length_bucket",
        ),
        CheckConstraint(
            "passage_type IN ('experimental', 'descriptive', 'hypothesis_driven') "
            "OR passage_type IS NULL",
            name="ck_question_features_passage_type",
        ),
        CheckConstraint(
            "distractor_difficulty IN ('low', 'medium', 'high')",
            name="ck_question_features_distractor_difficulty",
        ),
        CheckConstraint(
            "jargon_density IN ('low', 'medium', 'high')",
            name="ck_question_features_jargon_density",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    question_format: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning_type: Mapped[str] = mapped_column(Text, nullable=False)
    requires_calculation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    calculation_steps: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    involves_graph_or_figure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    involves_data_table: Mapped[bool] = mapped_column(Boolean, nullable=False)
    has_negative_phrasing: Mapped[bool] = mapped_column(Boolean, nullable=False)
    passage_length_bucket: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    passage_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    distractor_difficulty: Mapped[str] = mapped_column(Text, nullable=False)
    trap_distractor_present: Mapped[bool] = mapped_column(Boolean, nullable=False)
    common_misconception: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    jargon_density: Mapped[str] = mapped_column(Text, nullable=False)
    key_concept_summary: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    extractor_version: Mapped[str] = mapped_column(Text, nullable=False)
