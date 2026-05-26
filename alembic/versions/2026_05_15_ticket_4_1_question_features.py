"""ticket 4.1 question features

Revision ID: 20260515_t41
Revises: 20260515_t35
Create Date: 2026-05-15

Creates the question_features table — one row per Question, holding the
content-agnostic features (graph interpretation, calc steps, distractor
difficulty, etc.) produced by the Phase 4.2 LLM feature extractor.

Schema only: no row backfill, no Question changes. CARS gets its own
table in a follow-up when CARS captures arrive.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260515_t41"
down_revision: Union[str, None] = "20260515_t35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_features",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("question_format", sa.Text(), nullable=False),
        sa.Column("reasoning_type", sa.Text(), nullable=False),
        sa.Column("requires_calculation", sa.Boolean(), nullable=False),
        sa.Column(
            "calculation_steps",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("involves_graph_or_figure", sa.Boolean(), nullable=False),
        sa.Column("involves_data_table", sa.Boolean(), nullable=False),
        sa.Column("has_negative_phrasing", sa.Boolean(), nullable=False),
        sa.Column("passage_length_bucket", sa.Text(), nullable=True),
        sa.Column("passage_type", sa.Text(), nullable=True),
        sa.Column("distractor_difficulty", sa.Text(), nullable=False),
        sa.Column("trap_distractor_present", sa.Boolean(), nullable=False),
        sa.Column("common_misconception", sa.Text(), nullable=True),
        sa.Column("jargon_density", sa.Text(), nullable=False),
        sa.Column("key_concept_summary", sa.Text(), nullable=False),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("extractor_version", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "question_format IN ('discrete', 'passage_based')",
            name="ck_question_features_question_format",
        ),
        sa.CheckConstraint(
            "reasoning_type IN ('recall', 'comprehension', 'application', 'analysis', 'inference')",
            name="ck_question_features_reasoning_type",
        ),
        sa.CheckConstraint(
            "calculation_steps >= 0",
            name="ck_question_features_calculation_steps_nonneg",
        ),
        sa.CheckConstraint(
            "passage_length_bucket IN ('short', 'medium', 'long') OR passage_length_bucket IS NULL",
            name="ck_question_features_passage_length_bucket",
        ),
        sa.CheckConstraint(
            "passage_type IN ('experimental', 'descriptive', 'hypothesis_driven') "
            "OR passage_type IS NULL",
            name="ck_question_features_passage_type",
        ),
        sa.CheckConstraint(
            "distractor_difficulty IN ('low', 'medium', 'high')",
            name="ck_question_features_distractor_difficulty",
        ),
        sa.CheckConstraint(
            "jargon_density IN ('low', 'medium', 'high')",
            name="ck_question_features_jargon_density",
        ),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("question_id", name="uq_question_features_question_id"),
    )


def downgrade() -> None:
    op.drop_table("question_features")
