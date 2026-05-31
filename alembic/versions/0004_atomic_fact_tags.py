"""T30 — atomic_fact_tags canonical tag table (V-T1, V-T2, V-T3, §I.schema)

Materializes the §I-line-94 canonical `<target>_tags` shape for atomic
facts — the edge table connecting atomic_facts ↔ outline_nodes with
per-node provenance (source / calibrated confidence / manual_review /
extractor_version). Byte-identical constraint shape to question_tags /
anki_note_tags; only the FK target differs.

`source` is TEXT + CHECK (no named ENUM TYPE), so V-MIG1's leaked-type
downgrade trap does not apply — `drop_table` is a clean teardown.

Revision ID: 0004_atomic_fact_tags
Revises: 0003_kb_substrate
Create Date: 2026-05-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_atomic_fact_tags"
down_revision: Union[str, None] = "0003_kb_substrate"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "atomic_fact_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("atomic_fact_id", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("extractor_version", sa.Text(), nullable=True),
        sa.Column("manual_review", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("is_overridden", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("overridden_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        # V-T3: confidence required iff source='llm'.
        sa.CheckConstraint(
            "(source = 'llm' AND confidence IS NOT NULL) "
            "OR (source <> 'llm' AND confidence IS NULL)",
            name="ck_atomic_fact_tags_confidence_when_llm",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)",
            name="ck_atomic_fact_tags_confidence_range",
        ),
        # V-T3: low-confidence surfaces for review, ⊥ silently dropped.
        sa.CheckConstraint(
            "confidence IS NULL OR confidence >= 0.5 OR manual_review",
            name="ck_atomic_fact_tags_low_conf_flagged",
        ),
        sa.CheckConstraint(
            "source IN ('schema_map', 'llm', 'manual')",
            name="ck_atomic_fact_tags_source",
        ),
        sa.ForeignKeyConstraint(["atomic_fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["outline_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "atomic_fact_id", "node_id", "source", name="uq_atomic_fact_tags_node_source"
        ),
    )
    op.create_index(
        "ix_atomic_fact_tags_atomic_fact_id",
        "atomic_fact_tags",
        ["atomic_fact_id"],
        unique=False,
    )
    op.create_index("ix_atomic_fact_tags_node_id", "atomic_fact_tags", ["node_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_atomic_fact_tags_node_id", table_name="atomic_fact_tags")
    op.drop_index("ix_atomic_fact_tags_atomic_fact_id", table_name="atomic_fact_tags")
    op.drop_table("atomic_fact_tags")
