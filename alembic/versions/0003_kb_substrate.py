"""T24 — KB substrate tables (V-KB1, §I.schema)

Adds the P2 knowledge-base substrate: pdf_sources, atomic_facts,
content_embeddings, concept_edges, notion_pages, discriminator_factors.
The `embedding` column on content_embeddings is JSONB placeholder pending
T25's pgvector dep + extension + image swap — T25 ALTERs to
`vector(N)` where N matches `EMBEDDING_MODEL`.

Revision ID: 0003_kb_substrate
Revises: 0002_source_discriminator
Create Date: 2026-05-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_kb_substrate"
down_revision: Union[str, None] = "0002_source_discriminator"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pdf_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending','parsing','ingested','failed')",
            name="ck_pdf_sources_status",
        ),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sha256", name="uq_pdf_sources_sha256"),
    )
    op.create_index("ix_pdf_sources_course_id", "pdf_sources", ["course_id"], unique=False)
    op.create_index("ix_pdf_sources_status", "pdf_sources", ["status"], unique=False)

    op.create_table(
        "atomic_facts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("pdf_source_id", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("node_id", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pdf_source_id"], ["pdf_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["outline_nodes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("course_id", "content_hash", name="uq_atomic_facts_course_hash"),
    )
    op.create_index("ix_atomic_facts_course_id", "atomic_facts", ["course_id"], unique=False)
    op.create_index(
        "ix_atomic_facts_pdf_source_id", "atomic_facts", ["pdf_source_id"], unique=False
    )
    op.create_index("ix_atomic_facts_node_id", "atomic_facts", ["node_id"], unique=False)

    # content_embeddings: T25 ALTERs `embedding` JSONB → vector(N) after
    # pgvector dep + extension land. Placeholder lets the rest of the
    # substrate land now without forcing the pgvector image swap.
    op.create_table(
        "content_embeddings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("entity_kind", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("embedding", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "entity_kind IN ('question','atomic_fact','outline_node')",
            name="ck_content_embeddings_entity_kind",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_kind",
            "entity_id",
            "embedding_version",
            name="uq_content_embeddings_entity_version",
        ),
    )
    op.create_index(
        "ix_content_embeddings_entity",
        "content_embeddings",
        ["entity_kind", "entity_id"],
        unique=False,
    )
    op.create_index(
        "ix_content_embeddings_version",
        "content_embeddings",
        ["embedding_version"],
        unique=False,
    )

    op.create_table(
        "concept_edges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("src_node_id", sa.Integer(), nullable=False),
        sa.Column("dst_node_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("score", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("kind IN ('similarity','manual')", name="ck_concept_edges_kind"),
        sa.CheckConstraint("src_node_id <> dst_node_id", name="ck_concept_edges_no_self_edge"),
        sa.CheckConstraint(
            "score IS NULL OR (score BETWEEN -1.0 AND 1.0)",
            name="ck_concept_edges_score_range",
        ),
        sa.ForeignKeyConstraint(["src_node_id"], ["outline_nodes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dst_node_id"], ["outline_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "src_node_id", "dst_node_id", "kind", name="uq_concept_edges_src_dst_kind"
        ),
    )
    op.create_index("ix_concept_edges_src", "concept_edges", ["src_node_id"], unique=False)
    op.create_index("ix_concept_edges_dst", "concept_edges", ["dst_node_id"], unique=False)
    op.create_index("ix_concept_edges_kind", "concept_edges", ["kind"], unique=False)

    op.create_table(
        "notion_pages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.Integer(), nullable=False),
        sa.Column("notion_page_id", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["node_id"], ["outline_nodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("node_id", name="uq_notion_pages_node_id"),
        sa.UniqueConstraint("notion_page_id", name="uq_notion_pages_notion_page_id"),
    )
    op.create_index("ix_notion_pages_node_id", "notion_pages", ["node_id"], unique=False)

    op.create_table(
        "discriminator_factors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("factor_text", sa.Text(), nullable=False),
        sa.Column("node_id", sa.Integer(), nullable=True),
        sa.Column("notion_block_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["node_id"], ["outline_nodes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "question_id",
            "factor_text",
            name="uq_discriminator_factors_question_text",
        ),
    )
    op.create_index(
        "ix_discriminator_factors_question_id",
        "discriminator_factors",
        ["question_id"],
        unique=False,
    )
    op.create_index(
        "ix_discriminator_factors_node_id",
        "discriminator_factors",
        ["node_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_discriminator_factors_node_id", table_name="discriminator_factors")
    op.drop_index("ix_discriminator_factors_question_id", table_name="discriminator_factors")
    op.drop_table("discriminator_factors")

    op.drop_index("ix_notion_pages_node_id", table_name="notion_pages")
    op.drop_table("notion_pages")

    op.drop_index("ix_concept_edges_kind", table_name="concept_edges")
    op.drop_index("ix_concept_edges_dst", table_name="concept_edges")
    op.drop_index("ix_concept_edges_src", table_name="concept_edges")
    op.drop_table("concept_edges")

    op.drop_index("ix_content_embeddings_version", table_name="content_embeddings")
    op.drop_index("ix_content_embeddings_entity", table_name="content_embeddings")
    op.drop_table("content_embeddings")

    op.drop_index("ix_atomic_facts_node_id", table_name="atomic_facts")
    op.drop_index("ix_atomic_facts_pdf_source_id", table_name="atomic_facts")
    op.drop_index("ix_atomic_facts_course_id", table_name="atomic_facts")
    op.drop_table("atomic_facts")

    op.drop_index("ix_pdf_sources_status", table_name="pdf_sources")
    op.drop_index("ix_pdf_sources_course_id", table_name="pdf_sources")
    op.drop_table("pdf_sources")
