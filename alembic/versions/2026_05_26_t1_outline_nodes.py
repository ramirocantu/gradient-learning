"""SPEC T1 — collapse Section/FC/CC/Topic → courses + outline_nodes (V-O1)

P0 schema generalize. The PoC's four dedicated outline tables
(sections → foundational_concepts → content_categories → topics) are
replaced by a domain-blind `courses` + recursive `outline_nodes` tree;
AAMC's four levels become `kind` labels on a four-deep instance (V-O1),
re-seeded as an uploaded schema in T9 (V-O3).

Fresh-start re-architecture (SPEC §O): old MCAT outline data is not
migrated. The DROPs use CASCADE so inbound FK constraints from the tag
tables (question_tags / anki_note_tags → topics / content_categories) are
removed here; those tag tables are restructured onto `node_id` in T2.

Downgrade recreates the four tables' structure only (cf §B / T95 precedent)
— data and inbound FK constraints are not restored.

Revision ID: 20260526_t1
Revises: 20260524_t95
Create Date: 2026-05-26
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260526_t1"
down_revision: Union[str, None] = "20260524_t95"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # CASCADE removes inbound FK constraints from tag tables; T2 restructures
    # those onto node_id. Children-first order is moot under CASCADE.
    op.execute("DROP TABLE IF EXISTS topics CASCADE")
    op.execute("DROP TABLE IF EXISTS content_categories CASCADE")
    op.execute("DROP TABLE IF EXISTS foundational_concepts CASCADE")
    op.execute("DROP TABLE IF EXISTS sections CASCADE")

    op.execute(
        """
        CREATE TABLE courses (
            id SERIAL PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE outline_nodes (
            id SERIAL PRIMARY KEY,
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            parent_id INTEGER NULL REFERENCES outline_nodes(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            depth INTEGER NOT NULL,
            position INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_outline_nodes_course_parent_name
                UNIQUE (course_id, parent_id, name)
        )
        """
    )
    op.execute("CREATE INDEX ix_outline_nodes_course_id ON outline_nodes (course_id)")
    op.execute("CREATE INDEX ix_outline_nodes_parent_id ON outline_nodes (parent_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS outline_nodes")
    op.execute("DROP TABLE IF EXISTS courses")

    # Structure-only reversal. Data + inbound FK constraints not restored.
    op.execute(
        """
        CREATE TABLE sections (
            id SERIAL PRIMARY KEY,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            position INTEGER NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE foundational_concepts (
            id SERIAL PRIMARY KEY,
            section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            position INTEGER NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE content_categories (
            id SERIAL PRIMARY KEY,
            foundational_concept_id INTEGER NOT NULL
                REFERENCES foundational_concepts(id) ON DELETE CASCADE,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT NULL,
            position INTEGER NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE topics (
            id SERIAL PRIMARY KEY,
            content_category_id INTEGER NOT NULL
                REFERENCES content_categories(id) ON DELETE RESTRICT,
            parent_topic_id INTEGER NULL REFERENCES topics(id) ON DELETE SET NULL,
            name TEXT NOT NULL,
            disciplines TEXT[] NOT NULL DEFAULT '{}',
            depth INTEGER NOT NULL,
            position INTEGER NOT NULL,
            CONSTRAINT uq_topic_cc_parent_name
                UNIQUE (content_category_id, parent_topic_id, name)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_topic_content_category_id ON topics (content_category_id)"
    )
    op.execute("CREATE INDEX ix_topic_parent_topic_id ON topics (parent_topic_id)")
