"""SPEC T31 — anki_card_tags gains content_category_id + 'aamc_cc' parsed_kind

Revision ID: 20260519_t31
Revises: 20260519_t2
Create Date: 2026-05-19
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260519_t31"
down_revision: Union[str, None] = "20260519_t2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD COLUMN IF NOT EXISTS content_category_id INTEGER NULL
                REFERENCES content_categories(id) ON DELETE SET NULL
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_card_tags_content_category_id "
        "ON anki_card_tags (content_category_id)"
    )
    # Drop + recreate parsed_kind CHECK to include 'aamc_cc'. Keep 'aamc_topic'
    # for back-compat (T32 LLM categorizer will produce 'aamc_topic_llm' rows).
    op.execute("ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_parsed_kind")
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_parsed_kind
            CHECK (parsed_kind IN ('aamc_topic', 'aamc_cc', 'uworld_qid', 'unparsed'))
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_parsed_kind")
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_parsed_kind
            CHECK (parsed_kind IN ('aamc_topic', 'uworld_qid', 'unparsed'))
        """
    )
    op.execute("DROP INDEX IF EXISTS ix_anki_card_tags_content_category_id")
    op.execute("ALTER TABLE anki_card_tags DROP COLUMN IF EXISTS content_category_id")
