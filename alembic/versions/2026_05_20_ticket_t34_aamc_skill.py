"""SPEC T34 — anki_card_tags gains skill_number + 'aamc_skill' parsed_kind

Revision ID: 20260520_t34
Revises: 20260519_t31
Create Date: 2026-05-20
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260520_t34"
down_revision: Union[str, None] = "20260519_t31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE anki_card_tags ADD COLUMN IF NOT EXISTS skill_number INTEGER NULL")
    # AAMC publishes 4 skills (1-4); a CHECK keeps junk out.
    op.execute(
        """
        ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_skill_number
        """
    )
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_skill_number
            CHECK (skill_number IS NULL OR skill_number BETWEEN 1 AND 4)
        """
    )
    op.execute("ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_parsed_kind")
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_parsed_kind
            CHECK (parsed_kind IN ('aamc_topic', 'aamc_cc', 'aamc_skill', 'uworld_qid', 'unparsed'))
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_parsed_kind")
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_parsed_kind
            CHECK (parsed_kind IN ('aamc_topic', 'aamc_cc', 'uworld_qid', 'unparsed'))
        """
    )
    op.execute(
        "ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_skill_number"
    )
    op.execute("ALTER TABLE anki_card_tags DROP COLUMN IF EXISTS skill_number")
