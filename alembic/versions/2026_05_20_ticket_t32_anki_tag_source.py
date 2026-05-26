"""SPEC T32 — anki_card_tags gains source/confidence/rationale/extractor_version

Mirrors QuestionTag pattern so 'how was this row derived' is an orthogonal
axis from 'what does it point at' (parsed_kind). Existing rows backfill
source='regex' (they were all written by the T3 regex sync).

Revision ID: 20260520_t32
Revises: 20260520_t34
Create Date: 2026-05-20
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260520_t32"
down_revision: Union[str, None] = "20260520_t34"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns nullable first so we can backfill, then tighten `source` to NOT NULL.
    op.execute("ALTER TABLE anki_card_tags ADD COLUMN IF NOT EXISTS source TEXT NULL")
    op.execute("ALTER TABLE anki_card_tags ADD COLUMN IF NOT EXISTS confidence NUMERIC NULL")
    op.execute("ALTER TABLE anki_card_tags ADD COLUMN IF NOT EXISTS rationale TEXT NULL")
    op.execute("ALTER TABLE anki_card_tags ADD COLUMN IF NOT EXISTS extractor_version TEXT NULL")
    # Backfill: every existing row was written by the T3 regex sync.
    op.execute("UPDATE anki_card_tags SET source = 'regex' WHERE source IS NULL")
    op.execute("ALTER TABLE anki_card_tags ALTER COLUMN source SET NOT NULL")
    op.execute("ALTER TABLE anki_card_tags ALTER COLUMN source SET DEFAULT 'regex'")
    op.execute(
        """
        ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_source
        """
    )
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_source
            CHECK (source IN ('regex', 'llm', 'manual'))
        """
    )
    op.execute(
        "ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_confidence_range"
    )
    op.execute(
        """
        ALTER TABLE anki_card_tags
            ADD CONSTRAINT ck_anki_card_tags_confidence_range
            CHECK (confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0))
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_card_tags_source ON anki_card_tags (source)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_anki_card_tags_source")
    op.execute(
        "ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_confidence_range"
    )
    op.execute("ALTER TABLE anki_card_tags DROP CONSTRAINT IF EXISTS ck_anki_card_tags_source")
    op.execute("ALTER TABLE anki_card_tags DROP COLUMN IF EXISTS extractor_version")
    op.execute("ALTER TABLE anki_card_tags DROP COLUMN IF EXISTS rationale")
    op.execute("ALTER TABLE anki_card_tags DROP COLUMN IF EXISTS confidence")
    op.execute("ALTER TABLE anki_card_tags DROP COLUMN IF EXISTS source")
