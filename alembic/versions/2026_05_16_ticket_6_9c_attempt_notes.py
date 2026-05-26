"""Ticket 6.9c — attempt_notes table for per-attempt notes and flag-for-review

Revision ID: 20260516_t69c
Revises: 20260516_t69b
Create Date: 2026-05-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260516_t69c"
down_revision: Union[str, None] = "20260516_t69b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS attempt_notes (
            id SERIAL PRIMARY KEY,
            attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
            note_text TEXT NOT NULL,
            flag_for_review BOOLEAN NOT NULL DEFAULT false,
            source TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_attempt_notes_source CHECK (source IN ('user', 'mcp'))
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_attempt_notes_attempt_id ON attempt_notes (attempt_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_attempt_notes_flagged "
        "ON attempt_notes (flag_for_review) WHERE flag_for_review = true"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS attempt_notes")
