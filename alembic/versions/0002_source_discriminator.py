"""T3 — open source discriminator on questions/attempts; open raw_captures

Adds an open `source` TEXT discriminator (default 'uworld') to questions +
attempts and drops the hard `CHECK source IN ('uworld')` on raw_captures, so
any registered source adapter (§A) may write captures. Identity renames
(qid→external_id, uworld_test_id→session_ref) are deferred to the T12–T14
reader ports per §I.

Revision ID: 0002_source_discriminator
Revises: 0001_initial
Create Date: 2026-05-26
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0002_source_discriminator"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE questions ADD COLUMN source TEXT NOT NULL DEFAULT 'uworld'")
    op.execute("ALTER TABLE attempts ADD COLUMN source TEXT NOT NULL DEFAULT 'uworld'")
    op.execute("CREATE INDEX ix_questions_source ON questions (source)")
    op.execute("CREATE INDEX ix_attempts_source ON attempts (source)")
    # Open the discriminator: drop the single-value closed enum.
    op.execute("ALTER TABLE raw_captures DROP CONSTRAINT ck_raw_captures_source")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE raw_captures ADD CONSTRAINT ck_raw_captures_source "
        "CHECK (source IN ('uworld'))"
    )
    op.execute("DROP INDEX ix_attempts_source")
    op.execute("DROP INDEX ix_questions_source")
    op.execute("ALTER TABLE attempts DROP COLUMN source")
    op.execute("ALTER TABLE questions DROP COLUMN source")
