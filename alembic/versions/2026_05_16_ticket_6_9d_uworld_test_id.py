"""Ticket 6.9d — uworld_test_id on raw_captures and attempts

Revision ID: 20260516_t69d
Revises: 20260517_t68
Create Date: 2026-05-16

Adds the `uworld_test_id` column (scraped from UWorld's review-page header)
to both `raw_captures` and `attempts`. Existing rows keep NULL — they
surface as the "Unsessioned" pseudo-row in the dashboard sessions view.

Load-bearing for Phase 8.2 (`get_session_summary(test_id)` MCP tool).
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260516_t69d"
down_revision: Union[str, None] = "20260516_t69c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE raw_captures ADD COLUMN IF NOT EXISTS uworld_test_id TEXT NULL")
    op.execute("ALTER TABLE attempts ADD COLUMN IF NOT EXISTS uworld_test_id TEXT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS ix_attempts_uworld_test_id ON attempts (uworld_test_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_attempts_uworld_test_id")
    op.execute("ALTER TABLE attempts DROP COLUMN IF EXISTS uworld_test_id")
    op.execute("ALTER TABLE raw_captures DROP COLUMN IF EXISTS uworld_test_id")
