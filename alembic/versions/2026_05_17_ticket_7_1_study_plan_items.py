"""Ticket 7.1 — study_plan_items table for dashboard-native study plan

Revision ID: 20260517_t71
Revises: 20260517_iss28
Create Date: 2026-05-17
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260517_t71"
down_revision: Union[str, None] = "20260517_iss28"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS study_plan_items (
            id SERIAL PRIMARY KEY,
            topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
            section_code TEXT,
            recommendation_score DOUBLE PRECISION NOT NULL DEFAULT 0,
            reason TEXT NOT NULL,
            note_text TEXT,
            scheduled_for DATE NOT NULL,
            completed_at TIMESTAMPTZ,
            auto_completed_from_attempt_id INTEGER REFERENCES attempts(id) ON DELETE SET NULL,
            source TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_study_plan_items_target CHECK (
                (topic_id IS NOT NULL) OR (section_code IS NOT NULL)
            ),
            CONSTRAINT ck_study_plan_items_source CHECK (source IN ('mcp', 'manual'))
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_study_plan_items_scheduled_for "
        "ON study_plan_items (scheduled_for)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_study_plan_items_open "
        "ON study_plan_items (scheduled_for) WHERE completed_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS study_plan_items")
