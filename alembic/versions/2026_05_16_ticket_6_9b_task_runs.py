"""Ticket 6.9b — task_runs table for scheduler history

Revision ID: 20260516_t69b
Revises: 20260517_t68
Create Date: 2026-05-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260516_t69b"
down_revision: Union[str, None] = "20260517_t68"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "DO $$ BEGIN "
        "CREATE TYPE task_run_status AS ENUM ('running', 'succeeded', 'failed'); "
        "EXCEPTION WHEN duplicate_object THEN NULL; "
        "END $$"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS task_runs (
            id SERIAL PRIMARY KEY,
            job_name VARCHAR(64) NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ,
            status task_run_status NOT NULL,
            items_processed INTEGER NOT NULL DEFAULT 0,
            cost_usd NUMERIC(10, 4),
            error_text TEXT
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_task_runs_job_name ON task_runs (job_name)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_task_runs_job_started ON task_runs (job_name, started_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_task_runs_job_started")
    op.execute("DROP INDEX IF EXISTS ix_task_runs_job_name")
    op.execute("DROP TABLE IF EXISTS task_runs")
    op.execute("DROP TYPE IF EXISTS task_run_status")
