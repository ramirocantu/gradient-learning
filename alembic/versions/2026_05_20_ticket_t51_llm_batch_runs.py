"""SPEC T51 — llm_batch_runs table for Anthropic Message Batches API tracking.

Revision ID: 20260520_t51
Revises: 20260520_t32
Create Date: 2026-05-20
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260520_t51"
down_revision: Union[str, None] = "20260520_t32"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_batch_runs (
            id SERIAL PRIMARY KEY,
            anthropic_batch_id TEXT NOT NULL UNIQUE,
            extractor TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            model TEXT NOT NULL,
            submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processing_status TEXT NOT NULL,
            total_requests INTEGER NOT NULL DEFAULT 0,
            succeeded_count INTEGER NOT NULL DEFAULT 0,
            errored_count INTEGER NOT NULL DEFAULT 0,
            canceled_count INTEGER NOT NULL DEFAULT 0,
            expired_count INTEGER NOT NULL DEFAULT 0,
            processing_count INTEGER NOT NULL DEFAULT 0,
            ended_at TIMESTAMPTZ NULL,
            total_cost_usd NUMERIC(10, 4) NULL,
            result_persisted_at TIMESTAMPTZ NULL,
            notes TEXT NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_llm_batch_runs_extractor_submitted
            ON llm_batch_runs (extractor, submitted_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_llm_batch_runs_extractor_submitted")
    op.execute("DROP TABLE IF EXISTS llm_batch_runs")
