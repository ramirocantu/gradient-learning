"""SPEC T58 — anki_assignments table (V51 lifecycle).

Revision ID: 20260521_t58
Revises: 20260520_t35
Create Date: 2026-05-21

V51 state machine encoded via CHECK on `status`:
  pending -> unlocked -> (completed | skipped | failed)

`card_ids INTEGER[]` snapshots the resolved card set at create-time per
V52 (no re-resolution at unlock). `priority` records the resolver mode
used for that snapshot. `failure_count` powers the V55 retry-with-cap
semantics (3x fail -> terminal status='failed').
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260521_t58"
down_revision: Union[str, None] = "20260520_t35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_assignments (
            id SERIAL PRIMARY KEY,
            study_plan_item_id INTEGER NULL
                REFERENCES study_plan_items(id) ON DELETE SET NULL,
            scope_kind TEXT NOT NULL,
            scope_value TEXT NOT NULL,
            scheduled_unlock_at TIMESTAMPTZ NOT NULL,
            actual_unlock_at TIMESTAMPTZ NULL,
            card_ids BIGINT[] NOT NULL,
            max_cards INTEGER NULL,
            priority TEXT NULL DEFAULT 'most_specific_first',
            status TEXT NOT NULL DEFAULT 'pending',
            error_text TEXT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_anki_assignments_scope_kind
                CHECK (scope_kind IN ('cc','topic')),
            CONSTRAINT ck_anki_assignments_status
                CHECK (status IN ('pending','unlocked','completed','skipped','failed')),
            CONSTRAINT ck_anki_assignments_max_cards_pos
                CHECK (max_cards IS NULL OR max_cards > 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_anki_assignments_status_scheduled
        ON anki_assignments (status, scheduled_unlock_at)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_anki_assignments_study_plan_item
        ON anki_assignments (study_plan_item_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_anki_assignments_actual_unlock
        ON anki_assignments (actual_unlock_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS anki_assignments")
