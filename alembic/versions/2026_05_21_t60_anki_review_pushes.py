"""SPEC T60 — anki_review_pushes table (V53 filtered-deck pushes).

Revision ID: 20260521_t60
Revises: 20260521_t58
Create Date: 2026-05-21

V53: one-off filtered-deck nudges for already-unsuspended cards on a
target date. UNIQUE(push_date, scope_slug) so re-push on the same key
is a delete+recreate of the named filtered deck rather than two
parallel decks. `deck_name` records the AnkiConnect filtered deck name
(`<ANKI_DECK_PREFIX>::review::YYYY-MM-DD::<scope_slug>`) so
service code can target the exact deck for delete-on-re-push without
re-deriving the name.

Sequenced before T59 (`anki_writes`) so its FK target exists.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260521_t60"
down_revision: Union[str, None] = "20260521_t58"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_review_pushes (
            id SERIAL PRIMARY KEY,
            push_date DATE NOT NULL,
            scope_slug TEXT NOT NULL DEFAULT 'ad-hoc',
            card_ids BIGINT[] NOT NULL,
            deck_name TEXT NOT NULL,
            study_plan_item_id INTEGER NULL
                REFERENCES study_plan_items(id) ON DELETE SET NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_text TEXT NULL,
            failure_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            pushed_at TIMESTAMPTZ NULL,
            CONSTRAINT ck_anki_review_pushes_status
                CHECK (status IN ('pending','pushed','failed')),
            CONSTRAINT uq_anki_review_pushes_date_slug
                UNIQUE (push_date, scope_slug)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_anki_review_pushes_status_date
        ON anki_review_pushes (status, push_date)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS anki_review_pushes")
