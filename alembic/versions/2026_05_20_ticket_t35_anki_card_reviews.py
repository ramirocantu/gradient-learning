"""SPEC T35 — anki_card_reviews append-only review log for retention math (V26, V27)

Revision ID: 20260520_t35
Revises: 20260520_t51
Create Date: 2026-05-20

V26: append-only; sync uses `startID = MAX(review_id) + 1` incremental;
first-run backfill `startID=0`. `review_id` is Anki's revlog id (unix-ms),
globally unique per Anki, used as PK so re-sync is idempotent.

V27: stores `ease` + `type` to enable T37 retention.py to compute windowed
"true retention" (pass = ease ∈ {2,3,4}; exclude type='learn').
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260520_t35"
down_revision: Union[str, None] = "20260520_t51"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_card_reviews (
            review_id BIGINT PRIMARY KEY,
            card_id INTEGER NOT NULL
                REFERENCES anki_cards(id) ON DELETE CASCADE,
            reviewed_at TIMESTAMPTZ NOT NULL,
            ease INTEGER NOT NULL,
            type TEXT NOT NULL,
            interval_before INTEGER NULL,
            interval_after INTEGER NULL,
            time_ms INTEGER NULL,
            CONSTRAINT ck_anki_card_reviews_ease
                CHECK (ease BETWEEN 1 AND 4),
            CONSTRAINT ck_anki_card_reviews_type
                CHECK (type IN ('learn', 'review', 'relearn', 'cram'))
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_anki_card_reviews_card_reviewed
        ON anki_card_reviews (card_id, reviewed_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS anki_card_reviews")
