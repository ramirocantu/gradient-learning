"""SPEC T61 — anki_load_config singleton (V59).

Revision ID: 20260521_t61
Revises: 20260521_t59
Create Date: 2026-05-21

V59: singleton table backing the Anki load realism evaluator (T66).
`id INT PK CHECK (id=1)` enforces one row; the migration seeds the
default row (daily_card_review_budget=200, daily_minutes_budget=60)
so the evaluator's "read-or-create on first access" path always finds
a row in production. Service layer (T66) `set_anki_load_config` updates
the singleton in place.

Disjoint from `study_plan_configs` (V5, deferred T8) — different
domain (Anki review load vs MCAT plan exam-date / hour-budget); both
can coexist post-P12 un-defer per V59.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260521_t61"
down_revision: Union[str, None] = "20260521_t59"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_load_config (
            id INTEGER PRIMARY KEY,
            daily_card_review_budget INTEGER NOT NULL,
            daily_minutes_budget NUMERIC NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_anki_load_config_singleton
                CHECK (id = 1),
            CONSTRAINT ck_anki_load_config_budget_pos
                CHECK (daily_card_review_budget > 0),
            CONSTRAINT ck_anki_load_config_minutes_pos
                CHECK (daily_minutes_budget > 0)
        )
        """
    )
    op.execute(
        """
        INSERT INTO anki_load_config (id, daily_card_review_budget, daily_minutes_budget)
        VALUES (1, 200, 60)
        ON CONFLICT (id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS anki_load_config")
