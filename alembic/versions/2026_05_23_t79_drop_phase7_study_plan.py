"""T79 — drop Phase 7 study plan layer per V61.

Removes `study_plan_items` table + the `study_plan_item_id` FK columns on
`anki_assignments` and `anki_review_pushes` (T58/T60). No
`study_plan_configs` table to drop — V5 was declared but never migrated
(T8 stayed deferred and is now `X` cut).

Downgrade re-creates the dropped artifacts in their pre-T79 shape so
the alembic round-trip test passes and dev rollback chains stay
runnable, BUT V61 still forbids re-introducing Phase 7 features into
code. Treat downgrade as a schema-only escape hatch, not an invitation
to bring back the layer.

Revision ID: 20260523_t79
Revises: 20260523_t76
Create Date: 2026-05-23
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20260523_t79"
down_revision: Union[str, None] = "20260523_t76"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_anki_assignments_study_plan_item")
    op.execute("ALTER TABLE anki_assignments DROP COLUMN IF EXISTS study_plan_item_id")
    # Table renamed by T76 (anki_review_pushes -> anki_reviews); drop the
    # FK column from the new name. Use IF EXISTS so a hypothetical fresh
    # DB that never had the column still upgrades cleanly.
    op.execute("ALTER TABLE anki_reviews DROP COLUMN IF EXISTS study_plan_item_id")
    op.execute("DROP TABLE IF EXISTS study_plan_items")


def downgrade() -> None:
    # Re-create the dropped table in its pre-T79 shape (matches the 7.1
    # migration). FKs from anki_assignments / anki_review_pushes come
    # back as nullable columns w/ SET NULL behavior — same as T58/T60.
    # V61 still forbids reintroducing Phase 7 features in code; this
    # is a schema-only round-trip path for tests + emergency rollback.
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
    op.execute(
        "ALTER TABLE anki_assignments ADD COLUMN IF NOT EXISTS study_plan_item_id INTEGER "
        "REFERENCES study_plan_items(id) ON DELETE SET NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_assignments_study_plan_item "
        "ON anki_assignments (study_plan_item_id)"
    )
    op.execute(
        "ALTER TABLE anki_reviews ADD COLUMN IF NOT EXISTS study_plan_item_id INTEGER "
        "REFERENCES study_plan_items(id) ON DELETE SET NULL"
    )
