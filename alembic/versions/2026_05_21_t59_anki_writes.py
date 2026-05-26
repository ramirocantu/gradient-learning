"""SPEC T59 — anki_writes audit log (V50, V55).

Revision ID: 20260521_t59
Revises: 20260521_t60
Create Date: 2026-05-21

V50: every AnkiConnect write attempt (success or fail) appends a row
here. Used by /admin to verify the V50 allowlist held in practice and
by V55 to drive retry-with-cap (failure_count++ on per-row failure).

Sequenced after T58 + T60 so FKs to `anki_assignments` and
`anki_review_pushes` resolve.

Index is (occurred_at DESC) to match the dominant scan pattern
(most-recent-first audit page). The ORM-level Index in `app/models/anki.py`
omits direction so `Base.metadata.create_all` produces a usable index
in test DBs; production-DB DESC ordering lives only in this migration.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260521_t59"
down_revision: Union[str, None] = "20260521_t60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_writes (
            id SERIAL PRIMARY KEY,
            action TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            response_json JSONB NULL,
            status TEXT NOT NULL,
            error_text TEXT NULL,
            source TEXT NOT NULL,
            assignment_id INTEGER NULL
                REFERENCES anki_assignments(id) ON DELETE SET NULL,
            review_push_id INTEGER NULL
                REFERENCES anki_review_pushes(id) ON DELETE SET NULL,
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_anki_writes_status
                CHECK (status IN ('succeeded','failed')),
            CONSTRAINT ck_anki_writes_source
                CHECK (source IN ('mcp','scheduler','manual','test'))
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_anki_writes_occurred_at
        ON anki_writes (occurred_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS anki_writes")
