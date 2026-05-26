"""SPEC T76 — rename anki_review_pushes → anki_reviews (V53 amended).

Revision ID: 20260523_t76
Revises: 20260521_t61
Create Date: 2026-05-23

V53 amended 2026-05-23: reviews are standalone (no FK to assignments),
filtered-deck name = `<ANKI_DECK_PREFIX>::review::{anki_reviews.id}`,
no UNIQUE(push_date, scope_slug) constraint (tags-as-log accepts dup
reviews per day; idempotency lives in UI debounce per T75 design A).

Migration steps:
  1. Abort if any rows exist in anki_review_pushes — active-scope only,
     no prod data on these tables; presence of rows is a refusal signal
     so manual reconciliation can happen before destructive rename.
  2. ALTER TABLE anki_review_pushes RENAME TO anki_reviews.
  3. ALTER TABLE anki_reviews RENAME COLUMN push_date TO review_date.
  4. DROP CONSTRAINT uq_anki_review_pushes_date_slug (UNIQUE gone).
  5. ALTER TABLE anki_reviews DROP COLUMN scope_slug.
  6. ALTER TABLE anki_writes RENAME COLUMN review_push_id TO review_id.

Index names (`ix_anki_review_pushes_status_date`) and CHECK constraint
names (`ck_anki_review_pushes_status`) are cosmetic post-rename — left
under their old names so PG metadata stays minimally churned. The ORM
will reference them via the same SQLAlchemy `Index` / `CheckConstraint`
definitions under the new model class.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260523_t76"
down_revision: Union[str, None] = "20260521_t61"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "anki_review_pushes" in inspector.get_table_names():
        existing = bind.execute(sa.text("SELECT COUNT(*) FROM anki_review_pushes")).scalar_one()
        if existing and existing > 0:
            raise RuntimeError(
                f"T76 refuses to rename anki_review_pushes — {existing} row(s) exist. "
                "T76 scope assumes active-scope-only (no prod data); reconcile rows "
                "manually before re-running this migration."
            )
        op.execute(
            "ALTER TABLE anki_review_pushes DROP CONSTRAINT IF EXISTS uq_anki_review_pushes_date_slug"
        )
        op.execute("ALTER TABLE anki_review_pushes RENAME COLUMN push_date TO review_date")
        op.execute("ALTER TABLE anki_review_pushes DROP COLUMN IF EXISTS scope_slug")
        op.execute("ALTER TABLE anki_review_pushes RENAME TO anki_reviews")
    if "anki_writes" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("anki_writes")}
        if "review_push_id" in cols and "review_id" not in cols:
            op.execute("ALTER TABLE anki_writes RENAME COLUMN review_push_id TO review_id")


def downgrade() -> None:
    op.execute("ALTER TABLE anki_writes RENAME COLUMN review_id TO review_push_id")
    op.execute("ALTER TABLE anki_reviews RENAME TO anki_review_pushes")
    op.execute(
        "ALTER TABLE anki_review_pushes ADD COLUMN scope_slug TEXT NOT NULL DEFAULT 'ad-hoc'"
    )
    op.execute("ALTER TABLE anki_review_pushes RENAME COLUMN review_date TO push_date")
    op.execute(
        "ALTER TABLE anki_review_pushes ADD CONSTRAINT uq_anki_review_pushes_date_slug "
        "UNIQUE (push_date, scope_slug)"
    )
