"""Ticket 6.8 — lift categorizer depth-0 constraint (Issue #18)

Revision ID: 20260517_t68
Revises: 20260516_bug16
Create Date: 2026-05-17

EXTRACTOR_VERSION bumped to v3-leaf-first: the canonical topic enum now
includes sub-topics up to depth 3, enabling more specific categorization.
All LLM-assigned tags are wiped and every question is re-queued so the
worker can re-categorize with the new, more precise prompt.

Decision: DELETE all source='llm' rows including is_overridden=True. The
soft-deleted rows represent user corrections to the *old* under-specific
categorizations; after re-categorization the whole categorization history
is reset. The audit trail for *future* re-categorizations starts fresh.
If preserving soft-deleted history is preferred, change the WHERE clause
to `WHERE source='llm' AND is_overridden = false`.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260517_t68"
down_revision: Union[str, None] = "20260516_bug16"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DELETE FROM question_tags WHERE source = 'llm'")
    op.execute("UPDATE questions SET needs_categorization = true")


# downgrade: no-op — cannot restore deleted categorizations; re-run the
# categorizer worker to rebuild from scratch if a rollback is needed.
def downgrade() -> None:
    pass
