"""ticket 3.5 canonicalize llm categorizer outputs

Revision ID: 20260515_t35
Revises: 20260514_t33
Create Date: 2026-05-15

No table changes. Data-only migration: wipes the existing source='llm'
QuestionTag rows (which carry non-canonical, free-text identifiers from
3.3/3.4) and re-queues every question for re-categorization under the
new EXTRACTOR_VERSION ("v2-canonical-identifiers") that emits
fully-qualified topic paths from the AAMC outline.

downgrade() does NOT restore the deleted rows — the wipe is irreversible.
The schema is unchanged on rollback.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260515_t35"
down_revision: Union[str, None] = "20260514_t33"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    deleted = bind.execute(sa.text("DELETE FROM question_tags WHERE source = 'llm'")).rowcount
    queued = bind.execute(
        sa.text(
            "UPDATE questions SET needs_categorization = true, "
            "last_updated_at = now() WHERE needs_categorization = false"
        )
    ).rowcount
    print(
        f"ticket_3_5_canonicalize_outputs: deleted {deleted} llm tags, "
        f"queued {queued} questions for re-categorization"
    )


def downgrade() -> None:
    # NOTE: data wipe is not reversed; deleted llm rows are gone.
    pass
