"""ticket 3.3 llm categorizer

Revision ID: 20260514_t33
Revises: 20260513_t21
Create Date: 2026-05-14

Adds question_tags.rationale and question_tags.extractor_version columns
to support LLM-driven categorization. Also runs a one-time data migration
that wipes the deterministic uworld_map tags from 3.1/3.2 and re-queues
every question for LLM categorization.

The data wipe in upgrade() is NOT reversible: downgrade() restores the
columns to NULL but cannot bring back the deleted uworld_map rows.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260514_t33"
down_revision: Union[str, None] = "20260513_t21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("question_tags", sa.Column("rationale", sa.Text(), nullable=True))
    op.add_column(
        "question_tags",
        sa.Column("extractor_version", sa.Text(), nullable=True),
    )

    # IRREVERSIBLE DATA MIGRATION:
    # 3.1/3.2 deterministic tags are wiped. The LLM categorizer (3.3) replaces them.
    # Every question is re-queued for re-categorization on the next worker run.
    bind = op.get_bind()
    deleted = bind.execute(
        sa.text("DELETE FROM question_tags WHERE source = 'uworld_map'")
    ).rowcount
    queued = bind.execute(
        sa.text(
            "UPDATE questions SET needs_categorization = true, "
            "last_updated_at = now() WHERE needs_categorization = false"
        )
    ).rowcount
    print(
        f"ticket_3_3_llm_categorizer: deleted {deleted} uworld_map tags, "
        f"queued {queued} questions for re-categorization"
    )


def downgrade() -> None:
    # NOTE: the data wipe is not reversed; the deleted uworld_map rows are gone.
    op.drop_column("question_tags", "extractor_version")
    op.drop_column("question_tags", "rationale")
