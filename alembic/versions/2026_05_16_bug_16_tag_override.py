"""bug #16 — soft delete for question tags

Revision ID: 20260516_bug16
Revises: 20260515_t41
Create Date: 2026-05-16

Adds is_overridden and overridden_at to question_tags so LLM-assigned tags
can be marked as removed without losing the audit trail.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260516_bug16"
down_revision: Union[str, None] = "20260515_t41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_tags",
        sa.Column("is_overridden", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "question_tags",
        sa.Column("overridden_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("question_tags", "overridden_at")
    op.drop_column("question_tags", "is_overridden")
