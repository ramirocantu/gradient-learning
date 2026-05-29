"""T55 — questions.correct_choice → NULLABLE (V-CAP1)

A deferred-answer capture source (e.g. a future web-capture adapter that scrapes
a question before its answer is revealed) may record a question with no known
correct answer; NULL = answer pending. Plain ALTER COLUMN nullability flip — no
named ENUM TYPE, so V-MIG1's leaked-type trap does not apply.

Downgrade re-imposes NOT NULL; it fails if any NULL rows exist (expected — by
then a deferred-answer source has written them).

Revision id kept <=32 chars (alembic_version.version_num is varchar(32) — V-MIG2).

Revision ID: 0006_correct_choice_nullable
Revises: 0005_atomic_fact_extractor
Create Date: 2026-05-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_correct_choice_nullable"
down_revision: Union[str, None] = "0005_atomic_fact_extractor"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "questions",
        "correct_choice",
        existing_type=sa.Text(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "questions",
        "correct_choice",
        existing_type=sa.Text(),
        nullable=False,
    )
