"""T56 — course-scope captures: questions.course_id + raw_captures.course_id (V-CAP2)

A capture is tagged to a course at ingest. The extension sends ``course_slug``;
the adapter resolves it to ``course_id`` and stamps it on the persisted
``RawCapture`` + ``Question``. The grounded-tag categorizer then scopes a
question's recall to its own course (retiring the "exactly one course" guard for
course-stamped rows).

``ondelete=SET NULL`` (not CASCADE): captures/attempts are user history — a
deleted course nulls the link rather than wiping the history; a NULL course_id
falls back to the single-course tagging rule.

No named ENUM TYPE here, so V-MIG1's leaked-type trap does not apply. Revision
id ``0007_question_course_id`` is 23 chars (≤32 — V-MIG2).

Revision ID: 0007_question_course_id
Revises: 0006_correct_choice_nullable
Create Date: 2026-05-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007_question_course_id"
down_revision: Union[str, None] = "0006_correct_choice_nullable"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("questions", sa.Column("course_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_questions_course_id",
        "questions",
        "courses",
        ["course_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_questions_course_id", "questions", ["course_id"])

    op.add_column("raw_captures", sa.Column("course_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_raw_captures_course_id",
        "raw_captures",
        "courses",
        ["course_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_raw_captures_course_id", "raw_captures", ["course_id"])


def downgrade() -> None:
    op.drop_index("ix_raw_captures_course_id", table_name="raw_captures")
    op.drop_constraint("fk_raw_captures_course_id", "raw_captures", type_="foreignkey")
    op.drop_column("raw_captures", "course_id")

    op.drop_index("ix_questions_course_id", table_name="questions")
    op.drop_constraint("fk_questions_course_id", "questions", type_="foreignkey")
    op.drop_column("questions", "course_id")
