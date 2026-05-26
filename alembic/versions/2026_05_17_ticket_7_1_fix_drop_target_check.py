"""Ticket 7.1 fix — drop ck_study_plan_items_target CHECK constraint

Rationale: the original 7.1 schema combined ON DELETE SET NULL on
study_plan_items.topic_id with a CHECK constraint that required at least
one of (topic_id, section_code) to be non-null. Those two are
incompatible: deleting a topic referenced by a topic-only plan item
triggers the SET NULL, which then violates the CHECK, so the topic
delete fails outright. That contradicts 7.1's stated intent ("keep the
plan-item row with a null topic if the outline drops the topic").

Fix: drop the CHECK. The Pydantic model_validator
`StudyPlanItemCreate.exactly_one_target` still enforces "exactly one of
topic_id or section_code" at insert time. The DB only relaxes the
constraint for post-insert referential cleanup (topic deletion). A
plan-item with both nulls after a topic delete still has its original
`reason` text as provenance and the dashboard's `_label` fallback
renders "(no target)".

Revision ID: 20260517_t71fix
Revises: 20260517_t71
Create Date: 2026-05-17
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260517_t71fix"
down_revision: Union[str, None] = "20260517_t71"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE study_plan_items DROP CONSTRAINT IF EXISTS ck_study_plan_items_target")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE study_plan_items "
        "ADD CONSTRAINT ck_study_plan_items_target CHECK ("
        "(topic_id IS NOT NULL) OR (section_code IS NOT NULL)"
        ")"
    )
