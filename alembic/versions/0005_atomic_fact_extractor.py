"""T54 — atomic_facts.extractor_version (V-KB3, notes-ingress redesign)

Stamps which extraction prompt/schema version produced each grounded fact,
parallel to `<target>_tags.extractor_version` (V-T2). Plain nullable TEXT —
no named ENUM TYPE, so V-MIG1's leaked-type downgrade trap does not apply;
`drop_column` is a clean teardown.

Revision id kept ≤32 chars (alembic_version.version_num is varchar(32)).

Revision ID: 0005_atomic_fact_extractor
Revises: 0004_atomic_fact_tags
Create Date: 2026-05-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_atomic_fact_extractor"
down_revision: Union[str, None] = "0004_atomic_fact_tags"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "atomic_facts",
        sa.Column("extractor_version", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("atomic_facts", "extractor_version")
