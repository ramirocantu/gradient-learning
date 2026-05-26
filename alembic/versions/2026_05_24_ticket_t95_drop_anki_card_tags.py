"""SPEC T95 — note-as-unit contract: note_ids snapshots + DROP anki_card_tags

Contract step of the note-as-unit cutover (T93 expand -> T94 sync -> T95
read/resolve + contract). By now every reader, the resolver (worker + batch),
sync, assignment + review services read/write `anki_note_tags`; nothing
references `anki_card_tags`, so the table is dropped.

Also adds the `note_ids BIGINT[]` snapshot columns to anki_assignments +
anki_reviews (§V75: notes are the canonical addTags target; card_ids stays the
unsuspend / filtered-deck target). Existing rows backfill empty via DEFAULT
'{}' — the next unlock/review run is keyed off the live snapshot anyway.

Downgrade recreates the `anki_card_tags` structure (constraints + indexes) but
NOT its data — a dropped table's rows are unrecoverable. The note-level data
lives on in anki_note_tags regardless.

Revision ID: 20260524_t95
Revises: 20260524_t93
Create Date: 2026-05-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260524_t95"
down_revision: Union[str, None] = "20260524_t93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE anki_assignments ADD COLUMN IF NOT EXISTS note_ids BIGINT[] "
        "NOT NULL DEFAULT '{}'"
    )
    op.execute(
        "ALTER TABLE anki_reviews ADD COLUMN IF NOT EXISTS note_ids BIGINT[] NOT NULL DEFAULT '{}'"
    )
    # Contract: tags are note-keyed now (anki_note_tags). Nothing reads
    # anki_card_tags after T93->T95.
    op.execute("DROP TABLE IF EXISTS anki_card_tags")


def downgrade() -> None:
    # Recreate the table structure (final T2+T31+T32+T34 shape). Data is not
    # restored — this is a contract-step reversal of structure only.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_card_tags (
            id SERIAL PRIMARY KEY,
            anki_card_id INTEGER NOT NULL
                REFERENCES anki_cards(id) ON DELETE CASCADE,
            tag_raw TEXT NOT NULL,
            topic_id INTEGER NULL
                REFERENCES topics(id) ON DELETE SET NULL,
            content_category_id INTEGER NULL
                REFERENCES content_categories(id) ON DELETE SET NULL,
            skill_number INTEGER NULL,
            question_qid TEXT NULL,
            parsed_kind TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'regex',
            confidence NUMERIC NULL,
            rationale TEXT NULL,
            extractor_version TEXT NULL,
            CONSTRAINT ck_anki_card_tags_parsed_kind
                CHECK (parsed_kind IN
                    ('aamc_topic', 'aamc_cc', 'aamc_skill', 'uworld_qid', 'unparsed')),
            CONSTRAINT ck_anki_card_tags_skill_number
                CHECK (skill_number IS NULL OR skill_number BETWEEN 1 AND 4),
            CONSTRAINT ck_anki_card_tags_source
                CHECK (source IN ('regex', 'llm', 'manual')),
            CONSTRAINT ck_anki_card_tags_confidence_range
                CHECK (confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)),
            CONSTRAINT uq_anki_card_tags_card_tag UNIQUE (anki_card_id, tag_raw)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_card_tags_topic_id ON anki_card_tags (topic_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_card_tags_content_category_id "
        "ON anki_card_tags (content_category_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_card_tags_question_qid ON anki_card_tags (question_qid)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_card_tags_source ON anki_card_tags (source)")
    op.execute("ALTER TABLE anki_reviews DROP COLUMN IF EXISTS note_ids")
    op.execute("ALTER TABLE anki_assignments DROP COLUMN IF EXISTS note_ids")
