"""SPEC T2 — anki_cards + anki_card_tags tables for Phase 11 Anki integration

Revision ID: 20260519_t2
Revises: 20260517_t71fix
Create Date: 2026-05-19
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260519_t2"
down_revision: Union[str, None] = "20260517_t71fix"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_cards (
            id SERIAL PRIMARY KEY,
            anki_card_id BIGINT NOT NULL,
            deck_name TEXT NOT NULL,
            note_id BIGINT NULL,
            model_name TEXT NULL,
            fields_json JSONB NULL,
            due_date DATE NULL,
            interval_days INTEGER NULL,
            ease INTEGER NULL,
            lapses INTEGER NULL,
            queue INTEGER NULL,
            sync_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_anki_cards_deck_card UNIQUE (deck_name, anki_card_id)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_cards_due_date ON anki_cards (due_date)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_card_tags (
            id SERIAL PRIMARY KEY,
            anki_card_id INTEGER NOT NULL
                REFERENCES anki_cards(id) ON DELETE CASCADE,
            tag_raw TEXT NOT NULL,
            topic_id INTEGER NULL
                REFERENCES topics(id) ON DELETE SET NULL,
            question_qid TEXT NULL,
            parsed_kind TEXT NOT NULL,
            CONSTRAINT ck_anki_card_tags_parsed_kind
                CHECK (parsed_kind IN ('aamc_topic', 'uworld_qid', 'unparsed')),
            CONSTRAINT uq_anki_card_tags_card_tag UNIQUE (anki_card_id, tag_raw)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_card_tags_topic_id ON anki_card_tags (topic_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_card_tags_question_qid ON anki_card_tags (question_qid)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS anki_card_tags")
    op.execute("DROP TABLE IF EXISTS anki_cards")
