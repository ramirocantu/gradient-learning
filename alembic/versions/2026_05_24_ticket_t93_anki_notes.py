"""SPEC T93 — note-as-unit: anki_notes + anki_note_tags + backfill (§V75)

Expand step of the note-as-unit cutover (T93 -> T94 -> T95). Creates the
note-level tables, collapses the per-card tag fan-out into one tag set per
note (regex+llm+manual all preserved per §V43; §V24 orthogonal cols carried
across), and adds the `anki_cards.note_id -> anki_notes.note_id` FK.

`anki_card_tags` is intentionally NOT dropped here. The readers + sync still
reference it until T94 (sync) + T95 (reads/resolve) cut over; the contract
step (DROP anki_card_tags) lands in the T95 migration once nothing reads it.
Dropping it here would red the suite mid-cutover (cf §V75 "sequence
T93->T94->T95").

Tests use Base.metadata.create_all (not this migration), so the backfill SQL
below is behaviourally re-exercised in tests/test_anki_note_schema.py against
a seeded create_all schema.

Revision ID: 20260524_t93
Revises: 20260523_t79
Create Date: 2026-05-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260524_t93"
down_revision: Union[str, None] = "20260523_t79"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. anki_notes — note_id is Anki's native BIGINT note id used directly as
    #    the PK (no local SERIAL bridge; §B11 overflow lesson only bites the
    #    card-id INTEGER FK pattern, not a BIGINT PK).
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_notes (
            note_id BIGINT PRIMARY KEY,
            deck_name TEXT NULL,
            model_name TEXT NULL,
            fields_json JSONB NULL
        )
        """
    )

    # 2. anki_note_tags — note-level mirror of anki_card_tags (§V24 cols).
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS anki_note_tags (
            id SERIAL PRIMARY KEY,
            note_id BIGINT NOT NULL
                REFERENCES anki_notes(note_id) ON DELETE CASCADE,
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
            CONSTRAINT ck_anki_note_tags_parsed_kind
                CHECK (parsed_kind IN
                    ('aamc_topic', 'aamc_cc', 'aamc_skill', 'uworld_qid', 'unparsed')),
            CONSTRAINT ck_anki_note_tags_skill_number
                CHECK (skill_number IS NULL OR skill_number BETWEEN 1 AND 4),
            CONSTRAINT ck_anki_note_tags_source
                CHECK (source IN ('regex', 'llm', 'manual')),
            CONSTRAINT ck_anki_note_tags_confidence_range
                CHECK (confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)),
            CONSTRAINT uq_anki_note_tags_note_tag UNIQUE (note_id, tag_raw)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_note_tags_topic_id ON anki_note_tags (topic_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_note_tags_content_category_id "
        "ON anki_note_tags (content_category_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_anki_note_tags_question_qid ON anki_note_tags (question_qid)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_anki_note_tags_source ON anki_note_tags (source)")

    # 3. Backfill anki_notes from the cards' note-level fields. model_name +
    #    fields_json are note-level in Anki (consistent across a note's cards);
    #    deck_name can vary per card so DISTINCT ON keeps a representative.
    #    Cards with NULL note_id (none after a sync) cannot map to a note and
    #    are skipped — the FK in step 5 leaves them NULL.
    op.execute(
        """
        INSERT INTO anki_notes (note_id, deck_name, model_name, fields_json)
        SELECT DISTINCT ON (note_id)
            note_id, deck_name, model_name, fields_json
        FROM anki_cards
        WHERE note_id IS NOT NULL
        ORDER BY note_id, id
        ON CONFLICT (note_id) DO NOTHING
        """
    )

    # 4. Backfill anki_note_tags: collapse the per-card fan-out. Each note's N
    #    cards carry identical regex tag copies (Anki tags are note-level) ->
    #    one row per (note_id, tag_raw). regex+llm+manual all survive because
    #    their tag_raw namespaces differ (§V43). When the SAME tag_raw repeats
    #    with differing confidence (e.g. an llm topic resolved per-card across
    #    cloze siblings), the highest-confidence row wins the collapse
    #    (confidence DESC NULLS LAST), tie-broken by lowest id for determinism.
    op.execute(
        """
        INSERT INTO anki_note_tags (
            note_id, tag_raw, topic_id, content_category_id, skill_number,
            question_qid, parsed_kind, source, confidence, rationale,
            extractor_version
        )
        SELECT DISTINCT ON (c.note_id, act.tag_raw)
            c.note_id, act.tag_raw, act.topic_id, act.content_category_id,
            act.skill_number, act.question_qid, act.parsed_kind, act.source,
            act.confidence, act.rationale, act.extractor_version
        FROM anki_card_tags act
        JOIN anki_cards c ON c.id = act.anki_card_id
        WHERE c.note_id IS NOT NULL
        ORDER BY c.note_id, act.tag_raw, act.confidence DESC NULLS LAST, act.id
        ON CONFLICT (note_id, tag_raw) DO NOTHING
        """
    )

    # 5. anki_cards.note_id -> anki_notes.note_id FK. Added after the note
    #    backfill so every non-null note_id already has a parent row.
    op.execute("ALTER TABLE anki_cards DROP CONSTRAINT IF EXISTS fk_anki_cards_note_id")
    op.execute(
        """
        ALTER TABLE anki_cards
            ADD CONSTRAINT fk_anki_cards_note_id
            FOREIGN KEY (note_id) REFERENCES anki_notes(note_id)
            ON DELETE SET NULL
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE anki_cards DROP CONSTRAINT IF EXISTS fk_anki_cards_note_id")
    op.execute("DROP TABLE IF EXISTS anki_note_tags")
    op.execute("DROP TABLE IF EXISTS anki_notes")
