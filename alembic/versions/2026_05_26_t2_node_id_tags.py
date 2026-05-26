"""SPEC T2 — retarget tags to node_id; canonical <target>_tags shape (V-T1/2/3)

Collapses the PoC's 3-target tag shape (topic_id/content_category_id/skill[
_number]) on `question_tags` + `anki_note_tags` down to a single
`node_id` → outline_nodes (V-T1). `source` enum becomes {schema_map, llm,
manual} (V-T2; the old `uworld_map`/`regex` deterministic maps fold into
`schema_map`). `confidence` is NULL-able and DB-enforced to be present iff
`source='llm'`; `<0.5` requires `manual_review` (V-T3).

anki_note_tags keeps its plugin provenance columns (tag_raw, parsed_kind,
question_qid) and the raw-tag UQ for addTags idempotency; node_id is NULL-able
(unparsed / bare-qid tags resolve to no node). The MCAT-specific parsed_kind
CHECK is dropped — the AnKing tag-shape parser is a plugin (§A).

Fresh-start (SPEC §O): no data migrated. After T1, these tables still carried
the old topic/cc columns but with their inbound FKs already CASCADE-dropped;
this migration drops + recreates them clean. Downgrade restores the 3-target
structure WITHOUT the topic/cc FKs (topics/content_categories don't exist in
the T1-upgraded schema) — structure-only reversal (cf T95).

Revision ID: 20260526_t2
Revises: 20260526_t1
Create Date: 2026-05-26
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260526_t2"
down_revision: Union[str, None] = "20260526_t1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS question_tags")
    op.execute(
        """
        CREATE TABLE question_tags (
            id SERIAL PRIMARY KEY,
            question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
            node_id INTEGER NOT NULL REFERENCES outline_nodes(id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            confidence NUMERIC(3, 2) NULL,
            rationale TEXT NULL,
            extractor_version TEXT NULL,
            manual_review BOOLEAN NOT NULL DEFAULT false,
            is_overridden BOOLEAN NOT NULL DEFAULT false,
            overridden_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_question_tags_confidence_when_llm
                CHECK ((source = 'llm' AND confidence IS NOT NULL)
                    OR (source <> 'llm' AND confidence IS NULL)),
            CONSTRAINT ck_question_tags_confidence_range
                CHECK (confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)),
            CONSTRAINT ck_question_tags_low_conf_flagged
                CHECK (confidence IS NULL OR confidence >= 0.5 OR manual_review),
            CONSTRAINT ck_question_tags_source
                CHECK (source IN ('schema_map', 'llm', 'manual')),
            CONSTRAINT uq_question_tags_node_source UNIQUE (question_id, node_id, source)
        )
        """
    )
    op.execute("CREATE INDEX ix_question_tags_question_id ON question_tags (question_id)")
    op.execute("CREATE INDEX ix_question_tags_node_id ON question_tags (node_id)")

    op.execute("DROP TABLE IF EXISTS anki_note_tags")
    op.execute(
        """
        CREATE TABLE anki_note_tags (
            id SERIAL PRIMARY KEY,
            note_id BIGINT NOT NULL REFERENCES anki_notes(note_id) ON DELETE CASCADE,
            tag_raw TEXT NOT NULL,
            node_id INTEGER NULL REFERENCES outline_nodes(id) ON DELETE SET NULL,
            question_qid TEXT NULL,
            parsed_kind TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'schema_map',
            confidence NUMERIC(3, 2) NULL,
            rationale TEXT NULL,
            extractor_version TEXT NULL,
            manual_review BOOLEAN NOT NULL DEFAULT false,
            is_overridden BOOLEAN NOT NULL DEFAULT false,
            overridden_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_anki_note_tags_confidence_when_llm
                CHECK ((source = 'llm' AND confidence IS NOT NULL)
                    OR (source <> 'llm' AND confidence IS NULL)),
            CONSTRAINT ck_anki_note_tags_confidence_range
                CHECK (confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)),
            CONSTRAINT ck_anki_note_tags_low_conf_flagged
                CHECK (confidence IS NULL OR confidence >= 0.5 OR manual_review),
            CONSTRAINT ck_anki_note_tags_source
                CHECK (source IN ('schema_map', 'llm', 'manual')),
            CONSTRAINT uq_anki_note_tags_node_source UNIQUE (note_id, node_id, source),
            CONSTRAINT uq_anki_note_tags_note_tag UNIQUE (note_id, tag_raw)
        )
        """
    )
    op.execute("CREATE INDEX ix_anki_note_tags_node_id ON anki_note_tags (node_id)")
    op.execute("CREATE INDEX ix_anki_note_tags_question_qid ON anki_note_tags (question_qid)")
    op.execute("CREATE INDEX ix_anki_note_tags_source ON anki_note_tags (source)")


def downgrade() -> None:
    # Structure-only reversal to the post-T1 3-target shape. topic_id /
    # content_category_id are plain INTs here (no FK) — topics /
    # content_categories don't exist in the T1-upgraded schema.
    op.execute("DROP TABLE IF EXISTS question_tags")
    op.execute(
        """
        CREATE TABLE question_tags (
            id SERIAL PRIMARY KEY,
            question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
            topic_id INTEGER NULL,
            content_category_id INTEGER NULL,
            skill INTEGER NULL,
            confidence NUMERIC(3, 2) NOT NULL,
            source TEXT NOT NULL,
            rationale TEXT NULL,
            extractor_version TEXT NULL,
            is_overridden BOOLEAN NOT NULL DEFAULT false,
            overridden_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_question_tags_exactly_one_target
                CHECK (((topic_id IS NOT NULL)::int + (content_category_id IS NOT NULL)::int
                    + (skill IS NOT NULL)::int) = 1),
            CONSTRAINT ck_question_tags_skill_range
                CHECK (skill IS NULL OR (skill BETWEEN 1 AND 4)),
            CONSTRAINT ck_question_tags_confidence_range
                CHECK (confidence BETWEEN 0.0 AND 1.0),
            CONSTRAINT ck_question_tags_source
                CHECK (source IN ('uworld_map', 'llm', 'manual'))
        )
        """
    )

    op.execute("DROP TABLE IF EXISTS anki_note_tags")
    op.execute(
        """
        CREATE TABLE anki_note_tags (
            id SERIAL PRIMARY KEY,
            note_id BIGINT NOT NULL REFERENCES anki_notes(note_id) ON DELETE CASCADE,
            tag_raw TEXT NOT NULL,
            topic_id INTEGER NULL,
            content_category_id INTEGER NULL,
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
    op.execute("CREATE INDEX ix_anki_note_tags_topic_id ON anki_note_tags (topic_id)")
    op.execute(
        "CREATE INDEX ix_anki_note_tags_content_category_id "
        "ON anki_note_tags (content_category_id)"
    )
    op.execute("CREATE INDEX ix_anki_note_tags_question_qid ON anki_note_tags (question_qid)")
    op.execute("CREATE INDEX ix_anki_note_tags_source ON anki_note_tags (source)")
