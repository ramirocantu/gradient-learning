from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from decimal import Decimal

from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AnkiNote(Base):
    """One row per Anki note (§V75 note-as-unit).

    A note holds the user-authored content (`fields_json`) + its model + an
    indicative deck, and is the tag/topic-resolution target via `AnkiNoteTag`.
    Anki tags + content are note-level by design (`addTags`/`notesInfo`/
    `findNotes` are note-scoped); per-card SRS state lives on `AnkiCard`,
    which FKs back here via `note_id`.

    `note_id` is Anki's native note id (BIGINT, 13-digit timestamp-based)
    used directly as the PK — the same id passed to AnkiConnect `notesInfo` /
    `addTags(notes=...)`, so no local-vs-native bridge is needed (cf §B11 for
    the card-id INTEGER-overflow lesson).
    """

    __tablename__ = "anki_notes"

    note_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    deck_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fields_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    cards: Mapped[list[AnkiCard]] = relationship(back_populates="note")
    tags: Mapped[list[AnkiNoteTag]] = relationship(
        back_populates="note", cascade="all, delete-orphan"
    )


class AnkiCard(Base):
    """One row per Anki card seen during sync.

    Identity: (deck_name, anki_card_id). UNIQUE constraint enables
    idempotent upsert per SPEC §V1.

    Review-state columns (due_date, interval_days, ease, lapses, queue,
    sync_at) capture Anki's scheduler state at sync time per §V2. The
    T3 sync service flags rows as stale when `sync_at` is older than
    `ANKI_SYNC_INTERVAL_MINUTES * 2`.
    """

    __tablename__ = "anki_cards"
    __table_args__ = (
        UniqueConstraint("deck_name", "anki_card_id", name="uq_anki_cards_deck_card"),
        Index("ix_anki_cards_due_date", "due_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    anki_card_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    deck_name: Mapped[str] = mapped_column(Text, nullable=False)
    note_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        # Name the constraint so Base.metadata.create_all (tests) matches the
        # T93 migration's `fk_anki_cards_note_id` rather than emitting the
        # Postgres-default `anki_cards_note_id_fkey`.
        ForeignKey("anki_notes.note_id", ondelete="SET NULL", name="fk_anki_cards_note_id"),
        nullable=True,
    )
    model_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fields_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    interval_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ease: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    lapses: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    queue: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    sync_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # §V75: a card's tags live on its note. Viewonly relationship via the
    # shared note_id so existing readers (`selectinload(AnkiCard.tags)`,
    # `AnkiCardOut.tags`) keep working against anki_note_tags once
    # anki_card_tags is dropped. Read-only view: no back_populates / cascade.
    tags: Mapped[list[AnkiNoteTag]] = relationship(
        "AnkiNoteTag",
        primaryjoin="AnkiCard.note_id == foreign(AnkiNoteTag.note_id)",
        viewonly=True,
    )
    reviews: Mapped[list[AnkiCardReview]] = relationship(
        back_populates="card", cascade="all, delete-orphan"
    )
    note: Mapped[Optional[AnkiNote]] = relationship(back_populates="cards")


class AnkiNoteTag(Base):
    """Canonical tag (V-T1) attached to an AnkiNote (§V75) — sole target `node_id`.

    The PoC's 3-target (topic_id/content_category_id/skill_number) is retired.
    `node_id` is NULL-able: an unparsed tag or a bare qid reference resolves to
    no outline node. Anki-plugin provenance (`tag_raw`, `parsed_kind`,
    `question_qid`) is retained alongside the canonical core; `parsed_kind` is
    now plugin-defined free text (the MCAT-specific CHECK is gone — the AnKing
    tag-shape parser is a plugin per §A).

    `source` records HOW the tag was derived (V-T2): the deterministic
    tag-shape parser writes `schema_map` (was `regex`); the LLM topic resolver
    writes `llm`; `manual` for human edits. `confidence` required iff
    `source='llm'`, `<0.5` ⇒ `manual_review` (V-T3).

    `question_qid` is deliberately NOT a foreign key — AnKing carries qids for
    not-yet-scraped questions.
    """

    __tablename__ = "anki_note_tags"
    __table_args__ = (
        # V-T3: confidence required iff source='llm'.
        CheckConstraint(
            "(source = 'llm' AND confidence IS NOT NULL) "
            "OR (source <> 'llm' AND confidence IS NULL)",
            name="ck_anki_note_tags_confidence_when_llm",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence BETWEEN 0.0 AND 1.0)",
            name="ck_anki_note_tags_confidence_range",
        ),
        CheckConstraint(
            "confidence IS NULL OR confidence >= 0.5 OR manual_review",
            name="ck_anki_note_tags_low_conf_flagged",
        ),
        CheckConstraint(
            "source IN ('schema_map', 'llm', 'manual')",
            name="ck_anki_note_tags_source",
        ),
        # Canonical UQ (V-T1) + raw-tag UQ for sync idempotency (addTags).
        UniqueConstraint("note_id", "node_id", "source", name="uq_anki_note_tags_node_source"),
        UniqueConstraint("note_id", "tag_raw", name="uq_anki_note_tags_note_tag"),
        Index("ix_anki_note_tags_node_id", "node_id"),
        Index("ix_anki_note_tags_question_qid", "question_qid"),
        Index("ix_anki_note_tags_source", "source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    note_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("anki_notes.note_id", ondelete="CASCADE"),
        nullable=False,
    )
    tag_raw: Mapped[str] = mapped_column(Text, nullable=False)
    node_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("outline_nodes.id", ondelete="SET NULL"),
        nullable=True,
    )
    question_qid: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_kind: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="schema_map")
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extractor_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    manual_review: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    is_overridden: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    overridden_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    note: Mapped[AnkiNote] = relationship(back_populates="tags")


class AnkiCardReview(Base):
    """Append-only revlog mirror per SPEC §V26 / §V27.

    Backs T37 retention.py windowed "true retention" math (pass = ease ∈
    {2,3,4}; exclude type='learn'). `review_id` is Anki's revlog id
    (unix-ms, globally unique per Anki) used as PK so T36's incremental
    sync (`startID = MAX(review_id) + 1`) is idempotent on re-run.
    """

    __tablename__ = "anki_card_reviews"
    __table_args__ = (
        CheckConstraint("ease BETWEEN 1 AND 4", name="ck_anki_card_reviews_ease"),
        CheckConstraint(
            "type IN ('learn', 'review', 'relearn', 'cram')",
            name="ck_anki_card_reviews_type",
        ),
        Index("ix_anki_card_reviews_card_reviewed", "card_id", "reviewed_at"),
    )

    review_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    card_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("anki_cards.id", ondelete="CASCADE"),
        nullable=False,
    )
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ease: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    interval_before: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    interval_after: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    card: Mapped[AnkiCard] = relationship(back_populates="reviews")


class AnkiAssignment(Base):
    """Study-plan-driven promise to unsuspend N cards for a scope on a date.

    V51 lifecycle (status):
        pending -> unlocked -> (completed | skipped | failed)

    `card_ids` snapshots the resolved card set at create-time per V52 —
    drift between snapshot and live Anki state is expected and tolerated
    (unsuspend on an already-unsuspended card is a no-op). `priority`
    records the resolver mode that produced the snapshot for audit.
    """

    __tablename__ = "anki_assignments"
    __table_args__ = (
        CheckConstraint(
            "scope_kind IN ('cc','topic')",
            name="ck_anki_assignments_scope_kind",
        ),
        CheckConstraint(
            "status IN ('pending','unlocked','completed','skipped','failed')",
            name="ck_anki_assignments_status",
        ),
        CheckConstraint(
            "max_cards IS NULL OR max_cards > 0",
            name="ck_anki_assignments_max_cards_pos",
        ),
        Index(
            "ix_anki_assignments_status_scheduled",
            "status",
            "scheduled_unlock_at",
        ),
        Index("ix_anki_assignments_actual_unlock", "actual_unlock_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope_kind: Mapped[str] = mapped_column(Text, nullable=False)
    scope_value: Mapped[str] = mapped_column(Text, nullable=False)
    scheduled_unlock_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_unlock_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    card_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False)
    # §V75: notes whose cards back this assignment — the canonical target for
    # addTags(notes=...). card_ids stays the unsuspend target. server_default
    # '{}' backfills rows created before the note-as-unit cutover.
    note_ids: Mapped[list[int]] = mapped_column(
        ARRAY(BigInteger), nullable=False, server_default=text("'{}'")
    )
    max_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    priority: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, server_default="most_specific_first"
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AnkiReview(Base):
    """One-off filtered-deck review for a card set on a target date (V53
    amended 2026-05-23 per T76).

    Standalone — ⊥ FK to assignments; a review may target any card_ids
    regardless of assignment membership. Filtered-deck name = built from
    this row's own PK: `<ANKI_DECK_PREFIX>::review::{id}`. No UNIQUE
    constraint on `(review_date, *)` per V53 amend — tags-as-log accepts
    dup reviews per day; UX debounce protects against accidental
    double-click. Tag chain in `run_review_due` writes
    `coach::review:{id}` per V50.
    """

    __tablename__ = "anki_reviews"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','pushed','failed')",
            name="ck_anki_reviews_status",
        ),
        Index("ix_anki_reviews_status_date", "status", "review_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_date: Mapped[date] = mapped_column(Date, nullable=False)
    card_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), nullable=False)
    # §V75: notes whose cards back this review — target for addTags(notes=...).
    note_ids: Mapped[list[int]] = mapped_column(
        ARRAY(BigInteger), nullable=False, server_default=text("'{}'")
    )
    deck_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    pushed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class AnkiWrite(Base):
    """Append-only AnkiConnect write audit (V50, V55).

    One row per write attempt, success or fail. Drives /admin allowlist
    verification and V55 retry-with-cap (`anki_assignments.failure_count`
    increments alongside a failed row here).

    Migration sets a DESC index on `occurred_at` to match the dominant
    "most-recent first" scan pattern; the ORM-level Index here omits
    direction so `Base.metadata.create_all` produces a usable index in
    test DBs.
    """

    __tablename__ = "anki_writes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('succeeded','failed')",
            name="ck_anki_writes_status",
        ),
        CheckConstraint(
            "source IN ('mcp','scheduler','manual','test')",
            name="ck_anki_writes_source",
        ),
        Index("ix_anki_writes_occurred_at", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    payload_hash: Mapped[str] = mapped_column(Text, nullable=False)
    response_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    assignment_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("anki_assignments.id", ondelete="SET NULL"),
        nullable=True,
    )
    review_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("anki_reviews.id", ondelete="SET NULL"),
        nullable=True,
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AnkiLoadConfig(Base):
    """Singleton config for the Anki load realism evaluator (V59).

    `id INT PK CHECK (id=1)` enforces a single row. The T61 migration
    seeds defaults (200, 60); service-layer `set_anki_load_config`
    updates the singleton in place.
    """

    __tablename__ = "anki_load_config"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_anki_load_config_singleton"),
        CheckConstraint(
            "daily_card_review_budget > 0",
            name="ck_anki_load_config_budget_pos",
        ),
        CheckConstraint(
            "daily_minutes_budget > 0",
            name="ck_anki_load_config_minutes_pos",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    daily_card_review_budget: Mapped[int] = mapped_column(Integer, nullable=False)
    daily_minutes_budget: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
