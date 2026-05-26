"""Tests for the anki topic resolver Batches API adapter (SPEC §T51).

Mocks the Anthropic SDK at `client.messages.batches.*`. Exercises:
  - build_topic_resolver_batch_requests skip rules (cache hits, empty signal,
    no-candidate CCs)
  - persist_topic_resolver_batch_results writes one anki_note_tags row per
    pick mirroring the sync worker (§V75 note-scoped)
  - submit_topic_resolver_batch records an llm_batch_runs row
  - finalize_topic_resolver_batch refuses to run on a non-terminal batch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiCard, AnkiNote, AnkiNoteTag
from app.models.llm_batch import LlmBatchRun
from app.models.outline import ContentCategory, Topic
from app.services.anki.topic_resolver import EXTRACTOR_VERSION
from app.services.anki.topic_resolver_batch import (
    _build_custom_id,
    _parse_custom_id,
    build_topic_resolver_batch_requests,
    finalize_topic_resolver_batch,
    persist_topic_resolver_batch_results,
    submit_topic_resolver_batch,
)
from app.services.anki.topic_resolver_cache import AnkiTopicResolverCache


# --------------------------------------------------------------------------- #
# Anthropic batch stub
# --------------------------------------------------------------------------- #


@dataclass
class _Counts:
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


@dataclass
class _Batch:
    id: str = "batch_test"
    processing_status: str = "in_progress"
    request_counts: _Counts = field(default_factory=_Counts)
    created_at: Any = None
    ended_at: Any = None


@dataclass
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 80
    cache_read_input_tokens: int = 0


@dataclass
class _Message:
    content: list[Any]
    usage: _Usage = field(default_factory=_Usage)


@dataclass
class _SuccessResult:
    message: _Message
    type: str = "succeeded"


@dataclass
class _ErrorResult:
    type: str = "errored"
    error: Any = None


@dataclass
class _Individual:
    custom_id: str
    result: Any


class _StubBatches:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.retrieve_responses: dict[str, _Batch] = {}
        self.results_payload: dict[str, list[_Individual]] = {}
        self.next_batch: _Batch | None = None

    async def create(self, *, requests, **_):
        self.created.append({"requests": list(requests)})
        return self.next_batch or _Batch(
            id="batch_created",
            processing_status="in_progress",
            request_counts=_Counts(processing=len(self.created[-1]["requests"])),
        )

    async def retrieve(self, batch_id, **_):
        return self.retrieve_responses.get(batch_id, _Batch(id=batch_id))

    async def results(self, batch_id, **_):
        async def stream():
            for item in self.results_payload.get(batch_id, []):
                yield item

        return stream()


class _StubClient:
    def __init__(self) -> None:
        self.messages = type("M", (), {"batches": _StubBatches()})()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _cc_with_topic(session: AsyncSession) -> tuple[ContentCategory, Topic, str]:
    cc = (
        await session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    topic = (
        (
            await session.execute(
                select(Topic).where(
                    Topic.content_category_id == cc.id, Topic.parent_topic_id.is_(None)
                )
            )
        )
        .scalars()
        .first()
    )
    assert topic is not None
    return cc, topic, f"{cc.code} >> {topic.name}"


async def _seed_card_with_cc(
    session: AsyncSession,
    *,
    anki_card_id: int,
    cc: ContentCategory,
    text: str = "long enough text for resolver test ABC",
) -> AnkiCard:
    """§V75: seed a NOTE carrying the content + aamc_cc tag (the batch builder
    is note-scoped: `_candidate_notes`, `_note_text`, custom_id keyed on
    note_id), plus a card linked to it."""
    note = AnkiNote(
        note_id=anki_card_id,
        deck_name="AnKing MCAT Deck",
        fields_json={"Text": {"value": text, "order": 0}},
    )
    session.add(note)
    await session.flush()
    card = AnkiCard(
        anki_card_id=anki_card_id,
        deck_name="AnKing MCAT Deck",
        note_id=note.note_id,
        fields_json={"Text": {"value": text, "order": 0}},
    )
    session.add(card)
    await session.flush()
    session.add(
        AnkiNoteTag(
            note_id=note.note_id,
            tag_raw=f"#AK_MCAT_v2::#AAMC::Concepts::C/P::Foundational_Concept_04::{cc.code}-x",
            parsed_kind="aamc_cc",
            content_category_id=cc.id,
            source="regex",
        )
    )
    await session.flush()
    return card


def _temp_cache(tmp_path: Path) -> AnkiTopicResolverCache:
    return AnkiTopicResolverCache(tmp_path / "topic-cache.db")


def _make_message(picks: list[dict]) -> _Message:
    """Build a fake assistant message carrying a tool_use block w/ the
    multi-pick payload, shaped like the production schema."""
    from anthropic.types import ToolUseBlock

    block = ToolUseBlock.model_construct(
        id="toolu_x",
        name="submit_anki_topic",
        type="tool_use",
        input={"decline": False, "topic_picks": picks},
    )
    return _Message(content=[block])


# --------------------------------------------------------------------------- #
# custom_id round-trip
# --------------------------------------------------------------------------- #


def test_custom_id_round_trip() -> None:
    cid = _build_custom_id(12345, "4A")
    assert cid == "topic-12345-4A"
    assert _parse_custom_id(cid) == (12345, "4A")


def test_parse_custom_id_handles_garbage() -> None:
    assert _parse_custom_id("garbage") is None
    assert _parse_custom_id("topic-notanint-4A") is None
    assert _parse_custom_id("topic-1") is None


# --------------------------------------------------------------------------- #
# build_topic_resolver_batch_requests
# --------------------------------------------------------------------------- #


async def test_build_batch_requests_includes_pending_pairs(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    cc, _, _ = await _cc_with_topic(db_session)
    await _seed_card_with_cc(db_session, anki_card_id=701, cc=cc)
    await _seed_card_with_cc(db_session, anki_card_id=702, cc=cc)
    await db_session.commit()

    cache = _temp_cache(tmp_path)
    try:
        summary = await build_topic_resolver_batch_requests(db_session, cache=cache)
    finally:
        cache.close()

    assert len(summary.items) == 2
    assert summary.skipped_cache_hits == 0
    custom_ids = {it.custom_id for it in summary.items}
    assert custom_ids == {"topic-701-4A", "topic-702-4A"}
    # Each params dict must carry the cc-scoped system + tool def w/ cache_control.
    params = summary.items[0].params
    assert params["tool_choice"] == {"type": "tool", "name": "submit_anki_topic"}
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["tools"][0]["cache_control"] == {"type": "ephemeral"}


async def test_build_batch_requests_skips_cache_hits(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    from app.services.anki.topic_resolver import (
        TopicPick,
        _format_tag_payload,
        make_cache_key,
    )
    from app.services.anki.topic_resolver_worker import _filter_anking_tags, _note_text

    cc, _, topic_path = await _cc_with_topic(db_session)
    card = await _seed_card_with_cc(db_session, anki_card_id=801, cc=cc)
    await db_session.commit()
    # §V75: the cache key is note-scoped. Re-fetch the NOTE w/ eager-loaded
    # tags so _filter_anking_tags doesn't trip MissingGreenlet via lazy-load.
    from sqlalchemy.orm import selectinload

    note = (
        await db_session.execute(
            select(AnkiNote)
            .where(AnkiNote.note_id == card.note_id)
            .options(selectinload(AnkiNote.tags))
        )
    ).scalar_one()

    cache = _temp_cache(tmp_path)
    try:
        # Pre-populate cache for this note's expected key.
        from app.config import settings

        tag_payload = _format_tag_payload(_filter_anking_tags(note))
        key = make_cache_key(
            tag_payload,
            _note_text(note),
            "4A",
            settings.ANKI_TOPIC_RESOLVER_MODEL,
        )
        cache.put(
            key,
            [TopicPick(topic_path=topic_path, confidence=0.9, rationale="warm")],
            EXTRACTOR_VERSION,
            model=settings.ANKI_TOPIC_RESOLVER_MODEL,
            cost=0.0001,
        )

        summary = await build_topic_resolver_batch_requests(db_session, cache=cache)
    finally:
        cache.close()

    assert summary.items == []
    assert summary.skipped_cache_hits == 1


async def test_build_batch_requests_skips_cars(db_session: AsyncSession, tmp_path: Path) -> None:
    cc = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "CARS"))
    ).scalar_one()
    await _seed_card_with_cc(db_session, anki_card_id=901, cc=cc)
    await db_session.commit()

    cache = _temp_cache(tmp_path)
    try:
        summary = await build_topic_resolver_batch_requests(db_session, cache=cache)
    finally:
        cache.close()

    assert summary.items == []
    assert summary.skipped_no_candidates == 1


# --------------------------------------------------------------------------- #
# persist_topic_resolver_batch_results
# --------------------------------------------------------------------------- #


async def test_persist_results_writes_one_row_per_pick(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    cc = (
        await db_session.execute(select(ContentCategory).where(ContentCategory.code == "4A"))
    ).scalar_one()
    topics = (
        (
            await db_session.execute(
                select(Topic).where(
                    Topic.content_category_id == cc.id, Topic.parent_topic_id.is_(None)
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(topics) >= 2
    path_a = f"{cc.code} >> {topics[0].name}"
    path_b = f"{cc.code} >> {topics[1].name}"

    card = await _seed_card_with_cc(db_session, anki_card_id=1101, cc=cc)
    await db_session.commit()

    client = _StubClient()
    client.messages.batches.results_payload["batch_xyz"] = [
        _Individual(
            custom_id=_build_custom_id(card.note_id, "4A"),
            result=_SuccessResult(
                message=_make_message(
                    [
                        {"topic_path": path_a, "confidence": 0.9, "rationale": "primary"},
                        {"topic_path": path_b, "confidence": 0.7, "rationale": "secondary"},
                    ]
                ),
                type="succeeded",
            ),
        ),
    ]

    cache = _temp_cache(tmp_path)
    try:
        persist = await persist_topic_resolver_batch_results(
            db_session, anthropic_batch_id="batch_xyz", client=client, cache=cache
        )
    finally:
        cache.close()

    assert persist.succeeded == 1
    assert persist.persisted_rows == 2
    rows = (
        (
            await db_session.execute(
                select(AnkiNoteTag).where(
                    AnkiNoteTag.note_id == card.note_id,
                    AnkiNoteTag.source == "llm",
                    AnkiNoteTag.parsed_kind == "aamc_topic",
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert {r.topic_id for r in rows} == {topics[0].id, topics[1].id}


async def test_persist_results_counts_errored(db_session: AsyncSession, tmp_path: Path) -> None:
    cc, _, topic_path = await _cc_with_topic(db_session)
    await _seed_card_with_cc(db_session, anki_card_id=1102, cc=cc)
    await db_session.commit()

    client = _StubClient()
    client.messages.batches.results_payload["batch_err"] = [
        _Individual(
            custom_id="topic-1102-4A",
            result=_ErrorResult(error={"type": "overloaded"}),
        ),
    ]

    cache = _temp_cache(tmp_path)
    try:
        persist = await persist_topic_resolver_batch_results(
            db_session, anthropic_batch_id="batch_err", client=client, cache=cache
        )
    finally:
        cache.close()

    assert persist.errored == 1
    assert persist.persisted_rows == 0
    # Just to keep `topic_path` referenced for lint hygiene.
    _ = topic_path


# --------------------------------------------------------------------------- #
# submit + finalize
# --------------------------------------------------------------------------- #


async def test_submit_topic_resolver_batch_records_run_row(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    cc, _, _ = await _cc_with_topic(db_session)
    await _seed_card_with_cc(db_session, anki_card_id=1201, cc=cc)
    await db_session.commit()

    client = _StubClient()
    client.messages.batches.next_batch = _Batch(
        id="batch_submitted",
        processing_status="in_progress",
        request_counts=_Counts(processing=1),
    )

    cache = _temp_cache(tmp_path)
    try:
        row, build = await submit_topic_resolver_batch(db_session, client=client, cache=cache)
    finally:
        cache.close()

    assert row.anthropic_batch_id == "batch_submitted"
    assert row.total_requests == 1
    assert row.extractor == "anki_topic_resolver"
    assert build.skipped_cache_hits == 0


async def test_submit_topic_resolver_batch_raises_when_no_items(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    # No cards seeded → no candidates → no items → raises.
    client = _StubClient()
    cache = _temp_cache(tmp_path)
    try:
        with pytest.raises(ValueError, match="no items"):
            await submit_topic_resolver_batch(db_session, client=client, cache=cache)
    finally:
        cache.close()


async def test_finalize_refuses_non_terminal_batch(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    # Insert a run row directly.
    from datetime import datetime, timezone

    row = LlmBatchRun(
        anthropic_batch_id="batch_pending",
        extractor="anki_topic_resolver",
        extractor_version=EXTRACTOR_VERSION,
        model="claude-haiku-4-5-20251001",
        submitted_at=datetime.now(timezone.utc),
        processing_status="in_progress",
        total_requests=1,
    )
    db_session.add(row)
    await db_session.flush()

    client = _StubClient()
    client.messages.batches.retrieve_responses["batch_pending"] = _Batch(
        id="batch_pending",
        processing_status="in_progress",
        request_counts=_Counts(processing=1),
    )

    cache = _temp_cache(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="not in a terminal status"):
            await finalize_topic_resolver_batch(
                db_session, run_id=row.id, client=client, cache=cache
            )
    finally:
        cache.close()
