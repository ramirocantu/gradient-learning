"""Shared capture normalizer (§A plugin seam).

The source-agnostic `capture → normalized {Question, Attempt}` pipeline that
every browser-extension-style source adapter delegates to. It stamps
``payload.source`` onto every row it writes, so the same code serves UWorld,
generic web-Qbank, and manual entry without privileging any one source
(§A — the core is domain-blind). Adapters that need source-specific handling
wrap or replace steps; today UWorld, web-Qbank, and manual share it verbatim.

Runs entirely inside the caller's transaction — the endpoint owns
commit/rollback.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Passage, Question, RawCapture
from app.models.media import Media
from app.models.outline import Course
from app.schemas.captures import (
    CapturePayload,
    ChoiceItem,
    IngestResponse,
    MediaCapture,
    PassageCapture,
)
from app.services.adapters import UnknownCourseError
from app.services.media_store import relative_media_path, write_media

_NON_TAG_CHARS = re.compile(r"[^<>\s]")


def _html_has_text(html: str) -> bool:
    """Cheap heuristic: any non-whitespace char outside angle brackets."""
    stripped = re.sub(r"<[^>]*>", "", html)
    return bool(_NON_TAG_CHARS.search(stripped))


def _resolve_choice(choice: ChoiceItem, media_by_hash: dict[str, int]) -> dict[str, Any]:
    media_ids: list[int] = []
    for h in choice.media_content_hashes:
        if h not in media_by_hash:
            raise ValueError(f"choice {choice.key!r} references unknown media content_hash {h!r}")
        media_ids.append(media_by_hash[h])
    return {
        "key": choice.key,
        "html": choice.html,
        "plain": choice.plain,
        "media_ids": media_ids,
    }


async def _persist_media(
    session: AsyncSession,
    items: list[MediaCapture],
    warnings: list[dict[str, Any]],
) -> tuple[dict[str, int], list[int]]:
    by_hash: dict[str, int] = {}
    touched: list[int] = []

    for m in items:
        if not m.bytes_b64:
            warnings.append(
                {
                    "code": "media_bytes_empty",
                    "message": f"media {m.content_hash} bytes_b64 is empty",
                    "selector": None,
                }
            )
            continue

        try:
            await write_media(m.content_hash, m.mime_type, m.bytes_b64)
        except ValueError as exc:
            warnings.append(
                {
                    "code": "media_decode_failed",
                    "message": f"media {m.content_hash}: {exc}",
                    "selector": None,
                }
            )
            continue

        local_path = relative_media_path(m.content_hash, m.mime_type)
        byte_size = (len(m.bytes_b64) * 3) // 4 - m.bytes_b64.count("=", -2)

        ins = (
            pg_insert(Media)
            .values(
                content_hash=m.content_hash,
                local_path=local_path,
                original_url=m.original_url,
                mime_type=m.mime_type,
                width_px=m.width_px,
                height_px=m.height_px,
                byte_size=byte_size,
            )
            .on_conflict_do_nothing(index_elements=["content_hash"])
            .returning(Media.id)
        )
        result = await session.execute(ins)
        media_id = result.scalar_one_or_none()
        if media_id is None:
            existing = await session.execute(
                select(Media.id).where(Media.content_hash == m.content_hash)
            )
            media_id = existing.scalar_one()

        by_hash[m.content_hash] = media_id
        if media_id not in touched:
            touched.append(media_id)

    return by_hash, touched


async def _upsert_passage(
    session: AsyncSession,
    capture: PassageCapture,
) -> int:
    content_hash = hashlib.sha256(capture.html.encode("utf-8")).hexdigest()

    found: Passage | None = None
    if capture.uworld_passage_id is not None:
        found = (
            await session.execute(
                select(Passage).where(Passage.uworld_passage_id == capture.uworld_passage_id)
            )
        ).scalar_one_or_none()
    if found is None:
        found = (
            await session.execute(select(Passage).where(Passage.content_hash == content_hash))
        ).scalar_one_or_none()

    if found is None:
        row = Passage(
            uworld_passage_id=capture.uworld_passage_id,
            content_hash=content_hash,
            html=capture.html,
            plain_text=capture.plain,
        )
        session.add(row)
        await session.flush()
        return row.id

    if found.html != capture.html:
        await session.execute(
            update(Passage)
            .where(Passage.id == found.id)
            .values(
                html=capture.html,
                plain_text=capture.plain,
                content_hash=content_hash,
                last_updated_at=func.clock_timestamp(),
            )
        )
    return found.id


def _choices_equal(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x.get("key") != y.get("key"):
            return False
        if x.get("html") != y.get("html"):
            return False
        if x.get("plain") != y.get("plain"):
            return False
        if list(x.get("media_ids") or []) != list(y.get("media_ids") or []):
            return False
    return True


async def _upsert_question(
    session: AsyncSession,
    payload: CapturePayload,
    passage_id: int | None,
    resolved_choices: list[dict[str, Any]],
    course_id: int | None,
) -> Question:
    parsed = payload.parsed
    existing = (
        await session.execute(select(Question).where(Question.qid == payload.qid))
    ).scalar_one_or_none()

    if existing is None:
        q = Question(
            source=payload.source,
            qid=payload.qid,
            course_id=course_id,
            passage_id=passage_id,
            stem_html=parsed.stem_html,
            stem_plain=parsed.stem_plain,
            choices=resolved_choices,
            correct_choice=parsed.correct_choice,
            explanation_html=parsed.explanation_html,
            explanation_plain=parsed.explanation_plain,
            uworld_aamc_tags=list(parsed.uworld_aamc_tags) if parsed.uworld_aamc_tags else None,
            needs_categorization=True,
        )
        session.add(q)
        await session.flush()
        return q

    incoming_tags = list(parsed.uworld_aamc_tags) if parsed.uworld_aamc_tags else None
    stored_tags = list(existing.uworld_aamc_tags) if existing.uworld_aamc_tags else None
    tags_changed = incoming_tags != stored_tags

    content_changed = (
        existing.stem_html != parsed.stem_html
        or existing.correct_choice != parsed.correct_choice
        or existing.explanation_html != parsed.explanation_html
        or existing.passage_id != passage_id
        or not _choices_equal(existing.choices or [], resolved_choices)
        or tags_changed
    )

    if content_changed:
        existing.stem_html = parsed.stem_html
        existing.stem_plain = parsed.stem_plain
        existing.choices = resolved_choices
        existing.correct_choice = parsed.correct_choice
        existing.explanation_html = parsed.explanation_html
        existing.explanation_plain = parsed.explanation_plain
        existing.uworld_aamc_tags = incoming_tags
        existing.passage_id = passage_id
        if tags_changed:
            existing.needs_categorization = True
        existing.last_updated_at = func.clock_timestamp()
        await session.flush()

    # Course re-scope (V-CAP2): a capture may supply a course the existing row
    # lacked or differs from. Adopt it and re-flag for categorization so the
    # grounded-tag job recalls against the new course's outline. A capture with
    # no course_slug (course_id None) never clears an existing binding.
    if course_id is not None and existing.course_id != course_id:
        existing.course_id = course_id
        existing.needs_categorization = True
        existing.last_updated_at = func.clock_timestamp()
        await session.flush()

    return existing


def _sanity_warnings(payload: CapturePayload) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    def _check(field_name: str, html: str | None, plain: str | None) -> None:
        if html is None:
            return
        if (plain or "").strip():
            return
        if _html_has_text(html):
            warnings.append(
                {
                    "code": "empty_plain_with_nonempty_html",
                    "message": f"{field_name} plain is empty but html is not",
                    "selector": None,
                }
            )

    parsed = payload.parsed
    if parsed.passage is not None:
        _check("passage", parsed.passage.html, parsed.passage.plain)
    _check("stem", parsed.stem_html, parsed.stem_plain)
    _check("explanation", parsed.explanation_html, parsed.explanation_plain)
    for c in parsed.choices:
        _check(f"choice.{c.key}", c.html, c.plain)
    return warnings


async def _resolve_course_id(session: AsyncSession, course_slug: str | None) -> int | None:
    """Resolve ``course_slug`` → ``course_id`` against ``courses.slug`` (V-CAP2).

    None slug → None (unscoped, single-course fallback). Unknown slug →
    :class:`UnknownCourseError` (the endpoint maps it to 422 — ⊥ silent drop)."""
    if course_slug is None:
        return None
    course_id = (
        await session.execute(select(Course.id).where(Course.slug == course_slug))
    ).scalar_one_or_none()
    if course_id is None:
        raise UnknownCourseError(course_slug)
    return course_id


async def normalize_capture(
    payload: CapturePayload, session: AsyncSession
) -> IngestResponse:
    """Normalize one capture into {RawCapture, media, Passage, Question,
    Attempt}, stamping ``payload.source`` on each. Source-agnostic (§A) —
    the dispatching adapter only supplies the ``source`` key."""

    # Step 0: resolve the course (V-CAP2) before writing anything — an unknown
    # slug aborts the whole ingest (422) rather than persisting an unscoped row.
    course_id = await _resolve_course_id(session, payload.course_slug)

    # Step 1: RawCapture (warnings may be appended later).
    initial_warnings = (
        [w.model_dump() for w in payload.parse_warnings] if payload.parse_warnings else []
    )
    rc = RawCapture(
        source=payload.source,
        qid=payload.qid,
        course_id=course_id,
        captured_at=payload.captured_at,
        raw_html=payload.html,
        raw_json=payload.model_dump(mode="json"),
        parse_warnings=initial_warnings or None,
        extension_version=payload.extension_version,
        uworld_test_id=payload.uworld_test_id,
    )
    session.add(rc)
    await session.flush()

    accumulated_warnings: list[dict[str, Any]] = list(initial_warnings)

    # Step 2: media.
    media_by_hash, touched_media_ids = await _persist_media(
        session, payload.media, accumulated_warnings
    )

    # Step 3: passage.
    passage_id: int | None = None
    if payload.parsed.passage is not None:
        passage_id = await _upsert_passage(session, payload.parsed.passage)

    # Step 4: resolve choice media.
    resolved_choices = [_resolve_choice(c, media_by_hash) for c in payload.parsed.choices]

    # Step 5: question upsert.
    question = await _upsert_question(session, payload, passage_id, resolved_choices, course_id)

    # Step 6: attempt.
    attempt = Attempt(
        source=payload.source,
        question_id=question.id,
        attempted_at=payload.captured_at,
        selected_choice=payload.parsed.selected_choice,
        is_correct=payload.parsed.is_correct,
        time_seconds=payload.parsed.time_seconds,
        flagged=payload.parsed.flagged,
        uworld_test_id=payload.uworld_test_id,
    )
    session.add(attempt)
    await session.flush()

    # Sanity warnings.
    accumulated_warnings.extend(_sanity_warnings(payload))

    if accumulated_warnings != (rc.parse_warnings or []):
        rc.parse_warnings = accumulated_warnings or None
        await session.flush()

    return IngestResponse(
        capture_id=rc.id,
        question_id=question.id,
        attempt_id=attempt.id,
        passage_id=passage_id,
        media_ids=touched_media_ids,
    )
