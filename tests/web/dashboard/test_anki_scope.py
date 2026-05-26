"""Tests for `app.web.dashboard.services.anki_scope.attach_scope_labels`.

Confirms that topic / cc / missing scope values all render the right
label + URL when an AnkiAssignment is enriched in place.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.anki import AnkiAssignment
from app.models.outline import ContentCategory, Topic
from app.web.dashboard.services.anki_scope import attach_scope_labels


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _cc_by_code(session, code: str) -> ContentCategory:
    return (
        await session.execute(select(ContentCategory).where(ContentCategory.code == code))
    ).scalar_one()


async def _make_topic(
    session, *, cc: ContentCategory, name: str, parent_id: int | None = None, depth: int = 0
) -> Topic:
    t = Topic(
        content_category_id=cc.id,
        parent_topic_id=parent_id,
        name=name,
        disciplines=[],
        depth=depth,
        position=999,
    )
    session.add(t)
    await session.flush()
    return t


async def _make_assignment(session, *, scope_kind: str, scope_value: str) -> AnkiAssignment:
    a = AnkiAssignment(
        scope_kind=scope_kind,
        scope_value=scope_value,
        scheduled_unlock_at=_now() + timedelta(days=1),
        card_ids=[1, 2, 3],
        status="pending",
    )
    session.add(a)
    await session.flush()
    return a


@pytest.mark.asyncio
async def test_attach_scope_labels_topic_resolves_to_name_and_ancestor_url(session) -> None:
    cc = await _cc_by_code(session, "4B")  # any seeded CC
    root = await _make_topic(session, cc=cc, name="Root T")
    child = await _make_topic(session, cc=cc, name="Child T", parent_id=root.id, depth=1)
    leaf = await _make_topic(session, cc=cc, name="Leaf T", parent_id=child.id, depth=2)

    a = await _make_assignment(session, scope_kind="topic", scope_value=str(leaf.id))

    await attach_scope_labels(session, [a])

    assert a.scope_label == "Leaf T"
    assert a.scope_url == f"/mastery/{cc.code}/topics/{root.id}/{child.id}/{leaf.id}"


@pytest.mark.asyncio
async def test_attach_scope_labels_cc_resolves_to_code_name_and_cc_url(session) -> None:
    cc = await _cc_by_code(session, "4B")
    a = await _make_assignment(session, scope_kind="cc", scope_value=cc.code)

    await attach_scope_labels(session, [a])

    assert a.scope_label == f"{cc.code} — {cc.name}"
    assert a.scope_url == f"/mastery/{cc.code}"


@pytest.mark.asyncio
async def test_attach_scope_labels_missing_topic_falls_back_to_deleted_label(session) -> None:
    a = await _make_assignment(session, scope_kind="topic", scope_value="999999999")

    await attach_scope_labels(session, [a])

    assert a.scope_label == "Topic #999999999 (deleted)"
    assert a.scope_url is None


@pytest.mark.asyncio
async def test_attach_scope_labels_unknown_cc_falls_back(session) -> None:
    a = await _make_assignment(session, scope_kind="cc", scope_value="ZZZ-not-real")

    await attach_scope_labels(session, [a])

    assert a.scope_label == "CC: ZZZ-not-real"
    assert a.scope_url is None


@pytest.mark.asyncio
async def test_attach_scope_labels_non_numeric_topic_value_falls_back(session) -> None:
    a = await _make_assignment(session, scope_kind="topic", scope_value="not-a-number")

    await attach_scope_labels(session, [a])

    assert a.scope_label == "Topic #not-a-number (deleted)"
    assert a.scope_url is None
