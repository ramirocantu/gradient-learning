"""Resolve `AnkiAssignment.scope_kind/scope_value` into human labels + URLs.

Assignments store the scope as `(kind, value)` where value is a raw
identifier (CC code string, or topic id as string — topic ids are kept
numeric for token-efficient LLM prompts in the topic resolver). The
dashboard surfaces need a readable name + a link back to the mastery
view, so this helper enriches a batch of assignments with `scope_label`
and `scope_url` attributes the templates render directly.

Topic URL shape matches `/mastery/{cc_code}/topics/{id_path}` from
``app/web/dashboard/routes/topics.py`` — `id_path` is the slash-joined
ancestor chain ending at the leaf topic id (§V32 validates it server
side).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.anki import AnkiAssignment
from app.models.outline import ContentCategory, Topic


def _ancestor_chain(by_id: dict[int, Topic], topic_id: int) -> list[int]:
    chain: list[int] = []
    node = by_id.get(topic_id)
    if node is None:
        return chain
    chain.append(node.id)
    while node.parent_topic_id is not None and node.parent_topic_id in by_id:
        node = by_id[node.parent_topic_id]
        chain.append(node.id)
    chain.reverse()
    return chain


async def attach_scope_labels(session: AsyncSession, assignments: Sequence[AnkiAssignment]) -> None:
    """Mutate `assignments` in place, setting `scope_label` + `scope_url`.

    Single batch lookup per kind — no N+1. Unknown topic / CC values fall
    back to a "(deleted)" label with `scope_url=None` so the template can
    render flat text instead of a broken link.
    """
    topic_ids: set[int] = set()
    cc_codes: set[str] = set()
    for a in assignments:
        if a.scope_kind == "topic":
            try:
                topic_ids.add(int(a.scope_value))
            except (TypeError, ValueError):
                pass
        elif a.scope_kind == "cc":
            cc_codes.add(a.scope_value)

    topic_label_url: dict[int, tuple[str, str]] = {}
    if topic_ids:
        targets = (
            (await session.execute(select(Topic).where(Topic.id.in_(topic_ids)))).scalars().all()
        )
        target_cc_ids = {t.content_category_id for t in targets}
        all_topics: Iterable[Topic] = ()
        cc_by_id: dict[int, ContentCategory] = {}
        if target_cc_ids:
            all_topics = (
                (
                    await session.execute(
                        select(Topic).where(Topic.content_category_id.in_(target_cc_ids))
                    )
                )
                .scalars()
                .all()
            )
            cc_rows = (
                (
                    await session.execute(
                        select(ContentCategory).where(ContentCategory.id.in_(target_cc_ids))
                    )
                )
                .scalars()
                .all()
            )
            cc_by_id = {c.id: c for c in cc_rows}
        by_id = {t.id: t for t in all_topics}
        for t in targets:
            cc = cc_by_id.get(t.content_category_id)
            chain = _ancestor_chain(by_id, t.id)
            if cc is None or not chain:
                continue
            url = "/mastery/" + cc.code + "/topics/" + "/".join(str(x) for x in chain)
            topic_label_url[t.id] = (t.name, url)

    cc_label_url: dict[str, tuple[str, str]] = {}
    if cc_codes:
        cc_rows = (
            (
                await session.execute(
                    select(ContentCategory).where(ContentCategory.code.in_(cc_codes))
                )
            )
            .scalars()
            .all()
        )
        for c in cc_rows:
            cc_label_url[c.code] = (f"{c.code} — {c.name}", f"/mastery/{c.code}")

    for a in assignments:
        label: str
        url: str | None
        if a.scope_kind == "topic":
            try:
                tid = int(a.scope_value)
            except (TypeError, ValueError):
                tid = None
            if tid is not None and tid in topic_label_url:
                label, url = topic_label_url[tid]
            else:
                label = f"Topic #{a.scope_value} (deleted)"
                url = None
        elif a.scope_kind == "cc":
            if a.scope_value in cc_label_url:
                label, url = cc_label_url[a.scope_value]
            else:
                label = f"CC: {a.scope_value}"
                url = None
        else:
            label = f"{a.scope_kind}:{a.scope_value}"
            url = None
        a.scope_label = label  # type: ignore[attr-defined]
        a.scope_url = url  # type: ignore[attr-defined]
