from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Media, Passage, Question

_HASH_ATTR_RE = re.compile(r'data-media-content-hash="([^"]+)"')


def _hashes_in_html(*htmls: str | None) -> set[str]:
    found: set[str] = set()
    for h in htmls:
        if not h:
            continue
        found.update(_HASH_ATTR_RE.findall(h))
    return found


async def referenced_hashes_for_question(session: AsyncSession, question_id: int) -> set[str]:
    """All media content_hashes referenced by a question — choices.media_ids + inline HTML."""
    q = await session.get(Question, question_id)
    if q is None:
        return set()

    media_ids: set[int] = set()
    for choice in q.choices or []:
        for mid in choice.get("media_ids") or []:
            if isinstance(mid, int):
                media_ids.add(mid)

    hashes: set[str] = set()
    if media_ids:
        rows = await session.execute(select(Media.content_hash).where(Media.id.in_(media_ids)))
        hashes.update(rows.scalars().all())

    passage_html = None
    if q.passage_id is not None:
        passage = await session.get(Passage, q.passage_id)
        if passage is not None:
            passage_html = passage.html

    hashes |= _hashes_in_html(q.stem_html, q.explanation_html, passage_html)
    return hashes


async def media_by_hash_for_question(session: AsyncSession, question_id: int) -> dict[str, str]:
    """{content_hash: local_path} for the media a question references."""
    hashes = await referenced_hashes_for_question(session, question_id)
    if not hashes:
        return {}
    result = await session.execute(
        select(Media.content_hash, Media.local_path).where(Media.content_hash.in_(hashes))
    )
    return {row.content_hash: row.local_path for row in result}
