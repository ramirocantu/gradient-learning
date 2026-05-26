from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic


async def search_topics(
    session: AsyncSession, *, query: str, limit: int = 20
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    rows = (
        await session.execute(
            select(
                Topic.id,
                Topic.name,
                Topic.depth,
                ContentCategory.code.label("cc_code"),
                ContentCategory.name.label("cc_name"),
                Section.code.label("section_code"),
            )
            .join(ContentCategory, ContentCategory.id == Topic.content_category_id)
            .join(
                FoundationalConcept,
                FoundationalConcept.id == ContentCategory.foundational_concept_id,
            )
            .join(Section, Section.id == FoundationalConcept.section_id)
            .where(Topic.name.ilike(f"%{query}%"))
            .order_by(Topic.depth.asc(), Topic.name.asc())
            .limit(limit)
        )
    ).all()

    return [
        {
            "topic_id": r.id,
            "name": r.name,
            "depth": r.depth,
            "cc_code": r.cc_code,
            "cc_name": r.cc_name,
            "section_code": r.section_code,
        }
        for r in rows
    ]


async def get_aamc_outline(session: AsyncSession) -> dict[str, Any]:
    sections = (
        (await session.execute(select(Section).order_by(Section.position.asc()))).scalars().all()
    )
    fcs = (
        (
            await session.execute(
                select(FoundationalConcept).order_by(FoundationalConcept.position.asc())
            )
        )
        .scalars()
        .all()
    )
    ccs = (
        (await session.execute(select(ContentCategory).order_by(ContentCategory.position.asc())))
        .scalars()
        .all()
    )
    topics = (await session.execute(select(Topic).order_by(Topic.position.asc()))).scalars().all()

    topics_by_cc: dict[int, list[dict[str, Any]]] = {}
    for t in topics:
        topics_by_cc.setdefault(t.content_category_id, []).append(
            {
                "topic_id": t.id,
                "name": t.name,
                "depth": t.depth,
                "parent_topic_id": t.parent_topic_id,
                "disciplines": list(t.disciplines or []),
                "position": t.position,
            }
        )

    ccs_by_fc: dict[int, list[dict[str, Any]]] = {}
    for c in ccs:
        ccs_by_fc.setdefault(c.foundational_concept_id, []).append(
            {
                "cc_id": c.id,
                "code": c.code,
                "name": c.name,
                "description": c.description,
                "position": c.position,
                "topics": topics_by_cc.get(c.id, []),
            }
        )

    fcs_by_section: dict[int, list[dict[str, Any]]] = {}
    for fc in fcs:
        fcs_by_section.setdefault(fc.section_id, []).append(
            {
                "fc_id": fc.id,
                "code": fc.code,
                "name": fc.name,
                "position": fc.position,
                "content_categories": ccs_by_fc.get(fc.id, []),
            }
        )

    return {
        "sections": [
            {
                "section_id": s.id,
                "code": s.code,
                "name": s.name,
                "position": s.position,
                "foundational_concepts": fcs_by_section.get(s.id, []),
            }
            for s in sections
        ]
    }
