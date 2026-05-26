"""Idempotent seed script: loads aamc_outline.json into the topic tree tables.

CARS handling: CARS has no Foundational Concepts or Content Categories in the AAMC
document. This script synthesizes a placeholder FC (code='CARS-FC', name='CARS') and
a placeholder CC (code='CARS', name='Critical Analysis and Reasoning Skills') so that
the NOT NULL FK constraints on topics are satisfied. No topics are seeded for CARS
since the AAMC does not publish a CARS topic list. The JSON is not modified.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jsonschema import validate
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.outline import ContentCategory, FoundationalConcept, Section, Topic

_HERE = Path(__file__).parent
_SEEDS = _HERE.parent / "app" / "seeds"
OUTLINE_PATH = _SEEDS / "aamc_outline.json"
SCHEMA_PATH = _SEEDS / "aamc_outline.schema.json"


@dataclass
class SeedReport:
    sections_upserted: int
    fcs_upserted: int
    ccs_upserted: int
    topics_upserted: int
    max_depth_observed: int


async def _upsert_topic(
    session: AsyncSession,
    cc_id: int,
    parent_id: Optional[int],
    name: str,
    disciplines: list[str],
    depth: int,
    position: int,
) -> int:
    """Upsert a single topic row; returns the topic id.

    Uses SELECT + INSERT/UPDATE rather than ON CONFLICT because PostgreSQL unique
    constraints do not fire when any key column is NULL (parent_topic_id can be NULL),
    which would cause silent duplicates on re-run.
    """
    if parent_id is None:
        q = select(Topic.id).where(
            Topic.content_category_id == cc_id,
            Topic.parent_topic_id.is_(None),
            Topic.name == name,
        )
    else:
        q = select(Topic.id).where(
            Topic.content_category_id == cc_id,
            Topic.parent_topic_id == parent_id,
            Topic.name == name,
        )

    result = await session.execute(q)
    existing_id: Optional[int] = result.scalar_one_or_none()

    if existing_id is None:
        new_topic = Topic(
            content_category_id=cc_id,
            parent_topic_id=parent_id,
            name=name,
            disciplines=disciplines,
            depth=depth,
            position=position,
        )
        session.add(new_topic)
        await session.flush()
        return new_topic.id
    else:
        await session.execute(
            update(Topic)
            .where(Topic.id == existing_id)
            .values(disciplines=disciplines, depth=depth, position=position)
        )
        return existing_id


async def _seed_topics(
    session: AsyncSession,
    topics: list[dict],
    cc_id: int,
    parent_id: Optional[int],
    depth: int,
) -> tuple[int, int]:
    """Recursively upsert topics; returns (count_upserted, max_depth_observed)."""
    count = 0
    max_depth = depth - 1

    for pos, topic in enumerate(topics):
        topic_id = await _upsert_topic(
            session,
            cc_id=cc_id,
            parent_id=parent_id,
            name=topic["name"],
            disciplines=topic.get("disciplines", []),
            depth=depth,
            position=pos,
        )
        count += 1
        max_depth = max(max_depth, depth)

        children = topic.get("children", [])
        if children:
            child_count, child_max = await _seed_topics(
                session, children, cc_id, topic_id, depth + 1
            )
            count += child_count
            max_depth = max(max_depth, child_max)

    return count, max_depth


async def seed(session: AsyncSession) -> SeedReport:
    """Upsert the full AAMC outline into the database. Commits on success."""
    with open(OUTLINE_PATH) as f:
        data = json.load(f)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    validate(instance=data, schema=schema)

    sections_upserted = 0
    fcs_upserted = 0
    ccs_upserted = 0
    topics_upserted = 0
    max_depth = 0

    for s_pos, section in enumerate(data["sections"]):
        sec_ins = pg_insert(Section).values(
            code=section["code"], name=section["name"], position=s_pos
        )
        sec_upsert = sec_ins.on_conflict_do_update(
            index_elements=["code"],
            set_={"name": sec_ins.excluded.name, "position": sec_ins.excluded.position},
        ).returning(Section.id)
        result = await session.execute(sec_upsert)
        section_id: int = result.scalar_one()
        sections_upserted += 1

        if section["code"] == "CARS":
            # Synthesize placeholder FC and CC; CARS has no topic list in AAMC document.
            fc_ins = pg_insert(FoundationalConcept).values(
                section_id=section_id, code="CARS-FC", name="CARS", position=0
            )
            fc_upsert = fc_ins.on_conflict_do_update(
                index_elements=["code"],
                set_={
                    "section_id": fc_ins.excluded.section_id,
                    "name": fc_ins.excluded.name,
                    "position": fc_ins.excluded.position,
                },
            ).returning(FoundationalConcept.id)
            fc_result = await session.execute(fc_upsert)
            fc_id: int = fc_result.scalar_one()
            fcs_upserted += 1

            cc_ins = pg_insert(ContentCategory).values(
                foundational_concept_id=fc_id,
                code="CARS",
                name="Critical Analysis and Reasoning Skills",
                description=None,
                position=0,
            )
            cc_upsert = cc_ins.on_conflict_do_update(
                index_elements=["code"],
                set_={
                    "foundational_concept_id": cc_ins.excluded.foundational_concept_id,
                    "name": cc_ins.excluded.name,
                    "position": cc_ins.excluded.position,
                },
            ).returning(ContentCategory.id)
            await session.execute(cc_upsert)
            ccs_upserted += 1
            # No topics for CARS
        else:
            for fc_pos, fc in enumerate(section.get("foundational_concepts", [])):
                fc_ins = pg_insert(FoundationalConcept).values(
                    section_id=section_id,
                    code=fc["code"],
                    name=fc["name"],
                    position=fc_pos,
                )
                fc_upsert = fc_ins.on_conflict_do_update(
                    index_elements=["code"],
                    set_={
                        "section_id": fc_ins.excluded.section_id,
                        "name": fc_ins.excluded.name,
                        "position": fc_ins.excluded.position,
                    },
                ).returning(FoundationalConcept.id)
                fc_result = await session.execute(fc_upsert)
                fc_id = fc_result.scalar_one()
                fcs_upserted += 1

                for cc_pos, cc in enumerate(fc.get("content_categories", [])):
                    cc_ins = pg_insert(ContentCategory).values(
                        foundational_concept_id=fc_id,
                        code=cc["code"],
                        name=cc["name"],
                        description=cc.get("description"),
                        position=cc_pos,
                    )
                    cc_upsert = cc_ins.on_conflict_do_update(
                        index_elements=["code"],
                        set_={
                            "foundational_concept_id": cc_ins.excluded.foundational_concept_id,
                            "name": cc_ins.excluded.name,
                            "description": cc_ins.excluded.description,
                            "position": cc_ins.excluded.position,
                        },
                    ).returning(ContentCategory.id)
                    cc_result = await session.execute(cc_upsert)
                    cc_id: int = cc_result.scalar_one()
                    ccs_upserted += 1

                    t_count, t_max = await _seed_topics(
                        session, cc.get("topics", []), cc_id, None, 0
                    )
                    topics_upserted += t_count
                    max_depth = max(max_depth, t_max)

    await session.commit()
    return SeedReport(
        sections_upserted=sections_upserted,
        fcs_upserted=fcs_upserted,
        ccs_upserted=ccs_upserted,
        topics_upserted=topics_upserted,
        max_depth_observed=max_depth,
    )


async def _main() -> None:
    async with AsyncSessionLocal() as session:
        report = await seed(session)
    print(
        f"SeedReport("
        f"sections={report.sections_upserted}, "
        f"fcs={report.fcs_upserted}, "
        f"ccs={report.ccs_upserted}, "
        f"topics={report.topics_upserted}, "
        f"max_depth={report.max_depth_observed})"
    )


if __name__ == "__main__":
    asyncio.run(_main())
