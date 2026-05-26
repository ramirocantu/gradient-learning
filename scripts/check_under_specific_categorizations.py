"""Check for under-specific categorizations in the question_tags table.

Flags questions where a parent topic is assigned but none of its DB children
are also tagged on the same question. These questions are candidates for
re-categorization with the v3-leaf-first extractor version.

Output: TSV to stdout — qid, parent_topic_name, #_available_children, child_names_csv
Exit code: always 0 (informational, not a CI gate).

CLI:
    uv run python -m scripts.check_under_specific_categorizations
    uv run python -m scripts.check_under_specific_categorizations --cc-code 5A
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.captures import Question, QuestionTag
from app.models.outline import ContentCategory, Topic

logger = logging.getLogger(__name__)


async def run(session: AsyncSession, *, cc_code: str | None = None) -> int:
    """Scan and print under-specific categorizations; return flagged count."""
    topic_stmt = select(Topic)
    if cc_code is not None:
        topic_stmt = topic_stmt.join(
            ContentCategory, Topic.content_category_id == ContentCategory.id
        ).where(ContentCategory.code == cc_code)
    all_topics = (await session.execute(topic_stmt)).scalars().all()

    topics_by_id = {t.id: t for t in all_topics}
    children_by_parent: dict[int, list[Topic]] = {}
    for t in all_topics:
        if t.parent_topic_id is not None:
            children_by_parent.setdefault(t.parent_topic_id, []).append(t)

    parent_topic_ids = set(children_by_parent.keys())
    if not parent_topic_ids:
        print("No topics with children found.")
        return 0

    tag_stmt = select(QuestionTag).where(
        QuestionTag.topic_id.in_(parent_topic_ids),
        QuestionTag.is_overridden.is_(False),
    )
    parent_tags = (await session.execute(tag_stmt)).scalars().all()

    flagged: list[tuple[str, str, int, str]] = []
    for tag in parent_tags:
        parent_id = tag.topic_id
        assert parent_id is not None
        children = children_by_parent.get(parent_id, [])
        child_ids = {c.id for c in children}

        sibling_tags = (
            (
                await session.execute(
                    select(QuestionTag).where(
                        QuestionTag.question_id == tag.question_id,
                        QuestionTag.topic_id.in_(child_ids),
                        QuestionTag.is_overridden.is_(False),
                    )
                )
            )
            .scalars()
            .all()
        )

        if not sibling_tags:
            question = (
                await session.execute(select(Question).where(Question.id == tag.question_id))
            ).scalar_one_or_none()
            qid = question.qid if question else f"id={tag.question_id}"
            parent_name = (
                topics_by_id[parent_id].name if parent_id in topics_by_id else str(parent_id)
            )
            child_names = ", ".join(c.name for c in children)
            flagged.append((qid, parent_name, len(children), child_names))

    total_llm_tags_stmt = (
        select(func.count())
        .select_from(QuestionTag)
        .where(
            QuestionTag.source == "llm",
            QuestionTag.topic_id.is_not(None),
            QuestionTag.is_overridden.is_(False),
        )
    )
    total = (await session.execute(total_llm_tags_stmt)).scalar_one()

    for qid, parent_name, n_children, child_names in flagged:
        print(f"{qid}\t{parent_name}\t{n_children}\t{child_names}")

    print(f"Flagged {len(flagged)} questions out of {total} total LLM-tagged.")
    return len(flagged)


async def _main(*, cc_code: str | None) -> int:
    async with AsyncSessionLocal() as session:
        return await run(session, cc_code=cc_code)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cc-code",
        default=None,
        help="Filter to a single content category code (e.g. 5A).",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(_main(cc_code=args.cc_code))
    sys.exit(0)


if __name__ == "__main__":
    _cli()
