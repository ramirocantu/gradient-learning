"""Test for Ticket 6.8 migration SQL: wipe LLM tags, re-queue questions.

The project uses create_all (not alembic) for the test DB, so we test the
migration's SQL statements directly rather than running alembic commands.
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag
from app.models.outline import ContentCategory


async def test_migration_invalidates_llm_tags_only(seeded_report, test_engine):
    async with AsyncSession(test_engine) as s:
        cc_4a = (
            await s.execute(select(ContentCategory.id).where(ContentCategory.code == "4A"))
        ).scalar_one()
        cc_4b = (
            await s.execute(select(ContentCategory.id).where(ContentCategory.code == "4B"))
        ).scalar_one()
        cc_5a = (
            await s.execute(select(ContentCategory.id).where(ContentCategory.code == "5A"))
        ).scalar_one()

        q = Question(
            qid="q-migration-test-6-8",
            passage_id=None,
            stem_html="<p>Test</p>",
            stem_plain="Test stem",
            choices=[
                {
                    "key": "A",
                    "html": "<p>a</p>",
                    "plain": "a",
                    "media_content_hashes": [],
                }
            ],
            correct_choice="A",
            explanation_html=None,
            explanation_plain=None,
            uworld_aamc_tags=["Subject: Physics"],
            needs_categorization=False,
        )
        s.add(q)
        await s.flush()
        q_id = q.id

        manual_tag = QuestionTag(
            question_id=q_id,
            content_category_id=cc_4a,
            confidence=1.0,
            source="manual",
        )
        llm_tag = QuestionTag(
            question_id=q_id,
            content_category_id=cc_4b,
            confidence=0.9,
            source="llm",
            is_overridden=False,
        )
        llm_overridden_tag = QuestionTag(
            question_id=q_id,
            content_category_id=cc_5a,
            confidence=0.8,
            source="llm",
            is_overridden=True,
        )
        s.add_all([manual_tag, llm_tag, llm_overridden_tag])
        await s.commit()

    async with AsyncSession(test_engine) as s:
        await s.execute(text("DELETE FROM question_tags WHERE source = 'llm'"))
        await s.execute(text("UPDATE questions SET needs_categorization = true"))
        await s.commit()

    async with AsyncSession(test_engine) as s:
        tags = (
            (await s.execute(select(QuestionTag).where(QuestionTag.question_id == q_id)))
            .scalars()
            .all()
        )
        assert len(tags) == 1, f"Expected 1 tag (manual), got {len(tags)}"
        assert tags[0].source == "manual"

        q_row = (await s.execute(select(Question).where(Question.id == q_id))).scalar_one()
        assert q_row.needs_categorization is True

        # Cleanup so the test is idempotent across runs
        await s.delete(tags[0])
        await s.delete(q_row)
        await s.commit()
