"""Tests for check_under_specific_categorizations script — Ticket 6.8."""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag
from app.models.outline import ContentCategory, Topic
from scripts.check_under_specific_categorizations import run


async def _make_question(session: AsyncSession, qid: str) -> Question:
    q = Question(
        qid=qid,
        passage_id=None,
        stem_html="<p>Test</p>",
        stem_plain="Test stem",
        choices=[{"key": "A", "html": "<p>a</p>", "plain": "a", "media_content_hashes": []}],
        correct_choice="A",
        explanation_html=None,
        explanation_plain=None,
        uworld_aamc_tags=["Subject: General Chemistry"],
        needs_categorization=False,
    )
    session.add(q)
    await session.flush()
    return q


async def _cleanup_question(session: AsyncSession, qid: str) -> None:
    q = (await session.execute(select(Question).where(Question.qid == qid))).scalar_one_or_none()
    if q:
        await session.execute(delete(QuestionTag).where(QuestionTag.question_id == q.id))
        await session.delete(q)
        await session.commit()


async def _solubility_parent_id(session: AsyncSession) -> int:
    return (
        await session.execute(
            select(Topic.id)
            .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
            .where(
                Topic.name == "Solubility",
                ContentCategory.code == "5A",
                Topic.parent_topic_id.is_(None),
            )
        )
    ).scalar_one()


async def _ksp_child_id(session: AsyncSession) -> int:
    return (
        await session.execute(
            select(Topic.id)
            .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
            .where(
                Topic.name == "Solubility product constant; the equilibrium expression Ksp",
                ContentCategory.code == "5A",
            )
        )
    ).scalar_one()


async def test_check_script_flags_parent_without_child(seeded_report, test_engine, capsys):
    async with AsyncSession(test_engine) as s:
        q = await _make_question(s, "q-check-script-01")
        parent_id = await _solubility_parent_id(s)
        s.add(
            QuestionTag(
                question_id=q.id,
                topic_id=parent_id,
                confidence=0.9,
                source="llm",
                is_overridden=False,
            )
        )
        await s.commit()

    async with AsyncSession(test_engine) as s:
        flagged = await run(s)

    out = capsys.readouterr().out
    assert "q-check-script-01" in out
    assert "Solubility" in out
    assert flagged > 0

    async with AsyncSession(test_engine) as s:
        await _cleanup_question(s, "q-check-script-01")


async def test_check_script_does_not_flag_when_child_also_tagged(
    seeded_report, test_engine, capsys
):
    async with AsyncSession(test_engine) as s:
        q = await _make_question(s, "q-check-script-02")
        parent_id = await _solubility_parent_id(s)
        child_id = await _ksp_child_id(s)
        s.add(
            QuestionTag(
                question_id=q.id,
                topic_id=parent_id,
                confidence=0.9,
                source="llm",
                is_overridden=False,
            )
        )
        s.add(
            QuestionTag(
                question_id=q.id,
                topic_id=child_id,
                confidence=0.95,
                source="llm",
                is_overridden=False,
            )
        )
        await s.commit()

    async with AsyncSession(test_engine) as s:
        await run(s)

    out = capsys.readouterr().out
    assert "q-check-script-02" not in out

    async with AsyncSession(test_engine) as s:
        await _cleanup_question(s, "q-check-script-02")


async def test_check_script_ignores_leaf_topics(seeded_report, test_engine, capsys):
    async with AsyncSession(test_engine) as s:
        leaf_id = await _ksp_child_id(s)
        child_count = (
            await s.execute(
                select(func.count()).select_from(Topic).where(Topic.parent_topic_id == leaf_id)
            )
        ).scalar_one()
        if child_count > 0:
            return  # Not a leaf in this DB; skip gracefully

        q = await _make_question(s, "q-check-script-03")
        s.add(
            QuestionTag(
                question_id=q.id,
                topic_id=leaf_id,
                confidence=0.9,
                source="llm",
                is_overridden=False,
            )
        )
        await s.commit()

    async with AsyncSession(test_engine) as s:
        await run(s)

    out = capsys.readouterr().out
    assert "q-check-script-03" not in out

    async with AsyncSession(test_engine) as s:
        await _cleanup_question(s, "q-check-script-03")
