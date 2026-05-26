from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag
from app.models.features import QuestionFeatures
from app.models.outline import ContentCategory, Topic


class QuestionNotFoundError(Exception):
    pass


class AttemptNotFoundError(Exception):
    pass


async def _build_question_payload(session: AsyncSession, q: Question) -> dict[str, Any]:
    tag_rows = (
        await session.execute(
            select(QuestionTag, Topic.name, ContentCategory.code, ContentCategory.name)
            .join(Topic, Topic.id == QuestionTag.topic_id, isouter=True)
            .join(
                ContentCategory,
                ContentCategory.id == QuestionTag.content_category_id,
                isouter=True,
            )
            .where(QuestionTag.question_id == q.id)
            .where(QuestionTag.is_overridden.is_(False))
        )
    ).all()

    tags: list[dict[str, Any]] = []
    for tag, topic_name, cc_code, cc_label in tag_rows:
        tags.append(
            {
                "topic_id": tag.topic_id,
                "topic_name": topic_name,
                "content_category_code": cc_code,
                "content_category_label": cc_label,
                "skill": tag.skill,
                "confidence": float(tag.confidence) if tag.confidence is not None else None,
                "rationale": tag.rationale,
            }
        )

    features = (
        await session.execute(select(QuestionFeatures).where(QuestionFeatures.question_id == q.id))
    ).scalar_one_or_none()

    return {
        "qid": q.qid,
        "question_id": q.id,
        "stem": q.stem_plain,
        "choices": q.choices,
        "correct_choice": q.correct_choice,
        "explanation": q.explanation_plain,
        "tags": tags,
        "features": (
            {col.name: getattr(features, col.name) for col in features.__table__.columns}
            if features is not None
            else None
        ),
    }


async def get_question(session: AsyncSession, *, qid: str) -> dict[str, Any]:
    q = (await session.execute(select(Question).where(Question.qid == qid))).scalar_one_or_none()
    if q is None:
        raise QuestionNotFoundError(qid)
    return await _build_question_payload(session, q)


async def get_question_by_attempt_id(session: AsyncSession, *, attempt_id: int) -> dict[str, Any]:
    attempt = await session.get(Attempt, attempt_id)
    if attempt is None:
        raise AttemptNotFoundError(attempt_id)
    q = await session.get(Question, attempt.question_id)
    if q is None:
        raise QuestionNotFoundError(f"attempt {attempt_id} -> missing question")
    return await _build_question_payload(session, q)
