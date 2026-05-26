from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from app.models.captures import Attempt, Question, QuestionTag
from app.models.outline import Topic, ContentCategory


def get_relative_time(dt: datetime) -> str:
    """Format datetime as relative time."""
    now = datetime.now(timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    diff = now - dt_utc

    if diff.total_seconds() < 60:
        return "Just now"
    if diff.total_seconds() < 3600:
        mins = int(diff.total_seconds() / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if diff.total_seconds() < 86400:
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(diff.total_seconds() / 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


async def get_recent_activity(session: AsyncSession, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent attempts with their question, tag, and label."""
    recent_attempts_query = select(Attempt).order_by(Attempt.attempted_at.desc()).limit(limit)
    recent_result = await session.execute(recent_attempts_query)
    recent_attempts = list(recent_result.scalars().all())

    recent_activity = []
    if recent_attempts:
        q_ids = [a.question_id for a in recent_attempts]

        tags_query = (
            select(QuestionTag, Question)
            .join(Question, QuestionTag.question_id == Question.id)
            .where(QuestionTag.question_id.in_(q_ids))
            .order_by(QuestionTag.question_id, QuestionTag.confidence.desc())
        )
        tags_result = await session.execute(tags_query)
        tags_data = list(tags_result.all())

        q_tags = {}
        q_obj = {}
        for tag, question in tags_data:
            if tag.question_id not in q_tags:
                q_tags[tag.question_id] = tag
                q_obj[tag.question_id] = question

        for attempt in recent_attempts:
            tag = q_tags.get(attempt.question_id)
            question = q_obj.get(attempt.question_id)

            recent_activity.append(
                {
                    "attempt": attempt,
                    "relative_time": get_relative_time(attempt.attempted_at),
                    "question": question,
                    "tag": tag,
                    "label": "TBD",
                }
            )

    for activity in recent_activity:
        tag = activity["tag"]
        if not tag:
            activity["label"] = "Uncategorized"
            continue

        if tag.topic_id:
            t = await session.get(Topic, tag.topic_id)
            activity["label"] = t.name if t else f"Topic {tag.topic_id}"
        elif tag.content_category_id:
            cc = await session.get(ContentCategory, tag.content_category_id)
            activity["label"] = cc.name if cc else f"CC {tag.content_category_id}"
        elif tag.skill:
            activity["label"] = f"Skill {tag.skill}"
        else:
            activity["label"] = "Uncategorized"

    return recent_activity
