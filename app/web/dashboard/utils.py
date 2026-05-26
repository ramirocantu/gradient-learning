"""Dashboard utils — T14 partial port.

`get_recent_activity` labelled tags via the dropped `Topic` / `ContentCategory`
tables and the renamed-away `QuestionTag.topic_id` / `.content_category_id` /
`.skill` columns. With canonical `QuestionTag.node_id`, the label is the
outline node's `path_of(node_id)` via `OutlineLookup`. T14 follow-up wires
that in; for now the activity rows render with a placeholder label so the
home page doesn't 500.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Attempt, Question, QuestionTag


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
    """Fetch recent attempts with their question + a best-effort tag label.

    TODO(T14 follow-up): resolve `tag.node_id` → human label via OutlineLookup.
    """
    recent_attempts_query = select(Attempt).order_by(Attempt.attempted_at.desc()).limit(limit)
    recent_result = await session.execute(recent_attempts_query)
    recent_attempts = list(recent_result.scalars().all())

    recent_activity = []
    if recent_attempts:
        q_ids = [a.question_id for a in recent_attempts]

        # First tag per question (highest-confidence schema_map/manual or any llm).
        tags_query = (
            select(QuestionTag, Question)
            .join(Question, QuestionTag.question_id == Question.id)
            .where(QuestionTag.question_id.in_(q_ids))
            .order_by(QuestionTag.question_id, QuestionTag.confidence.desc().nullslast())
        )
        tags_result = await session.execute(tags_query)
        tags_data = list(tags_result.all())

        q_tags: dict[int, QuestionTag] = {}
        q_obj: dict[int, Question] = {}
        for tag, question in tags_data:
            if tag.question_id not in q_tags:
                q_tags[tag.question_id] = tag
                q_obj[tag.question_id] = question

        for attempt in recent_attempts:
            tag = q_tags.get(attempt.question_id)
            question = q_obj.get(attempt.question_id)
            label = "Uncategorized"
            if tag is not None:
                # Node-id label resolution is a T14 follow-up; surface node_id
                # so the UI shows something stable instead of "TBD".
                label = f"Node {tag.node_id}"
            recent_activity.append(
                {
                    "attempt": attempt,
                    "relative_time": get_relative_time(attempt.attempted_at),
                    "question": question,
                    "tag": tag,
                    "label": label,
                }
            )

    return recent_activity
