from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Attempt,
    ContentCategory,
    Media,
    Passage,
    Question,
    QuestionTag,
    RawCapture,
    Topic,
)
from app.web.dashboard.services.html_rewriter import rewrite_media_refs
from app.web.viewer.db import get_session
from app.web.viewer.services.html_rewriter import rewrite_choice_html
from app.web.viewer.services.refs import media_by_hash_for_question

router = APIRouter()


def _subject_from_tags(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    for t in tags:
        if isinstance(t, str) and t.startswith("Subject: "):
            return t[len("Subject: ") :].strip()
    return None


def _taxonomy_kv(tags: list[str] | None) -> list[tuple[str, str]]:
    if not tags:
        return []
    out: list[tuple[str, str]] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        if ": " in t:
            k, v = t.split(": ", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append(("(no field)", t.strip()))
    return out


async def _aamc_summaries_for_qids(session: AsyncSession, qids: list[int]) -> dict[int, list[str]]:
    """Compact AAMC labels per question for the list view (e.g. ['4A', '5D / Work', 'S2'])."""
    if not qids:
        return {}
    rows = (
        await session.execute(
            select(
                QuestionTag.question_id,
                QuestionTag.topic_id,
                QuestionTag.content_category_id,
                QuestionTag.skill,
            ).where(QuestionTag.question_id.in_(qids))
        )
    ).all()
    if not rows:
        return {}

    topic_ids = {r.topic_id for r in rows if r.topic_id is not None}
    cc_ids = {r.content_category_id for r in rows if r.content_category_id is not None}

    topics: dict[int, Topic] = {}
    if topic_ids:
        topics = {
            t.id: t
            for t in (await session.execute(select(Topic).where(Topic.id.in_(topic_ids)))).scalars()
        }
        cc_ids |= {t.content_category_id for t in topics.values()}

    cc_codes: dict[int, str] = {}
    if cc_ids:
        cc_codes = {
            cc.id: cc.code
            for cc in (
                await session.execute(select(ContentCategory).where(ContentCategory.id.in_(cc_ids)))
            ).scalars()
        }

    by_qid: dict[int, list[str]] = {}
    for qid, topic_id, cc_id, skill in rows:
        bucket = by_qid.setdefault(qid, [])
        if topic_id is not None:
            topic = topics.get(topic_id)
            if topic is not None:
                cc_code = cc_codes.get(topic.content_category_id, "?")
                bucket.append(f"{cc_code} / {topic.name}")
        elif cc_id is not None:
            bucket.append(cc_codes.get(cc_id, "?"))
        elif skill is not None:
            bucket.append(f"S{skill}")
    return by_qid


async def _aamc_tags_for_question(session: AsyncSession, question_id: int) -> list[dict[str, Any]]:
    """Full QuestionTag rendering for the detail view (one entry per row)."""
    tag_rows = (
        (
            await session.execute(
                select(QuestionTag)
                .where(QuestionTag.question_id == question_id)
                .order_by(QuestionTag.source, QuestionTag.id)
            )
        )
        .scalars()
        .all()
    )
    if not tag_rows:
        return []

    topic_ids = {t.topic_id for t in tag_rows if t.topic_id is not None}
    cc_ids = {t.content_category_id for t in tag_rows if t.content_category_id is not None}

    topics: dict[int, Topic] = {}
    if topic_ids:
        topics = {
            t.id: t
            for t in (await session.execute(select(Topic).where(Topic.id.in_(topic_ids)))).scalars()
        }
        cc_ids |= {t.content_category_id for t in topics.values()}

    ccs: dict[int, ContentCategory] = {}
    if cc_ids:
        ccs = {
            cc.id: cc
            for cc in (
                await session.execute(select(ContentCategory).where(ContentCategory.id.in_(cc_ids)))
            ).scalars()
        }

    out: list[dict[str, Any]] = []
    for t in tag_rows:
        entry: dict[str, Any] = {
            "tag_id": t.id,
            "source": t.source,
            "confidence": float(t.confidence),
            "rationale": t.rationale,
            "extractor_version": t.extractor_version,
        }
        if t.topic_id is not None:
            topic = topics.get(t.topic_id)
            cc = ccs.get(topic.content_category_id) if topic else None
            entry["kind"] = "topic"
            entry["cc_code"] = cc.code if cc else None
            entry["cc_name"] = cc.name if cc else None
            entry["label"] = topic.name if topic else f"topic#{t.topic_id}"
        elif t.content_category_id is not None:
            cc = ccs.get(t.content_category_id)
            entry["kind"] = "content_category"
            entry["cc_code"] = cc.code if cc else None
            entry["cc_name"] = cc.name if cc else None
            entry["label"] = cc.name if cc else f"cc#{t.content_category_id}"
        elif t.skill is not None:
            entry["kind"] = "skill"
            entry["cc_code"] = None
            entry["cc_name"] = None
            entry["label"] = f"Skill {t.skill}"
        else:
            entry["kind"] = "unknown"
            entry["label"] = "(no target)"
        out.append(entry)
    return out


def _has_media(q: Question) -> bool:
    for c in q.choices or []:
        if c.get("media_content_hashes") or c.get("media_ids"):
            return True
    for html in (q.stem_html, q.explanation_html):
        if html and "data-media-content-hash=" in html:
            return True
    return False


@router.get("/captures", response_class=HTMLResponse)
async def list_captures(
    request: Request,
    page: int = 1,
    per_page: int = 50,
    subject: str | None = None,
    has_passage: str | None = None,
    needs_categorization: str | None = None,
    since: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    page = max(1, page)
    per_page = max(1, min(per_page, 200))

    stmt = select(Question).order_by(desc(Question.last_updated_at), desc(Question.id))
    count_stmt = select(func.count(Question.id))

    if subject:
        tag_value = f"Subject: {subject}"
        stmt = stmt.where(Question.uworld_aamc_tags.cast(JSONB).contains([tag_value]))
        count_stmt = count_stmt.where(Question.uworld_aamc_tags.cast(JSONB).contains([tag_value]))
    if has_passage == "true":
        stmt = stmt.where(Question.passage_id.is_not(None))
        count_stmt = count_stmt.where(Question.passage_id.is_not(None))
    elif has_passage == "false":
        stmt = stmt.where(Question.passage_id.is_(None))
        count_stmt = count_stmt.where(Question.passage_id.is_(None))
    if needs_categorization == "true":
        stmt = stmt.where(Question.needs_categorization.is_(True))
        count_stmt = count_stmt.where(Question.needs_categorization.is_(True))
    elif needs_categorization == "false":
        stmt = stmt.where(Question.needs_categorization.is_(False))
        count_stmt = count_stmt.where(Question.needs_categorization.is_(False))
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
            stmt = stmt.where(Question.first_seen_at >= since_dt)
            count_stmt = count_stmt.where(Question.first_seen_at >= since_dt)
        except ValueError:
            pass

    total = (await session.execute(count_stmt)).scalar_one()
    stmt = stmt.limit(per_page).offset((page - 1) * per_page)
    questions = (await session.execute(stmt)).scalars().all()

    qids = [q.id for q in questions]
    attempt_counts: dict[int, int] = {}
    if qids:
        rows = await session.execute(
            select(Attempt.question_id, func.count(Attempt.id))
            .where(Attempt.question_id.in_(qids))
            .group_by(Attempt.question_id)
        )
        attempt_counts = {qid: n for qid, n in rows.all()}

    aamc_by_qid = await _aamc_summaries_for_qids(session, qids)

    rows = []
    for q in questions:
        rows.append(
            {
                "id": q.id,
                "qid": q.qid,
                "subject": _subject_from_tags(q.uworld_aamc_tags) or "",
                "n_attempts": attempt_counts.get(q.id, 0),
                "n_choices": len(q.choices or []),
                "has_passage": q.passage_id is not None,
                "has_media": _has_media(q),
                "needs_categorization": q.needs_categorization,
                "first_seen_at": q.first_seen_at,
                "last_updated_at": q.last_updated_at,
                "aamc_summary": aamc_by_qid.get(q.id, []),
            }
        )

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "captures_list.html",
        {
            "rows": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, (total + per_page - 1) // per_page),
            "filters": {
                "subject": subject or "",
                "has_passage": has_passage or "",
                "needs_categorization": needs_categorization or "",
                "since": since or "",
            },
        },
    )


@router.get("/captures/{question_id}", response_class=HTMLResponse)
async def detail(
    request: Request,
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    q = await session.get(Question, question_id)
    if q is None:
        raise HTTPException(status_code=404, detail="not found")

    passage = None
    passage_siblings: list[Question] = []
    if q.passage_id is not None:
        passage = await session.get(Passage, q.passage_id)
        sib_rows = await session.execute(
            select(Question.id, Question.qid)
            .where(Question.passage_id == q.passage_id, Question.id != q.id)
            .order_by(Question.qid)
        )
        passage_siblings = [{"id": r.id, "qid": r.qid} for r in sib_rows]

    attempts = (
        (
            await session.execute(
                select(Attempt)
                .where(Attempt.question_id == q.id)
                .order_by(desc(Attempt.attempted_at))
            )
        )
        .scalars()
        .all()
    )

    selected = attempts[0].selected_choice if attempts else None

    media_map = await media_by_hash_for_question(session, q.id)

    all_media_ids: set[int] = set()
    for c in q.choices or []:
        for mid in c.get("media_ids") or []:
            if isinstance(mid, int):
                all_media_ids.add(mid)

    media_by_id: dict[int, Media] = {}
    if all_media_ids:
        rows = await session.execute(select(Media).where(Media.id.in_(all_media_ids)))
        media_by_id = {m.id: m for m in rows.scalars().all()}

    media_rows: list[Media] = []
    if media_map:
        rows = await session.execute(select(Media).where(Media.content_hash.in_(media_map.keys())))
        media_rows = list(rows.scalars().all())
    for m in media_by_id.values():
        if m not in media_rows:
            media_rows.append(m)

    stem_html = rewrite_media_refs(q.stem_html or "", media_map)
    explanation_html = rewrite_media_refs(q.explanation_html or "", media_map)
    passage_html = rewrite_media_refs(passage.html if passage else "", media_map)

    rendered_choices: list[dict[str, Any]] = []
    for c in q.choices or []:
        choice_paths: list[str] = []
        for mid in c.get("media_ids") or []:
            m = media_by_id.get(mid) if isinstance(mid, int) else None
            if m is not None:
                choice_paths.append(m.local_path)
        rendered_choices.append(
            {
                "key": c.get("key") or c.get("label") or "",
                "html": rewrite_choice_html(c.get("html") or "", choice_paths),
                "plain": c.get("plain") or c.get("text") or "",
                "is_correct": (c.get("key") or c.get("label")) == q.correct_choice,
                "is_selected": (c.get("key") or c.get("label")) == selected,
                "media_ids": c.get("media_ids") or [],
            }
        )

    aamc_tags = await _aamc_tags_for_question(session, q.id)

    latest_raw = (
        await session.execute(
            select(RawCapture)
            .where(RawCapture.qid == q.qid)
            .order_by(desc(RawCapture.captured_at))
            .limit(1)
        )
    ).scalar_one_or_none()

    raw_html_preview = ""
    raw_html_has_more = False
    if latest_raw is not None:
        raw_html_preview = latest_raw.raw_html[:5000]
        raw_html_has_more = len(latest_raw.raw_html) > 5000

    raw_dump = {
        "id": q.id,
        "qid": q.qid,
        "passage_id": q.passage_id,
        "correct_choice": q.correct_choice,
        "choices": q.choices,
        "uworld_aamc_tags": q.uworld_aamc_tags,
        "needs_categorization": q.needs_categorization,
        "first_seen_at": q.first_seen_at.isoformat() if q.first_seen_at else None,
        "last_updated_at": q.last_updated_at.isoformat() if q.last_updated_at else None,
    }

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "capture_detail.html",
        {
            "q": q,
            "passage": passage,
            "passage_html": passage_html,
            "passage_siblings": passage_siblings,
            "passage_sibling_count": (len(passage_siblings) + 1) if passage else 0,
            "stem_html": stem_html,
            "explanation_html": explanation_html,
            "choices": rendered_choices,
            "attempts": attempts,
            "taxonomy": _taxonomy_kv(q.uworld_aamc_tags),
            "aamc_tags": aamc_tags,
            "media_rows": media_rows,
            "raw_html_preview": raw_html_preview,
            "raw_html_has_more": raw_html_has_more,
            "raw_capture_id": latest_raw.id if latest_raw else None,
            "raw_dump": raw_dump,
        },
    )


@router.get("/captures/{question_id}/raw-html", response_class=HTMLResponse)
async def raw_html(
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    q = await session.get(Question, question_id)
    if q is None:
        raise HTTPException(status_code=404, detail="not found")
    rc = (
        await session.execute(
            select(RawCapture)
            .where(RawCapture.qid == q.qid)
            .order_by(desc(RawCapture.captured_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    if rc is None:
        return HTMLResponse("<pre></pre>")
    from html import escape

    return HTMLResponse(f"<pre class='whitespace-pre-wrap text-xs'>{escape(rc.raw_html)}</pre>")
