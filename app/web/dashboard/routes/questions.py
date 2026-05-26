"""Standalone question detail routes (Ticket 6.6 + 6.9a + 6.9c).

GET    /questions/by-qid/{qid}          redirect to /questions/{id} for the UWorld qid
GET    /questions/{question_id}         full-page detail view by integer Question.id
GET    /questions/{question_id}/add-tag-form    inline add-tag form fragment
POST   /questions/{question_id}/add-tag         submit the add-tag form
DELETE /tags/{tag_id}                   remove a tag (hard-delete manual, soft-delete llm)
POST   /questions/{question_id}/attempts/{attempt_id}/notes        create a note (HTMX)
DELETE /questions/{question_id}/attempts/{attempt_id}/notes/{nid}  delete a note (HTMX)
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question, QuestionTag
from app.models.outline import ContentCategory, Topic
from app.services.admin_tags import (
    ManualTagConflictError,
    ManualTagValidationError,
    QuestionNotFoundError as ManualTagQuestionNotFoundError,
    TagDeleteForbiddenError,
    TagNotFoundError,
    create_manual_tag,
    delete_tag,
)
from app.services.anki.queries import list_cards_for_qid
from app.services.attempt_notes import (
    AttemptNotFoundError,
    NoteNotFoundError,
    create_note,
    delete_note,
    list_notes,
)
from app.web.dashboard.db import get_session
from app.web.dashboard.services.drilldown import (
    get_cc_info,
    get_question_detail,
    list_all_ccs,
    _tags_summaries_for,
)

router = APIRouter(prefix="/questions")
tags_router = APIRouter()


def _resolve_back_url(request: Request) -> str:
    """Return a safe path-only back URL derived from the request's Referer header.

    Falls back to /mastery when no referer is present, when the referer is
    another question detail page (avoid loops), or when the referer is malformed.
    """
    referer = request.headers.get("referer")
    if not referer:
        return "/mastery"
    try:
        parsed = urlsplit(referer)
    except ValueError:
        return "/mastery"
    path = parsed.path or "/mastery"
    if path.startswith("/questions/"):
        return "/mastery"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


@tags_router.delete("/tags/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_tag(
    tag_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        await delete_tag(session, tag_id)
        await session.commit()
    except TagNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TagDeleteForbiddenError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/by-qid/{qid}")
async def question_by_qid(
    qid: str,
    session: AsyncSession = Depends(get_session),
):
    row = (
        await session.execute(select(Question.id).where(Question.qid == qid).limit(1))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return RedirectResponse(url=f"/questions/{row}", status_code=302)


@router.get("/{question_id}/add-tag-form", response_class=HTMLResponse)
async def add_tag_form_fragment(
    request: Request,
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """Return the inline add-tag form fragment for the question detail page."""
    q = await session.get(Question, question_id)
    if q is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    all_ccs = await list_all_ccs(session)

    # Fetch existing topic_ids directly from QuestionTag (TagSummary does not expose topic_id).
    topic_id_rows = (
        (
            await session.execute(
                select(QuestionTag.topic_id)
                .where(QuestionTag.question_id == question_id)
                .where(QuestionTag.is_overridden == False)  # noqa: E712
                .where(QuestionTag.topic_id.is_not(None))
            )
        )
        .scalars()
        .all()
    )
    existing_topic_ids: set[int] = set(topic_id_rows)

    # Build full {cc_code: [{id, name, already_tagged}, ...]} map for client-side
    # filtering. ~33 CCs / ~1500 topics is small enough to ship to the browser in
    # one shot; cuts round trips and avoids HTMX orphan-<option> parser bugs.
    topic_rows = (
        await session.execute(
            select(Topic.id, Topic.name, ContentCategory.code)
            .join(ContentCategory, Topic.content_category_id == ContentCategory.id)
            .order_by(ContentCategory.code, Topic.name)
        )
    ).all()
    topics_by_cc: dict[str, list[dict]] = {}
    for t_id, t_name, cc_code in topic_rows:
        topics_by_cc.setdefault(cc_code, []).append(
            {
                "id": t_id,
                "name": t_name,
                "already_tagged": t_id in existing_topic_ids,
            }
        )

    # Derive CC codes and skills from TagSummary labels (stable format).
    tags_map = await _tags_summaries_for(session, [question_id])
    tags = tags_map.get(question_id, [])
    existing_cc_codes: set[str] = {
        t.label.split(" — ")[0] for t in tags if t.kind == "content_category" and " — " in t.label
    }
    existing_skills: set[int] = {
        int(t.label.split("Skill ")[-1]) for t in tags if t.kind == "skill"
    }

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/add_tag_form.html",
        context={
            "question_id": question_id,
            "all_ccs": all_ccs,
            "topics_by_cc": topics_by_cc,
            "existing_cc_codes": existing_cc_codes,
            "existing_skills": existing_skills,
        },
    )


@router.post("/{question_id}/add-tag", response_class=HTMLResponse)
async def add_tag_submit(
    request: Request,
    question_id: int,
    tag_kind: Annotated[str, Form()],
    cc_code: Annotated[str | None, Form()] = None,
    topic_id: Annotated[str | None, Form()] = None,
    skill: Annotated[str | None, Form()] = None,
    rationale: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """Submit the add-tag form; return the refreshed tags-section fragment on success."""
    templates = request.app.state.templates

    def _error(message: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="partials/add_tag_error.html",
            context={"message": message},
            status_code=200,
        )

    if tag_kind not in {"topic", "content_category", "skill"}:
        return _error(f"Unknown tag kind: {tag_kind!r}")

    # Resolve the one target field based on tag_kind; ignore the others.
    target_kwargs: dict = {}
    if tag_kind == "topic":
        topic_id_int = _parse_int(topic_id)
        if topic_id_int is None:
            return _error("A topic must be selected when tag kind is 'topic'.")
        target_kwargs = {"topic_id": topic_id_int}
    elif tag_kind == "content_category":
        if not cc_code:
            return _error("A content category must be selected.")
        cc = await get_cc_info(session, cc_code)
        if cc is None:
            return _error(f"Unknown content category: {cc_code!r}")
        target_kwargs = {"content_category_id": cc.id}
    else:  # skill
        skill_int = _parse_int(skill)
        if skill_int is None or skill_int not in range(1, 5):
            return _error("Skill must be an integer between 1 and 4.")
        target_kwargs = {"skill": skill_int}

    try:
        await create_manual_tag(session, question_id, rationale=rationale, **target_kwargs)
    except ManualTagConflictError:
        return _error("This tag already exists for the question.")
    except ManualTagValidationError as exc:
        return _error(str(exc))
    except ManualTagQuestionNotFoundError:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    await session.commit()

    detail = await get_question_detail(session, question_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    return templates.TemplateResponse(
        request=request,
        name="partials/tags_section.html",
        context={"tags": detail.tags, "question_id": question_id},
    )


def _parse_int(v: str | None) -> int | None:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


@router.post("/{question_id}/attempts/{attempt_id}/notes", response_class=HTMLResponse)
async def add_note_htmx(
    request: Request,
    question_id: int,
    attempt_id: int,
    note_text: Annotated[str, Form()],
    flag_for_review: Annotated[bool, Form()] = False,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    templates = request.app.state.templates
    if not note_text.strip():
        return templates.TemplateResponse(
            request=request,
            name="partials/attempt_notes_section.html",
            context={
                "notes": await list_notes(session, attempt_id=attempt_id),
                "attempt_id": attempt_id,
                "question_id": question_id,
                "error": "Note text must not be blank.",
            },
            status_code=422,
        )
    try:
        await create_note(
            session,
            attempt_id=attempt_id,
            note_text=note_text,
            flag_for_review=flag_for_review,
            source="user",
        )
    except AttemptNotFoundError:
        raise HTTPException(status_code=404, detail="attempt not found")
    await session.commit()
    notes = await list_notes(session, attempt_id=attempt_id)
    return templates.TemplateResponse(
        request=request,
        name="partials/attempt_notes_section.html",
        context={"notes": notes, "attempt_id": attempt_id, "question_id": question_id},
    )


@router.delete("/{question_id}/attempts/{attempt_id}/notes/{note_id}", response_class=HTMLResponse)
async def delete_note_htmx(
    request: Request,
    question_id: int,
    attempt_id: int,
    note_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    try:
        await delete_note(session, note_id=note_id)
    except NoteNotFoundError:
        raise HTTPException(status_code=404, detail="note not found")
    await session.commit()
    notes = await list_notes(session, attempt_id=attempt_id)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/attempt_notes_section.html",
        context={"notes": notes, "attempt_id": attempt_id, "question_id": question_id},
    )


@router.get("/{question_id}", response_class=HTMLResponse)
async def question_detail(
    request: Request,
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    detail = await get_question_detail(session, question_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    # SPEC §T26 / §V18: in-process Anki lookup for the question's UWorld qid.
    # Empty list is the normal answer for qids MileDown hasn't tagged.
    anki_cards = await list_cards_for_qid(session, qid=detail.question.qid)

    back_url = _resolve_back_url(request)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="question_detail.html",
        context={
            "detail": detail,
            "back_url": back_url,
            "anki_cards": anki_cards,
        },
    )
