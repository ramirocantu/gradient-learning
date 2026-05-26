"""Standalone question detail routes.

T14 partial port: the add-tag form's CC/topic/skill picker is gone (the
`Topic` / `ContentCategory` tables were dropped and tags are now node_id-only
per V-T1). The new add-tag UI takes a node-path (` >> `-delimited) which the
server resolves via `OutlineLookup.node_id_by_path`. The form/template/UX
rebuild is a T14 follow-up; this module keeps its route table loadable, the
notes endpoints fully functional, and the add-tag endpoints return a stub
explanation until the new UI lands.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.captures import Question
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
from app.services.categorizer.outline_lookup import OutlineLookup, OutlineNotSeededError
from app.web.dashboard.db import get_session
from app.web.dashboard.services.drilldown import get_question_detail

router = APIRouter(prefix="/questions")
tags_router = APIRouter()


def _resolve_back_url(request: Request) -> str:
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
    """Return a stub add-tag form fragment.

    TODO(T14 follow-up): the rebuilt picker accepts a `>>`-delimited node path
    (resolved server-side via OutlineLookup) instead of the retired CC/topic/
    skill triple.
    """
    q = await session.get(Question, question_id)
    if q is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")
    templates = request.app.state.templates
    # Render a minimal fragment so the dashboard doesn't 500. Real picker UX
    # ships with the dashboard SPA rebuild (SPEC §C frontend carve-out).
    return templates.TemplateResponse(
        request=request,
        name="partials/add_tag_form.html",
        context={
            "question_id": question_id,
            "all_ccs": [],
            "topics_by_cc": {},
            "existing_cc_codes": set(),
            "existing_skills": set(),
            "node_id_form_pending": True,
        },
    )


@router.post("/{question_id}/add-tag", response_class=HTMLResponse)
async def add_tag_submit(
    request: Request,
    question_id: int,
    node_path: Annotated[str | None, Form()] = None,
    rationale: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """Resolve `node_path` via OutlineLookup → create manual node_id tag."""
    templates = request.app.state.templates

    def _error(message: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="partials/add_tag_error.html",
            context={"message": message},
            status_code=200,
        )

    if not node_path or not node_path.strip():
        return _error("A node path is required (e.g. 'CP >> FC1 >> 1A >> Amino acids').")

    try:
        lookup = await OutlineLookup.load(session)
    except OutlineNotSeededError as exc:
        return _error(f"Outline not seeded: {exc}")

    node_id = lookup.node_id_by_path(node_path)
    if node_id is None:
        return _error(f"Unknown or ambiguous node path: {node_path!r}")

    try:
        await create_manual_tag(
            session, question_id, node_id=node_id, rationale=rationale
        )
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
