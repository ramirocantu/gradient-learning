"""Per-CC drilldown routes.

GET  /mastery/{cc_code}                                       drilldown page
GET  /mastery/{cc_code}/questions/{qid}/full                  expanded card fragment
GET  /mastery/{cc_code}/questions/{qid}/retag-form            re-tag form fragment
GET  /mastery/{cc_code}/topic-options                         topic <option> list
POST /mastery/{cc_code}/questions/{qid}/retag                 submit re-tag
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.admin import trigger_job_logic
from app.services.admin_tags import (
    ManualTagConflictError,
    ManualTagValidationError,
    QuestionNotFoundError as ManualTagQuestionNotFoundError,
    create_manual_tag,
)
from datetime import datetime, timedelta, timezone

from app.services.analytics import compute_mastery
from app.services.anki.assignment import (
    AssignmentError,
    create_assignment,
    resolve_card_ids,
)
from app.services.anki.queries import (
    list_cards_for_cc,
    list_review_queue_for_cc,
    list_review_queue_for_topic_subtree,
)
from app.web.dashboard.db import get_session
from app.web.dashboard.services.mastery import (
    cc_anki_overview,
    cc_header,
    cc_topics_tree,
    topic_anki_overview,
    topic_children_tree,
    topic_header,
    validate_topic_chain,
)
from app.web.dashboard.services.drilldown import (
    _filter_topics_for_cc,
    get_cc_info,
    get_full_question,
    get_question_card,
    get_questions_for_cc,
    get_questions_for_topic_subtree,
    list_all_ccs,
    list_question_cards,
    list_topics_for_cc,
)

router = APIRouter()

PER_PAGE = 20


# --------------------------------------------------------------------------- #
# Drilldown page
# --------------------------------------------------------------------------- #


@router.get("/mastery/{cc_code}", response_class=HTMLResponse)
async def cc_drilldown(
    request: Request,
    cc_code: str,
    page: int = 1,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    cc = await get_cc_info(session, cc_code)
    if cc is None:
        raise HTTPException(status_code=404, detail=f"unknown content category: {cc_code}")

    page = max(1, page)

    mastery = await compute_mastery(session)
    headline = next((s for s in mastery.by_content_category if s.target_id == cc.id), None)
    topic_rows = _filter_topics_for_cc(mastery.by_topic, cc.code)

    # §V30 two-axis header (UWorld Wilson+N | Anki retention_30d × unlock%).
    header = await cc_header(session, cc_code=cc.code)

    # §V34 layout — Anki state breakdown + CC-scoped review queue + topics tree.
    is_cars = cc.section_code == "CARS"
    state_breakdown = None if is_cars else await cc_anki_overview(session, cc_code=cc.code)
    due_before = datetime.now(tz=timezone.utc) + timedelta(days=1)
    review_queue = (
        []
        if is_cars
        else await list_review_queue_for_cc(
            session, cc_code=cc.code, due_before=due_before, limit=20
        )
    )
    topics_tree = await cc_topics_tree(session, cc_id=cc.id, cc_code=cc.code)

    qids = await get_questions_for_cc(session, cc.id)
    cards, total_questions = await list_question_cards(session, qids, page=page, per_page=PER_PAGE)
    pages = max(1, (total_questions + PER_PAGE - 1) // PER_PAGE)

    # SPEC §T27 / §V18: in-process Anki lookup for cards tagged under any
    # topic in this CC. Empty list is the normal "MileDown has no coverage
    # for this CC yet" answer.
    anki_cards = await list_cards_for_cc(session, cc_code=cc.code, limit=20)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="cc_drilldown.html",
        context={
            "cc": cc,
            "cc_code": cc.code,
            "is_cars": is_cars,
            "headline": headline,
            "header": header,
            "state_breakdown": state_breakdown,
            "review_queue": review_queue,
            "topics_tree": topics_tree,
            "topic_rows": topic_rows,
            "cards": cards,
            "page": page,
            "pages": pages,
            "total_questions": total_questions,
            "per_page": PER_PAGE,
            "anki_cards": anki_cards,
            "assign_scope_kind": "cc",
            "assign_scope_value": cc.code,
            "assign_redirect_to": f"/mastery/{cc.code}",
            "assign_today": datetime.now(tz=timezone.utc).date().isoformat(),
        },
    )


# --------------------------------------------------------------------------- #
# Topic drilldown page (§T45 / §V32 / §V33)
# --------------------------------------------------------------------------- #


@router.get("/mastery/{cc_code}/topics/{id_path:path}", response_class=HTMLResponse)
async def topic_drilldown(
    request: Request,
    cc_code: str,
    id_path: str,
    page: int = 1,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    # Parse id-path. §V32 requires server-side validation of the parent
    # chain — `validate_topic_chain` enforces:
    #   - every id exists,
    #   - ids[0].parent_topic_id IS NULL,
    #   - ids[0].content_category_id == cc.id,
    #   - ids[k].parent_topic_id == ids[k-1] for k > 0.
    try:
        ids = [int(x) for x in id_path.split("/") if x]
    except ValueError as e:
        raise HTTPException(status_code=404, detail="invalid topic path") from e
    if not ids:
        raise HTTPException(status_code=404, detail="empty topic path")

    chain = await validate_topic_chain(session, cc_code=cc_code, ids=ids)
    if chain is None:
        raise HTTPException(status_code=404, detail="invalid topic chain")
    topic_chain, breadcrumb = chain
    leaf = topic_chain[-1]

    # §V33 — subtree-scoped surfaces.
    header = await topic_header(session, cc_code=cc_code, topic=leaf)
    state_breakdown = await topic_anki_overview(session, topic_id=leaf.id)
    due_before = datetime.now(tz=timezone.utc) + timedelta(days=1)
    review_queue = await list_review_queue_for_topic_subtree(
        session, topic_id=leaf.id, due_before=due_before, limit=20
    )
    children_tree = await topic_children_tree(session, cc_code=cc_code, root_topic_id=leaf.id)

    # Q-grid scoped to subtree.
    page = max(1, page)
    qids = await get_questions_for_topic_subtree(session, leaf.id)
    cards, total_questions = await list_question_cards(session, qids, page=page, per_page=PER_PAGE)
    pages = max(1, (total_questions + PER_PAGE - 1) // PER_PAGE)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="topic_drilldown.html",
        context={
            "cc_code": cc_code,
            "topic_chain": topic_chain,
            "leaf": leaf,
            "breadcrumb": breadcrumb,
            "header": header,
            "state_breakdown": state_breakdown,
            "review_queue": review_queue,
            "children_tree": children_tree,
            "cards": cards,
            "page": page,
            "pages": pages,
            "total_questions": total_questions,
            "per_page": PER_PAGE,
            "assign_scope_kind": "topic",
            "assign_scope_value": str(leaf.id),
            "assign_redirect_to": f"/mastery/{cc_code}/topics/{id_path}",
            "assign_today": datetime.now(tz=timezone.utc).date().isoformat(),
        },
    )


# --------------------------------------------------------------------------- #
# Show-full fragment
# --------------------------------------------------------------------------- #


@router.get("/mastery/{cc_code}/questions/{question_id}/full", response_class=HTMLResponse)
async def question_full_fragment(
    request: Request,
    cc_code: str,
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    cc = await get_cc_info(session, cc_code)
    if cc is None:
        raise HTTPException(status_code=404, detail=f"unknown content category: {cc_code}")

    full = await get_full_question(session, question_id)
    if full is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    card = await get_question_card(session, question_id)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/question_full.html",
        context={"cc_code": cc.code, "full": full, "card": card},
    )


# --------------------------------------------------------------------------- #
# Collapsed card fragment (for the Collapse button on the expanded view)
# --------------------------------------------------------------------------- #


@router.get(
    "/mastery/{cc_code}/questions/{question_id}/card",
    response_class=HTMLResponse,
)
async def question_card_fragment(
    request: Request,
    cc_code: str,
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    cc = await get_cc_info(session, cc_code)
    if cc is None:
        raise HTTPException(status_code=404, detail=f"unknown content category: {cc_code}")

    card = await get_question_card(session, question_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/question_card.html",
        context={"cc_code": cc.code, "card": card},
    )


# --------------------------------------------------------------------------- #
# Re-tag form fragment
# --------------------------------------------------------------------------- #


@router.get(
    "/mastery/{cc_code}/questions/{question_id}/retag-form",
    response_class=HTMLResponse,
)
async def retag_form_fragment(
    request: Request,
    cc_code: str,
    question_id: int,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    cc = await get_cc_info(session, cc_code)
    if cc is None:
        raise HTTPException(status_code=404, detail=f"unknown content category: {cc_code}")

    card = await get_question_card(session, question_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    all_ccs = await list_all_ccs(session)
    initial_topics = await list_topics_for_cc(session, cc.code)

    existing_cc_codes = {
        t.label.split(" — ")[0]
        for t in card.tags
        if t.kind == "content_category" and " — " in t.label
    }
    existing_skills = {int(t.label.split("Skill ")[-1]) for t in card.tags if t.kind == "skill"}

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/retag_form.html",
        context={
            "cc_code": cc.code,
            "question_id": question_id,
            "all_ccs": all_ccs,
            "initial_topics": initial_topics,
            "card": card,
            "existing_cc_codes": existing_cc_codes,
            "existing_skills": existing_skills,
        },
    )


# --------------------------------------------------------------------------- #
# Topic options fragment (HTMX dynamic dropdown)
# --------------------------------------------------------------------------- #


@router.get("/mastery/{cc_code}/topic-options", response_class=HTMLResponse)
async def topic_options_fragment(
    request: Request,
    cc_code: str,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """Render <option> rows for the topic dropdown.

    The path's ``cc_code`` is the page context. The query param ``cc_code``
    (read directly off the request) is the *currently selected* CC inside the
    form — typically the CC the user just picked. Fall back to the page's CC
    when no query value is provided.
    """
    target_code = request.query_params.get("cc_code") or cc_code
    topics = await list_topics_for_cc(session, target_code)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/topic_options.html",
        context={"topics": topics},
    )


# --------------------------------------------------------------------------- #
# Re-tag submit
# --------------------------------------------------------------------------- #


@router.post(
    "/mastery/{cc_code}/questions/{question_id}/retag",
    response_class=HTMLResponse,
)
async def retag_submit(
    request: Request,
    cc_code: str,
    question_id: int,
    target_cc_code: Annotated[str, Form()],
    target_topic_id: Annotated[str | None, Form()] = None,
    target_skill: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    page_cc = await get_cc_info(session, cc_code)
    if page_cc is None:
        raise HTTPException(status_code=404, detail=f"unknown content category: {cc_code}")

    target_cc = await get_cc_info(session, target_cc_code)
    if target_cc is None:
        return _retag_error(request, page_cc.code, question_id, "Unknown content category.")

    topic_id_int = _parse_int(target_topic_id)
    skill_int = _parse_int(target_skill)

    targets: list[dict] = [{"content_category_id": target_cc.id}]
    if topic_id_int is not None:
        targets.append({"topic_id": topic_id_int})
    if skill_int is not None:
        targets.append({"skill": skill_int})

    created_any = False
    for kwargs in targets:
        try:
            await create_manual_tag(session, question_id, **kwargs)
            created_any = True
        except ManualTagConflictError:
            # Skip duplicates silently; surface only if nothing got created.
            continue
        except ManualTagValidationError as exc:
            return _retag_error(request, page_cc.code, question_id, str(exc))
        except ManualTagQuestionNotFoundError:
            raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    await session.commit()

    if not created_any:
        return _retag_error(
            request,
            page_cc.code,
            question_id,
            "All requested manual tags already exist.",
        )

    refreshed = await get_question_card(session, question_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail=f"question_id={question_id} not found")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/question_card.html",
        context={"cc_code": page_cc.code, "card": refreshed},
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


def _retag_error(request: Request, cc_code: str, question_id: int, message: str) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="partials/retag_error.html",
        context={"cc_code": cc_code, "question_id": question_id, "message": message},
        status_code=200,
    )


# --------------------------------------------------------------------------- #
# Assign cards (Anki state & retention widget — CC + topic mastery pages)
# --------------------------------------------------------------------------- #

_ASSIGN_PRIORITY = "most_specific_first"


def _safe_internal_path(target: str | None, fallback: str) -> str:
    """Only redirect back to an in-app absolute path. Rejects schema-relative
    (`//host`) and external URLs to close an open-redirect on the form's
    user-controllable `redirect_to` field."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target.split("?", 1)[0]
    return fallback


def _parse_unlock_date(raw: str | None) -> datetime:
    """`YYYY-MM-DD` → midnight-UTC datetime. Falls back to now() on blank or
    unparseable input (a today-or-past date makes the hourly unlock job fire
    on its next tick)."""
    if raw:
        try:
            return datetime.strptime(raw.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)


@router.post("/mastery/assign")
async def create_assignment_dashboard(
    scope_kind: Annotated[str, Form()],
    scope_value: Annotated[str, Form()],
    redirect_to: Annotated[str, Form()],
    unlock_date: Annotated[str | None, Form()] = None,
    max_cards: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    """In-process (V18) create-assignment from the mastery widget. Snapshots
    the suspended-card set for the scope and schedules the unsuspend for the
    chosen date; the AnkiConnect side-effect runs in the T63 unlock job."""
    dest = _safe_internal_path(redirect_to, "/mastery")
    if scope_kind not in ("cc", "topic"):
        return RedirectResponse(f"{dest}?assign_error=scope", status_code=303)

    max_n = _parse_int(max_cards)
    scheduled = _parse_unlock_date(unlock_date)

    # Guard: don't create an empty (no-op) assignment when no suspended cards
    # match the scope. resolve is deterministic for most_specific_first, so
    # this count matches the snapshot create_assignment takes below.
    try:
        candidate_ids = await resolve_card_ids(
            session,
            scope_kind=scope_kind,
            scope_value=scope_value,
            priority=_ASSIGN_PRIORITY,
            max_cards=max_n,
        )
    except AssignmentError:
        return RedirectResponse(f"{dest}?assign_error=scope", status_code=303)
    if not candidate_ids:
        return RedirectResponse(f"{dest}?assign_none=1", status_code=303)

    try:
        row = await create_assignment(
            session,
            scope_kind=scope_kind,
            scope_value=scope_value,
            scheduled_unlock_at=scheduled,
            max_cards=max_n,
            priority=_ASSIGN_PRIORITY,
        )
    except AssignmentError:
        return RedirectResponse(f"{dest}?assign_error=scope", status_code=303)
    await session.commit()

    # Nudge the T63 unlock job now instead of waiting up to a full
    # ANKI_ASSIGNMENT_UNLOCK_INTERVAL_MINUTES tick — a due-now assignment then
    # unsuspends within seconds of the click. The job itself filters on
    # scheduled_unlock_at <= now, so a future-dated assignment is a harmless
    # no-op. Best-effort: the assignment is already durably committed, so
    # swallow the same 409 (already in-flight) / 503 (scheduler off, e.g. in
    # tests) the /admin trigger tolerates — the interval job picks it up
    # regardless. The hardcoded job name is in _VALID_JOBS, so 404 is unreachable.
    try:
        await trigger_job_logic("run_anki_assignment_unlock")
    except HTTPException:
        pass
    return RedirectResponse(f"{dest}?assigned={len(row.card_ids)}", status_code=303)
