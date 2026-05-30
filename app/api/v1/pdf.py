"""POST /api/v1/pdf/ingest — upload a lecture-notes PDF for ingestion (T51).

One-shot counterpart to the inbox poller (``run_pdf_ingest_job``): the client
uploads a PDF for an explicit ``course_id`` and we run it through the
vision-ingest pipeline (``kb/pdf_ingest.ingest_pdf``) synchronously. Facts land
in ``atomic_facts`` with ``node_id`` NULL — the grounded-tag categorizer (T50)
assigns nodes later.

The OpenAI client is built via ``build_openai_client`` (the V16 SDK-boundary
seam tests patch). The upload is written to a temp file because ``ingest_pdf``
SHA-256s + renders from a path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.models.outline import Course
from app.schemas.pdf import PdfIngestResponse
from app.services.kb.pdf_ingest import ingest_pdf
from app.services.llm.client import build_openai_client

router = APIRouter()


@router.post("/pdf/ingest", response_model=PdfIngestResponse)
async def post_pdf_ingest(
    course_id: int = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),) -> PdfIngestResponse:
    course = (
        await session.execute(select(Course).where(Course.id == course_id))
    ).scalar_one_or_none()
    if course is None:
        raise HTTPException(status_code=404, detail=f"course id={course_id} not found")

    contents = await file.read()
    filename = file.filename or "upload.pdf"
    client = build_openai_client()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(filename).name
        path.write_bytes(contents)
        report = await ingest_pdf(
            session,
            course_id=course_id,
            path=path,
            vision_client=client,
        )

    return PdfIngestResponse(
        pdf_source_id=report.pdf_source_id,
        pages=report.pages,
        new_facts=report.new_facts,
        dup_facts=report.dup_facts,
        reused_pdf=report.reused_pdf,
        extractor_version=report.extractor_version,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        cached_tokens=report.cached_tokens,
    )
