"""Pydantic schema for the PDF-ingest endpoint (T51)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PdfIngestResponse(BaseModel):
    """Result of one ``POST /api/v1/pdf/ingest`` — mirrors
    ``kb.pdf_ingest.IngestReport`` (vision transcription → atomic facts)."""

    model_config = ConfigDict(extra="forbid")

    pdf_source_id: int
    pages: int
    new_facts: int
    dup_facts: int
    reused_pdf: bool
    extractor_version: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
